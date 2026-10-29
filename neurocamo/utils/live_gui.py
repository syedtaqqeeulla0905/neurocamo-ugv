"""
NeuroCamo - Live Rover Operator Console
=======================================
Streams a webcam (or a video file) through the spiking neural network and marks
camouflaged targets live with corner brackets.

    python -m utils.live_gui
    python -m utils.live_gui --video path\\to\\clip.mp4
    python -m utils.live_gui --camera 1

Threading model: capture + SNN inference live on a worker thread, results are
handed to the tkinter thread through a queue.Queue that a root.after() pump
drains. Only the UI thread ever touches widgets or builds a PhotoImage.
"""

import os
import sys
import time
import queue
import argparse
import threading
import collections

import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Add project root to sys.path so `models` / `utils` resolve when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.poisson_encoder import PoissonEncoder
from models.attention_bridge import NeuroCamoFullModel

ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DETECT_EVERY = 3      # run the SNN on every 3rd frame; reuse the last box between
INPUT_SIZE   = 128    # network input resolution
BBOX_EXPAND  = 3.0    # head w/h are small (trained on 128x128 crops) -> widen
BBOX_CAP     = 0.85   # never let a box swallow the whole frame

# Operator-console palette (BGR for cv2, hex for tk)
ACCENT_BGR = (60, 235, 90)     # green - DETECTED
MUTED_BGR  = (150, 150, 150)   # grey  - nothing found
BG_HEX     = "#07090c"
BAR_HEX    = "#111820"
TXT_HEX    = "#c9d4de"
ACC_HEX    = "#3ceb5a"


# ── Model loading (v2 first, always) ──────────────────────────────────────────
def resolve_checkpoint():
    """Prefer best_model_v2.pth; only fall back to best_model.pth if it is absent."""
    v2 = os.path.join(ROOT_DIR, "checkpoints", "best_model_v2.pth")
    v1 = os.path.join(ROOT_DIR, "checkpoints", "best_model.pth")
    if os.path.exists(v2):
        return v2, "best_model_v2.pth"
    if os.path.exists(v1):
        return v1, "best_model.pth"
    return None, "none (random weights)"


CKPT_PATH, CKPT_NAME = resolve_checkpoint()

encoder = PoissonEncoder(timesteps=16, gain=1.0).to(DEVICE)
model   = NeuroCamoFullModel(beta=0.9).to(DEVICE)

if CKPT_PATH:
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[OK] Loaded {CKPT_NAME} - epoch {ckpt.get('epoch', 1)}, "
          f"val_loss {ckpt.get('val_loss', 0.0):.4f}")
else:
    print("[WARN] No checkpoint found - running with random weights.")

model.eval()


# ── Scene filter (copied verbatim from utils/test_gui.py) ─────────────────────
TEXTURE_LOW_CUTOFF  = 150.0
TEXTURE_HIGH_CUTOFF = 400.0

def texture_score(bgr_img):
    """Laplacian variance = how complex/organic the image texture is."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())

def background_is_plain(bgr_img, corner_size=40, threshold=5.0):
    """True ONLY for artificial studio product photos (e.g. solid white/black paper bg).
    Checks all 4 corners with strict std < 5.0 and near 0 variance."""
    h, w = bgr_img.shape[:2]
    cs = corner_size
    corners = [
        bgr_img[0:cs, 0:cs],
        bgr_img[0:cs, w-cs:w],
        bgr_img[h-cs:h, 0:cs],
        bgr_img[h-cs:h, w-cs:w],
    ]
    avg_std = float(np.mean([c.std() for c in corners]))
    return avg_std < threshold

def scene_weight(bgr_img, tex):
    """Return 1.0 to let the SNN model speak directly. Suppress ONLY studio product photos."""
    if background_is_plain(bgr_img):
        return 0.0
    return 1.0


def imread_unicode(path):
    """cv2.imread returns None for non-ASCII Windows paths; this does not."""
    data = np.fromfile(path, np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ── Worker: capture + inference, never touches a widget ───────────────────────
class VisionWorker(threading.Thread):
    """Reads video frames, maintains a rolling 16-frame temporal sequence buffer, and runs SNN inference."""

    def __init__(self, source, out_q, is_video=False):
        super().__init__(daemon=True)
        self.source        = source
        self.out_q         = out_q
        self.is_video      = is_video
        self._stop         = threading.Event()
        self._paused       = threading.Event()
        self.last_det      = None
        self.seek_request  = None
        self.current_frame = 0
        self.total_frames  = 0
        self.video_fps     = 25.0
        self.frame_buffer  = collections.deque(maxlen=16)  # T=16 temporal sequence buffer

    def stop(self):
        self._stop.set()
        self._paused.clear()

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    @property
    def is_paused(self):
        return self._paused.is_set()

    def seek(self, seconds_delta):
        target = self.current_frame + int(seconds_delta * self.video_fps)
        target = max(0, min(self.total_frames - 1, target))
        self.seek_request = target

    def seek_to_fraction(self, frac):
        target = int(frac * self.total_frames)
        target = max(0, min(self.total_frames - 1, target))
        self.seek_request = target

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self._emit_error(f"Could not open {'video file' if self.is_video else f'camera index {self.source}'}")
            return

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0

        idx, t_prev, fps = 0, time.time(), 0.0
        while not self._stop.is_set():
            while self._paused.is_set() and not self._stop.is_set():
                if self.seek_request is not None:
                    break
                time.sleep(0.03)

            if self.seek_request is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_request)
                self.seek_request = None
                self.last_det = None
                self.prev_box = None
                self.frame_buffer.clear()

            ok, frame = cap.read()
            if not ok:
                if self.is_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame = 0
                    self.frame_buffer.clear()
                    continue
                self._emit_error("Camera stopped delivering frames")
                break

            self.current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Preprocess frame and add to rolling T=16 sequence buffer
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
            tensor_frame = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # (1, 3, 128, 128)

            # Fill buffer initially with first frame
            if len(self.frame_buffer) == 0:
                for _ in range(16):
                    self.frame_buffer.append(tensor_frame)
            else:
                self.frame_buffer.append(tensor_frame)

            if idx % DETECT_EVERY == 0:
                self.last_det = self._infer_sequence(frame)
            idx += 1

            now = time.time()
            dt  = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.8 * fps + 0.2 * (1.0 / dt) if fps else 1.0 / dt

            self._push({"frame": frame, "det": self.last_det, "fps": fps,
                        "pos": self.current_frame, "total": self.total_frames})

        cap.release()

    def _infer_sequence(self, frame_bgr):
        """One SNN pass, IDENTICAL to the working image path in test_gui.py.

        No temporal buffer, no spike-centroid blending, no size clamp — those were
        pulling the box off-target. We trust the trained regression head exactly as
        the image tester does, then expand 3x (capped at 0.85) to cover the target.
        """
        rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
        tensor  = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            spikes = encoder(tensor)                                   # (T, B, 3, 128, 128)
            bbox_pred, conf_pred, _class_logits, sparsity = model(spikes)  # class_logits is None (binary)

        xc, yc, bw, bh = [float(v) for v in bbox_pred[0].cpu().numpy()]
        snn_conf = float(conf_pred[0].item())
        sparsity = float(sparsity)

        # Expand head box (small from 128x128 training) to cover the full target
        bw = min(bw * BBOX_EXPAND, BBOX_CAP)
        bh = min(bh * BBOX_EXPAND, BBOX_CAP)

        tex = texture_score(frame_bgr)
        return {
            "snn_conf":  snn_conf,
            "weight":    scene_weight(frame_bgr, tex),
            "sparsity":  sparsity,
            "tex":       tex,
            "norm_box":  (xc, yc, bw, bh),
        }


    def _push(self, payload):
        """Keep the queue shallow so the UI always shows the newest frame."""
        if self.out_q.full():
            try:
                self.out_q.get_nowait()
            except queue.Empty:
                pass
        try:
            self.out_q.put_nowait(payload)
        except queue.Full:
            pass

    def _emit_error(self, msg):
        self._push({"error": msg})


# ── Tkinter GUI: Operator Console ─────────────────────────────────────────────
class NeuroCamoLiveGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuroCamo — Live Rover Operator Console")
        self.root.geometry("1100x720")
        self.root.configure(bg=BG_HEX)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.queue  = None
        self.worker = None

        self._build_header()
        self._build_connect_panel()   # Show connection panel first

    # ── Header (always visible) ────────────────────────────────────────────────
    def _build_header(self):
        header = tk.Frame(self.root, bg=BAR_HEX, height=56)
        header.pack(fill=tk.X)
        tk.Label(header, text="⚡  NeuroCamo Live Console",
                 font=("Consolas", 16, "bold"), fg=ACC_HEX, bg=BAR_HEX
                 ).pack(side=tk.LEFT, padx=18, pady=14)
        info = f"Device: {DEVICE}  |  Model: {CKPT_NAME}"
        tk.Label(header, text=info, font=("Consolas", 10),
                 fg=TXT_HEX, bg=BAR_HEX).pack(side=tk.RIGHT, padx=18)

    # ── Connection Panel ───────────────────────────────────────────────────────
    def _build_connect_panel(self):
        self.connect_frame = tk.Frame(self.root, bg=BG_HEX)
        self.connect_frame.pack(expand=True, fill=tk.BOTH)

        # Card
        card = tk.Frame(self.connect_frame, bg=BAR_HEX,
                        highlightbackground="#2d3748", highlightthickness=1)
        card.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=560, height=430)

        tk.Label(card, text="📡  Select Input Stream Source",
                 font=("Consolas", 15, "bold"), fg=ACC_HEX, bg=BAR_HEX
                 ).pack(pady=(22, 4))
        tk.Label(card, text="Choose live camera or upload a video clip for detection.",
                 font=("Consolas", 10), fg=TXT_HEX, bg=BAR_HEX).pack(pady=(0, 20))

        # ── Option 1: Webcam ──────────────────────────────────────────────────
        cam_row = tk.Frame(card, bg=BAR_HEX)
        cam_row.pack(fill=tk.X, padx=30, pady=4)
        tk.Label(cam_row, text="💻  Laptop / USB Webcam",
                 font=("Consolas", 11, "bold"), fg=TXT_HEX, bg=BAR_HEX
                 ).pack(side=tk.LEFT)
        self.cam_btn = tk.Button(
            cam_row, text="Connect Webcam",
            font=("Consolas", 10, "bold"), fg="#000000", bg=ACC_HEX,
            relief=tk.FLAT, padx=12, pady=4, cursor="hand2",
            command=self._connect_webcam)
        self.cam_btn.pack(side=tk.RIGHT)

        tk.Frame(card, bg="#2d3748", height=1).pack(fill=tk.X, padx=30, pady=12)

        # ── Option 2: IP Cam ─────────────────────────────────────────────────
        tk.Label(card, text="📱  Phone IP Camera  (IP Webcam App)",
                 font=("Consolas", 11, "bold"), fg=TXT_HEX, bg=BAR_HEX
                 ).pack(anchor=tk.W, padx=30, pady=(0, 4))

        url_row = tk.Frame(card, bg=BAR_HEX)
        url_row.pack(fill=tk.X, padx=30, pady=(0, 4))
        tk.Label(url_row, text="URL:", font=("Consolas", 10),
                 fg=TXT_HEX, bg=BAR_HEX).pack(side=tk.LEFT, padx=(0, 8))
        self.ip_var = tk.StringVar(value="http://192.168.x.x:8080/video")
        self.ip_entry = tk.Entry(
            url_row, textvariable=self.ip_var,
            font=("Consolas", 10), bg="#1a2332", fg=TXT_HEX,
            insertbackground=TXT_HEX, relief=tk.FLAT, width=32)
        self.ip_entry.pack(side=tk.LEFT, ipady=4)

        self.ip_btn = tk.Button(
            url_row, text="Connect IP",
            font=("Consolas", 10, "bold"), fg="#000000", bg=ACC_HEX,
            relief=tk.FLAT, padx=10, pady=3, cursor="hand2",
            command=self._connect_ip)
        self.ip_btn.pack(side=tk.RIGHT)

        tk.Frame(card, bg="#2d3748", height=1).pack(fill=tk.X, padx=30, pady=12)

        # ── Option 3: Upload Video Clip ───────────────────────────────────────
        row3 = tk.Frame(card, bg=BAR_HEX)
        row3.pack(fill=tk.X, padx=30, pady=4)
        tk.Label(row3, text="🎥  Upload Video File (.mp4, .avi, .mov)",
                 font=("Consolas", 11, "bold"), fg=TXT_HEX, bg=BAR_HEX
                 ).pack(side=tk.LEFT)
        tk.Button(
            row3, text="Select Video",
            font=("Consolas", 10, "bold"), fg="#000000", bg="#38bdf8",
            relief=tk.FLAT, padx=14, pady=4, cursor="hand2",
            command=self._upload_video).pack(side=tk.RIGHT)

        self.status_lbl = tk.Label(card, text="", font=("Consolas", 9),
                                    fg="#ef4444", bg=BAR_HEX)
        self.status_lbl.pack(pady=(12, 0))

    # ── Connection helpers ─────────────────────────────────────────────────────
    def _connect_webcam(self):
        self._launch_stream(source=0, is_video=False)

    def _connect_ip(self):
        url = self.ip_var.get().strip()
        if not url.startswith("http"):
            self.status_lbl.config(text="⚠  Enter a valid http://... URL first!")
            return
        self._launch_stream(source=url, is_video=True)

    def _upload_video(self):
        path = filedialog.askopenfilename(
            title="Select Video Clip",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if path:
            self._launch_stream(source=path, is_video=True)

    def _launch_stream(self, source, is_video):
        self.connect_frame.destroy()
        self.queue  = queue.Queue(maxsize=2)
        self.worker = VisionWorker(source=source, out_q=self.queue, is_video=is_video)
        self._build_live_ui()
        self.worker.start()
        self.root.after(20, self._pump)

    # ── Live Detection UI ──────────────────────────────────────────────────────
    def _build_live_ui(self):
        from tkinter import ttk

        self.canvas_frame = tk.Frame(self.root, bg=BG_HEX)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 0))

        self.video_label = tk.Label(self.canvas_frame, bg=BG_HEX)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # ── Transport controls bar ────────────────────────────────────────────
        self._scrub_user = False
        ctrl_bar = tk.Frame(self.root, bg="#111827", height=44)
        ctrl_bar.pack(fill=tk.X, padx=0, pady=0)
        ctrl_bar.pack_propagate(False)

        # ◀ -5s
        tk.Button(ctrl_bar, text="◀◀  -5s", font=("Consolas", 10, "bold"),
                  fg=TXT_HEX, bg="#1e293b", activebackground="#334155",
                  relief=tk.FLAT, padx=12, pady=0, cursor="hand2",
                  command=lambda: self.worker.seek(-5) if self.worker else None
                  ).pack(side=tk.LEFT, padx=(10, 2), pady=6)

        # ⏸ / ▶ Play-Pause button
        self.playpause_var = tk.StringVar(value="⏸")
        self.playpause_btn = tk.Button(
            ctrl_bar, textvariable=self.playpause_var,
            font=("Consolas", 13, "bold"),
            fg="#ffffff", bg=ACC_HEX, activebackground="#059669",
            relief=tk.FLAT, padx=14, pady=0, cursor="hand2",
            command=self._toggle_pause)
        self.playpause_btn.pack(side=tk.LEFT, padx=6, pady=6)

        # +5s ▶▶
        tk.Button(ctrl_bar, text="+5s  ▶▶", font=("Consolas", 10, "bold"),
                  fg=TXT_HEX, bg="#1e293b", activebackground="#334155",
                  relief=tk.FLAT, padx=12, pady=0, cursor="hand2",
                  command=lambda: self.worker.seek(+5) if self.worker else None
                  ).pack(side=tk.LEFT, padx=(2, 10), pady=6)

        # Time label
        self.time_var = tk.StringVar(value="0:00 / 0:00")
        tk.Label(ctrl_bar, textvariable=self.time_var, font=("Consolas", 10),
                 fg=TXT_HEX, bg="#111827", width=14).pack(side=tk.LEFT, padx=4)

        # Scrub / timeline slider using native tk.Scale for smooth responsiveness
        self.scrub_var = tk.DoubleVar(value=0.0)
        self.scrub = tk.Scale(
            ctrl_bar, from_=0.0, to=1.0, resolution=0.001,
            orient=tk.HORIZONTAL, variable=self.scrub_var,
            showvalue=False, bg="#111827", fg=TXT_HEX,
            troughcolor="#1e293b", activebackground=ACC_HEX,
            highlightthickness=0, bd=0, relief=tk.FLAT, cursor="hand2"
        )
        self.scrub.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.scrub.bind("<ButtonPress-1>",   lambda e: setattr(self, "_scrub_user", True))
        self.scrub.bind("<ButtonRelease-1>", self._on_scrub_release)

        # ── Stats footer ──────────────────────────────────────────────────────
        footer = tk.Frame(self.root, bg=BAR_HEX, height=50)
        footer.pack(fill=tk.X, side=tk.BOTTOM)

        self.stats_var = tk.StringVar(value="Connecting to stream…")
        tk.Label(footer, textvariable=self.stats_var, font=("Consolas", 11),
                 fg=TXT_HEX, bg=BAR_HEX).pack(side=tk.LEFT, padx=16, pady=12)

        self.verdict_var = tk.StringVar(value="SCANNING")
        self.verdict_label = tk.Label(
            footer, textvariable=self.verdict_var,
            font=("Consolas", 12, "bold"), fg="#000000",
            bg="#374151", padx=16, pady=6)
        self.verdict_label.pack(side=tk.RIGHT, padx=16, pady=10)

        # Spacebar = play/pause globally
        self.root.bind("<space>", lambda e: self._toggle_pause())
        self.root.bind("<Left>",  lambda e: self.worker.seek(-5) if self.worker else None)
        self.root.bind("<Right>", lambda e: self.worker.seek(+5) if self.worker else None)

    # ── Transport actions ──────────────────────────────────────────────────────
    def _toggle_pause(self):
        if not self.worker:
            return
        if self.worker.is_paused:
            self.worker.resume()
            self.playpause_var.set("⏸")
            self.playpause_btn.config(bg=ACC_HEX)
        else:
            self.worker.pause()
            self.playpause_var.set("▶")
            self.playpause_btn.config(bg="#2563eb")

    def _on_scrub_click(self, event):
        """Seek immediately when user clicks anywhere on the scrub bar."""
        w = self.scrub.winfo_width()
        if w <= 0 or not self.worker:
            return
        frac = max(0.0, min(1.0, event.x / w))
        self.worker.seek_to_fraction(frac)
        self.scrub_var.set(frac)

    def _on_scrub_release(self, event):
        if self.worker and self.worker.total_frames > 0:
            frac = self.scrub_var.get()
            self.worker.seek_to_fraction(frac)
        self._scrub_user = False

    @staticmethod
    def _fmt_time(frames, fps):
        secs = int(frames / fps) if fps > 0 else 0
        return f"{secs // 60}:{secs % 60:02d}"

    # ── Frame pump (runs every 20 ms) ─────────────────────────────────────────
    def _pump(self):
        try:
            while not self.queue.empty():
                payload = self.queue.get_nowait()
                if "error" in payload:
                    self.stats_var.set(f"❌  {payload['error']}")
                    break

                frame   = payload["frame"]
                det     = payload["det"]
                fps     = payload["fps"]
                pos     = payload.get("pos", 0)
                total   = payload.get("total", 0)

                # Update scrub bar (only when user isn't dragging)
                if not self._scrub_user and total > 0:
                    self.scrub_var.set(pos / total)
                    vfps = self.worker.video_fps if self.worker else 25.0
                    self.time_var.set(
                        f"{self._fmt_time(pos, vfps)} / {self._fmt_time(total, vfps)}")

                if det is not None:
                    raw_conf = det["snn_conf"]
                    weight   = det["weight"]
                    conf     = raw_conf * weight
                    sparsity = det["sparsity"]
                    xc, yc, bw, bh = det["norm_box"]
                    h, w = frame.shape[:2]

                    if conf >= 0.25:
                        x1 = int(max(0,      (xc - bw / 2) * w))
                        y1 = int(max(0,      (yc - bh / 2) * h))
                        x2 = int(min(w - 1,  (xc + bw / 2) * w))
                        y2 = int(min(h - 1,  (yc + bh / 2) * h))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), ACCENT_BGR, 2)
                        cv2.putText(frame, f"NeuroCamo: {conf*100:.1f}%",
                                    (x1, max(20, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT_BGR, 2)

                    # ── Spike heatmap debug overlay (bottom-left, 80×80 px) ──────
                    if "spike_map" in det:
                        sm = det["spike_map"]  # (16,16) float32
                        sm_norm = cv2.normalize(sm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        sm_color = cv2.applyColorMap(sm_norm, cv2.COLORMAP_JET)
                        ov_size = 80
                        sm_up = cv2.resize(sm_color, (ov_size, ov_size), interpolation=cv2.INTER_NEAREST)
                        # Place in bottom-left corner
                        oy = h - ov_size - 4
                        frame[oy:oy+ov_size, 4:4+ov_size] = sm_up
                        cv2.rectangle(frame, (4, oy), (4+ov_size, oy+ov_size), (200,200,200), 1)
                        cv2.putText(frame, "SNN spikes", (5, oy - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)

                    self.stats_var.set(
                        f"FPS: {fps:.1f}  |  Confidence: {conf*100:.1f}%  |  SNN Sparsity: {sparsity*100:.1f}%")

                    if conf >= 0.6:
                        self.verdict_var.set("🎯  TARGET DETECTED")
                        self.verdict_label.config(bg="#238636", fg="white")
                    elif conf >= 0.25:
                        self.verdict_var.set("⚠  POSSIBLE TARGET")
                        self.verdict_label.config(bg="#9e6a03", fg="white")
                    else:
                        self.verdict_var.set("✅  CLEAR / NO CAMO")
                        self.verdict_label.config(bg="#1f2937", fg="#9ca3af")

                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img   = Image.fromarray(rgb)
                cw    = self.canvas_frame.winfo_width()  or 1000
                ch    = self.canvas_frame.winfo_height() or 540
                img.thumbnail((cw, ch))
                photo = ImageTk.PhotoImage(img)
                self.video_label.config(image=photo)
                self.video_label.image = photo

        except Exception as e:
            print(f"[GUI Error] {e}")

        self.root.after(20, self._pump)

    def _on_close(self):
        if self.worker:
            self.worker.stop()
        self.root.destroy()

        self.canvas_frame = tk.Frame(self.root, bg=BG_HEX)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.video_label = tk.Label(self.canvas_frame, bg=BG_HEX)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        footer = tk.Frame(self.root, bg=BAR_HEX, height=58)
        footer.pack(fill=tk.X, side=tk.BOTTOM)

        self.stats_var = tk.StringVar(value="Connecting to stream…")
        tk.Label(footer, textvariable=self.stats_var, font=("Consolas", 11),
                 fg=TXT_HEX, bg=BAR_HEX).pack(side=tk.LEFT, padx=16, pady=16)

        self.verdict_var = tk.StringVar(value="SCANNING")
        self.verdict_label = tk.Label(
            footer, textvariable=self.verdict_var,
            font=("Consolas", 12, "bold"), fg="#000000",
            bg="#374151", padx=16, pady=6)
        self.verdict_label.pack(side=tk.RIGHT, padx=16, pady=12)

    # ── Frame pump (runs every 20 ms) ─────────────────────────────────────────
    def _pump(self):
        try:
            while not self.queue.empty():
                payload = self.queue.get_nowait()
                if "error" in payload:
                    self.stats_var.set(f"❌  {payload['error']}")
                    break

                frame   = payload["frame"]
                det     = payload["det"]
                fps     = payload["fps"]

                if det is not None:
                    raw_conf = det["snn_conf"]
                    weight   = det["weight"]
                    conf     = raw_conf * weight
                    sparsity = det["sparsity"]
                    xc, yc, bw, bh = det["norm_box"]
                    h, w = frame.shape[:2]

                    if conf >= 0.25:
                        x1 = int(max(0,          (xc - bw / 2) * w))
                        y1 = int(max(0,          (yc - bh / 2) * h))
                        x2 = int(min(w - 1, (xc + bw / 2) * w))
                        y2 = int(min(h - 1, (yc + bh / 2) * h))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), ACCENT_BGR, 2)
                        cv2.putText(frame, f"NeuroCamo: {conf*100:.1f}%",
                                    (x1, max(20, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT_BGR, 2)

                    self.stats_var.set(
                        f"FPS: {fps:.1f}  |  Confidence: {conf*100:.1f}%  |  SNN Sparsity: {sparsity*100:.1f}%")

                    if conf >= 0.6:
                        self.verdict_var.set("🎯  TARGET DETECTED")
                        self.verdict_label.config(bg="#238636", fg="white")
                    elif conf >= 0.25:
                        self.verdict_var.set("⚠  POSSIBLE TARGET")
                        self.verdict_label.config(bg="#9e6a03", fg="white")
                    else:
                        self.verdict_var.set("✅  CLEAR / NO CAMO")
                        self.verdict_label.config(bg="#1f2937", fg="#9ca3af")

                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img   = Image.fromarray(rgb)
                cw    = self.canvas_frame.winfo_width()  or 1000
                ch    = self.canvas_frame.winfo_height() or 580
                img.thumbnail((cw, ch))
                photo = ImageTk.PhotoImage(img)
                self.video_label.config(image=photo)
                self.video_label.image = photo

        except Exception as e:
            print(f"[GUI Error] {e}")

        self.root.after(20, self._pump)

    def _on_close(self):
        if self.worker:
            self.worker.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    NeuroCamoLiveGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
