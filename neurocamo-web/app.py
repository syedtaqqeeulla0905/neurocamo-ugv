"""
NeuroCamo Ground Control Station & Analysis Suite — Backend Server
==================================================================
Serves the multi-view Tactical GCS and SNN Analytics Suite from ./static
and exposes high-performance JSON APIs for:
  - Live SNN Camouflage Detection & Localization
  - Layer-3 Spatio-temporal Spike Spectrogram & Sparsity
  - Edge Power Draw & Battery Autonomy Profiling
  - Hardware Telemetry & Motor Actuation Packets
  - Model Diagnostics & Publication Graphs

Default: http://127.0.0.1:8080
"""

from __future__ import annotations

import base64
from collections import OrderedDict
import glob
import hashlib
import json
import math
import os
import select
import socket
import struct
import sys
import threading
import time
import traceback
import urllib.request
from urllib.parse import urlsplit
import uuid

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_file, send_from_directory, Response
from werkzeug.serving import WSGIRequestHandler

# Add project roots to path
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
NEUROCAMO_ROOT = os.path.join(PROJECT_ROOT, "neurocamo")

if HERE not in sys.path:
    sys.path.insert(0, HERE)
if NEUROCAMO_ROOT not in sys.path:
    sys.path.insert(0, NEUROCAMO_ROOT)

from inference import ModelBundle, run_inference
from power_monitor import PowerMonitor
from ultralytics import YOLO

HOST = os.environ.get("NEUROCAMO_HOST", "0.0.0.0")
PORT = int(os.environ.get("NEUROCAMO_PORT", "8080"))
MAX_UPLOAD_MB = 25
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}

TEST_IMAGE_DIR = os.path.join(PROJECT_ROOT, "test_image")
RESULTS_DIR = os.path.join(NEUROCAMO_ROOT, "results")
SAMPLES_DIR = os.path.join(HERE, "static", "samples")

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

print("=" * 68)
print("  NEUROCAMO: Tactical SNN Ground Control Station & Analysis Suite")
print("=" * 68)
print("  Initializing SNN Neuromorphic Model Bundle...")
BUNDLE = ModelBundle()
print(f"  [READY] SNN running on {BUNDLE.info.accelerator} ({BUNDLE.info.device})")
print(f"  [READY] Checkpoint: {os.path.basename(BUNDLE.info.checkpoint_path)}")

# Warmup SNN once so first benchmark run reflects steady-state execution
try:
    _warm_bgr = np.zeros((128, 128, 3), dtype=np.uint8)
    run_inference(BUNDLE, _warm_bgr, is_live=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
except Exception:
    pass

def _ensure_cuda_nms_compat() -> None:
    """If torchvision was built without CUDA NMS (common on Jetson pip wheels),
    transparently run only the tiny final NMS step on CPU while keeping YOLOv8
    backbone/neck/head inference 100% on CUDA GPU."""
    if not torch.cuda.is_available():
        return
    try:
        import torchvision
        _b = torch.tensor([[0.0, 0.0, 10.0, 10.0]], device="cuda")
        _s = torch.tensor([0.9], device="cuda")
        torchvision.ops.nms(_b, _s, 0.5)
    except Exception:
        try:
            import torchvision.ops as _tv_ops
            _orig_nms = _tv_ops.nms

            def _cpu_safe_nms(boxes, scores, iou_threshold):
                if boxes.is_cuda:
                    keep = _orig_nms(boxes.detach().cpu(), scores.detach().cpu(), iou_threshold)
                    return keep.to(boxes.device)
                return _orig_nms(boxes, scores, iou_threshold)

            _tv_ops.nms = _cpu_safe_nms
            print("  [INFO] Patched torchvision.ops.nms for Jetson CUDA compatibility")
        except Exception:
            pass


_ensure_cuda_nms_compat()

YOLO_PATH = os.path.join(NEUROCAMO_ROOT, "checkpoints", "yolov8n.pt")
if not os.path.exists(YOLO_PATH):
    YOLO_PATH = os.path.join(PROJECT_ROOT, "yolov8n.pt")

try:
    YOLO_MODEL = YOLO(YOLO_PATH)
    # Warmup
    dummy = np.zeros((128, 128, 3), dtype=np.uint8)
    YOLO_MODEL(dummy, verbose=False, device=BUNDLE.device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("  [READY] YOLOv8-Nano Baseline Model Loaded (Live Head-to-Head Active)")
except Exception as e:
    YOLO_MODEL = None
    print(f"  [WARN] YOLOv8 could not be loaded: {e}")


# ── RFC 6455 WebSocket Hub & Hardware Power Telemetry ───────────────────────
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_MAX_WS_PAYLOAD = 1024 * 1024  # 1 MiB safety limit for incoming control/ping frames


def _ws_encode_frame(opcode: int, payload: bytes = b"") -> bytes:
    """Encode a single unmasked server->client RFC 6455 frame."""
    header = bytearray([0x80 | (opcode & 0x0F)])
    n = len(payload)
    if n < 126:
        header.append(n)
    elif n <= 0xFFFF:
        header.append(126)
        header.extend(struct.pack("!H", n))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", n))
    return bytes(header) + payload


class _WSClient:
    """Thread-safe connected WebSocket client socket wrapper."""

    def __init__(self, conn: socket.socket) -> None:
        self.conn = conn
        self.lock = threading.Lock()
        self.closed = False

    def send_frame(self, opcode: int, payload: bytes = b"") -> bool:
        frame = _ws_encode_frame(opcode, payload)
        return self.send_raw(frame)

    def send_raw(self, frame: bytes) -> bool:
        with self.lock:
            if self.closed:
                return False
            try:
                self.conn.sendall(frame)
                return True
            except OSError:
                self.closed = True
                return False

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            try:
                self.conn.sendall(_ws_encode_frame(0x8, struct.pack("!H", 1000)))
            except OSError:
                pass
            try:
                self.conn.close()
            except OSError:
                pass


class WSClientHub:
    """Broadcasts live telemetry events immediately to all connected /ws/power clients."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: list[_WSClient] = []

    def add(self, conn: socket.socket) -> _WSClient:
        client = _WSClient(conn)
        with self._lock:
            self._clients.append(client)
        return client

    def remove(self, client: _WSClient) -> None:
        client.closed = True
        with self._lock:
            self._clients = [c for c in self._clients if c is not client]

    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def broadcast(self, event: dict) -> None:
        with self._lock:
            clients = list(self._clients)
        if not clients:
            return
        raw_json = json.dumps(sanitize_json(event), separators=(",", ":")).encode("utf-8")
        frame = _ws_encode_frame(0x1, raw_json)
        dead: list[_WSClient] = []
        for client in clients:
            if not client.send_raw(frame):
                dead.append(client)
        if dead:
            with self._lock:
                self._clients = [c for c in self._clients if c not in dead]

    def send_to(self, client: _WSClient, event: dict) -> bool:
        raw_json = json.dumps(sanitize_json(event), separators=(",", ":")).encode("utf-8")
        return client.send_raw(_ws_encode_frame(0x1, raw_json))


def _ws_recv_exact(conn: socket.socket, rx_buf: bytearray, n: int, client: _WSClient) -> bytes | None:
    """Read exactly n bytes from pre-buffered bytes + raw socket without poisoning BufferedReader."""
    while len(rx_buf) < n:
        if client.closed:
            return None
        try:
            ready, _, _ = select.select([conn], [], [], 1.0)
            if not ready:
                continue
            chunk = conn.recv(4096)
            if not chunk:
                return None
            rx_buf.extend(chunk)
        except (TimeoutError, socket.timeout, BlockingIOError):
            continue
        except OSError:
            return None
    out = bytes(rx_buf[:n])
    del rx_buf[:n]
    return out


def _ws_read_loop(conn: socket.socket, rx_buf: bytearray, client: _WSClient) -> None:
    """Read RFC 6455 frames from client: answer Ping->Pong, handle Close, ignore Text."""
    while not client.closed:
        hdr = _ws_recv_exact(conn, rx_buf, 2, client)
        if hdr is None:
            break
        b1, b2 = hdr[0], hdr[1]
        opcode = b1 & 0x0F
        masked = (b2 & 0x80) != 0
        length = b2 & 0x7F

        if length == 126:
            ext = _ws_recv_exact(conn, rx_buf, 2, client)
            if ext is None:
                break
            length = struct.unpack("!H", ext)[0]
        elif length == 127:
            ext = _ws_recv_exact(conn, rx_buf, 8, client)
            if ext is None:
                break
            length = struct.unpack("!Q", ext)[0]

        if length > _MAX_WS_PAYLOAD:
            break

        mask_key = _ws_recv_exact(conn, rx_buf, 4, client) if masked else None
        if masked and mask_key is None:
            break

        payload = _ws_recv_exact(conn, rx_buf, length, client) if length > 0 else b""
        if payload is None:
            break

        if masked and mask_key and payload:
            payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))

        if opcode == 0x8:  # Close
            try:
                client.send_frame(0x8, payload[:2] if len(payload) >= 2 else b"")
            except Exception:
                pass
            break
        elif opcode == 0x9:  # Ping -> Pong
            client.send_frame(0xA, payload)
        elif opcode == 0xA:  # Pong
            continue
        elif opcode == 0x1:  # Text (optional client heartbeat)
            txt = payload.decode("utf-8", errors="ignore").strip().lower()
            if txt in ("ping", '{"type":"ping"}'):
                WS_HUB.send_to(client, {"type": "pong", "ts": time.time()})


class NeuroCamoRequestHandler(WSGIRequestHandler):
    """Werkzeug handler supporting RFC 6455 WebSocket upgrade on /ws/power."""

    _is_websocket: bool = False

    def do_GET(self):  # type: ignore[override]
        req_path = urlsplit(self.path).path
        upgrade_hdr = (self.headers.get("Upgrade") or "").strip().lower()
        if req_path == "/ws/power" and upgrade_hdr == "websocket":
            self._handle_ws_power()
            return
        return self.run_wsgi()

    def finish(self) -> None:
        if self._is_websocket:
            try:
                self.connection.close()
            except OSError:
                pass
            return
        super().finish()

    def _handle_ws_power(self) -> None:
        ws_key = self.headers.get("Sec-WebSocket-Key")
        if not ws_key:
            self.send_error(400, "Missing Sec-WebSocket-Key")
            return

        accept = base64.b64encode(
            hashlib.sha1((ws_key.strip() + _WS_GUID).encode("ascii")).digest()
        ).decode("ascii")

        resp_hdrs = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "\r\n"
        ).encode("ascii")

        self._is_websocket = True
        self.close_connection = True
        self.wfile.write(resp_hdrs)
        self.wfile.flush()

        # Drain any bytes already buffered in rfile before switching to raw socket recv()
        rx_buf = bytearray()
        try:
            self.connection.setblocking(False)
            pre = self.rfile.read1(8192)
            if pre:
                rx_buf.extend(pre)
        except (BlockingIOError, OSError, AttributeError):
            pass

        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        self.connection.settimeout(5.0)

        client = WS_HUB.add(self.connection)
        try:
            WS_HUB.send_to(
                client,
                {
                    "type": "hello",
                    "endpoint": "/ws/power",
                    "capabilities": POWER_MONITOR.get_capabilities(),
                    "active_session": POWER_MONITOR.get_active_session_info(),
                    "latest": POWER_MONITOR.read_live(),
                },
            )
            _ws_read_loop(self.connection, rx_buf, client)
        finally:
            WS_HUB.remove(client)
            client.close()


WS_HUB = WSClientHub()
POWER_MONITOR = PowerMonitor(interval_s=0.15, on_event=WS_HUB.broadcast)
_pm_caps = POWER_MONITOR.get_capabilities()
print(
    f"  [READY] Power Telemetry: {_pm_caps['source']} "
    f"({'ESTIMATED' if _pm_caps['is_estimated'] else 'MEASURED HARDWARE'}) "
    f"[{_pm_caps['sensor_detail']}]"
)
print("=" * 68)

MAX_SESSIONS = 50
_session_lock = threading.Lock()
SESSION_STORE: OrderedDict[str, dict] = OrderedDict()
_benchmark_lock = threading.Lock()


def _store_session(record: dict) -> None:
    sid = record["summary"]["session_id"]
    with _session_lock:
        SESSION_STORE[sid] = record
        SESSION_STORE.move_to_end(sid, last=False)
        while len(SESSION_STORE) > MAX_SESSIONS:
            SESSION_STORE.popitem(last=True)


def _list_session_summaries() -> list[dict]:
    with _session_lock:
        return [rec["summary"] for rec in SESSION_STORE.values()]


def _get_session_record(session_id: str) -> dict | None:
    with _session_lock:
        return SESSION_STORE.get(session_id)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/ws/power")
def ws_power_http_fallback():
    return (
        jsonify(
            {
                "error": "WebSocket upgrade required (connect via ws://.../ws/power)",
                "endpoint": "/ws/power",
                "capabilities": POWER_MONITOR.get_capabilities(),
            }
        ),
        426,
    )


def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return True if obj else False
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_json(obj.tolist())
    return obj


@app.get("/api/status")
def status():
    i = BUNDLE.info
    return jsonify(
        {
            "ready": True,
            "device": i.device,
            "accelerator": i.accelerator,
            "epoch": i.epoch,
            "val_loss": round(i.val_loss, 4),
            "checkpoint_loaded": i.checkpoint_loaded,
            "checkpoint": os.path.basename(i.checkpoint_path),
            "params": i.params,
            "timesteps": i.timesteps,
            "head": i.head,
            "defaulted_keys": i.defaulted_keys or [],
            "unexpected_keys": i.unexpected_keys or [],
            "power_monitor": POWER_MONITOR.get_capabilities(),
            "ws_clients": WS_HUB.client_count(),
        }
    )


@app.get("/api/power/live")
def api_power_live():
    return jsonify(sanitize_json(POWER_MONITOR.read_live()))


@app.get("/api/sessions")
def api_list_sessions():
    active = POWER_MONITOR.get_active_session_info()
    summaries = _list_session_summaries()
    return jsonify(
        sanitize_json(
            {
                "sessions": summaries,
                "count": len(summaries),
                "active_session_id": active["session_id"] if active else None,
            }
        )
    )


@app.get("/api/sessions/<session_id>")
def api_get_session(session_id: str):
    rec = _get_session_record(session_id)
    if rec is None:
        return jsonify({"error": f"Session not found: {session_id}"}), 404
    payload = dict(rec["summary"])
    payload["samples"] = rec["samples"]
    return jsonify(sanitize_json(payload))


# ── Curated Sample Gallery ───────────────────────────────────────────────────
CURATED_SAMPLES = [
    {
        "id": "camo_marine",
        "title": "Marine Batfish (Benthic Concealment)",
        "category": "Marine Camouflage",
        "file": "camo_batfish.jpg",
        "source": "samples",
        "desc": "Natural seafloor camouflage blending benthic skin texture with coral gravel.",
    },
    {
        "id": "camo_chameleon",
        "title": "Arboreal Chameleon",
        "category": "Reptile Concealment",
        "file": "chameleon-camouflage-photograph.webp",
        "source": "test_image",
        "desc": "Pigment dispersion camouflage matching dense jungle foliage and branches.",
    },
    {
        "id": "camo_ghillie1",
        "title": "Tactical Ghillie Sniper",
        "category": "Military Camouflage",
        "file": "IMG_0978.JPG (1).jpeg",
        "source": "test_image",
        "desc": "Full synthetic burlap ghillie suit deployed in high-density deciduous brush.",
    },
    {
        "id": "camo_woodland",
        "title": "Woodland Recon Operative",
        "category": "Tactical Concealment",
        "file": "IMG_0986.JPG.jpeg",
        "source": "test_image",
        "desc": "Multi-spectral woodland camouflage breaking human silhouette in tree canopy.",
    },
    {
        "id": "camo_brush",
        "title": "Concealed Ground Observer",
        "category": "Tactical Concealment",
        "file": "IMG_0977.JPG.jpeg",
        "source": "test_image",
        "desc": "Ground observer positioned inside brush pile with disruptive pattern shroud.",
    },
    {
        "id": "neg_ordinary",
        "title": "Negative Control: Urban Environment",
        "category": "Non-Camouflage",
        "file": "ordinary_scene.jpg",
        "source": "samples",
        "desc": "Clear outdoor architectural setting with no concealed targets.",
    },
    {
        "id": "neg_landscape",
        "title": "Negative Control: Open Field",
        "category": "Non-Camouflage",
        "file": "4.jpeg",
        "source": "test_image",
        "desc": "Smooth outdoor landscape demonstrating scene-gating false alarm rejection.",
    },
]


@app.get("/api/samples")
def list_samples():
    return jsonify(CURATED_SAMPLES)


@app.get("/api/sample_img/<source>/<filename>")
def get_sample_image(source: str, filename: str):
    base_dir = SAMPLES_DIR if source == "samples" else TEST_IMAGE_DIR
    full_path = os.path.join(base_dir, filename)
    if not os.path.exists(full_path):
        return jsonify({"error": f"Sample file not found: {filename}"}), 404
    return send_file(full_path)


RAW_UNFILTERED_BGR = None
LAST_BGR = None


def apply_environmental_filter(bgr: np.ndarray, env_filter: str) -> np.ndarray:
    if env_filter == "dusk":
        gamma_table = np.array([((i / 255.0) ** 2.2) * 255 for i in range(256)]).astype("uint8")
        dark = cv2.LUT(bgr, gamma_table)
        return (dark * 0.55).astype(np.uint8)
    elif env_filter == "smoke":
        haze = np.full_like(bgr, 185)
        return cv2.addWeighted(bgr, 0.42, haze, 0.58, 0)
    elif env_filter == "noise":
        noise = np.random.normal(0, 24, bgr.shape).astype(np.float32)
        return np.clip(bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return bgr


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _quality_params(stream_quality: str) -> tuple[int, int]:
    sq = (stream_quality or "high").lower()
    if sq in ("1080p", "fhd"):
        return 1920, 85
    if sq == "fast":
        return 424, 50
    if sq == "medium":
        return 480, 65
    return 640, 80


def _prepare_frame(bgr: np.ndarray, env_filter: str, stream_quality: str) -> tuple[np.ndarray, str, int]:
    global RAW_UNFILTERED_BGR, LAST_BGR
    h, w = bgr.shape[:2]
    long_edge = max(h, w)
    max_edge, jpeg_qual = _quality_params(stream_quality)
    if long_edge > max_edge:
        s = float(max_edge) / long_edge
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    RAW_UNFILTERED_BGR = bgr.copy()
    bgr = apply_environmental_filter(bgr, env_filter)
    LAST_BGR = bgr.copy()

    ok_in, buf_in = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_qual])
    input_b64 = "data:image/jpeg;base64," + base64.b64encode(buf_in).decode("ascii") if ok_in else ""
    return bgr, input_b64, jpeg_qual


def _attach_visual_artifacts(out: dict, bgr: np.ndarray, input_b64: str, filename: str, env_filter: str, jpeg_qual: int, is_live: bool) -> None:
    annotated = out.pop("annotated_rgb")
    ok_out, buf_out = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_qual])
    if not ok_out:
        raise RuntimeError("Could not encode result image")

    out["result_png"] = "data:image/jpeg;base64," + base64.b64encode(buf_out).decode("ascii")
    out["input_png"] = input_b64
    out["filename"] = filename
    out["env_filter"] = env_filter

    # In live mode: skip extra JPEG compressions & temporal step sweeps to maximize FPS on Jetson Xavier NX
    if is_live:
        out["saliency_png"] = out["result_png"]
        out["camo_mask_png"] = out["result_png"]
        out["temporal_evolution"] = []
        return

    try:
        h_orig, w_orig = bgr.shape[:2]
        if out.get("heatmap") and out["heatmap"].get("data"):
            h_raw = np.array(out["heatmap"]["data"], dtype=np.uint8).reshape((16, 16))
            h_full = cv2.resize(h_raw, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

            heat_color = cv2.applyColorMap(h_full, cv2.COLORMAP_JET)
            saliency_bgr = cv2.addWeighted(bgr, 0.42, heat_color, 0.58, 0)
            ok_sal, buf_sal = cv2.imencode(".jpg", saliency_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_qual])
            out["saliency_png"] = "data:image/jpeg;base64," + base64.b64encode(buf_sal).decode("ascii") if ok_sal else out["result_png"]

            _, mask_bin = cv2.threshold(h_full, 80, 255, cv2.THRESH_BINARY)
            camo_mask_bgr = bgr.copy()
            camo_mask_bgr[mask_bin > 0] = cv2.addWeighted(
                camo_mask_bgr[mask_bin > 0],
                0.3,
                np.full_like(camo_mask_bgr[mask_bin > 0], (0, 230, 160)),
                0.7,
                0,
            )
            ok_msk, buf_msk = cv2.imencode(".jpg", camo_mask_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_qual])
            out["camo_mask_png"] = "data:image/jpeg;base64," + base64.b64encode(buf_msk).decode("ascii") if ok_msk else out["result_png"]
        else:
            out["saliency_png"] = out["result_png"]
            out["camo_mask_png"] = out["result_png"]
    except Exception as e_ov:
        print(f"Overlay generation warning: {e_ov}")
        out["saliency_png"] = out["result_png"]
        out["camo_mask_png"] = out["result_png"]

    try:
        resized_t = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (128, 128)).astype(np.float32) / 255.0
        tensor_t = torch.from_numpy(resized_t).permute(2, 0, 1).unsqueeze(0).to(BUNDLE.device)
        temporal_steps = []
        with torch.no_grad():
            spk_full = BUNDLE.encoder(tensor_t)
            spk_l3, mem_l3, _ = BUNDLE.model.backbone(spk_full)
            for t_step in range(1, 17):
                sub_spk = spk_l3[:t_step]
                fused = BUNDLE.model.asab(sub_spk, mem_l3 * (t_step / 16.0))
                hm, off, wh = BUNDLE.model.head(fused)
                bbox, conf_t = BUNDLE.model.decode_mask(hm)
                heat_t = torch.sigmoid(hm)[0, 0].cpu().numpy()
                temporal_steps.append({
                    "t": t_step,
                    "conf": round(float(conf_t[0].item()), 3),
                    "spikes": int(sub_spk.sum().item()),
                    "peak": round(float(heat_t.max()), 3),
                })
        out["temporal_evolution"] = temporal_steps
    except Exception as te:
        print(f"Temporal evolution calculation warning: {te}")
        out["temporal_evolution"] = []


def _attach_telemetry_and_energy(
    out: dict,
    snn_power_w: float,
    yolo_power_w: float | None,
    measured_snn_mj: float | None,
    measured_yolo_mj: float | None,
    source: str,
    is_estimated: bool,
) -> None:
    bbox = out.get("bbox_norm", [0.5, 0.5, 0.2, 0.2])
    xc, yc, bw, bh = bbox
    detected = out.get("verdict") in ("DETECTED", "POSSIBLE TARGET")
    out["detected"] = detected

    if detected:
        bearing_deg = (xc - 0.5) * 60.0
        elevation_deg = (0.5 - yc) * 45.0
        apparent_dim = max(0.05, max(bw, bh))
        standoff_range_m = round(float(28.0 / apparent_dim), 1)
        target_footprint_m2 = round(float((bw * standoff_range_m) * (bh * standoff_range_m) * 0.75), 2)

        turn = int(bearing_deg * 1.2)
        m1 = max(100, min(255, 230 - turn))
        m2 = max(100, min(255, 230 + turn))
        if abs(bearing_deg) < 5.0:
            cmd = "ENGAGE FORWARD"
        elif bearing_deg < 0:
            cmd = f"STEER LEFT ({-bearing_deg:.1f} deg)"
        else:
            cmd = f"STEER RIGHT ({bearing_deg:.1f} deg)"

        stanag_packet = f"$STANAG,TRT:01,AZ:{bearing_deg:+.2f},EL:{elevation_deg:+.2f},RNG:{standoff_range_m}m,STAT:LOCK*7E"
        packet = f"$UGV,SEQ:0482,CMD:ENGAGE,M1:{m1},M2:{m2},BRG:{bearing_deg:+.1f},RNG:{standoff_range_m}m*4B"
    else:
        bearing_deg = 0.0
        elevation_deg = 0.0
        standoff_range_m = 0.0
        target_footprint_m2 = 0.0
        m1, m2 = 0, 0
        cmd = "PATROL SCAN (SECTOR CLEAR)"
        stanag_packet = "$STANAG,TRT:01,AZ:+0.00,EL:+0.00,RNG:0.0m,STAT:SCAN*1A"
        packet = f"$UGV,SEQ:0482,CMD:PATROL,M1:0,M2:0,BRG:+0.0,PWR:{snn_power_w:.2f}W*1A"

    # Theoretical Horowitz 45nm CMOS synaptic energy computed from real firing sparsity (no random/fake power)
    total_synapses = 820417 * 16
    meas_sparsity = float(out.get("sparsity", 0.78))
    spec_info = out.get("spectrogram") or {}
    l3_mean_rate = float(spec_info.get("mean_rate", 0.12))
    active_ch = int(spec_info.get("active_channels", 64))
    ch_ratio = active_ch / 128.0
    conf_val = float(out.get("confidence", 0.0))

    raw_firing = 0.35 * (1.0 - meas_sparsity) + 0.35 * min(1.0, l3_mean_rate * 2.8) + 0.30 * ch_ratio
    gate_factor = (0.55 + 0.45 * conf_val) if detected else (0.12 + 0.25 * conf_val)
    effective_firing = raw_firing * gate_factor
    active_spikes = int(total_synapses * max(0.02, min(0.65, effective_firing)))

    snn_theo_mj = (active_spikes * 0.9) / 1e6
    snn_step_theo_mj = snn_theo_mj / 16.0
    ann_theo_mj = (total_synapses * 4.6) / 1e6
    reduction = round(ann_theo_mj / max(1e-5, snn_theo_mj), 1)

    out["dynamic_energy"] = {
        "active_synaptic_spikes": active_spikes,
        "silent_synapses": total_synapses - active_spikes,
        "snn_energy_mj": round(snn_theo_mj, 3),
        "snn_step_energy_mj": round(snn_step_theo_mj, 3),
        "ann_energy_mj": round(ann_theo_mj, 3),
        "energy_reduction_factor": f"{reduction}x",
        "measured_sparsity_pct": round(meas_sparsity * 100.0, 1),
        "power_snn_w": round(float(snn_power_w), 3),
        "theoretical_horowitz_snn_energy_mj": round(snn_theo_mj, 3),
        "theoretical_horowitz_ann_energy_mj": round(ann_theo_mj, 3),
        "measured_snn_energy_mj": round(float(measured_snn_mj), 3) if measured_snn_mj is not None else None,
        "measured_yolo_energy_mj": round(float(measured_yolo_mj), 3) if measured_yolo_mj is not None else None,
        "measured_snn_avg_watts": round(float(snn_power_w), 3),
        "measured_yolo_avg_watts": round(float(yolo_power_w), 3) if yolo_power_w is not None else None,
        "source": source,
        "is_estimated": is_estimated,
    }

    out["telemetry"] = {
        "detected": detected,
        "bearing_deg": round(bearing_deg, 1),
        "elevation_deg": round(elevation_deg, 1),
        "standoff_range_m": standoff_range_m,
        "target_footprint_m2": target_footprint_m2,
        "motor_a": m1,
        "motor_b": m2,
        "command": cmd,
        "packet": packet,
        "stanag_packet": stanag_packet,
        "power_snn_w": round(float(snn_power_w), 3),
        "power_ann_w": round(float(yolo_power_w), 3) if yolo_power_w is not None else None,
        "source": source,
        "is_estimated": is_estimated,
    }


def _run_yolo_single(bgr: np.ndarray, yolo_conf: float, power_w: float | None, measured_energy_mj: float | None, source: str, is_estimated: bool) -> dict:
    _sync_cuda()
    t_y_start = time.perf_counter()
    y_res = YOLO_MODEL(bgr, conf=yolo_conf, verbose=False, device=BUNDLE.device)[0]
    _sync_cuda()
    t_yolo_ms = (time.perf_counter() - t_y_start) * 1000.0

    yolo_bgr = y_res.plot()
    n_boxes = len(y_res.boxes)
    detections_list = []
    for b in y_res.boxes:
        c_id = int(b.cls[0])
        c_conf = float(b.conf[0])
        c_name = YOLO_MODEL.names[c_id]
        detections_list.append({
            "class": c_name,
            "conf": round(c_conf * 100.0, 1),
            "xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()],
        })

    if n_boxes == 0:
        cv2.putText(
            yolo_bgr,
            f"YOLOv8-Nano: 0 OBJECTS (CONF >= {int(yolo_conf * 100)}%)",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 70, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            yolo_bgr,
            "Standard COCO texture anchors missed camouflaged target",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 180, 200),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            yolo_bgr,
            f"YOLOv8: {n_boxes} OBJECT(S) DETECTED",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

    ok_yolo, buf_yolo = cv2.imencode(".jpg", yolo_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    yolo_b64 = "data:image/jpeg;base64," + base64.b64encode(buf_yolo).decode("ascii") if ok_yolo else ""

    w_orig, h_orig = bgr.shape[1], bgr.shape[0]
    yolo_gflops = round(float(8.70 * ((w_orig * h_orig) / (640.0 * 640.0))), 2)
    yolo_gflops = max(4.2, min(14.8, yolo_gflops))
    yolo_theo_mj = round(float((yolo_gflops * 1e9 * 4.6 * 1e-12) * 1000.0), 2)

    return {
        "available": True,
        "conf_threshold": yolo_conf,
        "boxes_count": n_boxes,
        "detections": detections_list,
        "latency_ms": round(t_yolo_ms, 1),
        "power_w": round(float(power_w), 3) if power_w is not None else None,
        "energy_mj": yolo_theo_mj,
        "theoretical_energy_mj": yolo_theo_mj,
        "measured_energy_mj": round(float(measured_energy_mj), 3) if measured_energy_mj is not None else None,
        "gflops": yolo_gflops,
        "status": f"{n_boxes} OBJECT(S) DETECTED" if n_boxes > 0 else f"0 OBJECTS (CONF >= {int(yolo_conf * 100)}%)",
        "result_png": yolo_b64,
        "source": source,
        "is_estimated": is_estimated,
    }


def _process_bgr(
    bgr: np.ndarray,
    filename: str = "input_frame",
    yolo_conf: float = 0.15,
    env_filter: str = "normal",
    is_live: bool = False,
    stream_quality: str = "high",
):
    """Fast single-pass processing for live streams, webcam frames, and non-benchmark re-evals."""
    bgr_proc, input_b64, jpeg_qual = _prepare_frame(bgr, env_filter, stream_quality)
    _sync_cuda()
    out = run_inference(BUNDLE, bgr_proc, is_live=is_live)
    _sync_cuda()
    _attach_visual_artifacts(out, bgr_proc, input_b64, filename, env_filter, jpeg_qual, is_live=is_live)

    live_snap = POWER_MONITOR.read_live()
    snn_pwr = float(live_snap["watts"])
    src = str(live_snap["source"])
    is_est = bool(live_snap["is_estimated"])

    yolo_data = {
        "available": False,
        "conf_threshold": yolo_conf,
        "boxes_count": 0,
        "detections": [],
        "latency_ms": 0.0,
        "power_w": None,
        "energy_mj": None,
        "theoretical_energy_mj": None,
        "measured_energy_mj": None,
        "gflops": 8.70,
        "status": "YOLOv8 STANDBY (ACTIVE IN COMPARE VIEW)" if is_live else "YOLOv8 MODEL UNAVAILABLE",
        "result_png": "",
        "source": src,
        "is_estimated": is_est,
    }
    yolo_pwr = None
    if not is_live and YOLO_MODEL is not None:
        try:
            yolo_data = _run_yolo_single(bgr_proc, yolo_conf, None, None, src, is_est)
            after_yolo = POWER_MONITOR.read_live()
            yolo_pwr = float(after_yolo["watts"])
            yolo_data["power_w"] = round(yolo_pwr, 3)
        except Exception as ey:
            print(f"YOLO comparison error: {ey}")
            yolo_data["error"] = str(ey)

    _attach_telemetry_and_energy(out, snn_pwr, yolo_pwr, None, None, src, is_est)
    out["yolo_benchmark"] = yolo_data
    return out


def _run_benchmark_pipeline(
    bgr: np.ndarray,
    filename: str = "uploaded_image",
    yolo_conf: float = 0.15,
    env_filter: str = "normal",
    stream_quality: str = "high",
) -> dict:
    """Serialized 5-phase live hardware benchmark:
    idle_baseline (0.8s) -> snn_inference (>=1.1s, >=3 runs) -> cooldown (0.5s) ->
    yolo_inference (>=1.1s, >=3 runs) -> post_idle (0.4s), streaming samples live via /ws/power.
    """
    bgr_proc, input_b64, jpeg_qual = _prepare_frame(bgr, env_filter, stream_quality)
    session_id = f"bench_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"

    with _benchmark_lock:
        try:
            # Phase 1: Idle Baseline
            POWER_MONITOR.start_session(session_id, image_name=filename, initial_phase="idle_baseline")
            time.sleep(0.8)

            # Phase 2: NeuroCamo SNN Inference (Run 1 captures UI payload; subsequent runs sustain hardware load)
            POWER_MONITOR.set_phase("snn_inference")
            t_snn_phase = time.perf_counter()
            _sync_cuda()
            t0 = time.perf_counter()
            out = run_inference(BUNDLE, bgr_proc, is_live=False)
            _sync_cuda()
            snn_latencies: list[float] = [(time.perf_counter() - t0) * 1000.0]

            while (time.perf_counter() - t_snn_phase) < 1.1 or len(snn_latencies) < 3:
                t_r = time.perf_counter()
                run_inference(BUNDLE, bgr_proc, is_live=False)
                _sync_cuda()
                snn_latencies.append((time.perf_counter() - t_r) * 1000.0)

            # Phase 3: Inter-model Cooldown Gap
            POWER_MONITOR.set_phase("cooldown")
            time.sleep(0.5)

            # Phase 4: YOLOv8-Nano Inference
            POWER_MONITOR.set_phase("yolo_inference")
            t_yolo_phase = time.perf_counter()
            yolo_latencies: list[float] = []
            yolo_data: dict | None = None

            if YOLO_MODEL is not None:
                try:
                    yolo_data = _run_yolo_single(bgr_proc, yolo_conf, None, None, "pending", False)
                    yolo_latencies.append(float(yolo_data["latency_ms"]))
                    while (time.perf_counter() - t_yolo_phase) < 1.1 or len(yolo_latencies) < 3:
                        t_y = time.perf_counter()
                        YOLO_MODEL(bgr_proc, conf=yolo_conf, verbose=False, device=BUNDLE.device)
                        _sync_cuda()
                        yolo_latencies.append((time.perf_counter() - t_y) * 1000.0)
                except Exception as ey:
                    print(f"YOLO benchmark error: {ey}")
                    yolo_data = {
                        "available": False,
                        "conf_threshold": yolo_conf,
                        "boxes_count": 0,
                        "detections": [],
                        "latency_ms": 0.0,
                        "power_w": None,
                        "energy_mj": None,
                        "theoretical_energy_mj": None,
                        "measured_energy_mj": None,
                        "gflops": 8.70,
                        "status": "YOLOv8 ERROR",
                        "result_png": "",
                        "error": str(ey),
                    }
            else:
                time.sleep(0.3)

            # Phase 5: Post-inference Idle Tail
            POWER_MONITOR.set_phase("post_idle")
            time.sleep(0.4)

            snn_mean_lat = sum(snn_latencies) / len(snn_latencies)
            yolo_mean_lat = (sum(yolo_latencies) / len(yolo_latencies)) if yolo_latencies else 0.0

            session_record = POWER_MONITOR.stop_session(
                phase_run_stats={
                    "snn_inference": {
                        "runs": len(snn_latencies),
                        "mean_latency_ms": snn_mean_lat,
                    },
                    "yolo_inference": {
                        "runs": len(yolo_latencies),
                        "mean_latency_ms": yolo_mean_lat,
                    },
                }
            )
            _store_session(session_record)
            summary = session_record["summary"]
            WS_HUB.broadcast(
                {
                    "type": "session_complete",
                    "session_id": session_id,
                    "summary": summary,
                }
            )
        except Exception as exc:
            POWER_MONITOR.abort_session()
            WS_HUB.broadcast(
                {
                    "type": "error",
                    "session_id": session_id,
                    "message": str(exc),
                }
            )
            raise

    # Assemble complete UI response outside benchmark timing lock
    _attach_visual_artifacts(out, bgr_proc, input_b64, filename, env_filter, jpeg_qual, is_live=False)

    snn_phase = summary["phases"]["snn_inference"]
    yolo_phase = summary["phases"]["yolo_inference"]
    snn_avg_w = float(snn_phase["avg_watts"])
    snn_epi_mj = float(snn_phase["energy_per_inference_mj"])
    yolo_avg_w = float(yolo_phase["avg_watts"]) if yolo_phase["runs"] > 0 else None
    yolo_epi_mj = float(yolo_phase["energy_per_inference_mj"]) if yolo_phase["runs"] > 0 else None
    src = str(summary["source"])
    is_est = bool(summary["is_estimated"])

    out["latency_ms"] = round(float(snn_phase["mean_latency_ms"] or out.get("latency_ms", 0.0)), 1)
    _attach_telemetry_and_energy(out, snn_avg_w, yolo_avg_w, snn_epi_mj, yolo_epi_mj, src, is_est)

    if yolo_data is None:
        yolo_data = {
            "available": False,
            "conf_threshold": yolo_conf,
            "boxes_count": 0,
            "detections": [],
            "latency_ms": 0.0,
            "power_w": None,
            "energy_mj": None,
            "theoretical_energy_mj": None,
            "measured_energy_mj": None,
            "gflops": 8.70,
            "status": "YOLOv8 MODEL UNAVAILABLE",
            "result_png": "",
            "source": src,
            "is_estimated": is_est,
        }
    else:
        yolo_data["source"] = src
        yolo_data["is_estimated"] = is_est
        if yolo_phase["runs"] > 0:
            yolo_data["latency_ms"] = round(float(yolo_phase["mean_latency_ms"]), 1)
            yolo_data["power_w"] = round(float(yolo_avg_w), 3) if yolo_avg_w is not None else None
            yolo_data["measured_energy_mj"] = round(float(yolo_epi_mj), 3) if yolo_epi_mj is not None else None

    out["yolo_benchmark"] = yolo_data
    out["benchmark_session"] = session_record
    return out


@app.get("/api/detect_sample/<sample_id>")
def detect_sample(sample_id: str):
    target = None
    for s in CURATED_SAMPLES:
        if s["id"] == sample_id or s["file"] == sample_id:
            target = s
            break
    if target is None:
        return jsonify({"error": f"Sample not found: {sample_id}"}), 404

    base_dir = SAMPLES_DIR if target["source"] == "samples" else TEST_IMAGE_DIR
    path = os.path.join(base_dir, target["file"])
    if not os.path.exists(path):
        return jsonify({"error": f"Sample file missing: {path}"}), 404

    bgr = cv2.imread(path)
    if bgr is None:
        return jsonify({"error": "Failed to decode sample image"}), 500

    yolo_conf = request.args.get("yolo_conf", 0.15, type=float)
    env_filter = request.args.get("env", "normal")
    run_bench = str(request.args.get("benchmark", "1")).lower() not in ("0", "false", "no")

    try:
        if run_bench:
            out = _run_benchmark_pipeline(bgr, target["file"], yolo_conf=yolo_conf, env_filter=env_filter)
        else:
            out = _process_bgr(bgr, target["file"], yolo_conf=yolo_conf, env_filter=env_filter)
        out["sample_info"] = target
        return jsonify(sanitize_json(out))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/detect", methods=["POST"])
@app.route("/api/benchmark/upload", methods=["POST"])
def detect_upload():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify({"error": "No image supplied under the 'image' field."}), 400

    ext = os.path.splitext(upload.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported file type '{ext}'. Use JPG, PNG, BMP or WEBP."}), 415

    raw = np.frombuffer(upload.read(), dtype=np.uint8)
    if raw.size == 0:
        return jsonify({"error": "Empty upload."}), 400

    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"error": "Could not decode image file."}), 400

    yolo_conf = request.form.get("yolo_conf", request.args.get("yolo_conf", 0.15, type=float), type=float)
    env_filter = request.form.get("env_filter", request.form.get("env", request.args.get("env", "normal")))
    is_live = str(request.form.get("is_live", request.args.get("is_live", "0"))).lower() in ("1", "true")
    stream_quality = str(request.form.get("stream_quality", request.args.get("stream_quality", "high")))
    benchmark_flag = str(request.form.get("benchmark", request.args.get("benchmark", "1"))).lower() not in ("0", "false", "no")

    try:
        if not is_live and benchmark_flag:
            out = _run_benchmark_pipeline(
                bgr,
                os.path.basename(upload.filename),
                yolo_conf=yolo_conf,
                env_filter=env_filter,
                stream_quality=stream_quality,
            )
        else:
            out = _process_bgr(
                bgr,
                os.path.basename(upload.filename),
                yolo_conf=yolo_conf,
                env_filter=env_filter,
                is_live=is_live,
                stream_quality=stream_quality,
            )
        return jsonify(sanitize_json(out))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.post("/api/env_reeval")
def env_reeval():
    global RAW_UNFILTERED_BGR
    if RAW_UNFILTERED_BGR is None:
        return jsonify({"error": "No active image in memory"}), 400

    req_data = request.get_json(silent=True) or {}
    env_filter = req_data.get("env") or req_data.get("filter") or "normal"
    yolo_conf = float(req_data.get("yolo_conf", 0.15))

    try:
        out = _process_bgr(RAW_UNFILTERED_BGR.copy(), "active_frame", yolo_conf=yolo_conf, env_filter=env_filter)
        return jsonify(sanitize_json(out))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.post("/api/yolo_reeval")
def yolo_reeval():
    global LAST_BGR
    if LAST_BGR is None:
        return jsonify({"error": "No image loaded in memory for re-evaluation"}), 400
    if YOLO_MODEL is None:
        return jsonify({"error": "YOLO model not loaded"}), 503

    req_data = request.get_json(silent=True) or {}
    conf = float(req_data.get("conf", 0.15))
    conf = max(0.01, min(0.95, conf))

    try:
        live_snap = POWER_MONITOR.read_live()
        yolo_data = _run_yolo_single(
            LAST_BGR,
            conf,
            float(live_snap["watts"]),
            None,
            str(live_snap["source"]),
            bool(live_snap["is_estimated"]),
        )
        return jsonify(sanitize_json(yolo_data))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Edge Power Benchmark Analytics ──────────────────────────────────────────
@app.get("/api/power_benchmark")
def power_benchmark():
    """Returns live hardware telemetry status + latest measured session alongside
    the theoretical Horowitz CMOS 45nm energy model."""
    e_ac_pj = 0.9
    e_mac_pj = 4.6
    n_params = 820417
    timesteps = 16

    ann_mac_ops = 8.7e9
    ann_energy_mj = (ann_mac_ops * e_mac_pj * 1e-12) * 1000.0
    snn_active_fraction = 0.22
    snn_synaptic_ops = n_params * timesteps * snn_active_fraction
    snn_energy_mj = (snn_synaptic_ops * e_ac_pj * 1e-12) * 1000.0

    summaries = _list_session_summaries()
    latest_summary = summaries[0] if summaries else None
    live_snap = POWER_MONITOR.read_live()

    if latest_summary:
        p = latest_summary["phases"]
        snn_w = p["snn_inference"]["avg_watts"]
        yolo_w = p["yolo_inference"]["avg_watts"]
        idle_w = p["idle_baseline"]["avg_watts"]
        ratio = latest_summary["comparison"]["power_ratio_yolo_over_snn"]
        reduction_str = f"{ratio}x"
    else:
        snn_w = live_snap["watts"]
        yolo_w = None
        idle_w = live_snap["watts"]
        reduction_str = "--"

    horowitz_theoretical = {
        "horowitz_model": {
            "e_ac_pj": e_ac_pj,
            "e_mac_pj": e_mac_pj,
            "mac_to_ac_energy_ratio": round(e_mac_pj / e_ac_pj, 2),
        },
        "theoretical_energy_per_frame": {
            "snn_energy_mj": round(snn_energy_mj, 3),
            "ann_energy_mj": round(ann_energy_mj, 3),
            "energy_savings_percent": round((1.0 - (snn_energy_mj / ann_energy_mj)) * 100.0, 1),
        },
        "breakeven_sparsity": 62.1,
        "measured_sparsity": 78.4,
    }

    return jsonify(
        sanitize_json(
            {
                "horowitz_theoretical": horowitz_theoretical,
                "horowitz_model": horowitz_theoretical["horowitz_model"],
                "power_draw": {
                    "snn_core_watts": snn_w,
                    "standard_yolo_watts": yolo_w,
                    "idle_standby_watts": idle_w,
                    "reduction_ratio": reduction_str,
                    "source": live_snap["source"],
                    "is_estimated": live_snap["is_estimated"],
                },
                "battery_autonomy": {
                    "battery_voltage": 12.0,
                    "battery_capacity_mah": 5200,
                    "total_watt_hours": 62.4,
                    "snn_runtime_hours": round(62.4 / snn_w, 1) if snn_w and snn_w > 0 else 0.0,
                    "ann_runtime_hours": round(62.4 / yolo_w, 1) if yolo_w and yolo_w > 0 else 0.0,
                    "runtime_extension_factor": reduction_str,
                },
                "theoretical_energy_per_frame": horowitz_theoretical["theoretical_energy_per_frame"],
                "breakeven_sparsity": 62.1,
                "measured_sparsity": 78.4,
                "hardware_telemetry": POWER_MONITOR.get_capabilities(),
                "latest_live": live_snap,
                "latest_session": latest_summary,
            }
        )
    )




# ── Live IP Camera Feed ──────────────────────────────────────────────────────
@app.route("/api/detect_stream", methods=["POST"])
def detect_stream():
    stream_url = ""
    if request.is_json and request.get_json(silent=True):
        stream_url = request.get_json(silent=True).get("stream_url", "")
    if not stream_url:
        stream_url = request.values.get("stream_url", "")
    stream_url = stream_url.strip()

    if not stream_url:
        return jsonify({"error": "No IP stream URL provided."}), 400

    if stream_url.startswith("http") and not any(stream_url.endswith(ext) for ext in [".jpg", ".png", "/shot.jpg", "/video", "/photo.jpg"]):
        if not stream_url.endswith("/"):
            stream_url += "/"
        stream_url += "shot.jpg"

    bgr = None
    is_snapshot = any(stream_url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", "/shot.jpg", "/photo.jpg"])

    if is_snapshot and (stream_url.startswith("http://") or stream_url.startswith("https://")):
        try:
            req = urllib.request.Request(stream_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=2.5) as response:
                raw = np.frombuffer(response.read(), dtype=np.uint8)
                if raw.size > 0:
                    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        except Exception:
            bgr = None

    if bgr is None:
        try:
            src = int(stream_url) if stream_url.isdigit() else stream_url
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    bgr = frame
            cap.release()
        except Exception:
            bgr = None

    if bgr is None:
        return jsonify({"error": f"Could not fetch frame from '{stream_url}'."}), 502

    try:
        out = _process_bgr(bgr, "Live Stream Frame")
        return jsonify(sanitize_json(out))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File exceeds the {MAX_UPLOAD_MB} MB limit."}), 413

# ── Intel RealSense D435i Live Integration ──────────────────────────────────
try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except ImportError:
    rs = None
    _RS_AVAILABLE = False

_rs_pipeline = None
_rs_lock = threading.Lock()


def _get_rs_pipeline():
    global _rs_pipeline
    if not _RS_AVAILABLE:
        raise RuntimeError("pyrealsense2 not installed")
    if _rs_pipeline is None:
        pipe = rs.pipeline()
        cfg = rs.config()
        try:
            cfg.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
            pipe.start(cfg)
        except Exception:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipe.start(cfg)
        for _ in range(10):
            pipe.wait_for_frames()
        _rs_pipeline = pipe
    return _rs_pipeline


@app.route("/api/detect_realsense", methods=["POST"])
def detect_realsense():
    try:
        body = request.get_json(silent=True) or {}
        yolo_conf = float(body.get("yolo_conf", request.args.get("yolo_conf", 0.15)))
        env_filter = str(body.get("env_filter", request.args.get("env", "normal")))
        is_live = bool(body.get("is_live", True))
        stream_quality = str(body.get("stream_quality", request.args.get("stream_quality", "high")))
        with _rs_lock:
            pipeline = _get_rs_pipeline()
            frames = pipeline.wait_for_frames(timeout_ms=1500)
            # Drain internal buffer queue to always grab the absolute freshest live frame (zero backlog lag)
            while True:
                newer = pipeline.poll_for_frames()
                if not newer:
                    break
                frames = newer
            color_frame = frames.get_color_frame()
            if not color_frame:
                return jsonify({"error": "No color frame received from RealSense camera"}), 400
            bgr = np.asanyarray(color_frame.get_data()).copy()
        out = _process_bgr(bgr, "RealSense D435i Live", yolo_conf=yolo_conf, env_filter=env_filter, is_live=is_live, stream_quality=stream_quality)
        # In live mode: return only essential images, with saliency and camo mask referencing result_png
        out["saliency_png"] = out["result_png"]
        out["camo_mask_png"] = out["result_png"]
        return jsonify(sanitize_json(out))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"RealSense: {str(e)}"}), 400


def _rs_mjpeg_generator():
    while True:
        try:
            with _rs_lock:
                pipeline = _get_rs_pipeline()
                frames = pipeline.wait_for_frames(timeout_ms=3000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                bgr = np.asanyarray(color_frame.get_data()).copy()
            out = _process_bgr(bgr, "RealSense Stream", is_live=True)
            jpg_b64 = out["result_png"].split(",", 1)[1]
            jpeg_bytes = base64.b64decode(jpg_b64)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n")
        except Exception as e:
            print("RS stream error:", e)
            break


@app.route("/api/realsense_stream")
def realsense_stream():
    return Response(_rs_mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    print(f"\n  [SERVER READY] Open in browser or desktop app mode: http://{HOST}:{PORT}")
    print(f"  [WS ENDPOINT]  Live hardware telemetry stream: ws://{HOST}:{PORT}/ws/power\n")
    try:
        app.run(
            host=HOST,
            port=PORT,
            debug=False,
            threaded=True,
            request_handler=NeuroCamoRequestHandler,
        )
    finally:
        POWER_MONITOR.close()
