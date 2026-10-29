"""
NeuroCamo Web — Inference core
==============================
Wraps the existing NeuroCamo SNN pipeline (Poisson encoder -> spiking CNN
backbone -> ASAB -> detection head) for use from a web request handler.

The detection logic is a direct port of neurocamo/utils/test_gui.py so the web
verdicts match the desktop Tkinter tool exactly. The one addition is the spike
spectrogram: the backbone is called explicitly so we can keep the layer-3 spike
train (T, B, 128, 16, 16) and reduce it to a channel x timestep firing-rate
matrix — the SNN analogue of an audio spectrogram.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch

# ── Locate the NeuroCamo project so its modules import unchanged ─────────────
NEUROCAMO_ROOT = os.environ.get(
    "NEUROCAMO_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "neurocamo"),
)
if NEUROCAMO_ROOT not in sys.path:
    sys.path.insert(0, NEUROCAMO_ROOT)

from utils.poisson_encoder import PoissonEncoder          # noqa: E402
from models.attention_bridge import (                      # noqa: E402
    NeuroCamoFullModel,
    NeuroCamoDenseModel,
)

ckpt_v4 = os.path.join(NEUROCAMO_ROOT, "checkpoints", "best_model_v4.pth")
ckpt_v3 = os.path.join(NEUROCAMO_ROOT, "checkpoints", "best_model_v3.pth")
ckpt_v2 = os.path.join(NEUROCAMO_ROOT, "checkpoints", "best_model_v2.pth")


def _default_checkpoint() -> str:
    for p in (ckpt_v4, ckpt_v3, ckpt_v2):
        if os.path.exists(p):
            return p
    return ckpt_v2


CHECKPOINT_PATH = os.environ.get("NEUROCAMO_CHECKPOINT", _default_checkpoint())

TIMESTEPS = 16
INPUT_SIZE = 128

# ── Texture heuristics (unchanged from test_gui.py) ──────────────────────────
TEXTURE_LOW_CUTOFF = 150.0
TEXTURE_HIGH_CUTOFF = 400.0


def texture_score(bgr_img: np.ndarray) -> float:
    """Laplacian variance = how complex/organic the image texture is."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def background_is_plain(bgr_img: np.ndarray, corner_size: int = 50, threshold: float = 15.0) -> bool:
    """True when the TOP corners are uniform — i.e. a product photo on a plain
    backdrop. Bottom corners are ignored because they often carry watermarks."""
    h, w = bgr_img.shape[:2]
    cs = min(corner_size, h, w // 2)
    corners = [bgr_img[0:cs, 0:cs], bgr_img[0:cs, w - cs:w]]
    return float(np.mean([c.std() for c in corners])) < threshold


def has_open_sky(
    bgr_img: np.ndarray,
    top_fraction: float = 0.20,
    brightness_thresh: float = 160.0,
    texture_thresh: float = 150.0,
) -> bool:
    """True when the top strip is bright and smooth (open sky). Camouflaged
    subjects are ground-level, so visible sky argues against a camo scene."""
    h = bgr_img.shape[0]
    top = bgr_img[0:max(1, int(h * top_fraction)), :]
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    tex = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness > brightness_thresh and tex < texture_thresh


# ── Model bundle ─────────────────────────────────────────────────────────────
@dataclass
class ModelInfo:
    device: str
    accelerator: str
    epoch: int
    val_loss: float
    checkpoint_loaded: bool
    checkpoint_path: str
    params: int
    timesteps: int
    # "dense-mask" (v4, mask-supervised heatmap) or "global-mlp" (v2/v3)
    head: str = "global-mlp"
    version: str = ""
    val_iou: float = 0.0
    # parameters present in the model but absent from the checkpoint, so they
    # kept their initialisation values instead of trained weights
    defaulted_keys: list = None
    unexpected_keys: list = None


class ModelBundle:
    """Loads the encoder + SNN once at process start and holds them for reuse.

    Two head families exist on disk. v2/v3 checkpoints belong to
    NeuroCamoFullModel, which regresses one box from globally-pooled features.
    v4 belongs to NeuroCamoDenseModel, whose 16x16 heatmap is trained as a
    downsampled object mask and whose box is read off that mask's extent. The
    checkpoint says which it is, so the right class is built before loading
    rather than after a failed load_state_dict.
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = PoissonEncoder(timesteps=TIMESTEPS, gain=1.0).to(self.device)

        ckpt = None
        if os.path.exists(CHECKPOINT_PATH):
            ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)

        head = str((ckpt or {}).get("head", ""))
        self.mask_mode = bool((ckpt or {}).get("mask_mode", False))
        self.dense = head == "NeuroCamoDenseHead"

        if self.dense:
            self.model = NeuroCamoDenseModel(
                beta=0.9, grid=int((ckpt or {}).get("grid", 16)),
                mask_mode=self.mask_mode,
            ).to(self.device)
        else:
            self.model = NeuroCamoFullModel(beta=0.9).to(self.device)

        epoch, val_loss, loaded = 0, 0.0, False
        defaulted: list[str] = []
        unexpected: list[str] = []
        if ckpt is not None:
            state = ckpt["model_state_dict"]
            try:
                self.model.load_state_dict(state)
            except RuntimeError:
                # The NeuroCamo model source can gain parameters after a
                # checkpoint was trained (e.g. the learned spatial-softmax
                # temperature). Fall back to a tolerant load so the site keeps
                # serving, and report exactly which tensors kept their
                # freshly-initialised defaults instead of trained values.
                result = self.model.load_state_dict(state, strict=False)
                defaulted = list(result.missing_keys)
                unexpected = list(result.unexpected_keys)
                print(f"[WARN] Checkpoint does not match the current model definition.")
                if defaulted:
                    print(f"       Using init defaults for: {', '.join(defaulted)}")
                if unexpected:
                    print(f"       Ignoring extra tensors: {', '.join(unexpected)}")
            epoch = int(ckpt.get("epoch", 1))
            val_loss = float(ckpt.get("val_loss", 0.0))
            loaded = True
            print(f"[OK] Loaded checkpoint - epoch {epoch}, val_loss {val_loss:.4f}"
                  f"{'  [dense-mask head]' if self.dense else ''}")
        else:
            print(f"[WARN] No checkpoint at {CHECKPOINT_PATH} - random weights.")

        self.model.eval()

        self.info = ModelInfo(
            device=self.device.type,
            accelerator=(
                torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
            ),
            epoch=epoch,
            val_loss=val_loss,
            checkpoint_loaded=loaded,
            checkpoint_path=CHECKPOINT_PATH,
            params=sum(p.numel() for p in self.model.parameters()),
            timesteps=TIMESTEPS,
            head="dense-mask" if self.dense else "global-mlp",
            version=str((ckpt or {}).get("version", "")),
            val_iou=float((ckpt or {}).get("val_iou", 0.0)),
            defaulted_keys=defaulted,
            unexpected_keys=unexpected,
        )


# ── Spike spectrogram ────────────────────────────────────────────────────────
def spike_spectrogram(spikes_l3: torch.Tensor) -> dict:
    """Reduce the layer-3 spike train to a channel x timestep firing-rate map.

    spikes_l3 is (T, B, C, H, W) binary. Averaging over the 16x16 spatial grid
    gives each of the 128 semantic channels a firing rate per timestep — one
    column per timestep, one row per channel. That is the direct analogue of an
    audio spectrogram's frequency-bin x frame matrix, so the same banded
    colormap and hover probe from the Echolyx spectrogram apply unchanged.
    """
    # (T, C) after collapsing batch and space
    rates = spikes_l3[:, 0].mean(dim=(2, 3))          # (T, C)
    mat = rates.transpose(0, 1).cpu().numpy()          # (C, T) rows = channels

    n_channels, n_steps = mat.shape
    peak = float(mat.max())
    scale = peak if peak > 1e-8 else 1.0

    quantized = np.clip(np.rint(mat / scale * 255.0), 0, 255).astype(np.uint8)

    # population firing rate per timestep — the "loudness envelope" of the train
    per_step = spikes_l3[:, 0].mean(dim=(1, 2, 3)).cpu().numpy()

    return {
        "n_channels": n_channels,
        "n_steps": n_steps,
        "peak_rate": peak,
        "mean_rate": float(mat.mean()),
        # row-major: row 0 = channel 0, each row has n_steps entries
        "data": quantized.flatten().tolist(),
        "per_step_rate": [round(float(v), 6) for v in per_step],
        "active_channels": int((mat.mean(axis=1) > 0.01).sum()),
    }


# ── Verdict palette, matched to the Echolyx design language ──────────────────
VERDICT_STYLE = {
    "DETECTED":          {"rgb": (0, 229, 160),  "band": "normal"},
    "POSSIBLE TARGET":   {"rgb": (245, 163, 0),  "band": "stress"},
    "NO CAMO DETECTED":  {"rgb": (255, 68, 85),  "band": "fault"},
}


def run_inference(bundle: ModelBundle, bgr: np.ndarray) -> dict:
    """Run the NeuroCamo SNN on one decoded BGR image.

    Returns a JSON-serialisable dict plus an 'annotated_rgb' numpy array that
    the caller encodes to PNG.
    """
    t0 = time.perf_counter()

    orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = bgr.shape[:2]

    # Step 1 — texture complexity of the full-resolution image
    tex = texture_score(bgr)

    # Step 2 — Poisson encode and run the spiking pipeline. The backbone is
    # called explicitly (rather than model(spikes)) so the layer-3 spike train
    # stays available for the spectrogram.
    resized = cv2.resize(orig_rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(bundle.device)

    with torch.no_grad():
        spikes = bundle.encoder(tensor)
        spikes_l3, mem_l3, sparsity = bundle.model.backbone(spikes)
        fused = bundle.model.asab(spikes_l3, mem_l3)
        if bundle.dense:
            hm, off, wh = bundle.model.head(fused)
            if bundle.mask_mode:
                bbox_pred, conf_pred = bundle.model.decode_mask(hm)
            else:
                bbox_pred, conf_pred = bundle.model.decode(hm, off, wh)
            heat = torch.sigmoid(hm)[0, 0].cpu().numpy()
        else:
            bbox_pred, conf_pred, _ = bundle.model.head(fused)
            heat = None

    xc, yc, bw, bh = [float(v) for v in bbox_pred[0].cpu().numpy()]
    spec = spike_spectrogram(spikes_l3)

    snn_conf = float(conf_pred[0].item())

    # Step 3 — camouflage scene gate.
    #
    # The gate exists only because the v2/v3 global-MLP head emitted ~0.998
    # objectness on literally any input, so a hand-written scene filter was the
    # only thing separating a hit from a miss. The v4 mask head produces a real
    # per-cell objectness, and these heuristics actively destroy correct
    # detections in flat or bright scenes (a snow hare, a sky-lined drone), so
    # they are bypassed entirely for the dense head.
    plain_bg = background_is_plain(bgr)
    sky = has_open_sky(bgr)

    if bundle.dense:
        tex_weight = 1.0
        gate_reason = "Dense mask head - objectness read from the heatmap, no scene gate"
    elif plain_bg:
        tex_weight, gate_reason = 0.0, "Plain backdrop - product-style photo, not a camo scene"
    elif sky:
        tex_weight, gate_reason = 0.0, "Open sky in frame - subject is not ground-concealed"
    elif tex < 300.0:
        tex_weight, gate_reason = 0.0, "Texture too flat - no organic concealment structure"
    else:
        tex_weight, gate_reason = 1.0, "Ground-level complex scene - camouflage candidate"

    conf = snn_conf * tex_weight

    # Step 4 — box straight from the head, no reshaping. The head's centre is
    # near-constant at (0.50, 0.50) on every image (predicted std 0.005 against
    # a ground-truth std of 0.06), so the box below is honest but not useful
    # until the head is retrained. Nothing here invents a size or a position.
    x1 = int(max(0, (xc - bw / 2.0) * w_orig))
    x2 = int(min(w_orig - 1, (xc + bw / 2.0) * w_orig))
    y1 = int(max(0, (yc - bh / 2.0) * h_orig))
    y2 = int(min(h_orig - 1, (yc + bh / 2.0) * h_orig))

    if conf >= 0.6:
        verdict = "DETECTED"
    elif conf >= 0.25:
        verdict = "POSSIBLE TARGET"
    else:
        verdict = "NO CAMO DETECTED"

    color = VERDICT_STYLE[verdict]["rgb"]

    # Annotate a clean copy at original resolution (NO HEATMAP OVERLAY)
    result = orig_rgb.copy()
    thickness = max(2, round(min(h_orig, w_orig) / 320))
    font_scale = max(0.5, min(h_orig, w_orig) / 900)

    if conf >= 0.25:
        # Draw bounding box & target badge
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        label = f"NEUROCAMO {conf * 100:.1f}%  {verdict}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        pad = max(4, thickness * 2)
        ty2 = max(th + pad * 2, y1)
        cv2.rectangle(result, (x1, max(0, ty2 - th - pad * 2)), (x1 + tw + pad * 2, ty2), color, -1)
        cv2.putText(result, label, (x1 + pad, ty2 - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (6, 6, 8), thickness)
    # If conf < 0.25: No camo detected — keep image clean without overlay

    latency_ms = (time.perf_counter() - t0) * 1000.0

    # The 16x16 objectness map, quantised to bytes for the browser. Unlike the
    # v2 head's single pooled number this is a genuine spatial saliency map: it
    # is the exact tensor the box was decoded from, so the overlay cannot
    # disagree with the box.
    heatmap = None
    if heat is not None:
        heatmap = {
            "grid": int(heat.shape[0]),
            "peak": round(float(heat.max()), 6),
            "mean": round(float(heat.mean()), 6),
            "data": np.clip(np.rint(heat * 255.0), 0, 255).astype(np.uint8).flatten().tolist(),
        }

    return {
        "annotated_rgb": result,
        "verdict": verdict,
        "band": VERDICT_STYLE[verdict]["band"],
        "confidence": round(conf, 6),
        "snn_confidence": round(snn_conf, 6),
        "head": bundle.info.head,
        "heatmap": heatmap,
        "texture_weight": tex_weight,
        "gate_reason": gate_reason,
        "texture_score": round(tex, 2),
        "sparsity": round(float(sparsity), 6),
        "plain_background": True if plain_bg else False,
        "open_sky": True if sky else False,
        "bbox_drawn": True if conf >= 0.25 else False,
        "bbox_px": [int(x1), int(y1), int(x2), int(y2)],
        "bbox_norm": [float(xc), float(yc), float(bw), float(bh)],
        "image_size": [int(w_orig), int(h_orig)],
        "latency_ms": round(latency_ms, 1),
        "spectrogram": spec,
        "thresholds": {
            "detected": 0.6,
            "possible": 0.25,
            "texture_low": TEXTURE_LOW_CUTOFF,
            "texture_high": TEXTURE_HIGH_CUTOFF,
            "texture_gate": 300.0,
        },
    }
