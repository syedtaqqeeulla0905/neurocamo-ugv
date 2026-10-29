"""
NeuroCamo - Dense-head training (v4)
====================================
Trains NeuroCamoDenseModel on COD10K + NC4K.

WHAT CHANGED VS train_cod10k_full.py, AND WHY
---------------------------------------------
1. Labels are correct now. cod10k_dataset._mask_to_bbox() used to return
   [0.5, 0.5, 0.1, 0.1] for an empty mask, which made has_object come out as
   1.0. COD10K's Train/Image folder is 2960 NonCAM backgrounds out of 6000, so
   half of every epoch taught "object present, dead centre, 10% of frame,
   confidence 1.0". That produced the collapsed detector.

2. The head is dense. Position is the argmax cell of a 16x16 heatmap plus a
   sub-cell offset, so it cannot regress to the dataset mean.

3. Negatives are in-domain. The 2960 NonCAM frames come from the same cameras
   and scenes as the positives and go through the identical /255.0 path, so
   there is no intensity shortcut. The 77 downloaded COCO photos are no longer
   needed and are not used.

4. Validation is honest. It contains negatives, reports REAL IoU (not
   `1 - ciou_loss`), and checkpoints are selected on a composite detection
   score rather than on val_loss.

    python training/train_dense.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.cod10k_dataset import COD10KDataset          # noqa: E402
from utils.poisson_encoder import PoissonEncoder            # noqa: E402
from models.attention_bridge import NeuroCamoDenseModel     # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOTS = [
    os.path.join(ROOT, "data", "COD10K"),
    os.path.join(ROOT, "data", "NC4K_extracted"),
]
CHECKPOINT_OUT = os.path.join(ROOT, "checkpoints", "best_model_v4.pth")
LAST_OUT = os.path.join(ROOT, "checkpoints", "last_model_v4.pth")

EPOCHS = 30
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
TIMESTEPS = 16
IMG_SIZE = 128
GRID = 16
NUM_WORKERS = 4
VAL_SUBSET = 1200          # cap the val pass so epochs stay reasonable
SPARSITY_TARGET = 0.85

W_HEATMAP, W_OFFSET, W_SIZE, W_SPARSITY = 1.0, 1.0, 5.0, 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Heatmap target construction ───────────────────────────────────────────────
def build_targets(bbox: torch.Tensor, has_obj: torch.Tensor, grid: int = GRID):
    """
    bbox    (B, 4) [xc, yc, w, h] in [0,1]; w=h=0 for a background frame
    has_obj (B, 1) 1.0 / 0.0

    Returns
      hm      (B, 1, G, G)  Gaussian-splatted centre target in [0, 1]
      reg     (B, 4)        [off_x, off_y, w, h] for the peak cell
      cell    (B,)          flat index of the peak cell
      pos     (B,) bool     which samples carry a box
    """
    B = bbox.shape[0]
    pos = has_obj.view(-1) > 0.5

    xs = (bbox[:, 0] * grid).clamp(0, grid - 1e-4)
    ys = (bbox[:, 1] * grid).clamp(0, grid - 1e-4)
    col = xs.floor()
    row = ys.floor()

    reg = torch.stack([xs - col, ys - row, bbox[:, 2], bbox[:, 3]], dim=1)
    cell = (row * grid + col).long()

    # Gaussian radius scaled to object size: big objects get a broader, more
    # forgiving peak; small ones a tight one. Same idea as CenterNet.
    sigma = ((bbox[:, 2] + bbox[:, 3]) * 0.5 * grid / 6.0).clamp(min=0.6)

    gy = torch.arange(grid, device=bbox.device).view(1, grid, 1).float() + 0.5
    gx = torch.arange(grid, device=bbox.device).view(1, 1, grid).float() + 0.5
    d2 = (gx - xs.view(B, 1, 1)) ** 2 + (gy - ys.view(B, 1, 1)) ** 2
    hm = torch.exp(-d2 / (2 * sigma.view(B, 1, 1) ** 2))

    # The Gaussian is centred on the true sub-cell location, so the peak cell's
    # value is generally < 1.0. Pin it to exactly 1.0: focal_heatmap_loss picks
    # positives with target.ge(0.999), and without this there would be no
    # positive term at all and the heatmap would only learn "background".
    hm = hm.view(B, -1)
    hm.scatter_(1, cell.view(B, 1), 1.0)
    hm = hm.view(B, grid, grid)

    hm = hm * pos.view(B, 1, 1).float()          # backgrounds stay all-zero
    return hm.unsqueeze(1), reg, cell, pos


def focal_heatmap_loss(logits: torch.Tensor, target: torch.Tensor,
                       alpha: float = 2.0, beta: float = 4.0):
    """CenterNet penalty-reduced focal loss. Normalised by the number of peaks
    so an all-background batch does not dominate."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    is_peak = target.ge(0.999).float()

    pos_loss = -((1 - p) ** alpha) * torch.log(p) * is_peak
    neg_loss = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * (1 - is_peak)

    n_peaks = is_peak.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_peaks


def gather_cell(maps: torch.Tensor, cell: torch.Tensor):
    """maps (B, C, G, G) -> (B, C) values at the given flat cell index."""
    B, C = maps.shape[0], maps.shape[1]
    return maps.view(B, C, -1).gather(2, cell.view(B, 1, 1).expand(B, C, 1)).squeeze(-1)


def compute_loss(hm, off, wh, bbox_gt, conf_gt, sparsity):
    hm_t, reg_t, cell, pos = build_targets(bbox_gt, conf_gt)

    loss_hm = focal_heatmap_loss(hm, hm_t)

    if pos.any():
        off_p = gather_cell(off, cell)[pos]
        wh_p = gather_cell(wh, cell)[pos]
        loss_off = F.l1_loss(off_p, reg_t[pos, :2])
        loss_wh = F.l1_loss(wh_p, reg_t[pos, 2:])
    else:
        loss_off = hm.sum() * 0.0
        loss_wh = hm.sum() * 0.0

    loss_sp = torch.relu(torch.tensor(SPARSITY_TARGET, device=hm.device) - sparsity)

    total = (W_HEATMAP * loss_hm + W_OFFSET * loss_off
             + W_SIZE * loss_wh + W_SPARSITY * loss_sp)
    return total, {
        "hm": float(loss_hm.item()),
        "off": float(loss_off.item()),
        "wh": float(loss_wh.item()),
    }


# ── Real IoU, on object-bearing frames only ───────────────────────────────────
def iou_xywh(p: np.ndarray, g: np.ndarray) -> np.ndarray:
    def to_xyxy(a):
        return np.stack([a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2,
                         a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2], axis=1)
    P, G = to_xyxy(p), to_xyxy(g)
    x1 = np.maximum(P[:, 0], G[:, 0]); y1 = np.maximum(P[:, 1], G[:, 1])
    x2 = np.minimum(P[:, 2], G[:, 2]); y2 = np.minimum(P[:, 3], G[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ap = np.clip(P[:, 2] - P[:, 0], 0, None) * np.clip(P[:, 3] - P[:, 1], 0, None)
    ag = np.clip(G[:, 2] - G[:, 0], 0, None) * np.clip(G[:, 3] - G[:, 1], 0, None)
    return inter / np.maximum(ap + ag - inter, 1e-9)


@torch.no_grad()
def validate(model, encoder, loader, max_samples=VAL_SUBSET):
    model.eval()
    preds, gts, confs, labs, losses = [], [], [], [], []
    seen = 0
    for imgs, bbox, conf in loader:
        imgs, bbox, conf = imgs.to(DEVICE), bbox.to(DEVICE), conf.to(DEVICE)
        b, c, aux, sp = model(encoder(imgs))
        loss, _ = compute_loss(aux["heatmap"], aux["offset"], aux["size"], bbox, conf, sp)
        losses.append(float(loss.item()))
        preds.append(b.cpu().numpy()); gts.append(bbox.cpu().numpy())
        confs.append(c.cpu().numpy().ravel()); labs.append(conf.cpu().numpy().ravel())
        seen += imgs.size(0)
        if seen >= max_samples:
            break

    P = np.concatenate(preds); G = np.concatenate(gts)
    C = np.concatenate(confs); L = np.concatenate(labs)
    pos = L > 0.5

    m_iou = float(iou_xywh(P[pos], G[pos]).mean()) if pos.any() else 0.0
    iou50 = float((iou_xywh(P[pos], G[pos]) >= 0.5).mean()) if pos.any() else 0.0
    sep = (float(C[pos].mean()) - float(C[~pos].mean())) if (pos.any() and (~pos).any()) else 0.0

    # Mean-box baseline on the same frames: what "ignore the image" scores.
    base = float(iou_xywh(np.tile(G[pos].mean(0), (pos.sum(), 1)), G[pos]).mean()) if pos.any() else 0.0

    return {
        "loss": float(np.mean(losses)),
        "iou": m_iou,
        "iou50": iou50,
        "baseline_iou": base,
        "sep": sep,
        "conf_pos": float(C[pos].mean()) if pos.any() else 0.0,
        "conf_neg": float(C[~pos].mean()) if (~pos).any() else 0.0,
        "centre_std": P[pos, :2].std(0).mean() if pos.any() else 0.0,
        "n_pos": int(pos.sum()), "n_neg": int((~pos).sum()),
    }


def main():
    print("=" * 74)
    print("  NeuroCamo v4 - dense heatmap head")
    print("=" * 74)
    print(f"  device: {DEVICE}"
          f"{' / ' + torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else ''}")

    train_ds = COD10KDataset(DATA_ROOTS, split="Train", img_size=IMG_SIZE, augment=True)
    val_ds = COD10KDataset(DATA_ROOTS, split="Test", img_size=IMG_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    # Deterministic, class-mixed validation subset. Taking the first N would be
    # all CAM frames (the Test folder sorts CAM before NonCAM), and shuffling
    # would make the epoch-to-epoch numbers noisy and non-comparable.
    stride = max(1, len(val_ds) // VAL_SUBSET)
    val_ds = torch.utils.data.Subset(val_ds, list(range(0, len(val_ds), stride)))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    print(f"  validation subset: {len(val_ds)} frames (stride {stride})")

    model = NeuroCamoDenseModel(beta=0.9, grid=GRID).to(DEVICE)
    encoder = PoissonEncoder(timesteps=TIMESTEPS, gain=1.0).to(DEVICE)
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=EPOCHS * len(train_loader), pct_start=0.25)

    os.makedirs(os.path.dirname(CHECKPOINT_OUT), exist_ok=True)
    best_score = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        run, parts, n = 0.0, {"hm": 0.0, "off": 0.0, "wh": 0.0}, 0

        for i, (imgs, bbox, conf) in enumerate(train_loader):
            imgs, bbox, conf = imgs.to(DEVICE, non_blocking=True), bbox.to(DEVICE), conf.to(DEVICE)
            _, _, aux, sp = model(encoder(imgs))
            loss, d = compute_loss(aux["heatmap"], aux["offset"], aux["size"], bbox, conf, sp)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()

            run += float(loss.item()); n += 1
            for k in parts:
                parts[k] += d[k]
            if (i + 1) % 100 == 0:
                print(f"    epoch {epoch:2d} step {i+1:4d}/{len(train_loader)}  "
                      f"loss {run/n:.4f}  hm {parts['hm']/n:.4f}  "
                      f"off {parts['off']/n:.4f}  wh {parts['wh']/n:.4f}")

        v = validate(model, encoder, val_loader)
        # Composite: localisation quality AND a usable objectness signal.
        score = v["iou"] + 0.5 * v["sep"]
        mark = ""
        if score > best_score:
            best_score = score
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_loss": v["loss"], "val_iou": v["iou"],
                "val_iou50": v["iou50"], "val_sep": v["sep"], "score": score,
                "version": "v4_dense_cod10k_nc4k", "grid": GRID, "img_size": IMG_SIZE,
                "head": "NeuroCamoDenseHead",
            }, CHECKPOINT_OUT)
            mark = "  <- saved"
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                    "version": "v4_dense_cod10k_nc4k", "head": "NeuroCamoDenseHead"}, LAST_OUT)

        print(f"  epoch {epoch:2d}/{EPOCHS}  {time.time()-t0:6.1f}s  "
              f"train {run/n:.4f}  val {v['loss']:.4f}  "
              f"IoU {v['iou']*100:5.1f}% (baseline {v['baseline_iou']*100:4.1f}%)  "
              f"IoU@50 {v['iou50']*100:5.1f}%  "
              f"conf {v['conf_pos']:.3f}/{v['conf_neg']:.3f} sep {v['sep']:+.3f}  "
              f"cstd {v['centre_std']:.3f}{mark}")

    print("\n  done. best composite score %.4f -> %s" % (best_score, CHECKPOINT_OUT))
    print("  sanity check:  python ../neurocamo-web/tools/diagnose_head.py "
          "checkpoints/best_model_v4.pth")


if __name__ == "__main__":
    main()
