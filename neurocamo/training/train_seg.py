"""
NeuroCamo - Mask-supervised training (v4-seg)
=============================================
Same model as train_dense.py, different SUPERVISION.

WHY
---
train_dense.py supervised the 16x16 heatmap with a centre Gaussian: roughly ONE
labelled cell per image. Measured over 4 epochs on 8884 images that gave train
loss 3.12 / val 3.34 - a 7% gap, i.e. underfitting, and IoU stuck at 26% against
a 38.5% mean-box baseline. The size head sat at L1 0.21, exactly the value for
predicting the dataset mean.

I ruled out the encoder: T=16 -> 32 -> 64 -> noise-free analog input all scored
26.8 / 26.3 / 26.0 / 23.4% IoU. Poisson noise is irrelevant. And a 24-image
overfit run reached 76% IoU, so the architecture and gradient path are fine.

The bottleneck is supervision density against a 93K-parameter backbone. COD10K
and NC4K both ship pixel masks that the box-only pipeline discarded. Training
the same heatmap channel as a downsampled OBJECT MASK gives all 256 cells a
label instead of one - a 256x increase in gradient signal per image, at zero
extra compute. The box is then read off the mask's extent (decode_mask), so
width and height are never regressed and cannot collapse to the dataset mean.

    python training/train_seg.py [epochs]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.cod10k_dataset import COD10KDataset          # noqa: E402
from utils.poisson_encoder import PoissonEncoder            # noqa: E402
from models.attention_bridge import NeuroCamoDenseModel     # noqa: E402
from training.train_dense import iou_xywh                   # noqa: E402

DATA_ROOTS = [
    os.path.join(ROOT, "data", "COD10K"),
    os.path.join(ROOT, "data", "NC4K_extracted"),
]
CHECKPOINT_OUT = os.path.join(ROOT, "checkpoints", "best_model_v4.pth")
LAST_OUT = os.path.join(ROOT, "checkpoints", "last_model_v4.pth")

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 9
# Weights to continue from. Supervision density was the bottleneck, not the
# optimiser, so a short run leaves real headroom: resuming keeps the 4-epoch
# solution and spends new epochs tightening the mask instead of relearning it.
RESUME = os.environ.get("NEUROCAMO_RESUME", "")
BATCH_SIZE = 16
LR = 2e-3
WEIGHT_DECAY = 1e-4
TIMESTEPS = 16
IMG_SIZE = 128
GRID = 16
NUM_WORKERS = 4
VAL_SUBSET = 640
SPARSITY_TARGET = 0.85
W_BCE, W_DICE, W_SPARSITY = 1.0, 1.0, 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seg_loss(logits: torch.Tensor, target: torch.Tensor, sparsity):
    """BCE + soft dice on the 16x16 occupancy mask.

    BCE alone is dominated by background cells (objects average ~25% of the
    grid, and 1/3 of frames are pure background). Dice is scale-invariant and
    keeps small objects from being ignored, so the two together give a stable
    gradient for both classes.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target)

    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    dice = 1.0 - (2 * inter + 1.0) / (p.sum(1) + t.sum(1) + 1.0)

    sp = torch.relu(torch.tensor(SPARSITY_TARGET, device=logits.device) - sparsity)
    total = W_BCE * bce + W_DICE * dice.mean() + W_SPARSITY * sp
    return total, {"bce": float(bce.item()), "dice": float(dice.mean().item())}


@torch.no_grad()
def validate(model, encoder, loader):
    model.eval()
    P, G, C, L, losses = [], [], [], [], []
    for imgs, bbox, conf, mask in loader:
        imgs, bbox = imgs.to(DEVICE), bbox.to(DEVICE)
        conf, mask = conf.to(DEVICE), mask.to(DEVICE)
        b, c, aux, sp = model(encoder(imgs))
        loss, _ = seg_loss(aux["heatmap"], mask, sp)
        losses.append(float(loss.item()))
        P.append(b.cpu().numpy()); G.append(bbox.cpu().numpy())
        C.append(c.cpu().numpy().ravel()); L.append(conf.cpu().numpy().ravel())

    P = np.concatenate(P); G = np.concatenate(G)
    C = np.concatenate(C); L = np.concatenate(L)
    pos = L > 0.5

    ious = iou_xywh(P[pos], G[pos]) if pos.any() else np.zeros(1)
    base = iou_xywh(np.tile(G[pos].mean(0), (pos.sum(), 1)), G[pos]).mean() if pos.any() else 0.0
    sep = (C[pos].mean() - C[~pos].mean()) if (pos.any() and (~pos).any()) else 0.0

    return {
        "loss": float(np.mean(losses)), "iou": float(ious.mean()),
        "iou50": float((ious >= 0.5).mean()), "baseline_iou": float(base),
        "sep": float(sep),
        "conf_pos": float(C[pos].mean()) if pos.any() else 0.0,
        "conf_neg": float(C[~pos].mean()) if (~pos).any() else 0.0,
        "centre_std": float(P[pos, :2].std(0).mean()) if pos.any() else 0.0,
        "wh_mean": P[pos, 2:].mean(0) if pos.any() else np.zeros(2),
    }


def main():
    print("=" * 78)
    print("  NeuroCamo v4-seg - mask-supervised dense head")
    print("=" * 78)
    print(f"  device: {DEVICE}"
          f"{' / ' + torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else ''}")

    train_ds = COD10KDataset(DATA_ROOTS, split="Train", img_size=IMG_SIZE,
                             augment=True, return_mask=True, mask_grid=GRID)
    val_full = COD10KDataset(DATA_ROOTS, split="Test", img_size=IMG_SIZE,
                             augment=False, return_mask=True, mask_grid=GRID)
    stride = max(1, len(val_full) // VAL_SUBSET)
    val_ds = Subset(val_full, list(range(0, len(val_full), stride)))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    print(f"  validation subset: {len(val_ds)} frames (stride {stride})")

    model = NeuroCamoDenseModel(beta=0.9, grid=GRID, mask_mode=True).to(DEVICE)
    encoder = PoissonEncoder(timesteps=TIMESTEPS, gain=1.0).to(DEVICE)
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  epochs: {EPOCHS}")

    if RESUME:
        ck = torch.load(RESUME, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        print(f"  resumed from {os.path.basename(RESUME)}"
              f" (epoch {ck.get('epoch', '?')}, IoU {ck.get('val_iou', 0.0)*100:.1f}%)")

    # A fresh OneCycle at max_lr would undo the resumed solution in its warm-up,
    # and `best` starting at -1.0 would let a worse epoch-1 model overwrite a
    # good checkpoint. Both are pinned to the resumed run's actual state.
    max_lr = LR * 0.3 if RESUME else LR
    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr, total_steps=EPOCHS * len(train_loader), pct_start=0.3)

    os.makedirs(os.path.dirname(CHECKPOINT_OUT), exist_ok=True)
    best = float(torch.load(RESUME, map_location="cpu", weights_only=False).get("score", -1.0)) if RESUME else -1.0
    if RESUME:
        print(f"  max_lr {max_lr:.1e}  /  must beat score {best:.4f} to overwrite")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        run, acc, n = 0.0, {"bce": 0.0, "dice": 0.0}, 0

        for i, (imgs, bbox, conf, mask) in enumerate(train_loader):
            imgs = imgs.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            _, _, aux, sp = model(encoder(imgs))
            loss, d = seg_loss(aux["heatmap"], mask, sp)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()

            run += float(loss.item()); n += 1
            for k in acc:
                acc[k] += d[k]
            if (i + 1) % 150 == 0:
                print(f"    epoch {epoch:2d} step {i+1:4d}/{len(train_loader)}  "
                      f"loss {run/n:.4f}  bce {acc['bce']/n:.4f}  dice {acc['dice']/n:.4f}")

        v = validate(model, encoder, val_loader)
        score = v["iou"] + 0.5 * v["sep"]
        mark = ""
        if score > best:
            best = score
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch,
                "val_loss": v["loss"], "val_iou": v["iou"], "val_iou50": v["iou50"],
                "val_sep": v["sep"], "score": score,
                "version": "v4seg_cod10k_nc4k", "grid": GRID, "img_size": IMG_SIZE,
                "head": "NeuroCamoDenseHead", "mask_mode": True,
            }, CHECKPOINT_OUT)
            mark = "  <- saved"
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                    "version": "v4seg_cod10k_nc4k", "head": "NeuroCamoDenseHead",
                    "mask_mode": True}, LAST_OUT)

        print(f"  epoch {epoch:2d}/{EPOCHS}  {time.time()-t0:6.1f}s  train {run/n:.4f}  "
              f"val {v['loss']:.4f}  IoU {v['iou']*100:5.1f}% (base {v['baseline_iou']*100:4.1f}%)  "
              f"IoU@50 {v['iou50']*100:5.1f}%  conf {v['conf_pos']:.3f}/{v['conf_neg']:.3f} "
              f"sep {v['sep']:+.3f}  cstd {v['centre_std']:.3f}  "
              f"wh {np.round(v['wh_mean'], 2)}{mark}")

    print(f"\n  done. best score {best:.4f} -> {CHECKPOINT_OUT}")


if __name__ == "__main__":
    main()
