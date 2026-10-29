"""
Where does the centre signal die?

The head is: fused -> conv_block -> SpatialSoftmax (32 keypoints -> 64 coords)
                                -> concat with GAP feats (64) -> MLP -> [xc,yc,w,h]

If spatial_coords varies across images but the final xc/yc does not, the MLP is
throwing position away. If spatial_coords itself is constant, the softmax over
the 16x16 grid is too flat (or the temperature has collapsed) and no position
information ever reaches the MLP.

    python tools/probe_centre.py [checkpoint.pth ...]
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NEUROCAMO = os.path.join(os.path.dirname(os.path.dirname(HERE)), "neurocamo")
if NEUROCAMO not in sys.path:
    sys.path.insert(0, NEUROCAMO)

from datasets.cod10k_dataset import COD10KDataset          # noqa: E402
from utils.poisson_encoder import PoissonEncoder            # noqa: E402
from models.attention_bridge import NeuroCamoFullModel      # noqa: E402

COD10K_ROOT = os.path.join(NEUROCAMO, "data", "COD10K")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N, BATCH = 96, 16


def probe(ckpt_path):
    model = NeuroCamoFullModel(beta=0.9).to(DEVICE)
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()
    enc = PoissonEncoder(timesteps=16, gain=1.0).to(DEVICE)
    ss = model.head.spatial_softmax

    print(f"\n{'=' * 72}\n  {os.path.basename(ckpt_path)}\n{'=' * 72}")
    print(f"  learned temperature   : {float(ss.temperature.item()):.4f}  (init was 10.0)")

    ds = COD10KDataset(COD10K_ROOT, split="Test", img_size=128, augment=False)
    idx = np.linspace(0, len(ds) - 1, N).astype(int)

    coords, boxes, gts, peaks = [], [], [], []
    for s in range(0, len(idx), BATCH):
        chunk = idx[s:s + BATCH]
        imgs = torch.stack([ds[int(i)][0] for i in chunk]).to(DEVICE)
        gts.append(torch.stack([ds[int(i)][1] for i in chunk]).numpy())
        with torch.no_grad():
            spk, mem, _ = model.backbone(enc(imgs))
            fused = model.asab(spk, mem)
            feats = model.head.conv_block(fused)
            c = ss(feats)
            b, _, _ = model.head(fused)
            # how peaked is the softmax the keypoints are read from?
            f = ss.conv(feats)
            B, C, H, W = f.shape
            sm = torch.softmax(f.view(B, C, -1) * torch.abs(ss.temperature), dim=-1)
            peaks.append(sm.max(dim=-1).values.mean().item())
        coords.append(c.cpu().numpy())
        boxes.append(b.cpu().numpy())

    K = np.concatenate(coords)      # (N, 64) = 32 x-coords then 32 y-coords
    P = np.concatenate(boxes)
    G = np.concatenate(gts)

    print(f"  softmax peak mass     : {np.mean(peaks):.4f}"
          f"   (1/256 = {1/256:.4f} means perfectly flat = no localisation)")
    print(f"  keypoint x  mean/std  : {K[:, :32].mean():.4f} / {K[:, :32].std(0).mean():.4f}")
    print(f"  keypoint y  mean/std  : {K[:, 32:].mean():.4f} / {K[:, 32:].std(0).mean():.4f}")
    print("     ^ std across images, averaged over the 32 keypoints")
    print(f"  predicted xc std      : {P[:, 0].std():.4f}   (ground truth {G[:, 0].std():.4f})")
    print(f"  predicted yc std      : {P[:, 1].std():.4f}   (ground truth {G[:, 1].std():.4f})")
    print(f"  corr(pred xc, gt xc)  : {np.corrcoef(P[:, 0], G[:, 0])[0, 1]:+.3f}")
    print(f"  corr(pred yc, gt yc)  : {np.corrcoef(P[:, 1], G[:, 1])[0, 1]:+.3f}")
    print(f"  corr(pred w,  gt w )  : {np.corrcoef(P[:, 2], G[:, 2])[0, 1]:+.3f}")
    print(f"  corr(pred h,  gt h )  : {np.corrcoef(P[:, 3], G[:, 3])[0, 1]:+.3f}")
    print("     ^ near 0 = that coordinate is not being predicted at all")


if __name__ == "__main__":
    targets = sys.argv[1:] or [
        os.path.join(NEUROCAMO, "checkpoints", "best_model_v2.pth"),
        os.path.join(NEUROCAMO, "checkpoints", "best_model_v3.pth"),
    ]
    print(f"device: {DEVICE}")
    for t in targets:
        probe(t)
