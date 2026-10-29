"""
Head diagnostic — is the detection head actually localising, or collapsing to a
constant box?

Runs a checkpoint over N COD10K test images and reports:
  * per-coordinate spread of the predictions (std across images)
  * true IoU against the mask-derived ground-truth box
  * what a fixed "predict the dataset mean box every time" baseline scores
  * confidence spread on positives vs. real (un-normalised) negatives

If the prediction std is tiny and the model's IoU is at or below the mean-box
baseline, the head is not looking at the image — it has regressed to the dataset
average, which is what "random spots" looks like on screen.

    python tools/diagnose_head.py [checkpoint.pth ...]
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NEUROCAMO = os.path.join(os.path.dirname(ROOT), "neurocamo")
for p in (NEUROCAMO,):
    if p not in sys.path:
        sys.path.insert(0, p)

from datasets.cod10k_dataset import COD10KDataset          # noqa: E402
from utils.poisson_encoder import PoissonEncoder            # noqa: E402
from models.attention_bridge import NeuroCamoFullModel      # noqa: E402

COD10K_ROOT = os.path.join(NEUROCAMO, "data", "COD10K")
NEG_DIR = os.path.join(NEUROCAMO, "data", "negatives")
N_IMAGES = 320
BATCH = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def xywh2xyxy(a):
    return np.stack(
        [a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2,
         a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2], axis=1
    )


def iou(pred, gt):
    p, g = xywh2xyxy(pred), xywh2xyxy(gt)
    x1 = np.maximum(p[:, 0], g[:, 0])
    y1 = np.maximum(p[:, 1], g[:, 1])
    x2 = np.minimum(p[:, 2], g[:, 2])
    y2 = np.minimum(p[:, 3], g[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ap = np.clip(p[:, 2] - p[:, 0], 0, None) * np.clip(p[:, 3] - p[:, 1], 0, None)
    ag = np.clip(g[:, 2] - g[:, 0], 0, None) * np.clip(g[:, 3] - g[:, 1], 0, None)
    return inter / np.maximum(ap + ag - inter, 1e-9)


def load(ckpt_path):
    model = NeuroCamoFullModel(beta=0.9).to(DEVICE)
    meta = {}
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        res = model.load_state_dict(ck["model_state_dict"], strict=False)
        meta = {
            "epoch": ck.get("epoch"),
            "val_loss": ck.get("val_loss"),
            "val_iou": ck.get("val_iou"),
            "version": ck.get("version"),
            "missing": list(res.missing_keys),
        }
    model.eval()
    return model, meta


def run(ckpt_path):
    name = os.path.basename(ckpt_path)
    model, meta = load(ckpt_path)
    enc = PoissonEncoder(timesteps=16, gain=1.0).to(DEVICE)

    ds = COD10KDataset(COD10K_ROOT, split="Test", img_size=128, augment=False)
    idx = np.linspace(0, len(ds) - 1, min(N_IMAGES, len(ds))).astype(int)

    preds, gts, confs, labels = [], [], [], []
    for s in range(0, len(idx), BATCH):
        chunk = idx[s:s + BATCH]
        imgs = torch.stack([ds[int(i)][0] for i in chunk]).to(DEVICE)
        box = torch.stack([ds[int(i)][1] for i in chunk]).numpy()
        lab = torch.stack([ds[int(i)][2] for i in chunk]).numpy().ravel()
        with torch.no_grad():
            b, c, _, _ = model(enc(imgs))
        preds.append(b.cpu().numpy())
        gts.append(box)
        labels.append(lab)
        confs.append(c.cpu().numpy().ravel())

    P_all = np.concatenate(preds)
    G_all = np.concatenate(gts)
    C_all = np.concatenate(confs)
    L = np.concatenate(labels)

    # COD10K/NC4K Test folders mix camouflaged frames with NonCAM backgrounds.
    # Box metrics are only meaningful on the frames that actually contain an
    # object; a zero-size ground-truth box has no IoU and no size ratio.
    pos = L > 0.5
    P, G, C = P_all[pos], G_all[pos], C_all[pos]

    mean_box = np.tile(G.mean(axis=0), (len(G), 1))

    print(f"\n{'=' * 72}\n  {name}   {meta}\n{'=' * 72}")
    print(f"  images                : {len(P_all)}  "
          f"({pos.sum()} with an object, {(~pos).sum()} NonCAM backgrounds)")
    print("  --- box metrics, object-bearing frames only ---")
    print("  prediction  mean      : " + np.array2string(P.mean(0), precision=4))
    print("  prediction  std       : " + np.array2string(P.std(0), precision=4)
          + "   <- near zero = constant output")
    print("  ground-truth mean     : " + np.array2string(G.mean(0), precision=4))
    print("  ground-truth std      : " + np.array2string(G.std(0), precision=4))
    print(f"  model      mean IoU   : {iou(P, G).mean() * 100:5.1f}%")
    print(f"  mean-box   mean IoU   : {iou(mean_box, G).mean() * 100:5.1f}%"
          "   <- baseline that ignores the image entirely")
    print(f"  centre error (px@128) : {np.abs(P[:, :2] - G[:, :2]).mean() * 128:5.1f}")
    print(f"  w,h ratio pred/gt     : {(P[:, 2:] / np.maximum(G[:, 2:], 1e-2)).mean():5.2f}"
          "   <- 1.0 = right size")
    for j, cn in enumerate(("xc", "yc", "w ", "h ")):
        print(f"  corr(pred {cn}, gt {cn})  : {np.corrcoef(P[:, j], G[:, j])[0, 1]:+.3f}"
              "   <- ~0 = coordinate not predicted at all")

    print("  --- objectness ---")
    print(f"  confidence  positives : mean {C.mean():.3f}  min {C.min():.3f}  max {C.max():.3f}")

    # In-distribution negatives: the NonCAM frames from the same split, which go
    # through the positives' own preprocessing path by construction.
    if (~pos).any():
        CN_in = C_all[~pos]
        print(f"  confidence  NonCAM    : mean {CN_in.mean():.3f}  min {CN_in.min():.3f}  "
              f"max {CN_in.max():.3f}   <- same-split backgrounds")
        print(f"  separation pos-NonCAM : {C.mean() - CN_in.mean():+.3f}")

    # External negatives (COCO downloads), fed the SAME way positives are
    # (plain /255, no ImageNet norm). Kept as a cross-domain sanity check; the
    # NonCAM numbers above are the ones that matter because they are in-domain.
    import cv2
    if not os.path.isdir(NEG_DIR):
        return
    negs = sorted(f for f in os.listdir(NEG_DIR) if f.lower().endswith((".jpg", ".png")))[:64]
    if not negs:
        return
    nc = []
    for s in range(0, len(negs), BATCH):
        arr = []
        for f in negs[s:s + BATCH]:
            im = cv2.imread(os.path.join(NEG_DIR, f))
            im = cv2.cvtColor(cv2.resize(im, (128, 128)), cv2.COLOR_BGR2RGB)
            arr.append(torch.from_numpy(im.astype(np.float32) / 255.0).permute(2, 0, 1))
        with torch.no_grad():
            _, c, _, _ = model(enc(torch.stack(arr).to(DEVICE)))
        nc.append(c.cpu().numpy().ravel())
    NC = np.concatenate(nc)
    print(f"  confidence  ext. negs : mean {NC.mean():.3f}  min {NC.min():.3f}  max {NC.max():.3f}"
          f"   <- {len(negs)} COCO photos")
    print(f"  separation pos - ext  : {C.mean() - NC.mean():+.3f}"
          "   <- ~0 means confidence is useless")


if __name__ == "__main__":
    targets = sys.argv[1:] or [
        os.path.join(NEUROCAMO, "checkpoints", "best_model_v2.pth"),
        os.path.join(NEUROCAMO, "checkpoints", "best_model_v3.pth"),
    ]
    print(f"device: {DEVICE}")
    for t in targets:
        run(t)
