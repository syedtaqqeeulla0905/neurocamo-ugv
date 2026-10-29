"""
NeuroCamo - Paper-Grade Evaluation
==================================
Generates the standard, citable figures & tables a detection / SNN paper needs,
using industry-standard libraries instead of any custom GUI readout:

  * torchmetrics.detection.MeanAveragePrecision  -> mAP, mAP@50, mAP@75  (COCO-style)
  * scikit-learn                                 -> confusion matrix + precision/recall/F1
  * snntorch.spikeplot                           -> spike raster (the SNN novelty figure)
  * captum.attr.LayerGradCam                     -> localization / explainability figure

All outputs are written to results/paper/ as 300-dpi PDF *and* PNG, plus a
metrics.csv, metrics.json and a ready-to-paste LaTeX table.

Run on the GPU machine (needs torch + the trained checkpoint):
  venv\\Scripts\\python.exe evaluation\\paper_eval.py
  (install deps first:  evaluation\\install_paper_deps.bat)
"""

import os
import sys
import json
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.military_yolo_dataset import get_military_dataloaders, MILITARY_CLASSES
from utils.poisson_encoder import PoissonEncoder
from models.attention_bridge import NeuroCamoFullModel

OUT_DIR = r"e:\major_project\neurocamo\results\paper"
CKPT = r"e:\major_project\neurocamo\checkpoints\best_military_model.pth"
DATA = r"e:\major_project\neurocamo\data\military_object_dataset"


def _xywh2xyxy(b):
    x, y, w, h = b
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def _savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")

def run(data_root=DATA, checkpoint_path=CKPT, output_dir=OUT_DIR,
        batch_size=16, timesteps=16, max_batches=0):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, val_loader = get_military_dataloaders(data_root, img_size=128, batch_size=batch_size)
    encoder = PoissonEncoder(timesteps=timesteps).to(device)
    model = NeuroCamoFullModel(beta=0.9, num_classes=len(MILITARY_CLASSES)).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint epoch {ckpt.get('epoch')} "
          f"(val_acc={ckpt.get('val_acc', 0):.1f}%)")

    # ── Collect predictions over the validation set ──────────────────────────
    map_preds, map_tgts = [], []      # for torchmetrics
    y_true, y_pred = [], []           # class ids, positives only (for confusion matrix)
    first_spikes = None               # cache one sample's spikes for the raster

    with torch.no_grad():
        for bi, (images, bboxes, labels, confs) in enumerate(val_loader):
            images = images.to(device)
            spikes = encoder(images)
            if first_spikes is None:
                first_spikes = spikes[:, 0].detach().cpu()  # (T, C, H, W) sample 0
            bbox_pred, conf_pred, class_logits, _ = model(spikes)

            bbox_pred = bbox_pred.cpu().numpy()
            conf_pred = conf_pred.squeeze(-1).cpu().numpy()
            cls_pred = class_logits.argmax(1).cpu().numpy()
            bboxes = bboxes.numpy()
            labels = labels.numpy()
            confs = confs.numpy()

            for i in range(len(images)):
                map_preds.append({
                    "boxes": torch.tensor([_xywh2xyxy(bbox_pred[i])], dtype=torch.float32),
                    "scores": torch.tensor([float(conf_pred[i])], dtype=torch.float32),
                    "labels": torch.tensor([int(cls_pred[i])], dtype=torch.int64),
                })
                if confs[i] > 0.5:  # positive image -> has a GT box
                    map_tgts.append({
                        "boxes": torch.tensor([_xywh2xyxy(bboxes[i])], dtype=torch.float32),
                        "labels": torch.tensor([int(labels[i])], dtype=torch.int64),
                    })
                    y_true.append(int(labels[i]))
                    y_pred.append(int(cls_pred[i]))
                else:               # negative image -> no GT objects
                    map_tgts.append({
                        "boxes": torch.zeros((0, 4), dtype=torch.float32),
                        "labels": torch.zeros((0,), dtype=torch.int64),
                    })
            if max_batches and bi + 1 >= max_batches:
                break

    _detection_metrics(map_preds, map_tgts, output_dir)
    _classification_report(y_true, y_pred, output_dir)
    _spike_raster(first_spikes, output_dir)
    _gradcam_figure(model, encoder, val_loader, device, output_dir)
    print(f"\nDone. Paper assets in {output_dir}")


def _detection_metrics(preds, tgts, output_dir):
    """COCO-style mAP via torchmetrics (the standard detection metric)."""
    print("\n[1/4] Detection metrics (torchmetrics mAP)...")
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except Exception as e:
        print(f"  torchmetrics not installed ({e}). Run install_paper_deps.bat")
        return
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.update(preds, tgts)
    res = {k: (float(v) if v.numel() == 1 else v.tolist())
           for k, v in metric.compute().items()}
    keep = {k: res[k] for k in ("map", "map_50", "map_75", "mar_1") if k in res}
    with open(os.path.join(output_dir, "detection_map.json"), "w") as f:
        json.dump(keep, f, indent=2)
    # LaTeX table
    with open(os.path.join(output_dir, "detection_map.tex"), "w") as f:
        f.write("\\begin{tabular}{lc}\n\\hline\nMetric & Value \\\\\n\\hline\n")
        for k, v in keep.items():
            f.write(f"{k.replace('_','@').upper()} & {v:.3f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    print(f"  mAP={keep.get('map', float('nan')):.3f}  "
          f"mAP@50={keep.get('map_50', float('nan')):.3f}")


def _classification_report(y_true, y_pred, output_dir):
    """Confusion matrix + precision/recall/F1 via scikit-learn."""
    print("\n[2/4] Classification report + confusion matrix (scikit-learn)...")
    if not y_true:
        print("  no positive samples — skipped.")
        return
    try:
        from sklearn.metrics import (confusion_matrix, classification_report,
                                     ConfusionMatrixDisplay)
    except Exception as e:
        print(f"  scikit-learn not installed ({e}). Run install_paper_deps.bat")
        return
    labels = list(range(len(MILITARY_CLASSES)))
    rep = classification_report(y_true, y_pred, labels=labels,
                                target_names=MILITARY_CLASSES, zero_division=0,
                                output_dict=True)
    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    with open(os.path.join(output_dir, "classification_report.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["class", "precision", "recall", "f1", "support"])
        for name in MILITARY_CLASSES:
            r = rep.get(name, {})
            wr.writerow([name, f"{r.get('precision',0):.3f}", f"{r.get('recall',0):.3f}",
                         f"{r.get('f1-score',0):.3f}", int(r.get('support',0))])

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=MILITARY_CLASSES).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True, values_format=".2f")
    ax.set_title("NeuroCamo — Normalized Confusion Matrix (val)")
    _savefig(fig, "confusion_matrix")


def _spike_raster(spikes, output_dir):
    """Spike raster using snnTorch's own plotting (the standard SNN figure)."""
    print("\n[3/4] Spike raster (snntorch.spikeplot)...")
    if spikes is None:
        print("  no spikes captured — skipped.")
        return
    try:
        import snntorch.spikeplot as splt
    except Exception as e:
        print(f"  snntorch.spikeplot unavailable ({e}).")
        return
    # (T, C, H, W) -> (T, N) flatten neurons, subsample to keep the plot readable
    T = spikes.shape[0]
    flat = spikes.reshape(T, -1)
    N = flat.shape[1]
    idx = np.linspace(0, N - 1, min(400, N)).astype(int)
    ras = flat[:, idx]
    fig, ax = plt.subplots(figsize=(9, 5))
    splt.raster(ras, ax, s=1.5, c="black")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Neuron index (subsampled)")
    ax.set_title(f"NeuroCamo — Input Spike Raster (T={T}, Poisson encoding)")
    _savefig(fig, "spike_raster")

    # Spike-rate per timestep (efficiency evidence)
    rate = flat.mean(1).numpy() * 100.0
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.bar(range(1, T + 1), rate, color="#2e7d32")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Active neurons (%)")
    ax2.set_title("Spike activity per timestep")
    _savefig(fig2, "spike_rate_per_step")


def _gradcam_figure(model, encoder, val_loader, device, output_dir):
    """Grad-CAM localization via Captum (Meta's official interpretability lib)."""
    print("\n[4/4] Grad-CAM localization (captum)...")
    try:
        from captum.attr import LayerGradCam, LayerAttribution
    except Exception as e:
        print(f"  captum not installed ({e}). Run install_paper_deps.bat")
        return

    images, bboxes, labels, confs = next(iter(val_loader))
    images = images.to(device)
    n = min(4, images.shape[0])

    # Wrap so Captum sees image-in -> class-logits-out
    class _Wrap(torch.nn.Module):
        def __init__(self, m, enc):
            super().__init__()
            self.m, self.enc = m, enc
        def forward(self, x):
            _, _, logits, _ = self.m(self.enc(x))
            return logits

    wrap = _Wrap(model, encoder).to(device)
    target_layer = model.asab.fuse_conv[0]     # last conv before the head
    gc = LayerGradCam(wrap, target_layer)

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    for i in range(n):
        x = images[i:i + 1].requires_grad_(True)
        with torch.no_grad():
            logits = wrap(x)
        cls = int(logits.argmax(1)[0].item())
        attr = gc.attribute(x, target=cls, relu_attributions=True)
        attr = LayerAttribution.interpolate(attr, (128, 128))
        cam = attr[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (np.ptp(cam) + 1e-8)
        img = images[i].detach().cpu().permute(1, 2, 0).numpy()

        axes[0, i].imshow(img); axes[0, i].axis("off")
        axes[0, i].set_title(MILITARY_CLASSES[cls], fontsize=9)
        axes[1, i].imshow(img); axes[1, i].imshow(cam, cmap="jet", alpha=0.5)
        axes[1, i].axis("off"); axes[1, i].set_title("Grad-CAM", fontsize=9)
    fig.suptitle("NeuroCamo — Grad-CAM Localization (Captum)")
    _savefig(fig, "gradcam_localization")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=CKPT)
    p.add_argument("--data-root", default=DATA)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-batches", type=int, default=0, help="0 = whole val set")
    a = p.parse_args()
    run(data_root=a.data_root, checkpoint_path=a.checkpoint,
        batch_size=a.batch_size, max_batches=a.max_batches)
