"""
NeuroCamo - Full Dataset End-to-End Retraining (COD10K + Negatives)
====================================================================
Retrains the NeuroCamo SNN detection head & ASAB bridge on 6,000+ COD10K
camouflage images + COCO negative backgrounds to achieve accurate spatial
bounding box regression (xc, yc, w, h) and prevent false positives.

Output Checkpoint: e:\\major_project\\neurocamo\\checkpoints\\best_model_v3.pth
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision.ops import complete_box_iou_loss

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.cod10k_dataset import COD10KDataset
from datasets.negative_dataset import NegativeDataset
from utils.poisson_encoder import PoissonEncoder
from models.attention_bridge import NeuroCamoFullModel

# ── Config ───────────────────────────────────────────────────────────────────
CHECKPOINT_IN  = r"e:\major_project\neurocamo\checkpoints\best_model.pth"
CHECKPOINT_OUT = r"e:\major_project\neurocamo\checkpoints\best_model_v3.pth"
COD10K_ROOT    = r"e:\major_project\neurocamo\data\COD10K"
NEG_DIR        = r"e:\major_project\neurocamo\data\negatives"

EPOCHS     = 20
BATCH_SIZE = 16
LR         = 5e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMESTEPS  = 16


class PositiveWrapper(torch.utils.data.Dataset):
    """Wraps COD10K so it returns (image, bbox, conf) matching NegativeDataset format."""
    def __init__(self, cod10k):
        self.cod10k = cod10k

    def __len__(self):
        return len(self.cod10k)

    def __getitem__(self, idx):
        image, bbox, _label = self.cod10k[idx]
        conf = torch.tensor([1.0], dtype=torch.float32)
        return image, bbox, conf


def xywh2xyxy(x):
    """Convert [x_center, y_center, width, height] -> [x1, y1, x2, y2]"""
    y = torch.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2.0
    y[:, 1] = x[:, 1] - x[:, 3] / 2.0
    y[:, 2] = x[:, 0] + x[:, 2] / 2.0
    y[:, 3] = x[:, 1] + x[:, 3] / 2.0
    return y


def compute_loss(bbox_pred, conf_pred, bbox_gt, conf_gt, sparsity):
    """
    Combined Loss:
      1. BBox Loss: Smooth L1 + CIoU Loss on positive samples (conf_gt > 0.5)
      2. BCE Loss: Objectness confidence (camo vs background)
      3. Sparsity Loss: Regularizes spike density
    """
    conf_gt_flat = conf_gt.float().view_as(conf_pred)
    conf_loss_fn = nn.BCELoss()
    conf_loss    = conf_loss_fn(conf_pred, conf_gt_flat)

    pos_mask = (conf_gt_flat.squeeze(-1) > 0.5)
    if pos_mask.any():
        pos_pred = bbox_pred[pos_mask]
        pos_gt   = bbox_gt[pos_mask]

        smooth_l1 = nn.SmoothL1Loss()(pos_pred, pos_gt)
        pred_xyxy = xywh2xyxy(torch.clamp(pos_pred, 0.001, 0.999))
        gt_xyxy   = xywh2xyxy(torch.clamp(pos_gt,   0.001, 0.999))
        ciou      = complete_box_iou_loss(pred_xyxy, gt_xyxy, reduction="mean")

        box_loss = ciou + 2.0 * smooth_l1
        with torch.no_grad():
            iou_metric = float(max(0.0, 1.0 - ciou.item()))
    else:
        box_loss = torch.tensor(0.0, device=DEVICE)
        iou_metric = 0.0

    if not isinstance(sparsity, torch.Tensor):
        sparsity = torch.tensor(sparsity, device=DEVICE, requires_grad=True)
    sparsity_loss = torch.relu(0.85 - sparsity)

    total_loss = 1.5 * box_loss + 1.0 * conf_loss + 0.1 * sparsity_loss
    return total_loss, box_loss, conf_loss, iou_metric


def train():
    print("=" * 65)
    print("  NeuroCamo - Full End-to-End Retraining (COD10K + Negatives)")
    print("=" * 65)
    print(f"  Device           : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU              : {torch.cuda.get_device_name(0)}")
    print(f"  Epochs           : {EPOCHS}")
    print(f"  Batch Size       : {BATCH_SIZE}")
    print(f"  Learning Rate    : {LR}")
    print(f"  Target Checkpoint: {CHECKPOINT_OUT}")
    print("=" * 65 + "\n")

    # 1. Load Datasets
    print("  [1/4] Building Datasets...")
    pos_train_raw = COD10KDataset(root_dir=COD10K_ROOT, split="Train", img_size=128, augment=True)
    pos_val_raw   = COD10KDataset(root_dir=COD10K_ROOT, split="Test",  img_size=128, augment=False)

    pos_train = PositiveWrapper(pos_train_raw)
    pos_val   = PositiveWrapper(pos_val_raw)
    neg_ds    = NegativeDataset(neg_dir=NEG_DIR, img_size=128)

    print(f"        Positive Train (COD10K): {len(pos_train)}")
    print(f"        Positive Test  (COD10K): {len(pos_val)}")
    print(f"        Negative COCO Samples  : {len(neg_ds)}")

    # Balanced Sampler for training
    train_combined = ConcatDataset([pos_train, neg_ds])
    train_weights  = [1.0] * len(pos_train) + [len(pos_train) / max(1, len(neg_ds))] * len(neg_ds)
    sampler        = WeightedRandomSampler(train_weights, num_samples=len(pos_train) + len(neg_ds), replacement=True)

    train_loader = DataLoader(train_combined, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(pos_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 2. Build Model & Encoder
    print("\n  [2/4] Initializing SNN Model Architecture...")
    encoder = PoissonEncoder(timesteps=TIMESTEPS, gain=1.0).to(DEVICE)
    model   = NeuroCamoFullModel(beta=0.9).to(DEVICE)

    if os.path.exists(CHECKPOINT_IN):
        ckpt = torch.load(CHECKPOINT_IN, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"        Loaded base weights from {CHECKPOINT_IN}")

    # Unfreeze ASAB and Detection Head for full fine-tuning
    for name, param in model.named_parameters():
        if "head" in name or "asab" in name:
            param.requires_grad = True
        else:
            param.requires_grad = True  # Train end-to-end for peak accuracy

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"        Trainable Parameters: {trainable_params:,}")

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 4. Training Loop
    print(f"\n  [3/4] Starting {EPOCHS} Training Epochs...")
    best_val_loss = float("inf")
    best_val_iou  = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        encoder.eval()

        run_loss, run_box, run_conf, run_iou = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for batch_idx, (images, bboxes, confs) in enumerate(train_loader):
            images = images.to(DEVICE)
            bboxes = bboxes.to(DEVICE)
            confs  = confs.to(DEVICE)

            optimizer.zero_grad()
            spikes = encoder(images)
            bbox_pred, conf_pred, _, sparsity = model(spikes)

            loss, box_loss, conf_loss, iou_m = compute_loss(bbox_pred, conf_pred, bboxes, confs, sparsity)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            run_loss += loss.item()
            run_box  += box_loss.item()
            run_conf += conf_loss.item()
            run_iou  += iou_m
            n_batches += 1

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Epoch [{epoch:02d}/{EPOCHS:02d}] Batch [{batch_idx+1:03d}/{len(train_loader):03d}] "
                      f"| Loss: {loss.item():.4f} (Box: {box_loss.item():.4f}, Conf: {conf_loss.item():.4f}) "
                      f"| IoU: {iou_m*100:.1f}%", flush=True)

        scheduler.step()
        train_time = time.time() - t0
        avg_loss = run_loss / max(1, n_batches)
        avg_box  = run_box  / max(1, n_batches)
        avg_conf = run_conf / max(1, n_batches)
        avg_iou  = run_iou  / max(1, n_batches)

        # Validation Pass
        model.eval()
        v_loss, v_iou, v_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for val_img, val_box, val_conf in val_loader:
                val_img = val_img.to(DEVICE)
                val_box = val_box.to(DEVICE)
                val_conf = val_conf.to(DEVICE)

                val_spikes = encoder(val_img)
                vb_pred, vc_pred, _, v_sparsity = model(val_spikes)
                vl, _, _, vi = compute_loss(vb_pred, vc_pred, val_box, val_conf, v_sparsity)
                v_loss += vl.item()
                v_iou  += vi
                v_batches += 1

        avg_v_loss = v_loss / max(1, v_batches)
        avg_v_iou  = v_iou  / max(1, v_batches)

        print(f"  Epoch [{epoch:02d}/{EPOCHS:02d}] {train_time:.1f}s | "
              f"Train Loss: {avg_loss:.4f} (Box: {avg_box:.4f}, IoU: {avg_iou*100:.1f}%) | "
              f"Val Loss: {avg_v_loss:.4f} (Val IoU: {avg_v_iou*100:.1f}%)")

        # Save Best Model Checkpoint
        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            best_val_iou  = avg_v_iou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_v_loss,
                "val_iou": avg_v_iou,
                "version": "v3_cod10k_retrained"
            }, CHECKPOINT_OUT)
            print(f"    --> Saved best checkpoint to {CHECKPOINT_OUT} (Val Loss: {best_val_loss:.4f}, Val IoU: {best_val_iou*100:.1f}%)")

    print("\n  [4/4] Training Complete!")
    print(f"        Best Val Loss: {best_val_loss:.4f}")
    print(f"        Best Val IoU : {best_val_iou*100:.1f}%")
    print(f"        Saved to     : {CHECKPOINT_OUT}")
    print("=" * 65)


if __name__ == "__main__":
    train()
