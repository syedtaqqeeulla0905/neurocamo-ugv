"""
NeuroCamo - Training Loop (BPTT + Surrogate Gradient)
======================================================
Trains the end-to-end NeuroCamo SNN model on the COD10K dataset using:
  - Poisson Rate Encoding (T=16 timesteps)
  - Spatio-Temporal Backpropagation (STBP) with ArcTan surrogate gradient
  - Combined Loss: CIoU Box Loss + BCE Confidence Loss + Sparsity Regularization
  - Adam Optimizer with Cosine Annealing Learning Rate Scheduler

Reference: Eshraghian et al. 2023 / Ahmed et al. 2024
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import complete_box_iou_loss

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.cod10k_dataset import get_dataloaders
from utils.poisson_encoder import PoissonEncoder
from models.attention_bridge import NeuroCamoFullModel


def xywh2xyxy(x):
    """Convert [x_center, y_center, width, height] -> [x1, y1, x2, y2]"""
    y = torch.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def compute_loss(bbox_pred, conf_pred, bbox_gt, conf_gt, sparsity, sparsity_target=0.85):
    """
    Combined Loss:
      1. BBox Loss: Smooth L1 (direct positional gradient) + CIoU (shape/aspect ratio)
         - Evaluated ONLY on positive samples (conf_gt > 0.5)
      2. BCE Confidence Loss (object present vs absent)
      3. Sparsity Loss (penalizes low neural sparsity for energy savings)
    """
    conf_gt_flat = conf_gt.float().view_as(conf_pred)

    # 1. BCE Confidence Loss
    conf_pred_clamped = torch.clamp(conf_pred, 1e-7, 1.0 - 1e-7)
    conf_loss_fn = nn.BCELoss()
    conf_loss = conf_loss_fn(conf_pred_clamped, conf_gt_flat)

    # 2. BBox Loss (Smooth L1 + CIoU) on positive samples only
    pos_mask = (conf_gt_flat.squeeze(-1) > 0.5)
    if pos_mask.any():
        pos_pred = bbox_pred[pos_mask]
        pos_gt   = bbox_gt[pos_mask]

        # Smooth L1 gives non-vanishing gradients even when initial IoU = 0
        smooth_l1 = nn.SmoothL1Loss()(pos_pred, pos_gt)

        # CIoU optimizes aspect ratio & fine overlap
        pred_xyxy = xywh2xyxy(torch.clamp(pos_pred, 0.001, 0.999))
        gt_xyxy   = xywh2xyxy(torch.clamp(pos_gt,   0.001, 0.999))
        ciou      = complete_box_iou_loss(pred_xyxy, gt_xyxy, reduction="mean")

        box_loss = ciou + 2.0 * smooth_l1

        with torch.no_grad():
            # Estimate mean IoU metric on positive samples
            iou_metric = float(max(0.0, 1.0 - ciou.item()))
    else:
        box_loss = torch.tensor(0.0, device=bbox_pred.device, requires_grad=True)
        iou_metric = 0.0

    # 3. Sparsity Loss (autograd preserved)
    if not isinstance(sparsity, torch.Tensor):
        sparsity = torch.tensor(sparsity, device=bbox_pred.device, requires_grad=True)
    sparsity_loss = torch.relu(sparsity_target - sparsity)

    # Total loss
    total_loss = box_loss + (1.0 * conf_loss) + (0.1 * sparsity_loss)
    return total_loss, box_loss, conf_loss, iou_metric


def train_neurocamo(
    data_root=r"e:\major_project\neurocamo\data\COD10K",
    epochs=10,
    batch_size=8,
    lr=1e-3,
    timesteps=16,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("  NeuroCamo - SNN Training Loop (Fixed Losses & BBox Gradient)")
    print("=" * 65)
    print(f"  Device           : {device}")
    if device.type == "cuda":
        print(f"  GPU              : {torch.cuda.get_device_name(0)}")
    print(f"  Epochs           : {epochs}")
    print(f"  Batch Size       : {batch_size}")
    print(f"  Learning Rate    : {lr}")
    print(f"  SNN Timesteps (T): {timesteps}")
    print("=" * 65 + "\n")

    # 1. DataLoaders
    print("  [Data] Loading COD10K Dataset + Negatives...")
    train_loader, val_loader = get_dataloaders(data_root, img_size=128, batch_size=batch_size)

    # 2. Pipeline Components
    encoder = PoissonEncoder(timesteps=timesteps, gain=1.0).to(device)
    model   = NeuroCamoFullModel(beta=0.9).to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Checkpoint dir
    os.makedirs(r"e:\major_project\neurocamo\checkpoints", exist_ok=True)
    best_val_loss = float("inf")

    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        running_box_loss = 0.0
        running_conf_loss = 0.0
        running_iou = 0.0
        running_sparsity = 0.0
        n_batches = 0

        for batch_idx, (images, bboxes, labels) in enumerate(train_loader):
            images = images.to(device)
            bboxes = bboxes.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Step A: Poisson Rate Encode (Image -> Spikes over T timesteps)
            spike_train = encoder(images)  # (T, B, 3, 128, 128)

            # Step B: Model Forward Pass
            bbox_pred, conf_pred, _class_logits, sparsity = model(spike_train)

            # Step C: Compute Loss with autograd gradients intact
            loss, box_loss, conf_loss, iou_m = compute_loss(bbox_pred, conf_pred, bboxes, labels, sparsity)

            # Step D: Backpropagation Through Time (BPTT)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_box_loss += box_loss.item()
            running_conf_loss += conf_loss.item()
            running_iou += iou_m
            running_sparsity += float(sparsity)
            n_batches += 1

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Epoch [{epoch:02d}/{epochs:02d}] Batch [{batch_idx+1:03d}/{len(train_loader):03d}] "
                      f"| Loss: {loss.item():.4f} (Box: {box_loss.item():.4f}, Conf: {conf_loss.item():.4f}) "
                      f"| IoU: {iou_m*100:.1f}% | Sparsity: {float(sparsity)*100:.1f}%")

        scheduler.step()
        epoch_time = time.time() - start_time
        avg_train_loss = running_loss / max(1, n_batches)
        avg_train_iou  = running_iou / max(1, n_batches)
        avg_sparsity   = (running_sparsity / max(1, n_batches)) * 100

        # Validation Pass
        model.eval()
        val_loss = 0.0
        val_iou  = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_images, val_bboxes, val_labels in val_loader:
                val_images = val_images.to(device)
                val_bboxes = val_bboxes.to(device)
                val_labels = val_labels.to(device)

                val_spikes = encoder(val_images)
                v_bbox_pred, v_conf_pred, _v_cls, v_sparsity = model(val_spikes)
                v_loss, _, _, v_iou = compute_loss(v_bbox_pred, v_conf_pred, val_bboxes, val_labels, v_sparsity)
                val_loss += v_loss.item()
                val_iou  += v_iou
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        avg_val_iou  = val_iou / max(1, val_batches)

        print("-" * 65)
        print(f"  [OK] Epoch {epoch:02d} Complete in {epoch_time:.1f}s | Train Loss: {avg_train_loss:.4f} (IoU: {avg_train_iou*100:.1f}%) | Val Loss: {avg_val_loss:.4f} (Val IoU: {avg_val_iou*100:.1f}%) | Sparsity: {avg_sparsity:.1f}%")
        print("-" * 65)

        # Save Best Model Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = r"e:\major_project\neurocamo\checkpoints\best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_iou": avg_val_iou,
                "sparsity": avg_sparsity
            }, checkpoint_path)
            print(f"  [Checkpoint] Saved new best model checkpoint to: {checkpoint_path}\n")

if __name__ == "__main__":
    # Train for 10 Epochs
    train_neurocamo(epochs=10, batch_size=8)
