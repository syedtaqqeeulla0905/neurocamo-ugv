"""
NeuroCamo - Confidence Head Fine-Tuning
==========================================
Fixes the "100% false positive" problem by training with:
  - Positive samples: COD10K camo images → conf target = 1.0
  - Negative samples: COCO regular images → conf target = 0.0

ONLY the detection head is trained (backbone + ASAB are FROZEN).
This makes training very fast: ~15-20 min on RTX 3050 Ti.

Usage:
  python training/finetune_negatives.py
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.poisson_encoder import PoissonEncoder
from models.attention_bridge import NeuroCamoFullModel
from datasets.cod10k_dataset import COD10KDataset
from datasets.negative_dataset import NegativeDataset, download_negatives

# ── Config ───────────────────────────────────────────────────────────────────
CHECKPOINT_IN  = r"e:\major_project\neurocamo\checkpoints\best_model.pth"
CHECKPOINT_OUT = r"e:\major_project\neurocamo\checkpoints\best_model_v2.pth"
COD10K_ROOT    = r"e:\major_project\neurocamo\data\COD10K"
NEG_DIR        = r"e:\major_project\neurocamo\data\negatives"

EPOCHS     = 10
BATCH_SIZE = 8
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMESTEPS  = 16


def build_positive_dataset():
    """Load COD10K training images — all are positive (conf=1.0)."""
    return COD10KDataset(
        root_dir=COD10K_ROOT,
        split="Train",
        img_size=128,
        augment=True,
    )


class PositiveWrapper(torch.utils.data.Dataset):
    """Wraps COD10K so it returns (image, bbox, conf) matching NegativeDataset format."""
    def __init__(self, cod10k):
        self.cod10k = cod10k

    def __len__(self):
        return len(self.cod10k)

    def __getitem__(self, idx):
        image, bbox, conf_orig = self.cod10k[idx]
        conf = torch.tensor([1.0])
    
        return image, bbox, conf


def freeze_backbone(model):
    """Freeze all layers EXCEPT the detection head."""
    for name, param in model.named_parameters():
        if "head" in name:
            param.requires_grad = True   # Train head only
        else:
            param.requires_grad = False  # Freeze backbone + ASAB

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}  (head only)")


def train():
    print("=" * 60)
    print("  NeuroCamo - Confidence Fine-Tuning (Negatives Fix)")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Step 1: Download negatives if not present ───────────────────────────
    existing = len([f for f in os.listdir(NEG_DIR) if f.endswith(".jpg")]) if os.path.exists(NEG_DIR) else 0
    if existing < 100:
        print(f"\n  Downloading negative images... (found {existing}, need 300)")
        download_negatives(NEG_DIR, count=300)
    else:
        print(f"\n  Found {existing} negative images already downloaded.")

    # ── Step 2: Build datasets ──────────────────────────────────────────────
    print("\n  Loading datasets...")
    pos_raw  = build_positive_dataset()
    pos_ds   = PositiveWrapper(pos_raw)
    neg_ds   = NegativeDataset(neg_dir=NEG_DIR, img_size=128)

    print(f"  Positive samples (COD10K): {len(pos_ds)}")
    print(f"  Negative samples (COCO):   {len(neg_ds)}")

    # Balance: 50% positives, 50% negatives using WeightedRandomSampler
    combined = ConcatDataset([pos_ds, neg_ds])
    weights  = [1.0] * len(pos_ds) + [1.0] * len(neg_ds)
    sampler  = WeightedRandomSampler(weights, num_samples=min(len(pos_ds), len(neg_ds)) * 2, replacement=True)

    loader = DataLoader(
        combined,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )

    # ── Step 3: Load model ──────────────────────────────────────────────────
    print("\n  Loading model from checkpoint...")
    encoder = PoissonEncoder(timesteps=TIMESTEPS).to(DEVICE)
    model   = NeuroCamoFullModel(beta=0.9).to(DEVICE)

    if os.path.exists(CHECKPOINT_IN):
        ckpt = torch.load(CHECKPOINT_IN, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint: Epoch {ckpt.get('epoch', '?')}, Val Loss: {ckpt.get('val_loss', 0):.4f}")
    else:
        print("  WARNING: No checkpoint found, starting from random weights!")

    # Freeze backbone, train head only
    freeze_backbone(model)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Loss functions
    bbox_loss_fn = nn.SmoothL1Loss()
    conf_loss_fn = nn.BCELoss()

    # ── Step 4: Fine-tune ───────────────────────────────────────────────────
    print(f"\n  Starting fine-tuning for {EPOCHS} epochs...")
    print(f"  Batches per epoch: {len(loader)}")
    print("-" * 60)

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        encoder.eval()  # Encoder is always frozen

        total_loss = 0.0
        total_conf_loss = 0.0
        total_bbox_loss = 0.0
        n_batches = 0

        for batch_idx, (images, bbox_targets, conf_targets) in enumerate(loader):
            images       = images.to(DEVICE)          # (B, 3, 128, 128)
            bbox_targets = bbox_targets.to(DEVICE)    # (B, 4)
            conf_targets = conf_targets.to(DEVICE)    # (B, 1)

            # Encode to spikes
            with torch.no_grad():
                spikes = encoder(images)              # (T, B, 3, 128, 128)

            # Forward pass
            bbox_pred, conf_pred, _, sparsity = model(spikes)

            # Compute losses
            # Only compute bbox loss for POSITIVE samples (where conf_target == 1.0)
            pos_mask = (conf_targets.squeeze() > 0.5)

            c_loss = conf_loss_fn(conf_pred, conf_targets)

            if pos_mask.sum() > 0:
                b_loss = bbox_loss_fn(bbox_pred[pos_mask], bbox_targets[pos_mask])
            else:
                b_loss = torch.tensor(0.0, device=DEVICE)

            loss = c_loss + 0.5 * b_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss      += loss.item()
            total_conf_loss += c_loss.item()
            total_bbox_loss += b_loss.item()
            n_batches       += 1

            if (batch_idx + 1) % 20 == 0:
                print(f"  Epoch [{epoch}/{EPOCHS}] Batch [{batch_idx+1}/{len(loader)}] "
                      f"Loss: {loss.item():.4f} | Conf: {c_loss.item():.4f} | BBox: {b_loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / n_batches
        avg_conf = total_conf_loss / n_batches
        avg_bbox = total_bbox_loss / n_batches

        print(f"\n  [Epoch {epoch}/{EPOCHS}] Avg Loss: {avg_loss:.4f} | Conf Loss: {avg_conf:.4f} | BBox Loss: {avg_bbox:.4f}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         avg_loss,
                "finetune":         True,
                "note":             "Fine-tuned with negatives to fix false positives",
            }, CHECKPOINT_OUT)
            print(f"  --> Saved best model (loss={best_loss:.4f}) to {CHECKPOINT_OUT}")

        print("-" * 60)

    print(f"\n  Fine-tuning complete!")
    print(f"  Best Loss: {best_loss:.4f}")
    print(f"  Model saved: {CHECKPOINT_OUT}")
    print("\n  To use in GUI, update CHECKPOINT_PATH to:")
    print(f"  {CHECKPOINT_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    train()
