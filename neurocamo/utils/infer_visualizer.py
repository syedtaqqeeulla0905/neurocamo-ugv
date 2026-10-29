"""
NeuroCamo - Visual Inference Evaluator
=======================================
Loads a trained checkpoint, runs inference on unseen COD10K test images,
and draws Ground-Truth (Red) vs Model Prediction (Green) bounding boxes.

Saved outputs land in: e:\major_project\neurocamo\results\
"""

import os
import sys
import cv2
import torch
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.cod10k_dataset import get_dataloaders
from utils.poisson_encoder import PoissonEncoder
from models.attention_bridge import NeuroCamoFullModel


def draw_box(img, bbox, color, label=""):
    """
    Draws a bounding box [x_center, y_center, width, height] normalized on image.
    """
    H, W, _ = img.shape
    xc, yc, w, h = bbox

    x1 = int((xc - w / 2) * W)
    y1 = int((yc - h / 2) * H)
    x2 = int((xc + w / 2) * W)
    y2 = int((yc + h / 2) * H)

    # Clamp coordinates to image boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return img


def run_visual_inference(
    data_root=r"e:\major_project\neurocamo\data\COD10K",
    checkpoint_path=r"e:\major_project\neurocamo\checkpoints\best_model.pth",
    output_dir=r"e:\major_project\neurocamo\results",
    num_samples=5,
    timesteps=16
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("  NeuroCamo - Visual Inference Evaluator")
    print("=" * 65)
    print(f"  Device    : {device}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output Dir: {output_dir}\n")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Test DataLoader
    _, val_loader = get_dataloaders(data_root, img_size=128, batch_size=num_samples)

    # 2. Load Pipeline
    encoder = PoissonEncoder(timesteps=timesteps, gain=1.0).to(device)
    model   = NeuroCamoFullModel(beta=0.9).to(device)

    # 3. Load Trained Weights
    if not os.path.exists(checkpoint_path):
        print(f"  ❌ Error: Checkpoint file not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"  ✅ Loaded trained weights from epoch {checkpoint.get('epoch', 1)} "
          f"(Val Loss: {checkpoint.get('val_loss', 0.0):.4f})\n")

    # 4. Fetch a batch of unseen test images
    val_images, val_bboxes, val_labels = next(iter(val_loader))

    with torch.no_grad():
        val_images_gpu = val_images.to(device)
        spike_train = encoder(val_images_gpu)
        bbox_preds, conf_preds, _class_logits, sparsity = model(spike_train)

    # 5. Process and save visual results
    for i in range(min(num_samples, len(val_images))):
        # Convert tensor image back to numpy RGB -> BGR for OpenCV
        img_np = val_images[i].permute(1, 2, 0).numpy()
        img_np = (img_np * 255.0).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        gt_box = val_bboxes[i].numpy()
        pred_box = bbox_preds[i].cpu().numpy()
        conf_score = conf_preds[i].item()

        # Draw Ground-Truth (Red) and Prediction (Green)
        img_out = draw_box(img_bgr.copy(), gt_box, (0, 0, 255), label="Ground Truth")
        img_out = draw_box(img_out, pred_box, (0, 255, 0), label=f"NeuroCamo ({conf_score:.2f})")

        save_path = os.path.join(output_dir, f"sample_{i+1}_pred.png")
        cv2.imwrite(save_path, img_out)
        print(f"  📸 Saved visual prediction result: {save_path}")
        print(f"     GT Box  : {np.round(gt_box, 3)}")
        print(f"     Pred Box: {np.round(pred_box, 3)} | Conf: {conf_score:.3f}\n")

    print("=" * 65)
    print("  ✅ Visual inference complete! Check results folder.")
    print("=" * 65)


if __name__ == "__main__":
    run_visual_inference(num_samples=5)
