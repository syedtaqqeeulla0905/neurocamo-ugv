"""
NeuroCamo - Real-Time Live Phone / Webcam Detection Demo
=========================================================
Runs real-time camouflage detection on live camera feeds:
  - Supports IP Webcam app stream (phone over Wi-Fi)
  - Supports standard Laptop Webcam (fallback)

USAGE:
  1. Open IP Webcam app on your phone and tap "Start server"
  2. Note the IP displayed on your phone screen (e.g. 192.168.1.5:8080)
  3. Run this script and enter your IP, or press Enter for webcam!
"""

import os
import sys
import cv2
import time
import torch
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.poisson_encoder import PoissonEncoder
from models.attention_bridge import NeuroCamoFullModel


def draw_detection_box(frame, bbox, conf_score, fps):
    """Draws green detection bounding box and live FPS stats on the camera frame."""
    H, W, _ = frame.shape
    xc, yc, w, h = bbox

    x1 = int((xc - w / 2) * W)
    y1 = int((yc - h / 2) * H)
    x2 = int((xc + w / 2) * W)
    y2 = int((yc + h / 2) * H)

    # Clamp coordinates
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)

    # Draw Green Bounding Box if confidence is high
    if conf_score > 0.3:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"NeuroCamo Target ({conf_score*100:.0f}%)"
        cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + 220, max(25, y1)), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 5, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Draw Live Performance HUD
    hud_bg = frame[:60, :280].copy()
    cv2.rectangle(frame, (0, 0), (280, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f} | SNN GPU (RTX 3050 Ti)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"Conf: {conf_score*100:.1f}% | Timesteps: T=16", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame


def run_live_demo(
    stream_url=None,
    checkpoint_path=r"e:\major_project\neurocamo\checkpoints\best_model.pth",
    timesteps=16
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("  NeuroCamo - Live Real-Time Detection Demo")
    print("=" * 65)
    print(f"  Device    : {device}")
    print(f"  Checkpoint: {checkpoint_path}\n")

    # 1. Load Model
    encoder = PoissonEncoder(timesteps=timesteps, gain=1.0).to(device)
    model   = NeuroCamoFullModel(beta=0.9).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  ✅ Loaded trained SNN weights (Val Loss: {checkpoint.get('val_loss', 0.0):.4f})")
    else:
        print("  ⚠️ Checkpoint not found! Running with un-trained initialized weights.")

    model.eval()

    # 2. Setup Camera Stream
    if stream_url is None or stream_url.strip() == "":
        print("\n  🎥 Opening Default Laptop Webcam (Index 0)...")
        cap = cv2.VideoCapture(0)
    else:
        # IP Webcam URL formatting
        if not stream_url.startswith("http"):
            stream_url = f"http://{stream_url}"
        if not stream_url.endswith("/video"):
            stream_url = stream_url.rstrip("/") + "/video"

        print(f"\n  📱 Connecting to IP Webcam stream at: {stream_url}")
        cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("  ❌ Error: Could not connect to camera stream!")
        print("     Make sure IP Webcam app 'Start server' is running on your phone,")
        print("     and both laptop & phone are on the SAME Wi-Fi network.")
        return

    print("\n  🚀 Live Detection Running! Press 'q' in video window to exit.\n")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  ⚠️ Frame dropped or stream disconnected.")
            break

        # Preprocess frame for SNN: BGR -> RGB -> resize to 128x128 -> normalize [0, 1]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        # Run SNN Inference on GPU
        with torch.no_grad():
            spike_train = encoder(img_tensor)  # (16, 1, 3, 128, 128)
            bbox_pred, conf_pred, _class_logits, sparsity = model(spike_train)

        bbox = bbox_pred[0].cpu().numpy()
        conf_score = conf_pred[0].item()

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # Draw overlays on original frame resolution
        output_frame = draw_detection_box(frame, bbox, conf_score, fps)

        cv2.imshow("NeuroCamo Real-Time Detection Feed", output_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("  👋 Live demo closed.")


if __name__ == "__main__":
    print("\nSelect Camera Mode:")
    print("  1. Laptop Webcam (Default)")
    print("  2. IP Webcam Phone Stream (Enter IP address)")
    choice = input("\nEnter choice (1 or IP address e.g. 192.168.1.5:8080): ").strip()

    if choice == "1" or choice == "":
        run_live_demo(stream_url=None)
    else:
        run_live_demo(stream_url=choice)
