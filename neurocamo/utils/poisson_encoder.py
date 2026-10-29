"""
NeuroCamo - Poisson Rate Encoder
=================================
Converts a normalized image (pixel values 0.0 to 1.0) into a spike train
over T timesteps using Poisson Rate Coding.

HOW IT WORKS (simple explanation):
  - Think of each pixel as a probability of firing
  - A bright pixel (value = 0.9) has a 90% chance of spiking each timestep
  - A dark pixel (value = 0.1) has only a 10% chance of spiking
  - Over 16 timesteps, bright pixels produce lots of 1s, dark pixels produce few 1s
  - This is how biological eyes convert light into nerve signals!

Output shape: (T, B, C, H, W) where T=timesteps, B=batch, C=channels, H=height, W=width
All values in output are either 0 or 1 (spikes)

Reference: Eshraghian et al. 2023 (snnTorch paper)
"""

import torch
import torch.nn as nn


class PoissonEncoder(nn.Module):
    """
    Converts a batch of images into spike trains using Poisson Rate Coding.

    Args:
        timesteps (int): Number of timesteps T. We use T=16 as per our paper.
        gain (float):    Scales the firing rate. 1.0 = pixel intensity maps
                         directly to firing probability.

    Input:  Image tensor of shape (B, C, H, W), values in [0.0, 1.0]
    Output: Spike tensor of shape (T, B, C, H, W), values in {0, 1}
    """

    def __init__(self, timesteps: int = 16, gain: float = 1.0):
        super().__init__()
        self.timesteps = timesteps
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Normalized image tensor (B, C, H, W), values in [0.0, 1.0]

        Returns:
            spike_train: Binary tensor (T, B, C, H, W)
        """
        # Step 1: Clamp to valid range [0, 1] after applying gain
        x = torch.clamp(x * self.gain, 0.0, 1.0)

        # Step 2: Expand to T timesteps -> shape (T, B, C, H, W)
        x_repeated = x.unsqueeze(0).repeat(self.timesteps, 1, 1, 1, 1)

        # Step 3: For each timestep, flip a biased coin for each pixel
        spike_train = torch.bernoulli(x_repeated)

        return spike_train

    def forward_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Encodes a video temporal sequence of T=16 consecutive frames into Poisson spikes.

        Args:
            x_seq: (T, B, C, H, W) normalized video frame sequence tensor in [0, 1]

        Returns:
            spike_train: (T, B, C, H, W) binary spike train
        """
        x_seq = torch.clamp(x_seq * self.gain, 0.0, 1.0)
        return torch.bernoulli(x_seq)


if __name__ == "__main__":
    print("=" * 50)
    print("  NeuroCamo - Poisson Encoder Test")
    print("=" * 50)

    # Use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Running on: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Create the encoder (T=16 timesteps)
    encoder = PoissonEncoder(timesteps=16, gain=1.0).to(device)

    # Simulate a batch of 4 images, 3 channels (RGB), 128x128 resolution
    # This is what one training batch looks like
    fake_images = torch.rand(4, 3, 128, 128).to(device)
    print(f"\n  Input  shape: {list(fake_images.shape)}  (batch=4, channels=3, 128x128)")

    # Encode into spike train
    spike_train = encoder(fake_images)
    print(f"  Output shape: {list(spike_train.shape)}  (16 timesteps, batch=4, channels=3, 128x128)")

    # Check sparsity (how many neurons are silent = 1 is a good thing for SNNs)
    sparsity = 1.0 - spike_train.mean().item()
    print(f"\n  Spike sparsity: {sparsity * 100:.1f}%")
    print(f"  (Higher sparsity = more energy efficient on Jetson Nano)")

    # Verify values are only 0 or 1
    unique_vals = spike_train.unique()
    print(f"\n  Unique values in output: {unique_vals.tolist()}")
    print(f"  (Should only be [0.0, 1.0])")

    print("\n  ✅ Poisson Encoder working correctly!")
    print("=" * 50)
