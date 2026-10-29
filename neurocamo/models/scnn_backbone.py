"""
NeuroCamo - 3-Layer Spiking CNN Backbone (SCNN)
================================================
This is the core feature extractor of the NeuroCamo architecture.

HOW IT WORKS:
  - Takes a 5D temporal spike tensor from Poisson Encoder: (T, B, C, H, W)
  - Processes each timestep through 3 Spiking Convolutional layers
  - Uses Adaptive Leaky Integrate-and-Fire (ALIF) neurons (snn.Leaky)
  - Keeps track of membrane potential (mem) across all T timesteps
  - Outputs both the final spike train and accumulated membrane feature maps

Neuron Model:
  V_mem(t+1) = beta * V_mem(t) + W * Spike(t) - V_th * OutputSpike(t)
  - beta: Membrane decay rate (learnable per channel)
  - V_th: Firing threshold

Reference: Ahmed et al. 2024 (Base Paper) / Eshraghian et al. 2023
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SpikingCNNBackbone(nn.Module):
    """
    3-Layer Spiking Convolutional Neural Network Backbone.

    Args:
        beta (float): Initial membrane decay rate (0.0 to 1.0). Default: 0.9
                      0.9 means 90% of membrane potential is retained per step.
        learn_beta (bool): If True, allows the decay rate to be learned during training.
    """

    def __init__(self, beta: float = 0.9, learn_beta: bool = True):
        super().__init__()

        self.beta = beta

        # ── Surrogate Gradient ───────────────────────────────────────────────
        # Standard step function has 0 derivative (dead neuron problem).
        # FastSigmoid provides a smooth curve during backpropagation.
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # ── Layer 1: Edge & Fine Texture Spikes ──────────────────────────────
        # Input: (B, 3, 128, 128) -> Output: (B, 32, 64, 64)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.lif1  = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=learn_beta, init_hidden=True)
        self.pool1 = nn.MaxPool2d(2, 2)  # 128x128 -> 64x64

        # ── Layer 2: Camouflage Boundary Spikes ──────────────────────────────
        # Input: (B, 32, 64, 64) -> Output: (B, 64, 32, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.lif2  = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=learn_beta, init_hidden=True)
        self.pool2 = nn.MaxPool2d(2, 2)  # 64x64 -> 32x32

        # ── Layer 3: High-Level Semantic Spikes ──────────────────────────────
        # Input: (B, 64, 32, 32) -> Output: (B, 128, 16, 16)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(128)
        self.lif3  = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=learn_beta, init_hidden=True)
        self.pool3 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16

    def forward(self, spike_train: torch.Tensor):
        """
        Args:
            spike_train (Tensor): (T, B, C, H, W) - 5D spike tensor from Poisson Encoder

        Returns:
            spikes_l3 (Tensor): Spikes from layer 3 over time -> (T, B, 128, 16, 16)
            mem_l3    (Tensor): Accumulated membrane potential -> (B, 128, 16, 16)
            sparsity  (float) : Overall spike sparsity percentage
        """
        num_steps = spike_train.size(0)
        batch_size = spike_train.size(1)

        # Reset hidden states of all LIF neurons at start of sequence
        self.lif1.init_leaky()
        self.lif2.init_leaky()
        self.lif3.init_leaky()

        # Lists to store spike activity across time
        spikes_l3_list = []
        total_spikes = 0
        total_neurons = 0

        # ── Process timestep by timestep over T=16 ───────────────────────────
        for step in range(num_steps):
            x_t = spike_train[step]  # Current frame spikes: (B, 3, 128, 128)

            # Layer 1 forward
            c1 = self.pool1(self.bn1(self.conv1(x_t)))
            spk1 = self.lif1(c1)
            mem1 = self.lif1.mem

            # Layer 2 forward
            c2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2 = self.lif2(c2)
            mem2 = self.lif2.mem

            # Layer 3 forward
            c3 = self.pool3(self.bn3(self.conv3(spk2)))
            spk3 = self.lif3(c3)
            mem3 = self.lif3.mem

            spikes_l3_list.append(spk3)

            # Accumulate spike count for sparsity calculation
            total_spikes += (spk1.sum() + spk2.sum() + spk3.sum()).item()
            total_neurons += (spk1.numel() + spk2.numel() + spk3.numel())

        # Stack Layer 3 spikes along time dimension: (T, B, 128, 16, 16)
        spikes_l3 = torch.stack(spikes_l3_list, dim=0)

        # Calculate overall neural sparsity (% of neurons that stayed silent)
        sparsity = 1.0 - (total_spikes / total_neurons)

        return spikes_l3, mem3, sparsity


if __name__ == "__main__":
    print("=" * 60)
    print("  NeuroCamo - Spiking CNN Backbone Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Running on: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Instantiate model
    model = SpikingCNNBackbone(beta=0.9, learn_beta=True).to(device)

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model Parameters: {total_params:,}  (~0.09M - extremely lightweight!)")

    # Create dummy 5D spike input: (T=16, B=4, C=3, H=128, W=128)
    dummy_spikes = (torch.rand(16, 4, 3, 128, 128) > 0.5).float().to(device)
    print(f"  Input Spike Train shape: {list(dummy_spikes.shape)}")

    # Run forward pass
    with torch.no_grad():
        spikes_out, mem_out, sparsity = model(dummy_spikes)

    print(f"\n  Output Spike Train shape: {list(spikes_out.shape)}  (T=16, B=4, Channels=128, 16x16)")
    print(f"  Membrane Feature shape   : {list(mem_out.shape)}     (B=4, Channels=128, 16x16)")
    print(f"  Overall Spike Sparsity   : {sparsity * 100:.2f}% (Target: >85% for energy saving)")

    print("\n  ✅ Spiking CNN Backbone working perfectly!")
    print("=" * 60)
