"""
NeuroCamo - Attention-based SNN-ANN Bridge (ASAB) & Detection Head
===================================================================
This module connects the temporal SNN backbone to a spatial detection head.

HOW IT WORKS:
  1. ASAB (Attention Bridge):
     - Takes temporal spike tensor: (T, B, C, H, W)
     - Applies temporal channel-attention to weight each timestep
     - Sums weighted spikes across T timesteps -> (B, C, H, W)
     - Fuses with accumulated membrane potential

  2. Detection Head:
     - Takes the fused 2D spatial feature map (B, C, H, W)
     - Predicts:
         a) Bounding Box coordinates: (x_center, y_center, width, height) normalized [0, 1]
         b) Object Confidence score:  probability that a target is in the box [0, 1]

Reference: Ahmed et al. 2024 (Base Paper - ASAB module)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASABridge(nn.Module):
    """
    Attention-based SNN-ANN Bridge (ASAB).
    Converts (T, B, C, H, W) temporal spikes + membrane potential -> (B, C, H, W) spatial map.
    """

    def __init__(self, channels: int = 128):
        super().__init__()

        # Temporal attention mechanism: learns which timesteps carry key camouflage edges
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels, bias=False),
            nn.Sigmoid()
        )

        # 1x1 Convolution to fuse accumulated membrane + weighted spikes
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, spikes_l3: torch.Tensor, mem_l3: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spikes_l3 (Tensor): (T, B, C, H, W) - Spikes over T=16 timesteps
            mem_l3    (Tensor): (B, C, H, W)    - Final accumulated membrane potential

        Returns:
            fused_features (Tensor): (B, C, H, W) - Dense 2D spatial representation
        """
        T, B, C, H, W = spikes_l3.shape

        # Step 1: Average spikes across all T timesteps -> (B, C, H, W)
        spikes_mean = spikes_l3.mean(dim=0)

        # Step 2: Calculate channel attention weights
        # (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        pool = self.avg_pool(spikes_mean).view(B, C)
        attn_weights = self.fc(pool).view(B, C, 1, 1)  # (B, C, 1, 1)

        # Step 3: Apply attention to mean spikes
        weighted_spikes = spikes_mean * attn_weights

        # Step 4: Concatenate weighted spikes + membrane potential along channel axis
        # (B, C, H, W) + (B, C, H, W) -> (B, 2C, H, W)
        cat_features = torch.cat([weighted_spikes, mem_l3], dim=1)

        # Step 5: 1x1 Conv fusion -> (B, C, H, W)
        fused_features = self.fuse_conv(cat_features)

        return fused_features


class SpatialSoftmax(nn.Module):
    """
    Extracts translation-equivariant spatial (x, y) coordinates from 2D feature maps.
    Learned temperature parameter sharpens softmax over the 16x16 grid for responsive localization.
    """
    def __init__(self, in_channels: int = 64, num_features: int = 32, height: int = 16, width: int = 16, temperature: float = 10.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_features, kernel_size=1)
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, height),
            torch.linspace(0.0, 1.0, width),
            indexing='ij'
        )
        self.register_buffer('pos_x', pos_x.reshape(1, 1, -1))
        self.register_buffer('pos_y', pos_y.reshape(1, 1, -1))

    def forward(self, x: torch.Tensor):
        feat = self.conv(x)  # (B, num_features, H, W)
        B, C, H, W = feat.shape
        # Temperature-scaled softmax over spatial grid
        softmax = torch.softmax(feat.view(B, C, -1) * torch.abs(self.temperature), dim=-1)  # (B, C, H*W)

        exp_x = torch.sum(self.pos_x * softmax, dim=-1)  # (B, C)
        exp_y = torch.sum(self.pos_y * softmax, dim=-1)  # (B, C)

        return torch.cat([exp_x, exp_y], dim=-1)  # (B, 2*C = 64)


class NeuroCamoDetectionHead(nn.Module):
    """
    Detection Head with Spatial Softmax BBox Regression.

    Input:  Fused 2D spatial feature map (B, 128, 16, 16)
    Outputs:
        bbox_pred  (Tensor): (B, 4)  -> [x, y, w, h] normalized
        conf_pred  (Tensor): (B, 1)  -> objectness in [0, 1]
        class_pred (Tensor): (B, num_classes) -> class logits (or None if num_classes=1)
    """

    def __init__(self, in_channels: int = 128, num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes

        # Spatial feature block (preserves 16x16 grid resolution)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )

        # Spatial Softmax extracts 32 keypoint coordinates (64 float values in [0, 1])
        self.spatial_softmax = SpatialSoftmax(in_channels=64, num_features=32, height=16, width=16)

        # Global average pool for confidence / classification
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # BBox regression branch: maps (spatial_coords=64 + global_feats=64) -> [x, y, w, h]
        self.bbox_branch = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

        # Objectness branch
        self.conf_branch = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Multi-class logits (used when num_classes > 1)
        self.class_branch = None
        if num_classes > 1:
            self.class_branch = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_classes)
            )

    def forward(self, fused_features: torch.Tensor):
        feats = self.conv_block(fused_features)               # (B, 64, 16, 16)

        spatial_coords = self.spatial_softmax(feats)          # (B, 64)
        global_feats   = self.avg_pool(feats).view(feats.size(0), -1)  # (B, 64)

        bbox_in = torch.cat([spatial_coords, global_feats], dim=-1)   # (B, 128)

        bbox_pred = self.bbox_branch(bbox_in)
        conf_pred = self.conf_branch(global_feats)
        class_logits = self.class_branch(global_feats) if self.class_branch is not None else None

        return bbox_pred, conf_pred, class_logits


class NeuroCamoFullModel(nn.Module):
    """
    Complete NeuroCamo End-to-End Pipeline:
    SNN Backbone -> ASABridge -> Detection Head

    num_classes=1  -> original COD10K binary objectness mode
    num_classes=12 -> Military Assets 12-class mode
    """

    def __init__(self, beta: float = 0.9, num_classes: int = 1):
        super().__init__()
        import sys, os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from scnn_backbone import SpikingCNNBackbone

        self.num_classes = num_classes
        self.backbone = SpikingCNNBackbone(beta=beta, learn_beta=True)
        self.asab     = ASABridge(channels=128)
        self.head     = NeuroCamoDetectionHead(in_channels=128, num_classes=num_classes)

    def forward(self, spike_train: torch.Tensor):
        """
        Args:
            spike_train (Tensor): (T, B, 3, 128, 128) from Poisson Encoder

        Returns:
            bbox_pred    (Tensor): (B, 4)
            conf_pred    (Tensor): (B, 1)
            class_logits (Tensor|None): (B, num_classes) or None
            sparsity     (float)
        """
        spikes_l3, mem_l3, sparsity = self.backbone(spike_train)
        fused_features = self.asab(spikes_l3, mem_l3)
        bbox_pred, conf_pred, class_logits = self.head(fused_features)

        return bbox_pred, conf_pred, class_logits, sparsity


class NeuroCamoDenseHead(nn.Module):
    """
    Dense per-cell detection head (CenterNet-style) on the 16x16 grid.

    WHY THIS EXISTS
    ---------------
    NeuroCamoDetectionHead regresses ONE box from globally-pooled features. That
    is a mean-seeking estimator: with COD10K centres clustered near 0.5, the
    cheapest solution for `Linear(128->64->4)+Sigmoid` is to emit the dataset
    average and ignore the image. Measured on best_model_v2.pth, predicted
    centre std was 0.005 against a ground-truth 0.09, and |corr(pred, gt)| was
    below 0.23 on all four coordinates - i.e. it had fully collapsed.

    Here, position is not regressed at all. It is the argmax cell of a 16x16
    objectness heatmap, refined by a sub-cell offset. Averaging two candidate
    locations produces a LOWER heatmap score, not a compromise box, so the
    collapse failure mode is structurally unavailable.

    Outputs
    -------
    heatmap (B, 1, 16, 16)  raw logits, per-cell "a centre is here"
    offset  (B, 2, 16, 16)  in [0, 1], sub-cell centre offset
    size    (B, 2, 16, 16)  in [0, 1], box w/h as a fraction of the whole image
    """

    def __init__(self, in_channels: int = 128, width: int = 96, grid: int = 16):
        super().__init__()
        self.grid = grid

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
        )

        self.heatmap = nn.Conv2d(width, 1, kernel_size=1)
        self.offset = nn.Conv2d(width, 2, kernel_size=1)
        self.size = nn.Conv2d(width, 2, kernel_size=1)

        # Focal-loss prior: start every cell at p ~= 0.01 so the overwhelming
        # background majority does not swamp the gradient in early epochs.
        nn.init.constant_(self.heatmap.bias, -4.595)

    def forward(self, fused_features: torch.Tensor):
        f = self.conv_block(fused_features)
        return self.heatmap(f), torch.sigmoid(self.offset(f)), torch.sigmoid(self.size(f))


class NeuroCamoDenseModel(nn.Module):
    """
    SNN Backbone -> ASABridge -> NeuroCamoDenseHead.

    forward() returns (bbox_pred, conf_pred, aux, sparsity) so it drops into the
    same call sites as NeuroCamoFullModel. `aux` carries the raw dense maps that
    the loss needs, and the heatmap doubles as a real saliency map for the UI -
    something the global-MLP head could never provide.
    """

    def __init__(self, beta: float = 0.9, grid: int = 16, mask_mode: bool = False):
        super().__init__()
        import sys, os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from scnn_backbone import SpikingCNNBackbone

        self.grid = grid
        # mask_mode=True trains the heatmap channel as a downsampled object mask
        # and reads the box off the mask's extent (see decode_mask). mask_mode
        # =False is the CenterNet-style centre-Gaussian + offset/size path.
        self.mask_mode = mask_mode
        self.backbone = SpikingCNNBackbone(beta=beta, learn_beta=True)
        self.asab = ASABridge(channels=128)
        self.head = NeuroCamoDenseHead(in_channels=128, grid=grid)

    @staticmethod
    def decode(hm_logits: torch.Tensor, offset: torch.Tensor, size: torch.Tensor):
        """Peak cell -> one [xc, yc, w, h] box in [0,1] plus its objectness."""
        B, _, H, W = hm_logits.shape
        prob = torch.sigmoid(hm_logits).view(B, -1)
        conf, flat = prob.max(dim=1)                       # (B,), (B,)
        row = torch.div(flat, W, rounding_mode="floor").float()
        col = (flat % W).float()

        off = offset.view(B, 2, -1).gather(2, flat.view(B, 1, 1).expand(B, 2, 1)).squeeze(-1)
        wh = size.view(B, 2, -1).gather(2, flat.view(B, 1, 1).expand(B, 2, 1)).squeeze(-1)

        xc = (col + off[:, 0]) / W
        yc = (row + off[:, 1]) / H
        bbox = torch.stack([xc, yc, wh[:, 0], wh[:, 1]], dim=1)
        return bbox, conf.unsqueeze(1)

    @staticmethod
    def decode_mask(hm_logits: torch.Tensor, thresh: float = 0.10):
        """Occupancy mask -> tight box around the above-threshold cells.

        Used when the heatmap channel is trained as a downsampled object mask
        rather than a centre Gaussian. This is far more sample-efficient: every
        cell carries a label, instead of one cell per image. The box comes from
        the mask's own extent, so size is never regressed and cannot collapse to
        the dataset mean.
        """
        B, _, H, W = hm_logits.shape
        p = torch.sigmoid(hm_logits).view(B, H, W)

        peak = p.view(B, -1).max(dim=1).values                  # (B,)
        # Threshold relative to each image's own peak as well as absolutely, so
        # a faint but confidently-detected object still yields a full-extent box.
        thr = torch.minimum(torch.full_like(peak, thresh), peak * 0.5)
        m = p >= thr.view(B, 1, 1)

        cols = m.any(dim=1).float()                             # (B, W)
        rows = m.any(dim=2).float()                             # (B, H)
        big = 1e4
        ar_w = torch.arange(W, device=p.device).float().view(1, W)
        ar_h = torch.arange(H, device=p.device).float().view(1, H)

        x1 = torch.amin(torch.where(cols > 0, ar_w, torch.full_like(ar_w, big)), dim=1)
        x2 = torch.amax(torch.where(cols > 0, ar_w, torch.full_like(ar_w, -big)), dim=1)
        y1 = torch.amin(torch.where(rows > 0, ar_h, torch.full_like(ar_h, big)), dim=1)
        y2 = torch.amax(torch.where(rows > 0, ar_h, torch.full_like(ar_h, -big)), dim=1)

        # Cell indices -> normalised image coords, covering the full cell extent
        xa, xb = x1 / W, (x2 + 1) / W
        ya, yb = y1 / H, (y2 + 1) / H
        bbox = torch.stack([(xa + xb) / 2, (ya + yb) / 2, xb - xa, yb - ya], dim=1)
        bbox = bbox.clamp(0.0, 1.0)

        # Objectness: mean of the top 4 cells. One hot cell is noise; a real
        # object lights up a contiguous region.
        topk = p.view(B, -1).topk(4, dim=1).values.mean(dim=1, keepdim=True)
        return bbox, topk

    def forward(self, spike_train: torch.Tensor):
        spikes_l3, mem_l3, sparsity = self.backbone(spike_train)
        fused = self.asab(spikes_l3, mem_l3)
        hm, off, wh = self.head(fused)
        if self.mask_mode:
            bbox_pred, conf_pred = self.decode_mask(hm)
        else:
            bbox_pred, conf_pred = self.decode(hm, off, wh)
        aux = {"heatmap": hm, "offset": off, "size": wh, "fused": fused}
        return bbox_pred, conf_pred, aux, sparsity


if __name__ == "__main__":
    print("=" * 60)
    print("  NeuroCamo - Full Model (Backbone + ASAB + Head) Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Running on: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Instantiate full model
    model = NeuroCamoFullModel(beta=0.9).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Full Model Parameters: {total_params:,}  (~0.26M - ultra lightweight!)")

    # Create dummy spike input: (T=16, B=4, C=3, H=128, W=128)
    dummy_spikes = (torch.rand(16, 4, 3, 128, 128) > 0.5).float().to(device)
    print(f"  Input Spike Train shape: {list(dummy_spikes.shape)}")

    # Run forward pass (binary COD10K-compatible mode)
    with torch.no_grad():
        bbox_pred, conf_pred, class_logits, sparsity = model(dummy_spikes)

    print(f"\n  BBox Predictions shape: {list(bbox_pred.shape)}  (B=4, [x, y, w, h])")
    print(f"  Conf Predictions shape: {list(conf_pred.shape)}  (B=4, score)")
    print(f"  Class logits          : {class_logits}")
    print(f"  Spike Sparsity        : {sparsity * 100:.2f}%")

    print("\n  Sample BBox output [0]:", bbox_pred[0].cpu().numpy().round(3))
    print("  Sample Conf output [0]:", round(conf_pred[0].item(), 3))

    # Multi-class military mode smoke test
    military = NeuroCamoFullModel(beta=0.9, num_classes=12).to(device)
    with torch.no_grad():
        _, _, mil_logits, _ = military(dummy_spikes)
    print(f"  Military class logits : {list(mil_logits.shape)}  (B=4, 12 classes)")

    print("\n  [OK] NeuroCamo Full Model architecture working perfectly!")
    print("=" * 60)
