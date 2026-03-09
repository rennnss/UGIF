"""
FCN Encoder backbone.

Implements a Fully Convolutional Network (FCN) encoder that produces
spatially-averaged feature presence maps per grid cell:

    f̄(O, S)_{g,i} = (1 / H×W) Σ_{x,y} σ( FCN_θ(X_fused)_{x,y,i} )
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ConvBNReLU(nn.Sequential):
    """Conv → BN → ReLU building block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class FCNEncoder(nn.Module):
    """Lightweight FCN encoder for change-detection feature extraction.

    Architecture (keeping spatial resolution at 1/8 of input):
        Conv-64 → Conv-128 → Conv-256 (stride 2) → Conv-128 → Conv-k (1×1)

    The output is a k-channel sigmoid activation map averaged spatially
    to produce a scalar presence confidence per feature per grid cell.

    Args:
        in_channels:  Number of input channels (optical + SAR fused).
        num_features: Number of semantic features (k) — e.g. 4.
        feature_dim:  Intermediate feature dimension.
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_features: int = 4,
        feature_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_features = num_features

        self.encoder = nn.Sequential(
            ConvBNReLU(in_channels, 64),
            ConvBNReLU(64, 128),
            ConvBNReLU(128, 256, stride=2),   # 1/2 spatial
            ConvBNReLU(256, 256, stride=2),   # 1/4 spatial
            ConvBNReLU(256, feature_dim),
            nn.Conv2d(feature_dim, num_features, kernel_size=1),  # 1×1 projection
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``(B, C, H, W)``.

        Returns:
            Spatially-averaged sigmoid map ``(B, k)`` — one scalar per
            semantic feature representing its normalised "presence" in
            the image patch.
        """
        feat_map = self.encoder(x)            # (B, k, H', W')
        activated = torch.sigmoid(feat_map)   # σ(·) per pixel
        # Average over spatial dims → (B, k)
        presence = activated.mean(dim=(-2, -1))
        return presence

    def forward_map(self, x: Tensor) -> Tensor:
        """Return the full spatial feature map (before averaging).

        Useful for SHAP computation and visualisation.

        Returns:
            ``(B, k, H', W')`` sigmoid-activated feature map.
        """
        return torch.sigmoid(self.encoder(x))
