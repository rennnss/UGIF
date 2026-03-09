"""
Siamese FCN (FC-CSN) — the core dual-stream network.

The architecture implements:
  • Shared-weight FCN encoder f_θ for both pre- and post-disaster streams
  • Contrastive Siamese Network (CSN) loss:
        L_CSN = (1-Y)·½·‖f(x₁;θ) − f(x₂;θ)‖²₂
              + Y·½·max(0, m − ‖f(x₁;θ) − f(x₂;θ)‖₂)²
  • Damage ratio: R_{g,i} = f̄(O_pre,S_pre)_{g,i} / (f̄(O_post,S_post)_{g,i} + ε)
  • Ratio clipping: R_{g,i} ← min(R_{g,i}, R_max)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.models.fcn import FCNEncoder


@dataclass
class SiameseOutput:
    """Container for Siamese forward pass outputs."""

    f_pre: Tensor   # (B, k) — pre-disaster presence map
    f_post: Tensor  # (B, k) — post-disaster presence map
    ratio: Tensor   # (B, k) — clipped damage ratios R_{g,i}
    distance: Tensor  # (B,) — Euclidean distance ‖f_pre − f_post‖₂


class SiameseFCN(nn.Module):
    """Fully Convolutional Siamese Network for change detection.

    Both the pre- and post-disaster images pass through the **same** FCNEncoder
    (shared weights). The forward pass returns feature presence maps plus
    derived damage metrics required for DII computation.

    Args:
        in_channels:  Input channels (optical + SAR fused).
        num_features: Number of semantic features k (e.g. 4).
        feature_dim:  FCN intermediate dimensionality.
        r_max:        Maximum ratio clip value.
        epsilon:      Numerical stability constant.
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_features: int = 4,
        feature_dim: int = 128,
        r_max: float = 10.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.r_max = r_max
        self.epsilon = epsilon

        # Shared encoder (weights are identical for both streams)
        self.encoder = FCNEncoder(
            in_channels=in_channels,
            num_features=num_features,
            feature_dim=feature_dim,
        )

        # Classification head: pre/post concat → binary change probability
        self.change_head = nn.Sequential(
            nn.Linear(num_features * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        pre_image: Tensor,
        post_image: Tensor,
    ) -> SiameseOutput:
        """Forward pass through both siamese streams.

        Args:
            pre_image:  ``(B, C, H, W)`` pre-disaster fused image.
            post_image: ``(B, C, H, W)`` post-disaster fused image.

        Returns:
            :class:`SiameseOutput` containing feature maps, ratio, and L2 distance.
        """
        f_pre  = self.encoder(pre_image)   # (B, k)
        f_post = self.encoder(post_image)  # (B, k)

        # Euclidean distance for CSN loss
        distance = torch.norm(f_pre - f_post, p=2, dim=1)  # (B,)

        # Damage ratio (clipped)
        ratio = (f_pre + self.epsilon) / (f_post + self.epsilon)  # (B, k)
        ratio = torch.clamp(ratio, max=self.r_max)

        return SiameseOutput(
            f_pre=f_pre,
            f_post=f_post,
            ratio=ratio,
            distance=distance,
        )

    def predict_change(self, pre_image: Tensor, post_image: Tensor) -> Tensor:
        """Return binary change probability for a pre/post pair.

        Returns:
            ``(B, 1)`` sigmoid probability of change.
        """
        out = self.forward(pre_image, post_image)
        feat = torch.cat([out.f_pre, out.f_post], dim=1)  # (B, 2k)
        return torch.sigmoid(self.change_head(feat))
