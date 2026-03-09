"""
Optical + SAR fusion utilities.

Implements the channel-wise fusion described in the paper:
    X_fused = O_i ⊕ S_i  ∈ ℝ^{H × W × (Co + Cs)}
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def fuse_optical_sar(optical: Tensor, sar: Tensor) -> Tensor:
    """Concatenate optical and SAR tensors along the channel dimension.

    Args:
        optical: ``(C_o, H, W)`` optical image tensor.  
        sar:     ``(C_s, H, W)`` SAR image tensor.

    Returns:
        Fused ``(C_o + C_s, H, W)`` tensor.

    Raises:
        ValueError: if spatial dimensions don't match.
    """
    if optical.shape[-2:] != sar.shape[-2:]:
        raise ValueError(
            f"Spatial dimensions mismatch: optical={optical.shape[-2:]} vs "
            f"sar={sar.shape[-2:]}"
        )
    return torch.cat([optical, sar], dim=0)


class SAROpticalFusionTransform:
    """Dataset transform that appends synthetic SAR channels to RGB images.

    When real SAR data is unavailable this transform simulates SAR-like
    texture by converting a greyscale copy of the image into VV and VH
    proxy channels through simple spectral manipulation.

    Args:
        num_sar_channels: Number of synthetic SAR channels to append (default 2).
    """

    def __init__(self, num_sar_channels: int = 2) -> None:
        self.num_sar_channels = num_sar_channels

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        pre = sample["pre_image"]   # (3, H, W)
        post = sample["post_image"]

        sample["pre_image"] = self._fuse(pre)
        sample["post_image"] = self._fuse(post)
        return sample

    def _fuse(self, rgb: Tensor) -> Tensor:
        """Append simulated SAR channels to an RGB tensor."""
        grey = rgb.mean(dim=0, keepdim=True)  # (1, H, W)
        sar_channels = [grey * (0.8 + 0.2 * i) for i in range(self.num_sar_channels)]
        sar = torch.cat(sar_channels, dim=0)  # (num_sar_channels, H, W)
        return fuse_optical_sar(rgb, sar)
