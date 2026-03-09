"""
TorchGeo-backed LightningDataModule for UGIF.

Uses torchgeo.datamodules.LEVIRCDDataModule under the hood, which:
  - Auto-downloads LEVIR-CD (~3.5 GB) on first run when download=True
  - Returns batches as {image: (B, 6, H, W), mask: (B, H, W)}
    where channels 0-2 = pre-event RGB, channels 3-5 = post-event RGB
  - Applies GPU Kornia augmentations via AugmentationSequential

After each batch is transferred to the GPU we append 2 synthetic SAR channels
to both the pre and post halves, yielding image shape (B, 10, H, W):
  channels 0-4  = pre  (3 RGB + 2 SAR)
  channels 5-9  = post (3 RGB + 2 SAR)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import kornia.augmentation as K
import torch
from torch import Tensor
from torchgeo.datamodules import LEVIRCDDataModule
from torchgeo.datasets import LEVIRCD
from torchgeo.transforms import AugmentationSequential


# ── Normalisation statistics from TorchGeo (LEVIR-CD imagery) ──────────────
# TorchGeo stores mean/std as tensors of shape (C,) in 0-10000 DN scale.
# We work in [0, 1] so divide by 10000.
_LEVIR_MEAN = torch.tensor([485.0, 456.0, 406.0, 485.0, 456.0, 406.0]) / 10000.0
_LEVIR_STD  = torch.tensor([229.0, 224.0, 225.0, 229.0, 224.0, 225.0]) / 10000.0

# SAR proxy channel statistics (empirical, Sentinel-1 dB-normalised)
_SAR_MEAN = torch.tensor([0.30, 0.25])
_SAR_STD  = torch.tensor([0.15, 0.12])


def _build_augmentation(train: bool) -> AugmentationSequential:
    """Return a Kornia AugmentationSequential for image + mask."""
    augs = []
    if train:
        augs += [
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=90, p=0.3),
        ]
    augs.append(K.Normalize(mean=_LEVIR_MEAN, std=_LEVIR_STD))
    return AugmentationSequential(*augs, data_keys=["image", "mask"])


def _simulate_sar(rgb: Tensor) -> Tensor:
    """Return 2-channel proxy SAR tensor from an RGB batch (B, 3, H, W)."""
    grey = rgb.mean(dim=1, keepdim=True)           # (B, 1, H, W)
    vv   = grey * 0.8
    vh   = grey * 0.6
    sar  = torch.cat([vv, vh], dim=1)              # (B, 2, H, W)
    sar  = (sar - _SAR_MEAN.view(1, 2, 1, 1).to(sar)) \
           / _SAR_STD.view(1, 2, 1, 1).to(sar)
    return sar


class UGIFDataModule(LEVIRCDDataModule):
    """TorchGeo LEVIRCDDataModule extended with SAR channel simulation.

    Batches returned by dataloaders have:
      - ``image``: ``(B, 10, H, W)``  — pre(5ch) + post(5ch)
      - ``mask``:  ``(B, H, W)``      — binary change map

    Args:
        root:        Path to store / load LEVIR-CD data.
        patch_size:  Spatial size of random crops (must be multiple of 32).
        batch_size:  Mini-batch size.
        num_workers: Dataloader worker count.
        download:    Download dataset if not already present.
    """

    def __init__(
        self,
        root: str = "./data/LEVIR-CD",
        patch_size: int = 256,
        batch_size: int = 8,
        num_workers: int = 2,
        download: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            root=root,
            patch_size=patch_size,
            batch_size=batch_size,
            num_workers=num_workers,
            download=download,
            **kwargs,
        )
        self._train_aug = _build_augmentation(train=True)
        self._val_aug   = _build_augmentation(train=False)

    # ------------------------------------------------------------------
    # Override augmentation methods from NonGeoDataModule
    # ------------------------------------------------------------------
    def on_after_batch_transfer(
        self, batch: Dict[str, Tensor], dataloader_idx: int
    ) -> Dict[str, Tensor]:
        """GPU-side: augment + append SAR channels."""
        image = batch["image"].float() / 10000.0   # DN → [0,1]
        mask  = batch["mask"].float()

        if self.trainer and self.trainer.training:
            aug = self._train_aug
        else:
            aug = self._val_aug

        result = aug({"image": image, "mask": mask})
        image  = result["image"]                   # (B, 6, H, W)
        mask   = result["mask"]                    # (B, 1, H, W) or (B, H, W)

        # Ensure mask is (B, H, W)
        if mask.dim() == 4:
            mask = mask.squeeze(1)

        # Append SAR to pre and post halves
        pre_rgb  = image[:, :3]                    # (B, 3, H, W)
        post_rgb = image[:, 3:]                    # (B, 3, H, W)
        pre_sar  = _simulate_sar(pre_rgb)          # (B, 2, H, W)
        post_sar = _simulate_sar(post_rgb)         # (B, 2, H, W)

        fused = torch.cat([pre_rgb, pre_sar, post_rgb, post_sar], dim=1)  # (B,10,H,W)

        batch["image"] = fused
        batch["mask"]  = mask
        return batch
