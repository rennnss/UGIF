"""
LightningDataModule for the UGIF pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.functional as TF
import random

from torchgeo.datasets import LEVIRCDPlus
from src.data.fusion import SAROpticalFusionTransform
from src.data.transforms import get_train_transforms, get_val_transforms, Compose


class TorchGeoAdapterTransform:
    def __init__(self, patch_size=256):
        self.patch_size = patch_size

    def __call__(self, sample):
        img = sample["image"]
        if img.ndim == 4 and img.shape[0] == 2:
            pre_image = img[0]
            post_image = img[1]
        elif img.ndim == 3 and img.shape[0] == 6:
            pre_image = img[:3]
            post_image = img[3:]
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        
        mask = sample["mask"]

        # Ensure correct type and scale
        pre_image = pre_image.float()
        if pre_image.max() > 1.0:
            pre_image /= 255.0
        post_image = post_image.float()
        if post_image.max() > 1.0:
            post_image /= 255.0
            
        mask = mask.float()
        if mask.max() > 1.0:
            mask = (mask > 127).float()

        # Random crop to patch_size
        _, h, w = pre_image.shape
        if h > self.patch_size or w > self.patch_size:
            i = random.randint(0, h - self.patch_size)
            j = random.randint(0, w - self.patch_size)
            pre_image = TF.crop(pre_image, i, j, self.patch_size, self.patch_size)
            post_image = TF.crop(post_image, i, j, self.patch_size, self.patch_size)
            mask = TF.crop(mask, i, j, self.patch_size, self.patch_size)

        return {
            "pre_image": pre_image,
            "post_image": post_image,
            "mask": mask
        }

def _compose_with_fusion(base_transforms: Compose, patch_size: int = 256, num_sar: int = 2) -> Compose:
    """Prepend the SAR fusion transform before normalisation."""
    adapter = TorchGeoAdapterTransform(patch_size=patch_size)
    fusion = SAROpticalFusionTransform(num_sar_channels=num_sar)
    fused_transforms = Compose([adapter, fusion] + base_transforms.transforms)
    return fused_transforms


class UGIFDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule wrapping LEVIR-CD with SAR fusion.

    Args:
        root:        Root directory for LEVIR-CD data.
        patch_size:  Spatial size of image patches.
        batch_size:  Batch size for dataloaders.
        num_workers: Number of dataloader workers.
        num_sar:     Number of synthetic SAR channels to append.
    """

    def __init__(
        self,
        root: str = "./data/LEVIR-CD",
        patch_size: int = 256,
        batch_size: int = 8,
        num_workers: int = 4,
        num_sar: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.root = root
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_sar = num_sar

    def setup(self, stage: Optional[str] = None) -> None:
        train_tfm = _compose_with_fusion(get_train_transforms(), self.patch_size, self.num_sar)
        val_tfm   = _compose_with_fusion(get_val_transforms(), self.patch_size, self.num_sar)

        if stage in ("fit", None):
            # LEVIRCDPlus has no built-in val split.
            # Use two separate dataset instances (each with the correct transform)
            # and split by index so they don't share transforms.
            full_train = LEVIRCDPlus(root=self.root, split="train", download=True)
            n_total   = len(full_train)
            val_size  = max(1, int(0.2 * n_total))
            train_size = n_total - val_size

            # Deterministic index split (no shuffle needed — indices are stable)
            g = torch.Generator().manual_seed(42)
            perm = torch.randperm(n_total, generator=g).tolist()
            train_indices = perm[:train_size]
            val_indices   = perm[train_size:]

            # Wrap with correct transforms via Subset over separately-transformed datasets
            train_ds = LEVIRCDPlus(root=self.root, split="train", transforms=train_tfm, download=True)
            val_ds   = LEVIRCDPlus(root=self.root, split="train", transforms=val_tfm,   download=True)

            self.train_dataset = Subset(train_ds, train_indices)
            self.val_dataset   = Subset(val_ds,   val_indices)

        if stage in ("test", None):
            self.test_dataset = LEVIRCDPlus(
                root=self.root,
                split="test",
                transforms=val_tfm,
                download=True,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
