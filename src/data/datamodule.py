"""
LightningDataModule for the UGIF pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.data.levir_dataset import LEVIRCDPatchDataset
from src.data.fusion import SAROpticalFusionTransform
from src.data.transforms import get_train_transforms, get_val_transforms, Compose


def _compose_with_fusion(base_transforms: Compose, num_sar: int = 2) -> Compose:
    """Prepend the SAR fusion transform before normalisation."""
    fusion = SAROpticalFusionTransform(num_sar_channels=num_sar)
    fused_transforms = Compose([fusion] + base_transforms.transforms)
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
        train_tfm = _compose_with_fusion(get_train_transforms(), self.num_sar)
        val_tfm   = _compose_with_fusion(get_val_transforms(), self.num_sar)

        if stage in ("fit", None):
            self.train_dataset = LEVIRCDPatchDataset(
                root=self.root,
                split="train",
                transform=train_tfm,
                patch_size=self.patch_size,
            )
            self.val_dataset = LEVIRCDPatchDataset(
                root=self.root,
                split="val",
                transform=val_tfm,
                patch_size=self.patch_size,
            )
        if stage in ("test", None):
            self.test_dataset = LEVIRCDPatchDataset(
                root=self.root,
                split="test",
                transform=val_tfm,
                patch_size=self.patch_size,
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
