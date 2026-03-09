"""
LEVIR-CD Dataset wrapper.

The LEVIR-CD (Change Detection) dataset contains pairs of pre/post
disaster Google Earth optical images at 0.5m/pixel resolution.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LEVIRCDPatchDataset(Dataset):
    """LEVIR-CD dataset that returns (pre_image, post_image, mask) triplets.

    Expected directory layout::

        root/
          train/
            A/   # pre-disaster images  (*.png)
            B/   # post-disaster images (*.png)
            label/  # binary change masks (*.png)
          val/
            ...
          test/
            ...

    If the official LEVIR-CD folder is not found the dataset falls back to
    generating random synthetic samples so that the rest of the pipeline can
    be exercised without downloading 3.5 GB of imagery.
    """

    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        patch_size: int = 256,
        synthetic_size: int = 200,
    ) -> None:
        assert split in self.SPLITS, f"split must be one of {self.SPLITS}"
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.patch_size = patch_size
        self.synthetic_size = synthetic_size

        self.split_dir = self.root / split
        self._use_synthetic = not (self.split_dir / "A").exists()

        if not self._use_synthetic:
            self.image_names = sorted(
                p.name for p in (self.split_dir / "A").glob("*.png")
            )
        else:
            # Synthetic fallback: deterministic random tensors
            self.image_names = [f"synthetic_{i:04d}.png" for i in range(synthetic_size)]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._use_synthetic:
            return self._synthetic_sample(idx)

        name = self.image_names[idx]
        pre = self._load_image(self.split_dir / "A" / name)
        post = self._load_image(self.split_dir / "B" / name)
        mask = self._load_mask(self.split_dir / "label" / name)

        sample = {"pre_image": pre, "post_image": post, "mask": mask}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    # ------------------------------------------------------------------
    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

    def _load_mask(self, path: Path) -> torch.Tensor:
        mask = Image.open(path).convert("L")
        arr = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)

    def _synthetic_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(seed=idx)
        H = W = self.patch_size
        pre  = torch.from_numpy(rng.random((3, H, W), dtype=np.float32))
        post = torch.from_numpy(rng.random((3, H, W), dtype=np.float32))
        mask = torch.from_numpy((rng.random((1, H, W)) > 0.7).astype(np.float32))
        sample = {"pre_image": pre, "post_image": post, "mask": mask}
        # Apply transform (including SAR fusion) — same as the real data path
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
