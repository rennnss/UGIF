"""
Data transforms for training and validation.

Normalisation statistics are approximate; tune with your dataset.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor
import torchvision.transforms.functional as TF
import random


# ── Normalisation means/stds ────────────────────────────────────────────────
# Optical (RGB): ImageNet-ish statistics, works well for EO fine-tuning
_OPTICAL_MEAN = [0.485, 0.456, 0.406]
_OPTICAL_STD  = [0.229, 0.224, 0.225]

# SAR (VV, VH): empirical log-scale Sentinel-1 statistics (dB normalised)
_SAR_MEAN = [0.3, 0.25]
_SAR_STD  = [0.15, 0.12]


class Normalize:
    """Per-image normalisation of pre/post images in a sample dict."""

    def __init__(self, mean: list[float], std: list[float]) -> None:
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std  = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, img: Tensor) -> Tensor:
        n_optical = len(_OPTICAL_MEAN)
        n_sar     = len(_SAR_MEAN)
        total_c   = img.shape[0]

        opt_mean = torch.tensor(_OPTICAL_MEAN).view(-1, 1, 1)
        opt_std  = torch.tensor(_OPTICAL_STD).view(-1, 1, 1)
        sar_mean = torch.tensor(_SAR_MEAN).view(-1, 1, 1)
        sar_std  = torch.tensor(_SAR_STD).view(-1, 1, 1)

        if total_c == n_optical:
            return (img - opt_mean) / opt_std
        elif total_c == n_sar:
            return (img - sar_mean) / sar_std
        else:
            # Fused tensor: normalise optical and SAR separately
            opt = (img[:n_optical] - opt_mean) / opt_std
            sar = (img[n_optical:n_optical + n_sar] - sar_mean) / sar_std
            rest = img[n_optical + n_sar:]
            parts = [opt, sar] + ([rest] if rest.shape[0] > 0 else [])
            return torch.cat(parts, dim=0)


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if random.random() < self.p:
            sample["pre_image"]  = TF.hflip(sample["pre_image"])
            sample["post_image"] = TF.hflip(sample["post_image"])
            sample["mask"]       = TF.hflip(sample["mask"])
        return sample


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if random.random() < self.p:
            sample["pre_image"]  = TF.vflip(sample["pre_image"])
            sample["post_image"] = TF.vflip(sample["post_image"])
            sample["mask"]       = TF.vflip(sample["mask"])
        return sample


class NormalizeSample:
    """Normalise pre and post images in a sample dict."""

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        _n = Normalize([], [])  # stateless; actual logic in __call__
        sample["pre_image"]  = _n(sample["pre_image"])
        sample["post_image"] = _n(sample["post_image"])
        return sample


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for t in self.transforms:
            sample = t(sample)
        return sample


def get_train_transforms() -> Compose:
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        NormalizeSample(),
    ])


def get_val_transforms() -> Compose:
    return Compose([NormalizeSample()])
