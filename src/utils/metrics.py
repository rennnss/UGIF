"""
Metrics for change detection evaluation.

All metrics operate on flattened binary prediction/target tensors.
"""
from __future__ import annotations

import torch
from torch import Tensor


def compute_iou(pred: Tensor, target: Tensor, eps: float = 1e-6) -> float:
    """Compute Intersection-over-Union (Jaccard index) for binary predictions.

    Args:
        pred:   Binary prediction tensor (values 0 or 1).
        target: Binary ground-truth tensor (values 0 or 1).

    Returns:
        Scalar IoU value in [0, 1].
    """
    pred   = pred.bool().flatten()
    target = target.bool().flatten()
    intersection = (pred & target).sum().float()
    union        = (pred | target).sum().float()
    return (intersection / (union + eps)).item()


def compute_f1(pred: Tensor, target: Tensor, eps: float = 1e-6) -> float:
    """Compute F1 score (harmonic mean of precision and recall).

    Args:
        pred:   Binary prediction tensor.
        target: Binary ground-truth tensor.

    Returns:
        Scalar F1 value in [0, 1].
    """
    pred   = pred.bool().flatten()
    target = target.bool().flatten()
    tp = (pred & target).sum().float()
    fp = (pred & ~target).sum().float()
    fn = (~pred & target).sum().float()
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    return (2 * precision * recall / (precision + recall + eps)).item()


def compute_precision_recall(
    pred: Tensor, target: Tensor, eps: float = 1e-6
) -> tuple[float, float]:
    """Compute precision and recall for binary predictions.

    Returns:
        Tuple of (precision, recall) scalars in [0, 1].
    """
    pred   = pred.bool().flatten()
    target = target.bool().flatten()
    tp = (pred & target).sum().float()
    fp = (pred & ~target).sum().float()
    fn = (~pred & target).sum().float()
    precision = (tp / (tp + fp + eps)).item()
    recall    = (tp / (tp + fn + eps)).item()
    return precision, recall
