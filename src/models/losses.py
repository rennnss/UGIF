"""
Loss functions for the UGIF training pipeline.

• ContrastiveLoss  — CSN objective from the architecture diagram:
      L_CSN = (1−Y)·½·‖f₁−f₂‖²₂ + Y·½·max(0, m − ‖f₁−f₂‖₂)²
• DiceLoss         — handles class imbalance in change masks
• BCEDiceLoss      — combined BCE + Dice for pixel-wise segmentation
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ContrastiveLoss(nn.Module):
    """Contrastive Siamese Network (CSN) loss.

    Args:
        margin: Minimum distance enforced for dissimilar pairs (default 1.0).
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, distance: Tensor, label: Tensor) -> Tensor:
        """Compute contrastive loss.

        Args:
            distance: ``(B,)`` Euclidean distances ‖f₁−f₂‖₂.
            label:    ``(B,)`` binary label. 0=similar, 1=dissimilar.

        Returns:
            Scalar contrastive loss.
        """
        Y = label.float()
        similar_loss    = (1 - Y) * 0.5 * distance.pow(2)
        dissimilar_loss = Y * 0.5 * F.relu(self.margin - distance).pow(2)
        return (similar_loss + dissimilar_loss).mean()


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation masks.

    Args:
        smooth: Laplace smoothing constant to avoid division by zero.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred:   ``(B, 1, H, W)`` logit predictions.
            target: ``(B, 1, H, W)`` binary ground-truth masks.
        """
        pred_sig = torch.sigmoid(pred)
        intersection = (pred_sig * target).sum(dim=(-2, -1))
        sum_pred   = pred_sig.sum(dim=(-2, -1))
        sum_target = target.sum(dim=(-2, -1))
        dice = (2 * intersection + self.smooth) / (sum_pred + sum_target + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Weighted combination of Binary Cross-Entropy and Dice loss.

    Args:
        bce_weight:  Weight for BCE term (default 0.5).
        dice_weight: Weight for Dice term (default 0.5).
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()
        self.bce  = nn.BCEWithLogitsLoss()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.bce_weight * self.bce(pred, target) + \
               self.dice_weight * self.dice(pred, target)
