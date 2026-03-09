"""
PyTorch Lightning Module for UGIF training.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.siamese import SiameseFCN
from src.models.losses import ContrastiveLoss, BCEDiceLoss
from src.utils.metrics import compute_iou, compute_f1


class UGIFLightningModule(pl.LightningModule):
    """Lightning module wrapping the SiameseFCN for change detection.

    Training objective is a combination of:
      - BCE + Dice loss on the pixel-wise change prediction
      - Contrastive loss on the feature distance (CSN term)

    Args:
        in_channels:  Input channels (optical + SAR).
        num_features: Number of semantic features k.
        feature_dim:  FCN hidden dimensionality.
        r_max:        Ratio clip ceiling.
        epsilon:      Division stability constant.
        lr:           Learning rate for AdamW.
        weight_decay: L2 regularisation.
        margin:       Contrastive loss margin.
        max_epochs:   Total epochs (for CosineAnnealingLR T_max).
        lambda_csn:   Weight for contrastive loss term.
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_features: int = 4,
        feature_dim: int = 128,
        r_max: float = 10.0,
        epsilon: float = 1e-6,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        margin: float = 1.0,
        max_epochs: int = 50,
        lambda_csn: float = 0.3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = SiameseFCN(
            in_channels=in_channels,
            num_features=num_features,
            feature_dim=feature_dim,
            r_max=r_max,
            epsilon=epsilon,
        )
        self.seg_loss = BCEDiceLoss()
        self.csn_loss = ContrastiveLoss(margin=margin)
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.lambda_csn = lambda_csn

    # ------------------------------------------------------------------
    def _shared_step(self, batch: Dict[str, Tensor], stage: str) -> Tensor:
        pre  = batch["pre_image"]   # (B, C, H, W)
        post = batch["post_image"]  # (B, C, H, W)
        mask = batch["mask"]        # (B, 1, H, W)

        # Forward through Siamese network
        out = self.model(pre, post)

        # ── Change-label from mask ───────────────────────────────────
        # A patch is "changed" if >5% of pixels are labelled as changed
        change_label = (mask.mean(dim=(-2, -1, -3)) > 0.05).float()  # (B,)

        # ── Segmentation-style loss ──────────────────────────────────
        # Upscale the change probability to match mask size
        change_prob = self.model.predict_change(pre, post)  # (B, 1)
        # Broadcast to (B, 1, H, W) for loss computation
        change_map = change_prob.view(-1, 1, 1, 1).expand_as(mask)
        seg_l = self.seg_loss(change_map, mask)

        # ── Contrastive (CSN) loss ───────────────────────────────────
        csn_l = self.csn_loss(out.distance, change_label)

        total = seg_l + self.lambda_csn * csn_l

        # ── Metrics ─────────────────────────────────────────────────
        with torch.no_grad():
            pred_bin = (change_prob.squeeze(1) > 0.5).float()  # (B,)
            iou = compute_iou(pred_bin, change_label)
            f1  = compute_f1(pred_bin, change_label)

        self.log(f"{stage}_loss", total, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_seg_loss",  seg_l,  on_epoch=True)
        self.log(f"{stage}_csn_loss",  csn_l,  on_epoch=True)
        self.log(f"{stage}_iou",  iou,  prog_bar=(stage == "val"), on_epoch=True)
        self.log(f"{stage}_f1",   f1,   prog_bar=False, on_epoch=True)
        return total

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "test")

    # ------------------------------------------------------------------
    def configure_optimizers(self) -> Dict[str, Any]:
        opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = CosineAnnealingLR(opt, T_max=self.max_epochs, eta_min=1e-7)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }
