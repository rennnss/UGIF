"""
PyTorch Lightning training callbacks.
"""
from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


def get_callbacks(output_dir: str = "./outputs") -> list[pl.Callback]:
    """Return the default set of training callbacks.

    Args:
        output_dir: Directory to save checkpoints.

    Returns:
        List of Lightning callbacks.
    """
    checkpoint = ModelCheckpoint(
        dirpath=f"{output_dir}/checkpoints",
        filename="ugif-{epoch:02d}-{val_iou:.4f}",
        monitor="val_iou",
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )

    early_stop = EarlyStopping(
        monitor="val_iou",
        patience=10,
        mode="max",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    return [checkpoint, early_stop, lr_monitor]
