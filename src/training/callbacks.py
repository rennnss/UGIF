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


def get_callbacks(output_dir: str = "./outputs", patience: int = 10) -> list[pl.Callback]:
    """Return the default set of training callbacks.

    Args:
        output_dir: Directory to save checkpoints.
        patience: Epochs to wait for early stopping (0 to disable).

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

    callbacks = [checkpoint]

    if patience > 0:
        early_stop = EarlyStopping(
            monitor="val_iou",
            patience=patience,
            mode="max",
            verbose=True,
        )
        callbacks.append(early_stop)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    return callbacks
