"""
Main training entrypoint for UGIF.

Usage::

    python src/training/train.py                      # default config
    python src/training/train.py training.max_epochs=2 training.fast_dev_run=True

Hydra automatically resolves configs/default.yaml.
"""
from __future__ import annotations

# ── Make 'src' importable when this file is run as a script ─────────────────
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ────────────────────────────────────────────────────────────────────────────

import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from src.data.datamodule import UGIFDataModule
from src.training.lightning_module import UGIFLightningModule
from src.training.callbacks import get_callbacks


@hydra.main(config_path="../../configs", config_name="default", version_base="1.3")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(42, workers=True)

    # ── DataModule ───────────────────────────────────────────────────
    dm = UGIFDataModule(
        root=cfg.data.root,
        patch_size=cfg.data.patch_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    # ── Model ────────────────────────────────────────────────────────
    model = UGIFLightningModule(
        in_channels=cfg.model.in_channels,
        num_features=cfg.model.num_features,
        r_max=cfg.model.r_max,
        epsilon=cfg.model.epsilon,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        margin=cfg.training.margin,
        max_epochs=cfg.training.max_epochs,
    )

    # ── Trainer ──────────────────────────────────────────────────────
    callbacks = get_callbacks(
        output_dir=cfg.output.dir,
        patience=cfg.training.get("patience", 0)
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        logger=pl.loggers.TensorBoardLogger(
            save_dir=cfg.output.log_dir,
            name="ugif",
        ),
        deterministic=True,
        fast_dev_run=cfg.training.get("fast_dev_run", False),
        accelerator="auto",
        devices="auto",
    )

    trainer.fit(model, datamodule=dm)
    # In fast_dev_run mode no checkpoint is saved, so use current weights
    ckpt = None if cfg.training.fast_dev_run else "best"
    trainer.test(model, datamodule=dm, ckpt_path=ckpt)


if __name__ == "__main__":
    train()
