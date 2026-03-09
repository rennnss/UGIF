"""
Visualization utilities for damage maps and SHAP importance.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor


def plot_damage_map(
    pre: Tensor,
    post: Tensor,
    mask: Optional[Tensor] = None,
    prediction: Optional[Tensor] = None,
    dii_score: Optional[float] = None,
    save_path: Optional[str] = None,
) -> None:
    """Visualise pre/post image pair with optional ground-truth and prediction overlays.

    Args:
        pre:        ``(3, H, W)`` or ``(C, H, W)`` pre-disaster image (RGB first 3 ch).
        post:       ``(3, H, W)`` or ``(C, H, W)`` post-disaster image.
        mask:       ``(1, H, W)`` binary ground-truth change mask.
        prediction: ``(1, H, W)`` predicted change probability map.
        dii_score:  Scalar DII value to display in title.
        save_path:  Path to save figure (shows in interactive mode if None).
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    def to_rgb(t: Tensor) -> np.ndarray:
        rgb = t[:3].permute(1, 2, 0).cpu().numpy()
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        return np.clip(rgb, 0, 1)

    n_cols = 2 + (mask is not None) + (prediction is not None)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(to_rgb(pre))
    axes[0].set_title("Pre-Disaster (Optical)", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(to_rgb(post))
    axes[1].set_title("Post-Disaster (Optical)", fontsize=12)
    axes[1].axis("off")

    col = 2
    if mask is not None:
        axes[col].imshow(mask.squeeze().cpu().numpy(), cmap="Reds", vmin=0, vmax=1)
        axes[col].set_title("Ground Truth Mask", fontsize=12)
        axes[col].axis("off")
        col += 1

    if prediction is not None:
        axes[col].imshow(prediction.squeeze().detach().cpu().numpy(),
                         cmap="jet", vmin=0, vmax=1)
        axes[col].set_title("Predicted Change Map", fontsize=12)
        axes[col].axis("off")

    title = "UGIF — Damage Assessment"
    if dii_score is not None:
        title += f" | DII = {dii_score:.3f}"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Damage map saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_shap_importance(
    phi_norm: Tensor,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Bar chart of normalised SHAP feature importances.

    Args:
        phi_norm:     ``(k,)`` normalised importance weights.
        feature_names: Labels for each feature.
        save_path:    Path to save figure.
    """
    import matplotlib.pyplot as plt

    if feature_names is None:
        feature_names = ["Buildings", "Roads", "Vegetation", "Infrastructure"]

    values = phi_norm.cpu().numpy()
    colours = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"][: len(values)]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(feature_names, values, color=colours, edgecolor="white", height=0.6)
    ax.set_xlabel("Normalised SHAP Importance (φᵢⁿᵒʳᵐ)", fontsize=11)
    ax.set_title("Feature Importance for DII Computation", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.2)
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"SHAP chart saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
