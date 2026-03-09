"""
Disaster Impact Index (DII) computation.

Implements the DII_improved formula from the architecture diagram:

    DII_improved = (1/N) Σ_{g=1}^{N} Σ_{i=1}^{k} φᵢⁿᵒʳᵐ · |f̄(O_pre, S_pre)_{g,i} + ε|
                                                              ─────────────────────────────
                                                              |f̄(O_post, S_post)_{g,i} + ε|

Where:
  • N     = number of grid cells
  • k     = number of semantic features (Buildings, Roads, Vegetation, Infrastructure)
  • φᵢⁿᵒʳᵐ = normalised SHAP importance weight for feature i
  • f̄     = spatially averaged sigmoid feature presence (from FCNEncoder)
  • ε     = numerical stability constant (1e-6)
"""
from __future__ import annotations

from typing import List

import torch
from torch import Tensor


def compute_dii_improved(
    f_pre: Tensor,
    f_post: Tensor,
    phi_norm: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    """Compute the improved Disaster Impact Index per batch sample.

    Args:
        f_pre:    ``(B, k)`` or ``(N, k)`` pre-disaster feature presence maps.
        f_post:   ``(B, k)`` or ``(N, k)`` post-disaster feature presence maps.
        phi_norm: ``(k,)`` normalised SHAP importance weights  Σ φᵢⁿᵒʳᵐ = 1.
        epsilon:  Numerical stability constant.

    Returns:
        ``(B,)`` DII score per sample (or scalar if batch-averaged).

    Notes:
        A DII >> 1 indicates severe damage (pre features >> post features).
        A DII ≈ 1 indicates little change.
        A DII < 1 may indicate recovery or vegetation regrowth.
    """
    # Ensure phi_norm sums to 1 (re-normalise defensively)
    phi = phi_norm / (phi_norm.sum() + epsilon)  # (k,)

    numerator   = (f_pre  + epsilon).abs()  # (B, k)
    denominator = (f_post + epsilon).abs()  # (B, k)

    ratio = numerator / denominator         # (B, k)

    # Weighted sum over features
    weighted = ratio * phi.unsqueeze(0)     # (B, k) * (1, k)
    dii = weighted.sum(dim=1)              # (B,)
    return dii


def compute_dii_grid(
    f_pre_grid: Tensor,
    f_post_grid: Tensor,
    phi_norm: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    """Compute DII_improved averaged over N grid cells (as in the paper formula).

    Args:
        f_pre_grid:  ``(N, k)`` pre-disaster features for N grid cells.
        f_post_grid: ``(N, k)`` post-disaster features for N grid cells.
        phi_norm:    ``(k,)`` normalised SHAP weights.
        epsilon:     Stability constant.

    Returns:
        Scalar DII score for the entire region.
    """
    per_cell = compute_dii_improved(f_pre_grid, f_post_grid, phi_norm, epsilon)
    return per_cell.mean()


# ── Interpretation ──────────────────────────────────────────────────────────

_DII_THRESHOLDS = [
    (0.90, "No damage"),
    (1.25, "Minor damage"),
    (1.75, "Moderate damage"),
    (2.50, "Severe damage"),
    (float("inf"), "Catastrophic damage"),
]


def interpret_dii(score: float) -> str:
    """Map a scalar DII score to a human-readable severity label.

    Args:
        score: DII_improved value.

    Returns:
        String label describing damage severity.
    """
    for threshold, label in _DII_THRESHOLDS:
        if score <= threshold:
            return label
    return "Catastrophic damage"
