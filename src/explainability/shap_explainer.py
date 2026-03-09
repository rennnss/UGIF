"""
SHAP-based explainability for the FCNEncoder / SiameseFCN.

Computes per-feature importance weights φᵢ and normalises them:

    φᵢⁿᵒʳᵐ = |φᵢ| / Σⱼ |φⱼ|

These weights feed directly into the DII_improved formula.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

from src.models.siamese import SiameseFCN


class SHAPExplainer:
    """Gradient-based SHAP explainer for the UGIF Siamese encoder.

    Uses ``shap.GradientExplainer`` which supports PyTorch natively.
    A reference background dataset is required (typically 50-100 samples).

    Args:
        model:       Trained :class:`SiameseFCN` instance (eval mode).
        background:  ``(N, C, H, W)`` background tensor for SHAP baseline.
        feature_names: Names for the k semantic features.
    """

    def __init__(
        self,
        model: SiameseFCN,
        background: Tensor,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        try:
            import shap
            self._shap = shap
        except ImportError:
            raise ImportError("Install shap: pip install shap")

        self.model = model.eval()
        self.feature_names = feature_names or [
            "Buildings", "Roads", "Vegetation", "Infrastructure"
        ]

        # Wrap encoder to return (B, k) presence values
        self._explainer = shap.GradientExplainer(
            model=self._encoder_wrapper,
            data=background,
        )

    def _encoder_wrapper(self, x: Tensor) -> Tensor:
        return self.model.encoder(x)  # (B, k)

    def explain(self, x: Tensor) -> np.ndarray:
        """Compute SHAP values for a batch of images.

        Args:
            x: ``(B, C, H, W)`` input tensor.

        Returns:
            ``(B, C, H, W)`` SHAP values per pixel per input channel.
        """
        shap_values = self._explainer.shap_values(x)
        return shap_values

    def feature_importance(self, x: Tensor) -> Tensor:
        """Compute normalised per-feature importance φᵢⁿᵒʳᵐ.

        Aggregates spatial SHAP values to obtain a scalar importance per
        semantic feature, then L1-normalises.

        Args:
            x: ``(B, C, H, W)`` input batch.

        Returns:
            ``(k,)`` normalised importance weights summing to 1.
        """
        shap_vals = self.explain(x)  # list of (B, C, H, W) — one per output node

        # Each element corresponds to one feature fi; collapse to |mean| per feature
        phi = torch.zeros(len(shap_vals))
        for i, sv in enumerate(shap_vals):
            phi[i] = abs(float(np.mean(sv)))

        # L1 normalise
        phi_norm = phi / (phi.sum() + 1e-9)
        return phi_norm

    @staticmethod
    def synthetic_phi_norm(num_features: int = 4) -> Tensor:
        """Return uniform importance weights when SHAP is not available.

        Useful for testing the DII pipeline without a trained model.
        """
        return torch.ones(num_features, dtype=torch.float32) / num_features
