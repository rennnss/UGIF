"""Tests for DII_improved formula correctness."""
import pytest
import torch
from src.explainability.dii import compute_dii_improved, compute_dii_grid, interpret_dii
from src.explainability.shap_explainer import SHAPExplainer


class TestDII:
    def test_no_change_dii_near_one(self):
        """Identical pre/post features → DII ≈ 1."""
        f = torch.tensor([[0.5, 0.4, 0.3, 0.6]])
        phi = SHAPExplainer.synthetic_phi_norm(4)
        dii = compute_dii_improved(f, f, phi)
        assert abs(dii.item() - 1.0) < 1e-4

    def test_severe_damage_dii_high(self):
        """Near-zero post features → DII >> 1."""
        f_pre  = torch.tensor([[0.8, 0.7, 0.6, 0.9]])
        f_post = torch.tensor([[0.05, 0.05, 0.05, 0.05]])
        phi = SHAPExplainer.synthetic_phi_norm(4)
        dii = compute_dii_improved(f_pre, f_post, phi)
        assert dii.item() > 5.0

    def test_phi_normalisation_enforced(self):
        """DII should be unaffected by phi scale (it re-normalises internally)."""
        f_pre  = torch.tensor([[0.6, 0.5, 0.4, 0.3]])
        f_post = torch.tensor([[0.3, 0.25, 0.2, 0.15]])
        phi1 = torch.tensor([0.25, 0.25, 0.25, 0.25])
        phi2 = torch.tensor([1.0, 1.0, 1.0, 1.0])   # 4× scale
        dii1 = compute_dii_improved(f_pre, f_post, phi1)
        dii2 = compute_dii_improved(f_pre, f_post, phi2)
        assert abs(dii1.item() - dii2.item()) < 1e-4

    def test_grid_dii_averages(self):
        """Grid DII should be the mean over N cells."""
        f_pre  = torch.rand(5, 4)
        f_post = torch.rand(5, 4)
        phi = SHAPExplainer.synthetic_phi_norm(4)
        per_cell = compute_dii_improved(f_pre, f_post, phi)
        grid_dii = compute_dii_grid(f_pre, f_post, phi)
        assert abs(grid_dii.item() - per_cell.mean().item()) < 1e-5

    def test_batch_shape(self):
        f_pre  = torch.rand(4, 4)
        f_post = torch.rand(4, 4)
        phi = SHAPExplainer.synthetic_phi_norm(4)
        dii = compute_dii_improved(f_pre, f_post, phi)
        assert dii.shape == (4,)


class TestInterpretDII:
    def test_no_damage(self):
        assert interpret_dii(0.85) == "No damage"

    def test_minor(self):
        assert interpret_dii(1.10) == "Minor damage"

    def test_moderate(self):
        assert interpret_dii(1.50) == "Moderate damage"

    def test_severe(self):
        assert interpret_dii(2.10) == "Severe damage"

    def test_catastrophic(self):
        assert interpret_dii(10.0) == "Catastrophic damage"
