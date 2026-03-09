"""Tests for LEVIR-CD dataset and SAR fusion pipeline."""
import pytest
import torch
from src.data.levir_dataset import LEVIRCDPatchDataset
from src.data.fusion import fuse_optical_sar, SAROpticalFusionTransform


class TestLEVIRCDDataset:
    def test_synthetic_fallback_length(self):
        ds = LEVIRCDPatchDataset(root="/nonexistent", synthetic_size=10)
        assert len(ds) == 10

    def test_sample_shapes(self):
        ds = LEVIRCDPatchDataset(root="/nonexistent", patch_size=64, synthetic_size=5)
        sample = ds[0]
        assert sample["pre_image"].shape  == (3, 64, 64)
        assert sample["post_image"].shape == (3, 64, 64)
        assert sample["mask"].shape       == (1, 64, 64)

    def test_mask_binary(self):
        ds = LEVIRCDPatchDataset(root="/nonexistent", synthetic_size=5)
        mask = ds[0]["mask"]
        assert set(mask.unique().tolist()).issubset({0.0, 1.0})

    def test_deterministic(self):
        ds = LEVIRCDPatchDataset(root="/nonexistent", synthetic_size=5)
        s1 = ds[2]["pre_image"]
        s2 = ds[2]["pre_image"]
        assert torch.allclose(s1, s2)


class TestSARFusion:
    def test_fuse_channels(self):
        optical = torch.rand(3, 64, 64)
        sar     = torch.rand(2, 64, 64)
        fused   = fuse_optical_sar(optical, sar)
        assert fused.shape == (5, 64, 64)

    def test_spatial_mismatch_raises(self):
        optical = torch.rand(3, 64, 64)
        sar     = torch.rand(2, 32, 32)   # different spatial size
        with pytest.raises(ValueError):
            fuse_optical_sar(optical, sar)

    def test_fusion_transform(self):
        sample = {
            "pre_image":  torch.rand(3, 64, 64),
            "post_image": torch.rand(3, 64, 64),
            "mask":       torch.zeros(1, 64, 64),
        }
        transform = SAROpticalFusionTransform(num_sar_channels=2)
        out = transform(sample)
        assert out["pre_image"].shape[0]  == 5  # 3 RGB + 2 SAR
        assert out["post_image"].shape[0] == 5
