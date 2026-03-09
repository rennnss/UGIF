"""Tests for LEVIR-CD dataset, SAR fusion, and TorchGeo batch splitting."""
import pytest
import torch
from src.data.fusion import fuse_optical_sar, SAROpticalFusionTransform, split_batch


class TestSARFusion:
    def test_fuse_channels(self):
        optical = torch.rand(3, 64, 64)
        sar     = torch.rand(2, 64, 64)
        fused   = fuse_optical_sar(optical, sar)
        assert fused.shape == (5, 64, 64)

    def test_spatial_mismatch_raises(self):
        optical = torch.rand(3, 64, 64)
        sar     = torch.rand(2, 32, 32)
        with pytest.raises(ValueError):
            fuse_optical_sar(optical, sar)

    def test_fusion_transform(self):
        """SAROpticalFusionTransform: (3,H,W) → (5,H,W) per image."""
        sample = {
            "pre_image":  torch.rand(3, 64, 64),
            "post_image": torch.rand(3, 64, 64),
            "mask":       torch.zeros(1, 64, 64),
        }
        transform = SAROpticalFusionTransform(num_sar_channels=2)
        out = transform(sample)
        assert out["pre_image"].shape[0]  == 5   # 3 RGB + 2 SAR
        assert out["post_image"].shape[0] == 5


class TestSplitBatch:
    def test_split_shapes(self):
        """split_batch: (B,10,H,W) → two (B,5,H,W) tensors."""
        image = torch.rand(4, 10, 64, 64)
        pre, post = split_batch(image)
        assert pre.shape  == (4, 5, 64, 64)
        assert post.shape == (4, 5, 64, 64)

    def test_split_content(self):
        image = torch.rand(2, 10, 32, 32)
        pre, post = split_batch(image)
        assert torch.allclose(pre,  image[:, :5])
        assert torch.allclose(post, image[:, 5:])


class TestSyntheticFallback:
    def test_no_transform_shapes(self):
        """Bare LEVIRCDPatchDataset still produces 3-channel images."""
        from src.data.levir_dataset import LEVIRCDPatchDataset
        ds = LEVIRCDPatchDataset(root="/nonexistent", patch_size=64, synthetic_size=5)
        sample = ds[0]
        assert sample["pre_image"].shape  == (3, 64, 64)
        assert sample["post_image"].shape == (3, 64, 64)
        assert sample["mask"].shape       == (1, 64, 64)

    def test_with_fusion_transform(self):
        """With SAROpticalFusionTransform the shapes grow to 5-ch."""
        from src.data.levir_dataset import LEVIRCDPatchDataset
        ds = LEVIRCDPatchDataset(
            root="/nonexistent",
            patch_size=64,
            synthetic_size=5,
            transform=SAROpticalFusionTransform(num_sar_channels=2),
        )
        sample = ds[0]
        assert sample["pre_image"].shape[0]  == 5
        assert sample["post_image"].shape[0] == 5
