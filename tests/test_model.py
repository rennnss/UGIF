"""Tests for FCN encoder and Siamese FCN forward passes."""
import pytest
import torch
from src.models.fcn import FCNEncoder
from src.models.siamese import SiameseFCN
from src.models.losses import ContrastiveLoss, DiceLoss, BCEDiceLoss


class TestFCNEncoder:
    def setup_method(self):
        self.model = FCNEncoder(in_channels=5, num_features=4, feature_dim=64)

    def test_forward_shape(self):
        x = torch.rand(2, 5, 64, 64)
        out = self.model(x)
        assert out.shape == (2, 4)  # (B, k) presence map

    def test_output_range(self):
        x = torch.rand(2, 5, 64, 64)
        out = self.model(x)
        assert out.min() >= 0.0 and out.max() <= 1.0  # sigmoid output

    def test_feature_map_shape(self):
        x = torch.rand(2, 5, 64, 64)
        fmap = self.model.forward_map(x)
        assert fmap.shape[0] == 2
        assert fmap.shape[1] == 4


class TestSiameseFCN:
    def setup_method(self):
        self.model = SiameseFCN(in_channels=5, num_features=4, feature_dim=64)

    def test_forward_output(self):
        pre  = torch.rand(2, 5, 64, 64)
        post = torch.rand(2, 5, 64, 64)
        out = self.model(pre, post)
        assert out.f_pre.shape   == (2, 4)
        assert out.f_post.shape  == (2, 4)
        assert out.ratio.shape   == (2, 4)
        assert out.distance.shape == (2,)

    def test_ratio_clipped(self):
        pre  = torch.rand(2, 5, 64, 64)
        post = torch.zeros(2, 5, 64, 64)  # extreme: post is zero
        out = self.model(pre, post)
        assert out.ratio.max().item() <= self.model.r_max + 1e-5

    def test_shared_weights(self):
        """Both streams should share the same encoder parameters."""
        pre  = torch.rand(1, 5, 32, 32)
        post = torch.rand(1, 5, 32, 32)
        out  = self.model(pre, post)
        # Feature maps from identical input should be identical
        identical_out = self.model(pre, pre)
        assert torch.allclose(identical_out.f_pre, identical_out.f_post, atol=1e-5)

    def test_predict_change(self):
        pre  = torch.rand(2, 5, 64, 64)
        post = torch.rand(2, 5, 64, 64)
        prob = self.model.predict_change(pre, post)
        assert prob.shape == (2, 1)
        assert prob.min() >= 0.0 and prob.max() <= 1.0


class TestLosses:
    def test_contrastive_similar(self):
        loss_fn = ContrastiveLoss(margin=1.0)
        dist  = torch.tensor([0.1, 0.2])
        label = torch.tensor([0, 0])  # similar pairs
        loss = loss_fn(dist, label)
        assert loss.item() >= 0

    def test_contrastive_dissimilar(self):
        loss_fn = ContrastiveLoss(margin=1.0)
        dist  = torch.tensor([0.05, 0.05])   # very similar but labelled dissimilar
        label = torch.tensor([1, 1])
        loss = loss_fn(dist, label)
        assert loss.item() > 0

    def test_dice_perfect(self):
        dice = DiceLoss()
        mask = torch.ones(2, 1, 16, 16)
        pred = torch.ones(2, 1, 16, 16) * 10  # large logit → sigmoid ≈ 1
        assert dice(pred, mask).item() < 0.05

    def test_bce_dice_combined(self):
        loss_fn = BCEDiceLoss()
        pred   = torch.rand(2, 1, 16, 16)
        target = (torch.rand(2, 1, 16, 16) > 0.5).float()
        loss = loss_fn(pred, target)
        assert loss.item() >= 0
