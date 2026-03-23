"""
Debug script to understand what raw values the DII model is producing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from src.training.lightning_module import UGIFLightningModule
from src.data.fusion import SAROpticalFusionTransform

_OPT_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_OPT_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_FUSE = SAROpticalFusionTransform(num_sar_channels=2)

def load(path):
    img = Image.open(path).convert("RGB")
    t = TF.to_tensor(img)
    return (t - _OPT_MEAN) / _OPT_STD

def fuse(rgb):
    sample = {"pre_image": rgb, "post_image": rgb, "mask": torch.zeros(1, *rgb.shape[-2:])}
    return _FUSE(sample)["pre_image"]

ckpt = "ugif-epoch=01-val_iou=0.5591.ckpt"
pre_path  = "data/inference/italy_sar/pre_image.png"
post_path = "data/inference/italy_sar/post_image.png"

model = UGIFLightningModule.load_from_checkpoint(ckpt, map_location="cpu")
model.eval()

pre_rgb  = load(pre_path)
post_rgb = load(post_path)
print(f"Image shape: {pre_rgb.shape}")
print(f"Pre  pixel range: [{pre_rgb.min():.3f}, {pre_rgb.max():.3f}]")
print(f"Post pixel range: [{post_rgb.min():.3f}, {post_rgb.max():.3f}]")
print(f"Pixel-level diff max = {(pre_rgb - post_rgb).abs().max():.4f}")

pre_fused  = fuse(pre_rgb)
post_fused = fuse(post_rgb)
print(f"\nFused shape: {pre_fused.shape}")

epsilon = 1e-6
with torch.no_grad():
    p = pre_fused.unsqueeze(0)
    q = post_fused.unsqueeze(0)
    f_pre  = model.model.encoder.forward_map(p)
    f_post = model.model.encoder.forward_map(q)
    print(f"\nEncoder output shape: {f_pre.shape}")
    print(f"f_pre  range: [{f_pre.min():.6f}, {f_pre.max():.6f}]")
    print(f"f_post range: [{f_post.min():.6f}, {f_post.max():.6f}]")

    k = f_pre.shape[1]
    phi = 1.0 / k
    ratio = torch.abs(f_pre + epsilon) / torch.abs(f_post + epsilon)
    dii_spatial = (phi * ratio).sum(dim=1).squeeze()
    print(f"\nDII spatial map shape: {dii_spatial.shape}")
    print(f"DII spatial range: [{dii_spatial.min():.6f}, {dii_spatial.max():.6f}]")
    print(f"DII spatial mean:  {dii_spatial.mean():.6f}")
    print(f"DII spatial std:   {dii_spatial.std():.6f}")
    
    # Check actual pixel-level image difference
    pre_np  = np.array(Image.open(pre_path).convert("RGB")).astype(float) / 255.0
    post_np = np.array(Image.open(post_path).convert("RGB")).astype(float) / 255.0
    raw_diff = np.abs(pre_np - post_np).mean(axis=2)
    print(f"\nRaw image pixel-diff range: [{raw_diff.min():.4f}, {raw_diff.max():.4f}]")
    print(f"Top 1% raw pixel diff: {np.percentile(raw_diff, 99):.4f}")
