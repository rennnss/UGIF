"""
Deep DII diagnostic - understand the distribution of raw values in detail.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from src.training.lightning_module import UGIFLightningModule
from src.data.fusion import SAROpticalFusionTransform

_OPT_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_OPT_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_FUSE = SAROpticalFusionTransform(num_sar_channels=2)

def load(path):
    t = TF.to_tensor(Image.open(path).convert("RGB"))
    return (t - _OPT_MEAN) / _OPT_STD

def fuse(rgb):
    s = {"pre_image": rgb, "post_image": rgb, "mask": torch.zeros(1, *rgb.shape[-2:])}
    return _FUSE(s)["pre_image"]

model = UGIFLightningModule.load_from_checkpoint(
    "ugif-epoch=01-val_iou=0.5591.ckpt", map_location="cpu")
model.eval()

pre  = load("data/inference/kerala_2018_flood/optical/pre_image.png")
post = load("data/inference/kerala_2018_flood/optical/post_image.png")

print(f"Image shape: {pre.shape}")

epsilon = 1e-6
with torch.no_grad():
    pf = fuse(pre).unsqueeze(0)
    qf = fuse(post).unsqueeze(0)
    fp = model.model.encoder.forward_map(pf)
    fq = model.model.encoder.forward_map(qf)
    print(f"Feature map shape: {fp.shape}")
    
    ratio = torch.abs(fp + epsilon) / torch.abs(fq + epsilon)
    dii   = (ratio / ratio.shape[1]).sum(dim=1).squeeze().numpy()
    
    print(f"\nRaw DII spatial map stats:")
    print(f"  shape  = {dii.shape}")
    print(f"  min    = {dii.min():.6f}")
    print(f"  max    = {dii.max():.6f}")
    print(f"  mean   = {dii.mean():.6f}")
    print(f"  std    = {dii.std():.6f}")
    print(f"  median = {np.median(dii):.6f}")
    print(f"\n  Percentiles:")
    for p in [50, 80, 90, 95, 98, 99, 99.5, 99.9]:
        print(f"    p{p:5.1f} = {np.percentile(dii, p):.6f}")
    
    deviation = np.abs(dii - 1.0)
    print(f"\nDeviation from 1.0 stats:")
    print(f"  max deviation = {deviation.max():.6f}")
    print(f"  std deviation = {deviation.std():.6f}")
    for p in [80, 90, 95, 99]:
        print(f"    dev p{p} = {np.percentile(deviation, p):.6f}")
    
    print(f"\nFraction of pixels with |DII-1| > 0.05 : {(deviation>0.05).mean()*100:.1f}%")
    print(f"Fraction of pixels with |DII-1| > 0.10 : {(deviation>0.10).mean()*100:.1f}%")
    print(f"Fraction of pixels with |DII-1| > 0.20 : {(deviation>0.20).mean()*100:.1f}%")
