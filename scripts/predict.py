"""
UGIF Real-time Inference Script
================================
Usage:
    python scripts/predict.py \
        --ckpt  path/to/ugif-epoch=XX-val_iou=X.XXXX.ckpt \
        --pre   path/to/pre_image.png \
        --post  path/to/post_image.png \
        [--threshold 0.5]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from src.training.lightning_module import UGIFLightningModule
from src.data.fusion import SAROpticalFusionTransform


# ── Normalisation constants (same as training) ─────────────────────────────
_OPT_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_OPT_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_image(path: str) -> torch.Tensor:
    """Load a PNG/JPG as a normalised (3, H, W) float32 tensor."""
    img = Image.open(path).convert("RGB")
    t = TF.to_tensor(img)          # [0, 1] float32, (3, H, W)
    t = (t - _OPT_MEAN) / _OPT_STD
    return t


def predict(ckpt_path: str, pre_path: str, post_path: str,
            threshold: float = 0.5, device: str = "cpu") -> dict:
    """
    Run change-detection inference on a pre/post image pair.

    Returns a dict with:
        - 'change_prob'  : float in [0, 1] — probability of change
        - 'changed'      : bool — whether the patch is classified as changed
        - 'damage_ratio' : np.ndarray (k,) — per-feature damage ratios
    """
    # ── Load model from checkpoint ─────────────────────────────────────────
    model = UGIFLightningModule.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)

    # ── Load and preprocess images ─────────────────────────────────────────
    pre  = load_image(pre_path).to(device)   # (3, H, W)
    post = load_image(post_path).to(device)

    # Append synthetic SAR channels (same as training)
    _fuse = SAROpticalFusionTransform(num_sar_channels=2)
    sample = {"pre_image": pre, "post_image": post, "mask": torch.zeros(1, *pre.shape[-2:])}
    sample = _fuse(sample)
    pre  = sample["pre_image"].unsqueeze(0)   # (1, 5, H, W)
    post = sample["post_image"].unsqueeze(0)

    # ── Inference ──────────────────────────────────────────────────────────
    with torch.no_grad():
        out    = model.model(pre, post)
        feat   = torch.cat([out.f_pre, out.f_post], dim=1)   # (1, 2k)
        logit  = model.model.change_head(feat).squeeze()       # scalar
        prob   = torch.sigmoid(logit).item()

    result = {
        "change_prob":  round(prob, 4),
        "changed":      prob >= threshold,
        "damage_ratio": out.ratio.squeeze().cpu().numpy(),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="UGIF change detection inference")
    parser.add_argument("--ckpt",      required=True, help="Path to .ckpt file")
    parser.add_argument("--pre",       required=True, help="Pre-event image path")
    parser.add_argument("--post",      required=True, help="Post-event image path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Change probability threshold")
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading checkpoint : {args.ckpt}")
    print(f"Device             : {args.device}")
    print()

    result = predict(args.ckpt, args.pre, args.post, args.threshold, args.device)

    print(f"Change probability : {result['change_prob']:.4f}")
    print(f"Classified as      : {'CHANGED ⚠️' if result['changed'] else 'UNCHANGED ✅'}")
    print(f"Damage ratios (k)  : {result['damage_ratio'].round(3)}")


if __name__ == "__main__":
    main()
