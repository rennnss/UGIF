"""
UGIF Visual Change Detection — DII Heatmap Inference
======================================================
Divides the image into a spatial grid, runs the Siamese FCN on each cell,
computes the DII_improved score per cell using the paper formula, and
renders a red heatmap overlay showing which areas changed most.

Formula
-------
DII_improved = (1/N) Σ_g Σ_i  φ_i^norm · |f̄(O_pre, S_pre)_{g,i} + ε|
                                           ─────────────────────────────
                                           |f̄(O_post, S_post)_{g,i} + ε|

where g = grid cell, i = feature index (1…k), φ_i^norm = 1/k (uniform).

Usage
-----
python scripts/predict.py \\
    --ckpt   checkpoints/ugif-epoch=01-val_iou=0.5591.ckpt \\
    --pre    path/to/pre_image.png \\
    --post   path/to/post_image.png \\
    --out    change_map.png          # optional, default: change_map.png
    --stride 128                     # grid stride in pixels (smaller = finer map)
    --patch  256                     # patch size (must match training)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF

from src.training.lightning_module import UGIFLightningModule
from src.data.fusion import SAROpticalFusionTransform

# ── Normalisation (same as training) ───────────────────────────────────────
_OPT_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_OPT_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_FUSE = SAROpticalFusionTransform(num_sar_channels=2)


def load_and_preprocess(path: str) -> torch.Tensor:
    """Load image → normalized (3, H, W) float32 tensor."""
    img = Image.open(path).convert("RGB")
    t = TF.to_tensor(img)                    # [0,1], (3, H, W)
    return (t - _OPT_MEAN) / _OPT_STD       # ImageNet-normalised


def fuse(rgb: torch.Tensor) -> torch.Tensor:
    """Append 2 synthetic SAR channels → (5, H, W)."""
    sample = {"pre_image": rgb, "post_image": rgb,
              "mask": torch.zeros(1, *rgb.shape[-2:])}
    return _FUSE(sample)["pre_image"]


def compute_dii_grid(
    model: UGIFLightningModule,
    pre_rgb: torch.Tensor,
    post_rgb: torch.Tensor,
    patch_size: int = 256,
    stride: int = 128,
    epsilon: float = 1e-6,
    device: str = "cpu",
) -> np.ndarray:
    """
    Slide a window over the image and compute DII_improved for every cell.

    Returns a float32 array of shape (H, W) with DII scores per pixel
    (each pixel gets the score of the cell it belongs to).
    """
    _, H, W = pre_rgb.shape
    dii_map   = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    # Precompute fused tensors
    pre_fused  = fuse(pre_rgb).to(device)
    post_fused = fuse(post_rgb).to(device)

    rows = range(0, max(1, H - patch_size + 1), stride)
    cols = range(0, max(1, W - patch_size + 1), stride)

    model.eval()
    with torch.no_grad():
        for r in rows:
            for c in cols:
                r2 = min(r + patch_size, H)
                c2 = min(c + patch_size, W)

                p  = pre_fused[:, r:r2, c:c2].unsqueeze(0)   # (1, 5, ph, pw)
                q  = post_fused[:, r:r2, c:c2].unsqueeze(0)

                # Spatial feature maps: (1, k, h', w')
                f_pre  = model.model.encoder.forward_map(p)   # sigmoid activated
                f_post = model.model.encoder.forward_map(q)

                k = f_pre.shape[1]
                phi = 1.0 / k   # uniform normalised weight per indicator

                # Pixel-perfect DII: Calculate per spatial location in (h', w')
                ratio = torch.abs(f_pre + epsilon) / torch.abs(f_post + epsilon)
                dii_spatial = (phi * ratio).sum(dim=1, keepdim=True)  # (1, 1, h', w')

                # Interpolate up to full patch size to restore pixel perfection
                patch_h, patch_w = r2 - r, c2 - c
                dii_upsampled = torch.nn.functional.interpolate(
                    dii_spatial, 
                    size=(patch_h, patch_w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze().cpu().numpy()

                dii_map[r:r2, c:c2]   += dii_upsampled
                count_map[r:r2, c:c2] += 1.0

    # Average overlapping cells
    count_map = np.maximum(count_map, 1)
    return dii_map / count_map


def make_visualization(
    pre_path: str,
    post_path: str,
    dii_map: np.ndarray,
    overall_dii: float,
    out_path: str,
    alpha: float = 0.55,
    color: str = "dark_red",
    threshold: float = 1.0
) -> None:
    """
    Save a side-by-side figure:
        [Pre image]  |  [Post image]  |  [Post + red DII heatmap]
    """
    pre_img  = Image.open(pre_path).convert("RGB")
    post_img = Image.open(post_path).convert("RGB")
    H, W     = np.array(pre_img).shape[:2]

    # Resize dii_map to image size
    dii_resized = np.array(
        Image.fromarray(dii_map).resize((W, H), Image.BILINEAR)
    )

    # Robust percentile-based normalisation (handles extreme outlier pixels)
    # Clip to [2nd, 98th] percentile so a few extreme spikes don't crush the rest to zero.
    p_lo = np.percentile(dii_resized, 2)
    p_hi = np.percentile(dii_resized, 98)
    
    if p_hi > p_lo + 1e-6:
        norm = np.clip((dii_resized - p_lo) / (p_hi - p_lo), 0, 1)
    else:
        norm = np.zeros_like(dii_resized)

    # Optional: if a hard threshold is set, zero out pixels that are below it
    if threshold > 0:
        base = np.percentile(dii_resized, threshold * 50)  # rough mapping
        norm = np.where(dii_resized >= base, norm, norm * 0.15)  # dim; not fully blank

    # Heatmap color mask: intensity = DII score
    color_layer = np.zeros((H, W, 4), dtype=np.uint8)
    if color == "red":
        color_layer[..., 0] = 255
    elif color == "dark_red":
        color_layer[..., 0] = 200   # Better contrast than 160 against dark SAR
    elif color == "cyan":
        color_layer[..., 1] = 255
        color_layer[..., 2] = 255
    elif color == "magenta":
        color_layer[..., 0] = 255
        color_layer[..., 2] = 255
    elif color == "yellow":
        color_layer[..., 0] = 255
        color_layer[..., 1] = 255
        
    color_layer[..., 3] = (norm * 255 * alpha).astype(np.uint8)  # alpha

    color_pil = Image.fromarray(color_layer, mode="RGBA")

    # Overlay heatmap on post image
    post_rgba = post_img.convert("RGBA")
    overlay   = Image.alpha_composite(post_rgba, color_pil).convert("RGB")

    # Compose side-by-side canvas
    pad   = 8
    total_w = W * 3 + pad * 4
    total_h = H + pad * 2 + 30   # extra height for label
    canvas  = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))

    canvas.paste(pre_img,  (pad,             pad))
    canvas.paste(post_img, (W + pad * 2,     pad))
    canvas.paste(overlay,  (W * 2 + pad * 3, pad))

    # Labels
    draw = ImageDraw.Draw(canvas)
    label_y = H + pad + 4
    draw.text((pad,             label_y), "Pre-event",          fill="white")
    draw.text((W + pad * 2,     label_y), "Post-event",         fill="white")
    draw.text((W * 2 + pad * 3, label_y),
              f"DII heatmap  (score: {overall_dii:.3f})",       fill=(255, 120, 100))

    canvas.save(out_path)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="UGIF DII visual change detection")
    parser.add_argument("--ckpt",   required=True, help=".ckpt checkpoint path")
    parser.add_argument("--pre",    required=True, help="Pre-event image")
    parser.add_argument("--post",   required=True, help="Post-event image")
    parser.add_argument("--out",    default="change_map.png")
    parser.add_argument("--patch",  type=int, default=256)
    parser.add_argument("--stride", type=int, default=128,
                        help="Smaller = finer heatmap, slower")
    parser.add_argument("--color",  type=str, default="dark_red", choices=["red", "dark_red", "cyan", "magenta", "yellow"],
                        help="Heatmap mask color (dark_red is default)")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Absolute DII damage threshold. Default 1.0.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device   : {args.device}")
    print(f"Checkpoint: {args.ckpt}")

    # Load model
    model = UGIFLightningModule.load_from_checkpoint(
        args.ckpt, map_location=args.device
    )
    model.to(args.device).eval()

    # Load images
    pre_rgb  = load_and_preprocess(args.pre)
    post_rgb = load_and_preprocess(args.post)

    # Compute DII heatmap
    print("Computing DII heatmap …")
    dii_map = compute_dii_grid(
        model, pre_rgb, post_rgb,
        patch_size=args.patch,
        stride=args.stride,
        device=args.device,
    )

    # Overall DII score = mean across all cells
    overall_dii = float(dii_map.mean())
    print(f"Overall DII score : {overall_dii:.4f}")
    print(f"  > 1.0 = damage detected | ≈ 1.0 = no significant change")

    # Save visualisation
    make_visualization(args.pre, args.post, dii_map, overall_dii, args.out, color=args.color, threshold=args.threshold)


if __name__ == "__main__":
    main()
