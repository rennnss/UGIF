"""
UGIF Fused SAR + Optical Change Detection (Cloud-Robust)
=========================================================
Downloads both Sentinel-1 SAR and Sentinel-2 Optical imagery for the same
event, runs DII inference on each, and saves two outputs:

  outputs/individual/<name>.png  — 3-row comparison grid
      Row 1: Optical Pre  |  Optical Post
      Row 2: SAR Pre      |  SAR Post
      Row 3: Optical heatmap result  |  SAR heatmap result

  outputs/fused/<name>.png       — Single fused image
      Post Optical overlaid with the max-fused SAR+Optical DII heatmap

MAX-FUSION: fused_dii[x,y] = max(dii_optical[x,y], dii_sar[x,y])
Wherever clouds block optical, SAR fills in (and vice versa).

Usage
-----
# From scratch (downloads both sensors automatically):
python scripts/predict_fused.py \\
    --project astute-baton-486304-r1 \\
    --ckpt ugif-epoch=01-val_iou=0.5591.ckpt \\
    --bbox 11.8 44.3 12.1 44.6 \\
    --pre-start 2023-03-01 --pre-end 2023-04-30 \\
    --post-start 2023-05-15 --post-end 2023-05-30 \\
    --scale 20 \\
    --name italy_flood

# Or re-use already-downloaded images:
python scripts/predict_fused.py \\
    --ckpt ugif-epoch=01-val_iou=0.5591.ckpt \\
    --pre-optical  data/inference/italy_optical/pre_image.png \\
    --post-optical data/inference/italy_optical/post_image.png \\
    --pre-sar      data/inference/italy_sar/pre_image.png \\
    --post-sar     data/inference/italy_sar/post_image.png \\
    --name italy_flood
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw

import torchvision.transforms.functional as TF

from src.training.lightning_module import UGIFLightningModule
from src.data.fusion import SAROpticalFusionTransform

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from download_gee import (
    init_ee, get_sentinel1_image, get_sentinel2_image, download_image_as_png,
)

# ── Normalisation (same as training) ─────────────────────────────────────────
_OPT_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_OPT_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_FUSE = SAROpticalFusionTransform(num_sar_channels=2)


def load_and_preprocess(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = TF.to_tensor(img)
    return (t - _OPT_MEAN) / _OPT_STD


def fuse_channels(rgb: torch.Tensor) -> torch.Tensor:
    sample = {"pre_image": rgb, "post_image": rgb,
              "mask": torch.zeros(1, *rgb.shape[-2:])}
    return _FUSE(sample)["pre_image"]


def compute_dii_grid(model, pre_rgb, post_rgb,
                     patch_size=256, stride=128, epsilon=1e-6, device="cpu") -> np.ndarray:
    """Pixel-perfect DII map (H, W) using sliding window FCN.

    Key invariants:
    - dii_map initialised to 1.0 (= no change baseline) so any pixel not
      covered by a window defaults to no-change rather than to 0 (which
      would appear as maximum deviation and create edge bars).
    - The row/col ranges always include a final edge-aligned patch so that
      every pixel in the right column and bottom row is visited at least once.
    """
    _, H, W = pre_rgb.shape
    dii_map   = np.ones((H, W), dtype=np.float32)   # default = no change (1.0)
    count_map = np.zeros((H, W), dtype=np.float32)

    pre_fused  = fuse_channels(pre_rgb).to(device)
    post_fused = fuse_channels(post_rgb).to(device)

    # Build row/col starts: regular stride + forced final edge patch
    def make_starts(dim):
        starts = list(range(0, max(1, dim - patch_size + 1), stride))
        last = dim - patch_size
        if last >= 0 and (not starts or starts[-1] != last):
            starts.append(last)
        return starts

    model.eval()
    with torch.no_grad():
        for r in make_starts(H):
            for c in make_starts(W):
                r2, c2 = min(r + patch_size, H), min(c + patch_size, W)
                p = pre_fused[:, r:r2, c:c2].unsqueeze(0)
                q = post_fused[:, r:r2, c:c2].unsqueeze(0)

                f_pre  = model.model.encoder.forward_map(p)
                f_post = model.model.encoder.forward_map(q)

                k   = f_pre.shape[1]
                phi = 1.0 / k
                ratio       = torch.abs(f_pre + epsilon) / torch.abs(f_post + epsilon)
                dii_spatial = (phi * ratio).sum(dim=1, keepdim=True)

                ph, pw = r2 - r, c2 - c
                dii_up = torch.nn.functional.interpolate(
                    dii_spatial, size=(ph, pw), mode='bilinear', align_corners=False
                ).squeeze().cpu().numpy()

                # Weighted accumulate (visited pixels gradually replace the 1.0 default)
                dii_map[r:r2, c:c2]   = (dii_map[r:r2, c:c2] * count_map[r:r2, c:c2] + dii_up) \
                                         / (count_map[r:r2, c:c2] + 1)
                count_map[r:r2, c:c2] += 1.0

    return dii_map


def normalize_dii(dii_map: np.ndarray, gamma: float = 0.45) -> np.ndarray:
    """
    Normalize the DII map to [0, 1] for heatmap rendering.

    Three-stage process designed to prevent blocky / uniform-wash outputs:

    1. Gaussian smooth  — eliminates hard edges caused by the FCN's 4× downsampled
                          feature grid being bilinearly upsampled back to image size.

    2. Deviation from 1 — only pixels that meaningfully differ from the "no change"
                          baseline (DII ≈ 1.0) are highlighted.  Uniform areas where
                          all values cluster around 1.0 become invisible instead of
                          a false red wash.

    3. Adaptive threshold — the highlight band starts at the 80th percentile of
                            deviations, ensuring only the top 20% most-changed
                            pixels are shown.  This self-adjusts across any dataset,
                            weak or strong signal.

    4. Gamma boost       — lifts mid-range values for visual clarity.
    """
    from PIL import ImageFilter

    # ── 1. Gaussian smooth to remove 4px feature-block edges ─────────────────
    # PIL GaussianBlur needs uint8; scale the DII range to [0,255] and back
    dii_min, dii_max = dii_map.min(), dii_map.max()
    if dii_max > dii_min:
        dii_u8 = ((dii_map - dii_min) / (dii_max - dii_min) * 255).astype(np.uint8)
    else:
        dii_u8 = np.zeros_like(dii_map, dtype=np.uint8)
    dii_blurred_u8 = np.array(
        Image.fromarray(dii_u8, mode='L').filter(ImageFilter.GaussianBlur(radius=4))
    )
    dii_smooth = dii_blurred_u8.astype(np.float32) / 255.0 * (dii_max - dii_min) + dii_min

    # ── 2. Deviation from the "no change" baseline (1.0) ─────────────────────
    deviation = np.abs(dii_smooth - 1.0)

    # ── 3. Adaptive threshold: highlight only top 20% deviating pixels ────────
    p_lo = np.percentile(deviation, 80)   # 80th percentile as lower gate
    p_hi = np.percentile(deviation, 99.5) # 99.5th prevents a single spike dominating

    if p_hi > p_lo + 1e-6:
        norm = np.clip((deviation - p_lo) / (p_hi - p_lo), 0, 1)
    else:
        # Completely flat map — nothing changed
        return np.zeros_like(dii_map)

    # ── 4. Gamma boost ─────────────────────────────────────────────────────────
    return np.power(norm, gamma)


def detect_cloud_mask(img: Image.Image,
                      brightness_threshold: int = 210,
                      blur_radius: int = 15) -> np.ndarray:
    """
    Detect clouds in an optical RGB image using brightness thresholding.
    
    Clouds appear as very bright near-white pixels (high R, G, B simultaneously).
    Returns a float32 mask in [0, 1] where 1.0 = definitely cloud, 0.0 = clear sky/land.
    A soft Gaussian blur creates a smooth gradient at cloud edges.
    
    Args:
        img: PIL RGB image
        brightness_threshold: pixels where all RGB channels > this are flagged as cloud
        blur_radius: how many pixels to soften the binary cloud edge (prevents harsh lines)
    """
    from PIL import ImageFilter
    arr = np.array(img, dtype=np.float32)
    
    # Cloud pixels: all 3 channels are above the threshold (very bright and grey/white)
    is_cloud = (
        (arr[..., 0] > brightness_threshold) &
        (arr[..., 1] > brightness_threshold) &
        (arr[..., 2] > brightness_threshold)
    ).astype(np.float32)
    
    # Soft the mask edges to avoid hard lines at cloud boundaries
    cloud_pil = Image.fromarray((is_cloud * 255).astype(np.uint8))
    cloud_blurred = cloud_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    soft_mask = np.array(cloud_blurred, dtype=np.float32) / 255.0
    
    cloud_pct = soft_mask.mean() * 100
    print(f"  Cloud coverage detected: {cloud_pct:.1f}%")
    return soft_mask


def cloud_aware_fusion(dii_optical: np.ndarray,
                       dii_sar: np.ndarray,
                       cloud_mask: np.ndarray) -> np.ndarray:
    """
    Smart cloud-aware DII fusion.
    At each pixel:
      fused = (1 - cloud_weight) * dii_optical + cloud_weight * dii_sar
    """
    h, w = dii_optical.shape
    cloud_r = np.array(Image.fromarray(cloud_mask).resize((w, h), Image.BILINEAR))
    cloud_r = np.clip(cloud_r, 0, 1)
    if dii_optical.shape != dii_sar.shape:
        dii_sar = np.array(Image.fromarray(dii_sar).resize((w, h), Image.BILINEAR))
    return (1.0 - cloud_r) * dii_optical + cloud_r * dii_sar


def compute_valid_mask(pre_path: str, post_path: str,
                       black_threshold: int = 8) -> np.ndarray:
    """
    Build a float mask [0, 1] from the pre and post images where 1 = valid data.
    Near-black pixels (all channels < black_threshold) in either image are nodata;
    those pixels are set to 0 in the mask so they never appear in the heatmap.
    The mask is blurred slightly to avoid harsh edges at nodata boundaries.
    """
    from PIL import ImageFilter
    pre  = np.array(Image.open(pre_path).convert("RGB"), dtype=np.float32)
    post = np.array(Image.open(post_path).convert("RGB"), dtype=np.float32)

    pre_valid  = ~((pre[...,0]  < black_threshold) & (pre[...,1]  < black_threshold) & (pre[...,2]  < black_threshold))
    post_valid = ~((post[...,0] < black_threshold) & (post[...,1] < black_threshold) & (post[...,2] < black_threshold))
    valid = (pre_valid & post_valid).astype(np.float32)

    # Erode edges slightly so boundary artifacts don't bleed into heatmap
    valid_pil = Image.fromarray((valid * 255).astype(np.uint8))
    valid_blurred = np.array(valid_pil.filter(ImageFilter.MinFilter(5)), dtype=np.float32) / 255.0

    nodata_frac = 1.0 - valid_blurred.mean()
    print(f"  No-data pixels masked out: {nodata_frac*100:.1f}%")
    return valid_blurred


def overlay_heatmap(base_img: Image.Image, dii_map: np.ndarray,
                    alpha: float = 0.75, rgb: tuple = (200, 30, 30)) -> Image.Image:
    """Overlay a heatmap on a PIL RGB image."""
    W, H = base_img.size
    norm = normalize_dii(dii_map)
    norm_r = np.array(Image.fromarray(norm.astype(np.float32)).resize((W, H), Image.BILINEAR))
    layer = np.zeros((H, W, 4), dtype=np.uint8)
    layer[..., 0] = rgb[0]
    layer[..., 1] = rgb[1]
    layer[..., 2] = rgb[2]
    layer[..., 3] = (norm_r * 255 * alpha).astype(np.uint8)
    return Image.alpha_composite(base_img.convert("RGBA"),
                                 Image.fromarray(layer, mode="RGBA")).convert("RGB")


def label_image(img: Image.Image, text: str,
                color=(255, 255, 255), bg=(30, 30, 30)) -> Image.Image:
    bar_h = 28
    W, H = img.size
    out = Image.new("RGB", (W, H + bar_h), color=bg)
    out.paste(img, (0, 0))
    ImageDraw.Draw(out).text((6, H + 5), text, fill=color)
    return out


def build_grid(images: list, rows: int, cols: int, pad: int = 10) -> Image.Image:
    W, H = images[0].size
    total_w = cols * W + (cols + 1) * pad
    total_h = rows * H + (rows + 1) * pad
    canvas = Image.new("RGB", (total_w, total_h), color=(20, 20, 20))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        canvas.paste(img, (pad + c * (W + pad), pad + r * (H + pad)))
    return canvas


def validate_image(path: str, label: str, black_threshold: int = 5,
                   max_black_fraction: float = 0.35) -> bool:
    """
    Validate a downloaded image before inference.
    Checks:
    - File exists and has non-trivial size (> 5 KB)
    - File opens without corruption
    - Image has content globally (pixel stddev > 1)
    - No large black no-data regions: checks a 4x4 grid of cells;
      fails if the overall fraction of near-black pixels exceeds max_black_fraction
    """
    if not os.path.exists(path):
        print(f"  ✗ [{label}] File missing: {path}")
        return False

    size_kb = os.path.getsize(path) / 1024
    if size_kb < 5:
        print(f"  ✗ [{label}] File suspiciously small ({size_kb:.1f} KB) — likely blank.")
        return False

    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32)
    except Exception as e:
        print(f"  ✗ [{label}] Corrupt or unreadable image: {e}")
        return False

    std = arr.std()
    mean = arr.mean()
    if std < 1.0:
        print(f"  ✗ [{label}] Image appears blank (mean={mean:.1f}, std={std:.2f}) — no content.")
        return False

    # Detect large no-data (near-black) regions
    is_black = (arr[..., 0] < black_threshold) & \
               (arr[..., 1] < black_threshold) & \
               (arr[..., 2] < black_threshold)
    black_frac = is_black.mean()
    if black_frac > max_black_fraction:
        print(f"  ✗ [{label}] Large black no-data region detected ({black_frac*100:.1f}% of pixels are black).")
        print(f"       Try different dates or a smaller bounding box.")
        return False

    print(f"  ✓ [{label}] Valid  |  size={size_kb:.0f} KB  mean={mean:.1f}  std={std:.1f}  black={black_frac*100:.1f}%")
    return True


def main():
    parser = argparse.ArgumentParser(description="UGIF Fused SAR+Optical Change Detection")

    parser.add_argument("--ckpt",   required=True)
    parser.add_argument("--name",   default="flood", help="Output filename stem")
    parser.add_argument("--patch",  type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--project",    default=None)
    parser.add_argument("--bbox",       nargs=4, type=float,
                        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--pre-start",  default=None)
    parser.add_argument("--pre-end",    default=None)
    parser.add_argument("--post-start", default=None)
    parser.add_argument("--post-end",   default=None)
    parser.add_argument("--scale",      type=float, default=20.0)

    parser.add_argument("--pre-optical",  default=None)
    parser.add_argument("--post-optical", default=None)
    parser.add_argument("--pre-sar",      default=None)
    parser.add_argument("--post-sar",     default=None)

    args = parser.parse_args()
    need_download = not all([args.pre_optical, args.post_optical, args.pre_sar, args.post_sar])

    if need_download:
        if not args.bbox or not args.pre_start:
            print("ERROR: Provide cached image paths OR download args.")
            sys.exit(1)
        init_ee(args.project)
        import ee
        min_lon, min_lat, max_lon, max_lat = args.bbox
        roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

        # Persistent output dirs under data/inference/<name>/
        opt_dir = os.path.join("data", "inference", args.name, "optical")
        sar_dir = os.path.join("data", "inference", args.name, "sar")
        os.makedirs(opt_dir, exist_ok=True)
        os.makedirs(sar_dir, exist_ok=True)
        print(f"\nSaving downloads to:")
        print(f"  Optical → {opt_dir}")
        print(f"  SAR     → {sar_dir}")

        paths = {}
        sensor_dirs = {"optical": opt_dir, "sar": sar_dir}
        for sensor in ["optical", "sar"]:
            for phase, start, end in [("pre", args.pre_start, args.pre_end),
                                       ("post", args.post_start, args.post_end)]:
                print(f"\n--- {phase.upper()} {sensor.upper()} ---")
                img = (get_sentinel2_image(roi, start, end) if sensor == "optical"
                       else get_sentinel1_image(roi, start, end))
                p = os.path.join(sensor_dirs[sensor], f"{phase}_image.png")
                download_image_as_png(img, roi, p, scale=args.scale)
                paths[f"{phase}_{sensor}"] = p

        # Validate all four images before spending time on inference
        print("\n─── Image Validation ───")
        valid = True
        for key, label in [("pre_optical",  "Optical Pre"),
                           ("post_optical", "Optical Post"),
                           ("pre_sar",      "SAR Pre"),
                           ("post_sar",     "SAR Post")]:
            if not validate_image(paths[key], label):
                valid = False
        if not valid:
            print("\n✗ One or more images failed validation. Aborting inference.")
            print("  Check the data/inference/ folder and re-run download with different dates or bbox.")
            sys.exit(1)
        print("All images are valid. Proceeding to inference...\n")

        pre_opt_path  = paths["pre_optical"];  post_opt_path = paths["post_optical"]
        pre_sar_path  = paths["pre_sar"];      post_sar_path = paths["post_sar"]
    else:
        pre_opt_path  = args.pre_optical;  post_opt_path = args.post_optical
        pre_sar_path  = args.pre_sar;      post_sar_path = args.post_sar
        # Validate pre-cached images too
        print("\n─── Image Validation ───")
        all_ok = all([
            validate_image(pre_opt_path,  "Optical Pre"),
            validate_image(post_opt_path, "Optical Post"),
            validate_image(pre_sar_path,  "SAR Pre"),
            validate_image(post_sar_path, "SAR Post"),
        ])
        if not all_ok:
            print("\n✗ One or more images failed validation. Aborting.")
            sys.exit(1)
        print("All images are valid. Proceeding to inference...\n")

    print(f"\nLoading model ({args.device})")
    model = UGIFLightningModule.load_from_checkpoint(args.ckpt, map_location=args.device)
    model.to(args.device).eval()
    opts = dict(patch_size=args.patch, stride=args.stride, device=args.device)

    print("\n[1/3] Optical DII...")
    dii_opt = compute_dii_grid(model,
                               load_and_preprocess(pre_opt_path),
                               load_and_preprocess(post_opt_path), **opts)
    print(f"  mean={dii_opt.mean():.4f}  max={dii_opt.max():.4f}")

    print("\n[2/3] SAR DII...")
    dii_sar = compute_dii_grid(model,
                               load_and_preprocess(pre_sar_path),
                               load_and_preprocess(post_sar_path), **opts)
    print(f"  mean={dii_sar.mean():.4f}  max={dii_sar.max():.4f}")

    print("\n[3/3] Smart cloud-aware fusion...")
    post_opt_for_cloud = Image.open(post_opt_path).convert("RGB")
    cloud_mask = detect_cloud_mask(post_opt_for_cloud)
    dii_fused = cloud_aware_fusion(dii_opt, dii_sar, cloud_mask)
    print(f"  Fused DII mean={dii_fused.mean():.4f}  max={dii_fused.max():.4f}")

    # ── Mask out no-data (black) pixels so they never appear in any heatmap ──
    print("\nBuilding valid-data mask...")
    opt_valid_mask = compute_valid_mask(pre_opt_path, post_opt_path)
    sar_valid_mask = compute_valid_mask(pre_sar_path, post_sar_path)

    def apply_mask(dii: np.ndarray, mask: np.ndarray) -> np.ndarray:
        H, W = dii.shape
        mask_r = np.array(Image.fromarray(mask).resize((W, H), Image.BILINEAR))
        return dii * np.clip(mask_r, 0, 1)

    dii_opt   = apply_mask(dii_opt, opt_valid_mask)
    dii_fused = apply_mask(dii_fused, np.minimum(opt_valid_mask, sar_valid_mask))

    # Resize SAR DII to match optical dims for panel rendering, then mask
    h_opt, w_opt = dii_opt.shape
    dii_sar_r = (np.array(Image.fromarray(dii_sar).resize((w_opt, h_opt), Image.BILINEAR))
                 if dii_opt.shape != dii_sar.shape else dii_sar)
    dii_sar_r = apply_mask(dii_sar_r, sar_valid_mask)

    # ── Load source images ────────────────────────────────────────────────────
    pre_opt_img  = Image.open(pre_opt_path).convert("RGB")
    post_opt_img = Image.open(post_opt_path).convert("RGB")
    W, H = pre_opt_img.size
    pre_sar_img  = Image.open(pre_sar_path).convert("RGB").resize((W, H), Image.BILINEAR)
    post_sar_img = Image.open(post_sar_path).convert("RGB").resize((W, H), Image.BILINEAR)

    # ── Heatmap panels ────────────────────────────────────────────────────────
    opt_heatmap = overlay_heatmap(post_opt_img.copy(), dii_opt,       rgb=(200, 30, 30))
    sar_heatmap = overlay_heatmap(post_sar_img.copy(), dii_sar_r,     rgb=(180, 90, 0))

    # ── Output directories ────────────────────────────────────────────────────
    os.makedirs("outputs/individual", exist_ok=True)
    os.makedirs("outputs/fused",      exist_ok=True)

    # ── Individual grid: 3 rows × 2 cols ─────────────────────────────────────
    panels = [
        label_image(pre_opt_img,  "Optical – Pre-event",  color=(180, 210, 255)),
        label_image(post_opt_img, "Optical – Post-event", color=(180, 210, 255)),
        label_image(pre_sar_img,  "SAR – Pre-event",      color=(255, 210, 140)),
        label_image(post_sar_img, "SAR – Post-event",     color=(255, 210, 140)),
        label_image(opt_heatmap,  f"Optical DII heatmap  (mean {dii_opt.mean():.3f})",
                    color=(255, 110, 90)),
        label_image(sar_heatmap,  f"SAR DII heatmap  (mean {dii_sar.mean():.3f})",
                    color=(255, 170, 70)),
    ]
    pw, ph = panels[0].size
    panels = [p.resize((pw, ph), Image.BILINEAR) for p in panels]

    grid = build_grid(panels, rows=3, cols=2, pad=10)
    individual_path = f"outputs/individual/{args.name}.png"
    grid.save(individual_path)
    print(f"\n✓ Saved individual grid  → {individual_path}")

    # ── Fused image: side-by-side pre-event vs fused heatmap ──────────────────
    fused_heatmap = overlay_heatmap(post_opt_img.copy(), dii_fused, rgb=(200, 20, 80))

    pre_labeled   = label_image(pre_opt_img,  "Optical – Pre-event",
                                color=(180, 210, 255))
    fused_labeled = label_image(fused_heatmap,
                                f"FUSED heatmap (SAR + Optical)   DII mean={dii_fused.mean():.3f}",
                                color=(255, 100, 150))

    # Make both the same height (label bar may differ by 1px)
    pw2, ph2 = pre_labeled.size
    fused_labeled = fused_labeled.resize((pw2, ph2), Image.BILINEAR)

    pad = 10
    fused_canvas = Image.new("RGB", (pw2 * 2 + pad * 3, ph2 + pad * 2), color=(20, 20, 20))
    fused_canvas.paste(pre_labeled,   (pad,           pad))
    fused_canvas.paste(fused_labeled, (pw2 + pad * 2, pad))

    fused_path = f"outputs/fused/{args.name}.png"
    fused_canvas.save(fused_path)
    print(f"✓ Saved fused result     → {fused_path}")


if __name__ == "__main__":
    main()
