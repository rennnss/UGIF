"""
Google Earth Engine Downloader for UGIF Inference
===================================================
Downloads pre- and post-disaster Sentinel-2 optical imagery 
as standard RGB PNGs ready for predict.py.

Usage:
  # 1. First time only, authenticate with Earth Engine
  earthengine authenticate

  # 2. Run the download script
  python scripts/download_gee.py \
      --bbox 80.0 12.0 81.0 13.5 \
      --pre-start 2023-01-01 --pre-end 2023-07-01 \
      --post-start 2023-08-01 --post-end 2023-08-31 \
      --out data/inference/chennai_flood
"""

import argparse
import os
import sys

try:
    import ee
    import requests
except ImportError:
    print("Error: Earth Engine Python API not installed.")
    print("Please run: pip install earthengine-api requests")
    sys.exit(1)

def init_ee(project=None):
    """Initialize Earth Engine, falling back to authentication if needed."""
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as e:
        print("Earth Engine not authenticated, or project not found.")
        print(f"Details: {e}")
        print("\nTry running:")
        print("  earthengine authenticate")
        print("If you have an Earth Engine project, pass it using --project <YOUR_PROJECT_ID>")
        sys.exit(1)

def mask_s2_clouds(image):
    """Masks clouds in Sentinel-2 images using the QA60 band."""
    qa = image.select('QA60')
    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
           qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask)

def get_sentinel2_image(roi, start_date, end_date, max_cloud=100.0):
    """
    Fetch a median Sentinel-2 SR image for the given ROI and date range,
    pixel-level cloud-masked and converted to an 8-bit RGB image.
    """
    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(roi)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud))
                  .map(mask_s2_clouds))
    
    # Calculate median composite across the date range (ignoring masked cloud pixels)
    median_img = collection.median()
    
    # Check if the collection is empty, to prevent blank masks
    
    # Convert B4 (Red), B3 (Green), B2 (Blue) to 8-bit RGB [0, 255]
    # Max reflectance of 3000 is a standard threshold for Sentinel-2 visualisations
    rgb_img = median_img.visualize(bands=['B4', 'B3', 'B2'], min=0, max=3000)
    
    return rgb_img

def get_sentinel1_image(roi, start_date, end_date):
    """
    Fetch a median Sentinel-1 SAR GRD image, which is completely cloud-free!
    Converts Radar backscatter (VV and VH) into a 3-channel RGB PNG format. 
    """
    collection = (ee.ImageCollection("COPERNICUS/S1_GRD")
                  .filterBounds(roi)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                  .filter(ee.Filter.eq('instrumentMode', 'IW')))
    
    # Calculate median composite across the date range
    median_img = collection.median()
    
    # Convert VV/VH decibel values (typically -25 to 0) into an 8-bit RGB [0, 255]
    # We use VV for Red, VH for Green, and VV for Blue as a standard SAR visualisation
    rgb_img = median_img.visualize(bands=['VV', 'VH', 'VV'], min=-25, max=0)
    
    return rgb_img

def download_image_as_png(image, roi, out_path, max_pixels=1e8, scale=10):
    """
    Download an ee.Image as a PNG and save to out_path.
    """
    print(f"Requesting download URL for {out_path}...")
    try:
        # Request a download URL from Earth Engine
        url = image.getDownloadURL({
            'format': 'png',
            'region': roi,
            'scale': scale, # 10m/px for Sentinel-2 and Sentinel-1
            'maxPixels': max_pixels
        })
        
        print(f"Downloading from {url[:50]}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(out_path, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)
                
        print(f"Successfully saved to: {out_path}")
        
    except Exception as e:
        print(f"Failed to download image. Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download pre/post optical or SAR images from GEE for inference.")
    
    # Bounding Box
    parser.add_argument("--bbox", nargs=4, type=float, required=True,
                        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                        help="Bounding box in EPSG:4326")
    
    # Date ranges
    parser.add_argument("--pre-start", required=True, help="Pre-event start date YYYY-MM-DD")
    parser.add_argument("--pre-end", required=True, help="Pre-event end date YYYY-MM-DD")
    parser.add_argument("--post-start", required=True, help="Post-event start date YYYY-MM-DD")
    parser.add_argument("--post-end", required=True, help="Post-event end date YYYY-MM-DD")
    
    # Output
    parser.add_argument("--out", required=True, help="Output directory path")
    
    # Project
    parser.add_argument("--project", default=None, help="Google Cloud Project ID for Earth Engine")
    
    # Scale (Resolution)
    parser.add_argument("--scale", type=float, default=30.0, 
                        help="Resolution in meters per pixel. Increase this (e.g., 30 or 50) for large bounding boxes to avoid size limits. Default is 30.")
    parser.add_argument("--max-cloud", type=float, default=100.0, 
                        help="Max cloudy pixel percentage per scene to include in the composite (default 100).")
    
    # Sensor 
    parser.add_argument("--sensor", choices=["optical", "sar"], default="optical", 
                        help="Choose 'optical' (Sentinel-2) or 'sar' (Sentinel-1) for cloud-free radar.")
    
    args = parser.parse_args()
    
    init_ee(args.project)
    
    # Create the ROI geometry
    min_lon, min_lat, max_lon, max_lat = args.bbox
    roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    
    # Ensure output directory exists
    os.makedirs(args.out, exist_ok=True)
    
    # 1. Process Pre-Event Image
    print(f"\n--- Processing Pre-Event Image ({args.sensor.upper()}) ---")
    if args.sensor == "sar":
        pre_img = get_sentinel1_image(roi, args.pre_start, args.pre_end)
    else:
        pre_img = get_sentinel2_image(roi, args.pre_start, args.pre_end, max_cloud=args.max_cloud)
        
    pre_out_path = os.path.join(args.out, "pre_image.png")
    download_image_as_png(pre_img, roi, pre_out_path, scale=args.scale)
    
    # 2. Process Post-Event Image
    print(f"\n--- Processing Post-Event Image ({args.sensor.upper()}) ---")
    if args.sensor == "sar":
        post_img = get_sentinel1_image(roi, args.post_start, args.post_end)
    else:
        post_img = get_sentinel2_image(roi, args.post_start, args.post_end, max_cloud=args.max_cloud)
        
    post_out_path = os.path.join(args.out, "post_image.png")
    download_image_as_png(post_img, roi, post_out_path, scale=args.scale)
    
    print("\nDone! You can now run prediction:")
    print(f"python scripts/predict.py --ckpt <your_ckpt> --pre {pre_out_path} --post {post_out_path}")

if __name__ == "__main__":
    main()
