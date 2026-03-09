"""
CLI entrypoint for Sentinel-1 SAR data download using copernicus_api.

Usage examples::

    # Basic area + date range download
    python scripts/download_sar.py \\
        --bbox 80.0 12.0 81.0 13.5 \\
        --start 2023-08-01 --end 2023-08-31 \\
        --out data/SAR/chennai_2023

    # Filter by orbit direction
    python scripts/download_sar.py \\
        --bbox 80.0 12.0 81.0 13.5 \\
        --start 2023-08-01 --end 2023-08-31 \\
        --orbit ASCENDING --max 10

    # Use a GeoJSON footprint file
    python scripts/download_sar.py \\
        --geojson path/to/area.geojson \\
        --start 2023-08-01 --end 2023-08-31

Credentials are read from env vars COPERNICUS_USER / COPERNICUS_PASS,
or from a .env file in the project root, or passed directly:

    python scripts/download_sar.py --user email@x.com --password mypass ...

Register free at: https://dataspace.copernicus.eu/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.sar_downloader import SARDownloader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Sentinel-1 SAR GRD data via Copernicus CDSE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Credentials ──────────────────────────────────────────────────
    cred = parser.add_argument_group("Credentials (or use env vars / .env)")
    cred.add_argument("--user",     default=None, help="CDSE account email")
    cred.add_argument("--password", default=None, help="CDSE password")

    # ── Spatial extent ────────────────────────────────────────────────
    spatial = parser.add_argument_group("Spatial extent (choose one)")
    spatial.add_argument(
        "--bbox", nargs=4, type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box in EPSG 4326",
    )
    spatial.add_argument(
        "--geojson", type=str,
        help="Path to a .geojson or .shp file defining the search footprint",
    )

    # ── Time range ────────────────────────────────────────────────────
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   required=True, help="End date   YYYY-MM-DD")

    # ── Query options ─────────────────────────────────────────────────
    parser.add_argument("--prod-type", default="GRD",
                        help="Product type keyword (default: GRD)")
    parser.add_argument("--orbit",     default=None,
                        choices=["ASCENDING", "DESCENDING"],
                        help="Filter by orbit direction")
    parser.add_argument("--max",       default=20, type=int,
                        help="Maximum number of products (default: 20)")
    parser.add_argument("--threads",   default=4,  type=int,
                        help="Parallel download threads (default: 4)")
    parser.add_argument("--out", default="data/SAR",
                        help="Output directory (default: data/SAR)")
    parser.add_argument("--query-only", action="store_true",
                        help="Only query and print results, do not download")

    args = parser.parse_args()

    if args.bbox is None and args.geojson is None:
        parser.error("Provide either --bbox or --geojson.")

    dl = SARDownloader(username=args.user, password=args.password)

    if args.query_only:
        bbox = tuple(args.bbox) if args.bbox else None
        products = dl.query(
            bbox=bbox,
            start_date=args.start,
            end_date=args.end,
            prod_type=args.prod_type,
            orbit_direction=args.orbit,
            max_products=args.max,
        )
        print(products[["Name", "ContentDate", "ContentLength"]].to_string())
        return

    if args.geojson:
        dl.download_from_geojson(
            geojson_path=args.geojson,
            start_date=args.start,
            end_date=args.end,
            out_dir=args.out,
            prod_type=args.prod_type,
            max_products=args.max,
        )
    else:
        dl.download(
            bbox=tuple(args.bbox),
            start_date=args.start,
            end_date=args.end,
            out_dir=args.out,
            prod_type=args.prod_type,
            orbit_direction=args.orbit,
            max_products=args.max,
            threads=args.threads,
        )


if __name__ == "__main__":
    main()
