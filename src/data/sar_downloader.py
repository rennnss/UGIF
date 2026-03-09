"""
SAR Data Downloader using armkhudinyan/copernicus_api.

This module wraps the Sentinel1API from the copernicus_api library
(cloned into third_party/copernicus_api) to search and download
Sentinel-1 GRD SAR data from the Copernicus Data Space Ecosystem (CDSE).

Registration (free): https://dataspace.copernicus.eu/

Usage::

    from src.data.sar_downloader import SARDownloader

    dl = SARDownloader(username="your@email.com", password="yourpass")
    dl.download(
        bbox=(80.0, 12.0, 81.0, 13.5),          # (min_lon, min_lat, max_lon, max_lat)
        start_date="2023-08-01",
        end_date="2023-08-31",
        out_dir="data/SAR/chennai_2023",
        orbit_direction="ASCENDING",             # optional filter
        max_products=5,
    )
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional

# ── Load copernicus_api via importlib to avoid 'src' package name collision ────
# Our project already has a 'src/' package; adding the third_party repo to
# sys.path would cause Python to pick up our src/__init__.py instead of theirs.
# We load the modules explicitly by file path using importlib.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_COP_API_DIR = _REPO_ROOT / "third_party" / "copernicus_api" / "src"

import importlib.util as _ilu
import types as _types

def _load_cop_module(name: str, path: Path) -> _types.ModuleType:
    """Load a module from an explicit file path and cache it in sys.modules."""
    full_name = f"_copernicus_api_vendor.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = _ilu.spec_from_file_location(full_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot find {name} at {path}")
    mod = _ilu.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod

# Load in dependency order (exceptions first so relative imports succeed)
try:
    # Create a fake parent package so relative imports inside copernicus_api work
    _pkg_name = "_copernicus_api_vendor"
    if _pkg_name not in sys.modules:
        _pkg = _types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(_COP_API_DIR)]  # type: ignore[assignment]
        _pkg.__package__ = _pkg_name
        sys.modules[_pkg_name] = _pkg

    _exc_mod = _load_cop_module("exceptions", _COP_API_DIR / "exceptions.py")
    _geo_mod = _load_cop_module("geo_utils",  _COP_API_DIR / "geo_utils.py")
    _api_mod = _load_cop_module("copernicus_api", _COP_API_DIR / "copernicus_api.py")

    Sentinel1API = _api_mod.Sentinel1API      # type: ignore
    to_openeo_wkt = _geo_mod.to_openeo_wkt   # type: ignore
except Exception as e:
    raise ImportError(
        "Failed to load copernicus_api from third_party/. Make sure you have run:\n"
        "  git clone https://github.com/armkhudinyan/copernicus_api.git "
        "third_party/copernicus_api\n"
        f"Original error: {e}"
    )


def _bbox_to_wkt_polygon(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> str:
    """Convert a bounding box to a WKT POLYGON string accepted by CDSE OData."""
    return (
        f"POLYGON(("
        f"{min_lon} {min_lat}, "
        f"{max_lon} {min_lat}, "
        f"{max_lon} {max_lat}, "
        f"{min_lon} {max_lat}, "
        f"{min_lon} {min_lat}"
        f"))"
    )


class SARDownloader:
    """High-level wrapper around Sentinel1API for UGIF SAR data collection.

    Credentials are loaded from:
      1. Constructor arguments  (highest priority)
      2. Environment variables  COPERNICUS_USER / COPERNICUS_PASS
      3. .env file in project root (loaded via python-dotenv if installed)

    Args:
        username: CDSE account email.
        password: CDSE account password.
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        # Try .env file first (silent if dotenv not installed)
        try:
            from dotenv import load_dotenv
            load_dotenv(_REPO_ROOT / ".env")
        except ImportError:
            pass

        self.username = username or os.environ.get("COPERNICUS_USER", "")
        self.password = password or os.environ.get("COPERNICUS_PASS", "")

        if not self.username or not self.password:
            raise ValueError(
                "CDSE credentials required. Provide them via:\n"
                "  - SARDownloader(username=..., password=...)\n"
                "  - Environment variables COPERNICUS_USER / COPERNICUS_PASS\n"
                "  - A .env file in the project root\n"
                "  Register free at https://dataspace.copernicus.eu/"
            )

        self._api = Sentinel1API(self.username, self.password)

    def query(
        self,
        bbox: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        prod_type: str = "GRD",
        orbit_direction: Optional[str] = None,
        relative_orbit: Optional[list[int]] = None,
        max_products: int = 20,
    ):
        """Search the CDSE catalogue for Sentinel-1 products.

        Args:
            bbox:            ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG 4326.
            start_date:      Query start in ``YYYY-MM-DD`` format.
            end_date:        Query end in ``YYYY-MM-DD`` format.
            prod_type:       Product type to match in file name (default ``"GRD"``).
            orbit_direction: Optional — ``"ASCENDING"`` or ``"DESCENDING"``.
            relative_orbit:  Optional list of relative orbit numbers, e.g. ``[52, 118]``.
            max_products:    Maximum number of results.

        Returns:
            ``pd.DataFrame`` of matching products (same as ``Sentinel1API.query``).
        """
        footprint = _bbox_to_wkt_polygon(*bbox)

        # Build optional attribute filters
        extra = {}
        if orbit_direction:
            extra["orbitDirection"] = [orbit_direction.upper()]
        if relative_orbit:
            extra["relativeOrbitNumber"] = relative_orbit

        print(
            f"[SARDownloader] Querying Sentinel-1 {prod_type} | "
            f"{start_date} → {end_date} | bbox={bbox}"
        )

        products = self._api.query(
            start_time=start_date,
            end_time=end_date,
            prod_type=prod_type,
            # NOTE: do NOT pass exclude='COG' — since 2022 CDSE distributes
            # ALL Sentinel-1 GRD products in COG format, so excluding COG
            # returns zero results every time.
            footprint=footprint,
            orderby="desc",
            limit=max_products,
            **extra,
        )

        print(f"[SARDownloader] Found {len(products)} products.")
        return products

    def download(
        self,
        bbox: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        out_dir: str | Path = "data/SAR",
        prod_type: str = "GRD",
        orbit_direction: Optional[str] = None,
        relative_orbit: Optional[list[int]] = None,
        max_products: int = 20,
        threads: int = 4,
    ) -> None:
        """Query and download Sentinel-1 SAR data to local disk.

        Args:
            bbox:            ``(min_lon, min_lat, max_lon, max_lat)`` bounding box.
            start_date:      Start date ``YYYY-MM-DD``.
            end_date:        End date ``YYYY-MM-DD``.
            out_dir:         Local output directory (created if absent).
            prod_type:       Product type filter (default ``"GRD"``).
            orbit_direction: ``"ASCENDING"`` or ``"DESCENDING"``.
            relative_orbit:  Filter by relative orbit numbers.
            max_products:    Max number of products to download.
            threads:         Parallel download threads.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        products = self.query(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            prod_type=prod_type,
            orbit_direction=orbit_direction,
            relative_orbit=relative_orbit,
            max_products=max_products,
        )

        if products.empty:
            print("[SARDownloader] No products matched. Nothing downloaded.")
            return

        print(f"[SARDownloader] Downloading {len(products)} products to {out_dir} ...")
        self._api.download_all(products, out_dir=out_dir, threads=threads)
        print("[SARDownloader] Done.")

    def download_from_geojson(
        self,
        geojson_path: str | Path,
        start_date: str,
        end_date: str,
        out_dir: str | Path = "data/SAR",
        prod_type: str = "GRD",
        max_products: int = 20,
    ) -> None:
        """Download SAR data using a GeoJSON bounding region file.

        Convenience method that calls ``to_openeo_wkt`` directly on a GeoJSON
        file (same helper used by the copernicus_api examples).

        Args:
            geojson_path: Path to a ``.geojson`` or ``.shp`` file.
            start_date:   Start date ``YYYY-MM-DD``.
            end_date:     End date ``YYYY-MM-DD``.
            out_dir:      Output directory.
            prod_type:    Product type filter.
            max_products: Maximum products to download.
        """
        footprint = to_openeo_wkt(str(geojson_path))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[SARDownloader] Querying from GeoJSON: {geojson_path} | "
            f"{start_date} → {end_date}"
        )

        products = self._api.query(
            start_time=start_date,
            end_time=end_date,
            prod_type=prod_type,
            footprint=footprint,
            orderby="desc",
            limit=max_products,
        )

        if products.empty:
            print("[SARDownloader] No products matched.")
            return

        print(f"[SARDownloader] Downloading {len(products)} products to {out_dir} ...")
        self._api.download_all(products, out_dir=out_dir)
        print("[SARDownloader] Done.")
