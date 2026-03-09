"""
GeoJSON FeatureCollection builder using Nominatim (OpenStreetMap geocoding).

No API key required.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple


def geocode_location(location: str) -> Optional[Tuple[float, float, float, float]]:
    """Geocode a location string to a bounding box using Nominatim.

    Args:
        location: Place name, e.g. "Chennai, India".

    Returns:
        ``(min_lon, min_lat, max_lon, max_lat)`` bounding box, or None if not found.
    """
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut

        geo = Nominatim(user_agent="ugif-disaster-assessment/0.1")
        time.sleep(1)  # Nominatim rate limit: 1 req/sec
        result = geo.geocode(location, exactly_one=True, timeout=10)
        if result and hasattr(result, "raw"):
            raw = result.raw
            bbox = raw.get("boundingbox")
            if bbox:
                min_lat, max_lat, min_lon, max_lon = map(float, bbox)
                return (min_lon, min_lat, max_lon, max_lat)
    except Exception as e:
        print(f"[GeoJSON] Geocoding failed: {e}")
    return None


def build_geojson(parsed_query: Dict) -> Dict:
    """Build a GeoJSON FeatureCollection from a parsed query dict.

    If geocoding succeeds the bounding box polygon is populated; otherwise
    a placeholder feature is returned.

    Args:
        parsed_query: Output from :func:`src.frontend.query_parser.parse_query`.

    Returns:
        GeoJSON-compatible dict.
    """
    location  = parsed_query.get("location") or "Unknown"
    bbox = geocode_location(location) if location != "Unknown" else None

    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        geometry = {
            "type": "Polygon",
            "coordinates": [[
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]],
        }
    else:
        geometry = {"type": "Point", "coordinates": [0.0, 0.0]}

    feature = {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "location":   location,
            "start_date": parsed_query.get("start_date"),
            "end_date":   parsed_query.get("end_date"),
            "task":       parsed_query.get("task"),
            "bbox":       list(bbox) if bbox else None,
        },
    }

    return {"type": "FeatureCollection", "features": [feature]}
