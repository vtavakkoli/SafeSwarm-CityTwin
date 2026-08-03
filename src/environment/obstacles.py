"""Real and synthetic city-layer ingestion with deterministic disk caching."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set, Tuple

import numpy as np

Cell = Tuple[int, int]
LayerDict = Dict[str, Any]
OSM_ATTRIBUTION = "© OpenStreetMap contributors"
OSM_LICENSE_URL = "https://www.openstreetmap.org/copyright"


def _clip_cell(x: int, y: int, grid_size: int) -> Optional[Cell]:
    if 0 <= x < grid_size and 0 <= y < grid_size:
        return (x, y)
    return None


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized[:60] or "city"


def _cache_path(cache_dir: str | Path, place_name: str, grid_size: int, radius_m: int) -> Path:
    digest = hashlib.sha256(f"{place_name}|{grid_size}|{radius_m}".encode()).hexdigest()[:10]
    return Path(cache_dir) / f"{_slug(place_name)}-g{grid_size}-r{radius_m}-{digest}.json"


def _priority_list(priority_cells: Mapping[Cell, float]) -> list[list[float]]:
    return [[int(x), int(y), float(score)] for (x, y), score in sorted(priority_cells.items())]


def save_layer_cache(path: str | Path, layers: Mapping[str, Any]) -> None:
    """Persist a compact, human-readable city snapshot."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "obstacles": [list(c) for c in sorted(layers["obstacles"])],
        "restricted_zones": [list(c) for c in sorted(layers["restricted_zones"])],
        "mission_zones": [list(c) for c in sorted(layers["mission_zones"])],
        "base_stations": [list(c) for c in sorted(layers["base_stations"])],
        "priority_cells": _priority_list(layers.get("priority_cells", {})),
        "metadata": dict(layers.get("metadata", {})),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_layer_cache(path: str | Path) -> LayerDict:
    """Load a city snapshot produced by :func:`save_layer_cache`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "obstacles": {tuple(map(int, c)) for c in payload["obstacles"]},
        "restricted_zones": {tuple(map(int, c)) for c in payload["restricted_zones"]},
        "mission_zones": {tuple(map(int, c)) for c in payload["mission_zones"]},
        "base_stations": {tuple(map(int, c)) for c in payload["base_stations"]},
        "priority_cells": {
            (int(x), int(y)): float(score) for x, y, score in payload.get("priority_cells", [])
        },
        "metadata": dict(payload.get("metadata", {})),
    }


def synthetic_city_layers(grid_size: int, seed: int = 42, place_name: str = "synthetic") -> LayerDict:
    """Create a deterministic fallback with weighted urban monitoring targets."""
    if grid_size < 8:
        raise ValueError("grid_size must be at least 8")

    place_offset = int(hashlib.sha256(place_name.encode("utf-8")).hexdigest()[:8], 16)
    effective_seed = int(seed) ^ place_offset
    rng = np.random.default_rng(effective_seed)
    obstacles: Set[Cell] = set()
    restricted: Set[Cell] = set()
    priority: dict[Cell, float] = {}

    for _ in range(max(4, grid_size // 8)):
        x0 = int(rng.integers(1, max(2, grid_size - 4)))
        y0 = int(rng.integers(1, max(2, grid_size - 4)))
        w = int(rng.integers(2, min(6, grid_size - x0) + 1))
        h = int(rng.integers(2, min(6, grid_size - y0) + 1))
        for x in range(x0, min(grid_size, x0 + w)):
            for y in range(y0, min(grid_size, y0 + h)):
                obstacles.add((x, y))

    for _ in range(max(2, grid_size // 18)):
        x0 = int(rng.integers(2, grid_size - 2))
        y0 = int(rng.integers(2, grid_size - 2))
        radius = int(rng.integers(1, 3))
        for x in range(max(0, x0 - radius), min(grid_size, x0 + radius + 1)):
            for y in range(max(0, y0 - radius), min(grid_size, y0 + radius + 1)):
                if (x - x0) ** 2 + (y - y0) ** 2 <= radius**2:
                    restricted.add((x, y))

    candidates = [
        (x, y)
        for x in range(grid_size)
        for y in range(grid_size)
        if (x, y) not in obstacles and (x, y) not in restricted
    ]
    rng.shuffle(candidates)
    for index, cell in enumerate(candidates[: max(12, grid_size // 2)]):
        priority[cell] = float([1.0, 0.85, 0.70, 0.55][index % 4])

    bases: Set[Cell] = set()
    for corner in [(0, 0), (grid_size - 1, grid_size - 1), (0, grid_size - 1)]:
        bases.add(_nearest_walkable(corner, grid_size, obstacles, restricted))

    obstacles -= bases
    restricted -= bases
    mission = set(priority)
    return {
        "obstacles": obstacles,
        "restricted_zones": restricted,
        "mission_zones": mission,
        "base_stations": bases,
        "priority_cells": priority,
        "metadata": {
            "source": "synthetic",
            "place_name": place_name,
            "feature_count": 0,
            "grid_size": grid_size,
            "seed": seed,
            "effective_seed": effective_seed,
            "attribution": "Synthetic deterministic fallback; no external data used.",
        },
    }


def _nearest_walkable(
    origin: Cell, grid_size: int, obstacles: Set[Cell], restricted: Set[Cell]
) -> Cell:
    if origin not in obstacles and origin not in restricted:
        return origin
    ox, oy = origin
    for radius in range(1, grid_size):
        for x in range(max(0, ox - radius), min(grid_size, ox + radius + 1)):
            for y in range(max(0, oy - radius), min(grid_size, oy + radius + 1)):
                cell = (x, y)
                if cell not in obstacles and cell not in restricted:
                    return cell
    return (0, 0)


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    try:
        if bool(np.isnan(value)):
            return False
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    return text not in {"", "nan", "none", "false"}


def _cell_from_geometry(geometry: Any, bounds: tuple[float, float, float, float], grid_size: int) -> Optional[Cell]:
    if geometry is None or geometry.is_empty:
        return None
    point = geometry if geometry.geom_type == "Point" else geometry.representative_point()
    minx, miny, maxx, maxy = bounds
    dx = max(maxx - minx, 1e-12)
    dy = max(maxy - miny, 1e-12)
    x = int(np.clip(round((float(point.x) - minx) / dx * (grid_size - 1)), 0, grid_size - 1))
    y = int(np.clip(round((float(point.y) - miny) / dy * (grid_size - 1)), 0, grid_size - 1))
    return _clip_cell(x, y, grid_size)


def _priority_for_row(row: Any) -> float:
    amenity = str(row.get("amenity", "")).lower()
    if amenity in {"hospital", "fire_station", "police", "clinic"}:
        return 1.0
    if amenity in {"school", "kindergarten", "nursing_home", "community_centre"}:
        return 0.85
    if _is_present(row.get("public_transport")) or str(row.get("railway", "")).lower() in {
        "station",
        "halt",
        "tram_stop",
    }:
        return 0.75
    if _is_present(row.get("tourism")):
        return 0.65
    if str(row.get("leisure", "")).lower() in {"park", "playground", "sports_centre"}:
        return 0.55
    if _is_present(row.get("shop")) or _is_present(row.get("office")):
        return 0.45
    return 0.0


def _download_osm_layers(place_name: str, grid_size: int, radius_m: int, seed: int) -> LayerDict:
    import osmnx as ox

    center = ox.geocode(place_name)
    tags = {
        "building": True,
        "landuse": ["industrial", "railway"],
        "natural": ["water", "wetland"],
        "water": True,
        "amenity": [
            "hospital",
            "clinic",
            "fire_station",
            "police",
            "school",
            "kindergarten",
            "nursing_home",
            "community_centre",
        ],
        "public_transport": True,
        "railway": ["station", "halt", "tram_stop"],
        "leisure": ["park", "playground", "sports_centre"],
        "tourism": True,
        "shop": True,
        "office": True,
    }
    gdf = ox.features_from_point(center, tags=tags, dist=radius_m)
    if gdf.empty:
        raise RuntimeError(f"OpenStreetMap returned no features for {place_name!r}")

    bounds = tuple(float(v) for v in gdf.total_bounds)
    obstacles: Set[Cell] = set()
    restricted: Set[Cell] = set()
    bases: Set[Cell] = set()
    priority: dict[Cell, float] = {}

    for _, row in gdf.iterrows():
        cell = _cell_from_geometry(row.geometry, bounds, grid_size)
        if cell is None:
            continue

        amenity = str(row.get("amenity", "")).lower()
        landuse = str(row.get("landuse", "")).lower()
        natural = str(row.get("natural", "")).lower()

        if _is_present(row.get("building")):
            obstacles.add(cell)
        if landuse in {"industrial", "railway"} or natural in {"water", "wetland"} or _is_present(row.get("water")):
            restricted.add(cell)
        if amenity in {"hospital", "clinic", "fire_station", "police"}:
            bases.add(cell)

        score = _priority_for_row(row)
        if score > 0:
            priority[cell] = max(priority.get(cell, 0.0), score)

    # Monitoring targets and response bases must remain reachable in the grid abstraction.
    mission = set(priority)
    obstacles -= mission | bases
    restricted -= mission | bases

    if not bases:
        bases = {
            _nearest_walkable((0, 0), grid_size, obstacles, restricted),
            _nearest_walkable((grid_size - 1, grid_size - 1), grid_size, obstacles, restricted),
        }
    if not mission:
        synth = synthetic_city_layers(grid_size, seed, place_name)
        mission = synth["mission_zones"]
        priority = synth["priority_cells"]

    return {
        "obstacles": obstacles,
        "restricted_zones": restricted,
        "mission_zones": mission,
        "base_stations": bases,
        "priority_cells": priority,
        "metadata": {
            "source": "openstreetmap",
            "place_name": place_name,
            "center_lat": float(center[0]),
            "center_lon": float(center[1]),
            "radius_m": int(radius_m),
            "grid_size": int(grid_size),
            "feature_count": int(len(gdf)),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "attribution": OSM_ATTRIBUTION,
            "license_url": OSM_LICENSE_URL,
        },
    }


def load_real_city_layers(
    place_name: str,
    grid_size: int,
    seed: int = 42,
    *,
    radius_m: int = 1500,
    cache_dir: str | Path = "data/cache",
    allow_network: bool = True,
    force_refresh: bool = False,
) -> LayerDict:
    """Load a small real-city OSM snapshot or a clearly labelled synthetic fallback.

    Cached snapshots guarantee that every strategy is evaluated on the exact same
    city cells and avoid repeated calls to Nominatim/Overpass during benchmarks.
    """
    cache_path = _cache_path(cache_dir, place_name, grid_size, radius_m)
    if cache_path.exists() and not force_refresh:
        layers = load_layer_cache(cache_path)
        layers["metadata"]["cache_path"] = str(cache_path)
        layers["metadata"]["cache_hit"] = True
        return layers

    if allow_network:
        try:
            layers = _download_osm_layers(place_name, grid_size, radius_m, seed)
            layers["metadata"]["cache_path"] = str(cache_path)
            layers["metadata"]["cache_hit"] = False
            save_layer_cache(cache_path, layers)
            return layers
        except Exception as exc:  # network/geocoder/Overpass failures must not break CI
            fallback = synthetic_city_layers(grid_size, seed, place_name)
            fallback["metadata"]["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            fallback["metadata"]["cache_path"] = str(cache_path)
            return fallback

    fallback = synthetic_city_layers(grid_size, seed, place_name)
    fallback["metadata"]["fallback_reason"] = "network access disabled"
    fallback["metadata"]["cache_path"] = str(cache_path)
    return fallback
