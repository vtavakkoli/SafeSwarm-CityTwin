"""Obstacle and zone generation utilities."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Set, Tuple

import numpy as np

Cell = Tuple[int, int]


def _clip_cell(x: int, y: int, grid_size: int) -> Optional[Cell]:
    if 0 <= x < grid_size and 0 <= y < grid_size:
        return (x, y)
    return None


def synthetic_city_layers(grid_size: int, seed: int = 42) -> Dict[str, Set[Cell]]:
    """Create deterministic synthetic city layers as fallback."""
    rng = np.random.default_rng(seed)
    obstacles: Set[Cell] = set()
    restricted: Set[Cell] = set()
    mission: Set[Cell] = set()

    for _ in range(max(4, grid_size // 10)):
        x0 = int(rng.integers(0, grid_size - 3))
        y0 = int(rng.integers(0, grid_size - 3))
        w = int(rng.integers(2, 5))
        h = int(rng.integers(2, 5))
        for x in range(x0, min(grid_size, x0 + w)):
            for y in range(y0, min(grid_size, y0 + h)):
                obstacles.add((x, y))

    for _ in range(max(2, grid_size // 20)):
        y = int(rng.integers(0, grid_size))
        for x in range(grid_size):
            if rng.random() < 0.2:
                restricted.add((x, y))

    for _ in range(max(8, grid_size // 4)):
        cell = (int(rng.integers(0, grid_size)), int(rng.integers(0, grid_size)))
        if cell not in obstacles and cell not in restricted:
            mission.add(cell)

    base_stations: Set[Cell] = {(0, 0), (grid_size - 1, grid_size - 1)}
    return {
        "obstacles": obstacles,
        "restricted_zones": restricted,
        "mission_zones": mission,
        "base_stations": base_stations,
    }


@lru_cache(maxsize=32)
def _cached_real_city_layers(place_name: str, grid_size: int, seed: int) -> Tuple[frozenset, frozenset, frozenset, frozenset]:
    try:
        import osmnx as ox

        tags = {
            "building": True,
            "landuse": ["industrial"],
            "leisure": ["park"],
            "natural": ["water"],
            "amenity": ["hospital", "fire_station", "police"],
            "shop": True,
            "office": True,
        }
        gdf = ox.features_from_place(place_name, tags)
        if gdf.empty:
            synth = synthetic_city_layers(grid_size=grid_size, seed=seed)
            return tuple(frozenset(synth[k]) for k in ["obstacles", "restricted_zones", "mission_zones", "base_stations"])

        minx, miny, maxx, maxy = gdf.total_bounds
        dx = max(maxx - minx, 1e-9)
        dy = max(maxy - miny, 1e-9)

        def geom_to_cell(cx: float, cy: float) -> Optional[Cell]:
            gx = int((cx - minx) / dx * (grid_size - 1))
            gy = int((cy - miny) / dy * (grid_size - 1))
            return _clip_cell(gx, gy, grid_size)

        obstacles: Set[Cell] = set()
        restricted: Set[Cell] = set()
        mission: Set[Cell] = set()
        bases: Set[Cell] = {(0, 0), (grid_size - 1, grid_size - 1)}

        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            c = geom.centroid
            cell = geom_to_cell(c.x, c.y)
            if cell is None:
                continue

            if row.get("building") is not None:
                obstacles.add(cell)
            if row.get("natural") == "water" or row.get("landuse") == "industrial" or row.get("leisure") == "park":
                restricted.add(cell)
            if row.get("shop") is not None or row.get("office") is not None:
                mission.add(cell)
            if row.get("amenity") in {"hospital", "fire_station", "police"}:
                bases.add(cell)

        if not mission:
            mission = set(list(obstacles)[: min(10, len(obstacles))])

        return frozenset(obstacles), frozenset(restricted), frozenset(mission), frozenset(bases)
    except Exception:
        synth = synthetic_city_layers(grid_size=grid_size, seed=seed)
        return tuple(frozenset(synth[k]) for k in ["obstacles", "restricted_zones", "mission_zones", "base_stations"])


def load_real_city_layers(place_name: str, grid_size: int, seed: int = 42) -> Dict[str, Set[Cell]]:
    obstacles, restricted, mission, bases = _cached_real_city_layers(place_name, grid_size, seed)
    return {
        "obstacles": set(obstacles),
        "restricted_zones": set(restricted),
        "mission_zones": set(mission),
        "base_stations": set(bases),
    }
