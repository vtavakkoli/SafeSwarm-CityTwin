"""Geographic split and starting-zone helpers for city experiments."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

Cell = tuple[int, int]

_ZONE_FRACTIONS: dict[str, tuple[float, float]] = {
    "north_west": (0.18, 0.18),
    "north_east": (0.82, 0.18),
    "south_west": (0.18, 0.82),
    "south_east": (0.82, 0.82),
    "center": (0.50, 0.50),
    "west": (0.12, 0.50),
    "east": (0.88, 0.50),
    "north": (0.50, 0.12),
    "south": (0.50, 0.88),
}


def load_protocol(path: str | Path = "configs/real_city_protocol.json") -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "cities" not in payload or "start_zones" not in payload:
        raise ValueError("Protocol must define cities and start_zones")
    return payload


def select_cities(protocol: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    cities = [dict(city) for city in protocol["cities"] if city.get("split") == split]
    if not cities:
        raise ValueError(f"No cities configured for split={split!r}")
    return cities


def start_zones_for_split(protocol: Mapping[str, Any], split: str) -> list[str]:
    zones = list(protocol.get("start_zones", {}).get(split, []))
    if not zones:
        raise ValueError(f"No start zones configured for split={split!r}")
    unknown = sorted(set(zones) - set(_ZONE_FRACTIONS))
    if unknown:
        raise ValueError(f"Unknown geographic start zones: {unknown}")
    return zones


def _anchor(grid_size: int, zone: str) -> Cell:
    if zone not in _ZONE_FRACTIONS:
        raise ValueError(f"Unknown start zone: {zone}")
    fx, fy = _ZONE_FRACTIONS[zone]
    limit = max(0, grid_size - 1)
    return int(round(fx * limit)), int(round(fy * limit))


def apply_start_zone(
    layers: Mapping[str, Any],
    grid_size: int,
    zone: str,
    *,
    n_bases: int = 4,
) -> dict[str, Any]:
    """Return a layer copy whose base stations lie in the requested geographic zone."""
    result = copy.deepcopy(dict(layers))
    obstacles = {tuple(cell) for cell in result.get("obstacles", set())}
    restricted = {tuple(cell) for cell in result.get("restricted_zones", set())}
    blocked = obstacles | restricted
    anchor = _anchor(grid_size, zone)

    traversable = [
        (x, y)
        for x in range(grid_size)
        for y in range(grid_size)
        if (x, y) not in blocked
    ]
    if not traversable:
        raise ValueError("City layer contains no traversable cells")

    ordered = sorted(
        traversable,
        key=lambda cell: (
            abs(cell[0] - anchor[0]) + abs(cell[1] - anchor[1]),
            (cell[0] - anchor[0]) ** 2 + (cell[1] - anchor[1]) ** 2,
            cell,
        ),
    )

    selected: list[Cell] = []
    min_spacing = max(1, grid_size // 12)
    for cell in ordered:
        if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) >= min_spacing for other in selected):
            selected.append(cell)
            if len(selected) >= max(1, n_bases):
                break
    if not selected:
        selected = [ordered[0]]

    result["base_stations"] = set(selected)
    metadata = dict(result.get("metadata", {}))
    metadata["start_zone"] = zone
    metadata["start_zone_anchor"] = list(anchor)
    metadata["spawn_base_count"] = len(selected)
    result["metadata"] = metadata
    return result


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, bool]:
    train_cities = {city["name"] for city in select_cities(protocol, "train")}
    validation_cities = {city["name"] for city in select_cities(protocol, "validation")}
    test_cities = {city["name"] for city in select_cities(protocol, "test")}
    train_zones = set(start_zones_for_split(protocol, "train"))
    test_zones = set(start_zones_for_split(protocol, "test"))
    return {
        "city_splits_disjoint": not (
            train_cities & validation_cities
            or train_cities & test_cities
            or validation_cities & test_cities
        ),
        "test_start_zones_unseen_during_training": not (train_zones & test_zones),
    }
