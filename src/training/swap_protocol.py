"""SWAP: Seeded World Alternate Protocol for anti-overfitting evaluation.

Changing the environment RNG seed alone does *not* create a new real OSM test
map because cached city layers are keyed by place/grid/radius. SWAP therefore
creates deterministic seed-indexed mission views from one immutable physical
OSM snapshot. Obstacles, restricted geography and city provenance stay fixed;
only the held-out monitoring-target subset changes.

SWAP is evaluation-only. Its seed-indexed target views must never be used to
select training weights, PRISM patterns, or hybrid parameters.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping

import numpy as np

Cell = tuple[int, int]


def _effective_seed(metadata: Mapping[str, Any], dataset_seed: int) -> int:
    place = str(metadata.get("place_name", "unknown-city"))
    digest = hashlib.sha256(f"{place}|SWAP|{int(dataset_seed)}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32 - 1)


def _signature(cells: set[Cell]) -> str:
    """Hash the actual target set, not the requested seed.

    This is deliberately seed-independent: two seeds that accidentally produce
    the same target set must have the same signature so CI can detect that the
    dataset did not really change.
    """

    body = ";".join(f"{x},{y}" for x, y in sorted(cells))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def seeded_mission_view(
    layers: Mapping[str, Any],
    dataset_seed: int,
    *,
    target_fraction: float = 0.72,
    min_targets: int = 6,
) -> dict[str, Any]:
    """Return a deterministic alternate target view of a cached city layer.

    Sampling is stratified by target priority so a new seed changes the mission
    set without collapsing the task into only high- or low-priority targets.
    The physical map is not modified.
    """

    result = copy.deepcopy(dict(layers))
    priority = {
        tuple(cell): float(score)
        for cell, score in dict(result.get("priority_cells", {})).items()
    }
    original = set(result.get("mission_zones", set()))
    candidates = sorted(original & set(priority)) or sorted(original)
    if not candidates:
        raise ValueError("SWAP requires at least one mission cell")

    fraction = float(np.clip(target_fraction, 0.20, 0.95))
    effective = _effective_seed(result.get("metadata", {}), int(dataset_seed))
    rng = np.random.default_rng(effective)

    buckets: dict[float, list[Cell]] = {}
    for cell in candidates:
        bucket = round(float(priority.get(cell, 0.5)), 2)
        buckets.setdefault(bucket, []).append(cell)

    selected: set[Cell] = set()
    for bucket in sorted(buckets):
        cells = list(buckets[bucket])
        order = rng.permutation(len(cells))
        take = max(1, int(round(len(cells) * fraction)))
        selected.update(cells[int(index)] for index in order[:take])

    required = min(len(candidates), max(1, int(min_targets)))
    if len(selected) < required:
        remainder = [cell for cell in candidates if cell not in selected]
        if remainder:
            order = rng.permutation(len(remainder))
            selected.update(remainder[int(i)] for i in order[: required - len(selected)])

    result["mission_zones"] = set(selected)
    result["priority_cells"] = {
        cell: float(priority.get(cell, 0.5)) for cell in selected
    }
    metadata = dict(result.get("metadata", {}))
    metadata.update(
        {
            "dataset_variant": "SWAP",
            "swap_seed": int(dataset_seed),
            "swap_effective_seed": int(effective),
            "swap_target_fraction": fraction,
            "swap_original_target_count": int(len(original)),
            "swap_target_count": int(len(selected)),
            "swap_dataset_changed": bool(set(selected) != original),
            "swap_signature": _signature(selected),
            "swap_selection": "priority-stratified deterministic target subset",
        }
    )
    result["metadata"] = metadata
    return result


def swap_seed_series(base_seed: int = 2042, count: int = 3) -> list[int]:
    """Return well-separated deterministic dataset seeds."""

    count = max(1, int(count))
    return [int(base_seed) + 1009 * index for index in range(count)]
