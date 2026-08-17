"""Observable-only guidance helpers for fair partially observable benchmarks.

Primary benchmark policies must not read ``mission_zones``, ``priority_map`` or
``priority_cells`` to steer movement before those locations have been sensed.
These helpers derive search guidance only from information available at runtime:
observations, uncertainty, pheromones and visit history.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src.environment.city_twin import Cell, CityTwinEnvironment


def observable_priority(env: CityTwinEnvironment, cell: Cell) -> float:
    """Return confidence-weighted sensed priority evidence for one cell."""

    confidence = float(np.clip(1.0 - env.uncertainty_map[cell], 0.0, 1.0))
    value = float(env.observation_map[cell]) * confidence
    if cell in env.discovered_missions:
        # A resolved target should not remain an attractive oracle-like beacon.
        value *= 0.15
    return value


def frontier_map(env: CityTwinEnvironment) -> np.ndarray:
    """Build a normalized exploration-frontier field from sensed coverage."""

    observed = (env.uncertainty_map < 0.99).astype(float)
    unseen = 1.0 - observed
    padded = np.pad(observed, 1, mode="constant", constant_values=0.0)
    neighbor_observed = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    ) / 4.0
    novelty = 1.0 / (1.0 + env.visit_counts.astype(float))
    field = neighbor_observed * unseen * novelty
    for cell in env.obstacles | env.restricted_zones:
        field[cell] = 0.0
    return np.clip(field, 0.0, 1.0)


def observable_search_utility(env: CityTwinEnvironment, cell: Cell) -> float:
    """Search utility usable by fixed baselines without hidden target labels."""

    frontier = frontier_map(env)
    novelty = 1.0 / (1.0 + float(env.visit_counts[cell]))
    return float(
        1.30 * observable_priority(env, cell)
        + 1.10 * float(env.uncertainty_map[cell]) * novelty
        + 0.85 * float(frontier[cell])
        + 0.35 * float(np.clip(env.pheromone_map[cell], 0.0, 2.0)) * novelty
    )


def observable_guidance_cells(env: CityTwinEnvironment, *, limit: int = 32) -> list[Cell]:
    """Return promising *observable* cells for long-range guidance.

    This is intentionally not a replacement for the hidden mission list. The
    candidates are current frontiers and sensed evidence only.
    """

    frontier = frontier_map(env)
    scores: list[tuple[float, Cell]] = []
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            cell = (x, y)
            if cell in env.obstacles or cell in env.restricted_zones:
                continue
            score = (
                1.4 * observable_priority(env, cell)
                + float(frontier[cell])
                + 0.2 * float(env.pheromone_map[cell]) / (1.0 + float(env.visit_counts[cell]))
            )
            if score > 1e-9:
                scores.append((float(score), cell))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [cell for _, cell in scores[: max(1, int(limit))]]


def observable_target_distance(env: CityTwinEnvironment, cell: Cell) -> float:
    candidates = observable_guidance_cells(env)
    if not candidates:
        return 0.0
    return min(float(np.hypot(tx - cell[0], ty - cell[1])) for tx, ty in candidates)


def nearest_observable_goal(env: CityTwinEnvironment, current: Cell) -> Cell | None:
    candidates = observable_guidance_cells(env)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda cell: observable_search_utility(env, cell)
        / (1.0 + abs(cell[0] - current[0]) + abs(cell[1] - current[1])),
    )
