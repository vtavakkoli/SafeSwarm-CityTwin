"""Metrics and ranking utilities for city-algorithm comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class EpisodeMetrics:
    mission_success_rate: float
    weighted_target_discovery: float
    coverage_ratio: float
    number_of_safety_violations: int
    safety_interventions: int
    actual_safety_incidents: int
    collision_count: int
    restricted_zone_entries: int
    battery_failures: int
    energy_consumption: float
    redundant_coverage: float
    communication_efficiency: float
    distance_travelled: int
    runtime_seconds: float
    runtime_overhead: float
    blocked_moves: int

    def to_dict(self, strategy: str, episode: int, **metadata: object) -> dict:
        return {"strategy": strategy, "episode": episode, **metadata, **asdict(self)}


def episode_operational_score(row: pd.Series | dict) -> float:
    """Hardware-independent operational score in [0, 1]."""
    get = row.get
    safety_rate = 1.0 - min(1.0, float(get("actual_safety_incidents", 0)) / max(1.0, float(get("steps", 1)) * float(get("agents", 1))))
    energy_efficiency = 1.0 - min(1.0, float(get("energy_consumption", 0.0)) / max(1.0, 100.0 * float(get("agents", 1))))
    coordination = 1.0 - min(1.0, float(get("redundant_coverage", 0.0)))
    score = (
        0.35 * float(get("weighted_target_discovery", 0.0))
        + 0.20 * float(get("coverage_ratio", 0.0))
        + 0.20 * safety_rate
        + 0.10 * energy_efficiency
        + 0.10 * coordination
        + 0.05 * float(get("communication_efficiency", 0.0))
    )
    return float(np.clip(score, 0.0, 1.0))


def rank_algorithms(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-city and overall rankings without mixing episode counts."""
    if records.empty:
        raise ValueError("records must not be empty")

    frame = records.copy()
    frame["operational_score"] = frame.apply(episode_operational_score, axis=1)
    numeric_columns = [
        "operational_score",
        "mission_success_rate",
        "weighted_target_discovery",
        "coverage_ratio",
        "number_of_safety_violations",
        "safety_interventions",
        "actual_safety_incidents",
        "collision_count",
        "restricted_zone_entries",
        "battery_failures",
        "energy_consumption",
        "redundant_coverage",
        "communication_efficiency",
        "distance_travelled",
        "runtime_seconds",
        "runtime_overhead",
        "blocked_moves",
    ]
    city_summary = (
        frame.groupby(["city", "data_source", "strategy"], as_index=False)[numeric_columns]
        .mean(numeric_only=True)
        .sort_values(["city", "operational_score"], ascending=[True, False])
    )
    city_summary["city_rank"] = city_summary.groupby("city")["operational_score"].rank(
        method="dense", ascending=False
    ).astype(int)

    overall = (
        city_summary.groupby("strategy", as_index=False)[numeric_columns]
        .mean(numeric_only=True)
        .sort_values("operational_score", ascending=False)
        .reset_index(drop=True)
    )
    overall.insert(0, "rank", np.arange(1, len(overall) + 1))
    overall["cities_evaluated"] = city_summary.groupby("strategy")["city"].nunique().reindex(overall["strategy"]).to_numpy()
    return city_summary, overall
