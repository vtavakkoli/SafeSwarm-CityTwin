"""Evaluation metrics for safety-feasibility experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeMetrics:
    mission_success_rate: float
    number_of_safety_violations: int
    collision_count: int
    restricted_zone_entries: int
    battery_failures: int
    coverage_ratio: float
    runtime_overhead: float

    def to_dict(self, strategy: str, episode: int) -> dict:
        return {
            "strategy": strategy,
            "episode": episode,
            "mission_success_rate": self.mission_success_rate,
            "number_of_safety_violations": self.number_of_safety_violations,
            "collision_count": self.collision_count,
            "restricted_zone_entries": self.restricted_zone_entries,
            "battery_failures": self.battery_failures,
            "coverage_ratio": self.coverage_ratio,
            "runtime_overhead": self.runtime_overhead,
        }
