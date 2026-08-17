"""Observable-only wrappers for legacy MARL execution baselines.

The legacy classes are retained for reproducibility, but primary experiments use
these wrappers so fixed baselines and learned policies receive the same
partially-observed information contract.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.agents.marl_baselines import (
    GRPOPolicy,
    HAPPOPolicy,
    IPPOPolicy,
    MADDPGPolicy,
    MAPPOPolicy,
    MATPolicy,
    QMIXPolicy,
)
from src.agents.observable_utils import observable_priority, observable_target_distance
from src.environment.city_twin import Cell, CityTwinEnvironment


class _ObservableFeaturesMixin:
    def _features(
        self,
        env: CityTwinEnvironment,
        cell: Cell,
        current: Cell,
        positions: Dict[int, Cell],
        agent_id: int,
    ) -> dict[str, float]:
        features = super()._features(env, cell, current, positions, agent_id)
        features["priority"] = observable_priority(env, cell)
        features["target_distance"] = float(
            observable_target_distance(env, cell) / max(1.0, 2.0 * env.grid_size)
        )
        return features


class ObservableIPPOPolicy(_ObservableFeaturesMixin, IPPOPolicy):
    name = "IPPO-Safe"


class ObservableMAPPOPolicy(_ObservableFeaturesMixin, MAPPOPolicy):
    name = "MAPPO-Safe"


class ObservableQMIXPolicy(_ObservableFeaturesMixin, QMIXPolicy):
    name = "QMIX-Safe"


class ObservableMADDPGPolicy(_ObservableFeaturesMixin, MADDPGPolicy):
    name = "MADDPG-Safe"


class ObservableHAPPOPolicy(_ObservableFeaturesMixin, HAPPOPolicy):
    name = "HAPPO-Safe"


class ObservableMATPolicy(_ObservableFeaturesMixin, MATPolicy):
    name = "MAT-Safe"


class ObservableGRPOPolicy(_ObservableFeaturesMixin, GRPOPolicy):
    name = "GRPO-Safe"

    def _behavior_scores(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        positions: Dict[int, Cell],
    ) -> np.ndarray:
        current = env.agents[agent_id].position
        neighbors = env.get_neighbors(current)
        priority_max = max(observable_priority(env, cell) for cell in neighbors)
        uncertainty_max = max(float(env.uncertainty_map[cell]) for cell in neighbors)
        pheromone_max = max(float(env.pheromone_map[cell]) for cell in neighbors)
        visit_min = min(float(env.visit_counts[cell]) for cell in neighbors)
        local_priority = observable_priority(env, current)
        local_uncertainty = float(env.uncertainty_map[current])
        team_spread = self._spread(current, agent_id, positions, env.grid_size)
        battery = float(env.agents[agent_id].battery_level / 100.0)
        scores = np.asarray(
            [
                1.20 * uncertainty_max + 0.20 * team_spread,
                1.45 * priority_max,
                1.00 * pheromone_max + 0.25 * priority_max,
                0.90 * local_uncertainty + 0.40 * local_priority,
                1.10 * local_priority + 0.70 * local_uncertainty,
                1.00 / (1.0 + visit_min) + 0.30 * team_spread,
                1.20 * max(0.0, 0.35 - battery),
            ],
            dtype=float,
        )
        active = scores[: self.group_size].copy()
        active -= float(np.mean(active))
        std = float(np.std(active))
        if std > 1e-8:
            active /= std
        scores[: self.group_size] = active
        return scores
