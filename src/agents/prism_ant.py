"""PRISM-Ant: global PRISM allocation fused with AntSwarm local search.

The v4 real-city result exposed complementary strengths:

* AntSwarmSafe had excellent target discovery and very low redundant coverage;
* PRISM covered more of the map and provided explicit global region allocation.

PRISM-Ant preserves PRISM's observable-only probability memory and disjoint
region/pattern goals, but chooses each safe local move with an adaptive fusion
of PRISM progress and Ant-style novelty/pheromone/anti-revisit scoring.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.agents.observable_utils import observable_priority
from src.agents.prism_pattern import PRISMPolicy
from src.environment.city_twin import Cell, CityTwinEnvironment


@dataclass
class PRISMAntConfig:
    ant_blend: float = 0.62
    ant_priority_weight: float = 1.45
    ant_uncertainty_weight: float = 0.55
    ant_pheromone_weight: float = 0.42
    ant_novelty_weight: float = 1.05
    ant_revisit_penalty: float = 0.82
    ant_congestion_penalty: float = 0.72
    prism_progress_weight: float = 2.15
    prism_probability_weight: float = 0.35


class PRISMAntPolicy(PRISMPolicy):
    """Adaptive global-region + local-ant fusion policy."""

    name = "PRISM-Ant-Safe"

    def __init__(
        self,
        *args: Any,
        hybrid_config: PRISMAntConfig | None = None,
        **kwargs: Any,
    ) -> None:
        model_path = kwargs.get("model_path")
        super().__init__(*args, **kwargs)
        self.hybrid_config = hybrid_config or PRISMAntConfig()
        if model_path:
            path = Path(model_path)
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                raw = dict(payload.get("hybrid_config", {}))
                if raw:
                    defaults = asdict(PRISMAntConfig())
                    defaults.update({k: float(v) for k, v in raw.items() if k in defaults})
                    self.hybrid_config = PRISMAntConfig(**defaults)
        self.name = kwargs.get("strategy_name") or "PRISM-Ant-Safe"

    def _adaptive_ant_blend(self, env: CityTwinEnvironment, cell: Cell) -> float:
        """Use more Ant behavior around observed evidence; more PRISM in unknown space."""
        base = float(self.hybrid_config.ant_blend)
        evidence = observable_priority(env, cell)
        unexplored = float(np.mean(env.uncertainty_map >= 0.99))
        return float(np.clip(base + 0.18 * evidence - 0.12 * unexplored, 0.25, 0.88))

    def _choose_safe_action(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        goal: Cell,
        planned_positions: Dict[int, Cell],
        positions: Dict[int, Cell],
    ) -> str:
        if env.agents[agent_id].done:
            return "STAY"
        safe = self.monitor.safe_actions(env, agent_id, planned_positions, count_masked=True)
        if not safe:
            return "STAY"
        assert self.search_probability_map is not None
        assert self.search_utility_map is not None

        current = env.agents[agent_id].position
        distances = self._distance_field(env, goal)
        current_distance = distances.get(current, env.grid_size * env.grid_size)
        h = self.hybrid_config

        def score(action: str) -> tuple[float, float, str]:
            cell = env.next_position(current, action)
            distance = distances.get(cell, env.grid_size * env.grid_size)
            progress = float(current_distance - distance)
            probability = float(self.search_probability_map[cell]) * env.traversable_cell_count
            prism_utility = float(
                np.nan_to_num(self.search_utility_map[cell], nan=-10.0, neginf=-10.0)
            )

            visits = float(env.visit_counts[cell])
            novelty = 1.0 / (1.0 + visits)
            priority = observable_priority(env, cell)
            uncertainty = float(env.uncertainty_map[cell])
            pheromone = float(np.clip(env.pheromone_map[cell], 0.0, 2.0))
            congestion = self._congestion(cell, agent_id, positions)

            prism_score = (
                h.prism_progress_weight * progress
                + h.prism_probability_weight * probability
                + 0.16 * prism_utility
                + 0.26 * novelty
                - self.config.congestion_penalty * congestion
            )
            ant_score = (
                h.ant_priority_weight * priority
                + h.ant_uncertainty_weight * uncertainty * novelty
                + h.ant_pheromone_weight * pheromone * novelty
                + h.ant_novelty_weight * novelty
                - h.ant_revisit_penalty * visits
                - h.ant_congestion_penalty * congestion
            )
            blend = self._adaptive_ant_blend(env, cell)
            combined = (
                (1.0 - blend) * prism_score
                + blend * ant_score
                - 0.05 * float(action == "STAY")
            )
            return combined, -float(distance), action

        return max(safe, key=score)

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "safeswarm-prism-ant-v1",
            "algorithm": "PRISM-Ant: PRISM global allocation + AntSwarm local search",
            "pattern_mode": self.pattern_mode,
            "config": self.config_dict(),
            "hybrid_config": asdict(self.hybrid_config),
            "metadata": dict(metadata or {}),
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def diagnostics(self) -> dict[str, Any]:
        data = super().diagnostics()
        data.update(
            {
                "name": self.name,
                "prism_ant_config": asdict(self.hybrid_config),
                "prism_ant_blend": float(self.hybrid_config.ant_blend),
            }
        )
        return data
