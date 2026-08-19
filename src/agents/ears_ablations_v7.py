"""Publication-only EARS mechanism ablations.

These policies never participate in checkpoint selection.  They load the frozen
EARS checkpoint and disable exactly one mechanism (or all but one trigger) for
post-selection scientific analysis.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.agents.ears_v6 import EARSPolicy
from src.environment.city_twin import Cell, CityTwinEnvironment


class EARSAblationPolicy(EARSPolicy):
    """Frozen EARS checkpoint with a named post-selection mechanism ablation."""

    def __init__(self, *args, ablation: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        allowed = {
            "stagnation_only",
            "revisit_only",
            "congestion_only",
            "no_energy_battery",
        }
        if ablation not in allowed:
            raise ValueError(f"Unknown EARS ablation {ablation!r}; expected {sorted(allowed)}")
        self.ablation = ablation
        labels = {
            "stagnation_only": "EARS-Ablation-StagnationOnly",
            "revisit_only": "EARS-Ablation-RevisitOnly",
            "congestion_only": "EARS-Ablation-CongestionOnly",
            "no_energy_battery": "EARS-Ablation-NoEnergyBatteryAware",
        }
        self.name = labels[ablation]

    def _event_reason(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        positions: Dict[int, Cell],
    ) -> str | None:
        if self.ablation == "no_energy_battery":
            return super()._event_reason(env, agent_id, positions)
        if env.steps < int(self.config.min_steps_before_reallocation):
            return None
        if env.steps < self._cooldown_until.get(agent_id, -1):
            return None
        current = env.agents[agent_id].position
        history = self._history_for(agent_id)
        history.append(current)

        if self.ablation == "stagnation_only":
            if len(history) >= max(4, int(0.75 * history.maxlen)):
                unique_ratio = len(set(history)) / max(1, len(history))
                if unique_ratio <= float(self.config.stagnation_unique_ratio):
                    return "stagnation"
            return None
        if self.ablation == "revisit_only":
            if self._local_revisit_ratio(env, current) >= float(self.config.local_revisit_trigger):
                return "revisit"
            return None
        if self.ablation == "congestion_only":
            if self._congestion(current, agent_id, positions) >= float(self.config.congestion_trigger):
                return "congestion"
            return None
        return None

    def _select_relocation_goals(
        self,
        env: CityTwinEnvironment,
        agents: list[int],
        positions: Dict[int, Cell],
    ) -> None:
        if self.ablation != "no_energy_battery":
            return super()._select_relocation_goals(env, agents, positions)
        if not agents:
            return
        utility = self._global_utility(env)
        finite_cells = [tuple(map(int, cell)) for cell in np.argwhere(np.isfinite(utility))]
        chosen: list[Cell] = []
        for aid in sorted(agents):
            current = env.agents[aid].position

            def score(cell: Cell) -> float:
                distance = float(self._manhattan(current, cell))
                congestion = self._congestion(cell, aid, positions)
                spacing_penalty = 0.0
                if chosen:
                    nearest_chosen = min(self._manhattan(cell, goal) for goal in chosen)
                    spacing_penalty = max(0.0, float(self.config.goal_spacing - nearest_chosen))
                # Deliberately remove both movement-energy cost and the battery
                # return-feasibility rejection while preserving distance,
                # congestion, novelty and observable frontier/evidence terms.
                return float(
                    utility[cell]
                    - self.config.global_distance_penalty * distance / max(1.0, env.grid_size)
                    - self.config.global_congestion_penalty * congestion
                    - 0.20 * spacing_penalty
                )

            goal = max(finite_cells, key=score) if finite_cells else current
            self._relocation_goals[aid] = goal
            self._relocation_until[aid] = int(env.steps + max(1, self.config.relocation_duration))
            self._cooldown_until[aid] = int(
                self._relocation_until[aid] + max(1, self.config.relocation_cooldown)
            )
            chosen.append(goal)

    def diagnostics(self) -> dict:
        data = super().diagnostics()
        data["ears_ablation"] = self.ablation
        return data
