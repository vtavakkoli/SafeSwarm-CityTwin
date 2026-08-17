"""Observable greedy city-monitoring baseline without a safety filter."""

from __future__ import annotations

from typing import Dict

from src.agents.observable_utils import nearest_observable_goal
from src.environment.city_twin import Cell, CityTwinEnvironment


class GreedyAgentPolicy:
    name = "GreedyAgent"

    @staticmethod
    def _direction(current: Cell, target: Cell) -> str:
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        if abs(dx) >= abs(dy):
            if dx > 0:
                return "RIGHT"
            if dx < 0:
                return "LEFT"
        if dy > 0:
            return "DOWN"
        if dy < 0:
            return "UP"
        return "STAY"

    def act(self, env: CityTwinEnvironment) -> Dict[int, str]:
        actions: Dict[int, str] = {}
        for aid, state in env.agents.items():
            if state.done:
                actions[aid] = "STAY"
                continue
            target = nearest_observable_goal(env, state.position)
            if target is None:
                candidates = env.get_neighbors(state.position)
                target = max(
                    candidates,
                    key=lambda cell: (
                        float(env.uncertainty_map[cell])
                        / (1.0 + float(env.visit_counts[cell]))
                    ),
                )
            actions[aid] = self._direction(state.position, target)
        return actions
