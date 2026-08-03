"""Priority-greedy city monitoring baseline without a safety filter."""

from __future__ import annotations

from typing import Dict, Tuple

from src.environment.city_twin import CityTwinEnvironment

Cell = Tuple[int, int]


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
        remaining = env.remaining_missions() or env.mission_zones
        if not remaining:
            return {aid: "STAY" for aid in env.agents}

        actions: Dict[int, str] = {}
        for aid, state in env.agents.items():
            target = max(
                remaining,
                key=lambda c: env.priority_cells.get(c, 0.5)
                / (1.0 + abs(c[0] - state.position[0]) + abs(c[1] - state.position[1])),
            )
            actions[aid] = self._direction(state.position, target)
        return actions
