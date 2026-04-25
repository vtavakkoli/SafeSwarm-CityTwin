"""Greedy baseline agent without safety filter."""

from __future__ import annotations

from typing import Dict, Tuple

from src.environment.city_twin import CityTwinEnvironment


class GreedyAgentPolicy:
    def _direction(self, current: Tuple[int, int], target: Tuple[int, int]) -> str:
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
        mission_targets = list(env.mission_zones)
        if not mission_targets:
            return {aid: "STAY" for aid in env.agents.keys()}

        for aid, state in env.agents.items():
            target = min(
                mission_targets,
                key=lambda c: abs(c[0] - state.position[0]) + abs(c[1] - state.position[1]),
            )
            actions[aid] = self._direction(state.position, target)
        return actions
