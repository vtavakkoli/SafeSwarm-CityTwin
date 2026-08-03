"""SafeSwarm policy with weighted target allocation and runtime assurance."""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.environment.city_twin import CityTwinEnvironment
from src.safety.runtime_monitor import RuntimeSafetyMonitor

Cell = Tuple[int, int]


class SafeSwarmAgentPolicy:
    name = "SafeSwarmAgent"

    def __init__(self, monitor: RuntimeSafetyMonitor | None = None) -> None:
        self.monitor = monitor or RuntimeSafetyMonitor()

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

    def _allocate_targets(self, env: CityTwinEnvironment) -> Dict[int, Cell]:
        unassigned: List[Cell] = list(env.remaining_missions() or env.mission_zones)
        assignments: Dict[int, Cell] = {}
        for aid, state in sorted(env.agents.items(), key=lambda item: item[1].battery_level):
            if not unassigned:
                assignments[aid] = min(
                    env.base_stations,
                    key=lambda b: abs(b[0] - state.position[0]) + abs(b[1] - state.position[1]),
                )
                continue
            target = max(
                unassigned,
                key=lambda c: env.priority_cells.get(c, 0.5)
                / (1.0 + abs(c[0] - state.position[0]) + abs(c[1] - state.position[1])),
            )
            assignments[aid] = target
            unassigned.remove(target)
        return assignments

    def act(self, env: CityTwinEnvironment) -> Dict[int, str]:
        assignments = self._allocate_targets(env)
        proposed: Dict[int, str] = {}
        for aid, state in env.agents.items():
            target = assignments[aid]
            if state.battery_level < 25:
                target = min(
                    env.base_stations,
                    key=lambda b: abs(b[0] - state.position[0]) + abs(b[1] - state.position[1]),
                )
            proposed[aid] = self._direction(state.position, target)
        return self.monitor.filter_actions(env, proposed)
