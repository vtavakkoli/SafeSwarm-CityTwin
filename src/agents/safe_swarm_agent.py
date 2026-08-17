"""SafeSwarm policy with observable frontier allocation and runtime assurance."""

from __future__ import annotations

from typing import Dict, List

from src.agents.observable_utils import observable_guidance_cells, observable_search_utility
from src.environment.city_twin import Cell, CityTwinEnvironment
from src.safety.runtime_monitor import RuntimeSafetyMonitor


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
        # Allocate only observable frontiers/evidence. Ground-truth mission
        # coordinates are evaluation labels and must never steer a policy.
        unassigned: List[Cell] = observable_guidance_cells(env, limit=max(16, 4 * len(env.agents)))
        assignments: Dict[int, Cell] = {}
        for aid, state in sorted(env.agents.items(), key=lambda item: item[1].battery_level):
            if state.done:
                assignments[aid] = state.position
                continue
            if state.battery_level < 30:
                assignments[aid] = min(
                    env.base_stations,
                    key=lambda b: abs(b[0] - state.position[0]) + abs(b[1] - state.position[1]),
                )
                continue
            if not unassigned:
                candidates = env.get_neighbors(state.position)
                assignments[aid] = max(
                    candidates,
                    key=lambda cell: observable_search_utility(env, cell),
                )
                continue
            target = max(
                unassigned,
                key=lambda cell: observable_search_utility(env, cell)
                / (1.0 + abs(cell[0] - state.position[0]) + abs(cell[1] - state.position[1])),
            )
            assignments[aid] = target
            unassigned.remove(target)
        return assignments

    def act(self, env: CityTwinEnvironment) -> Dict[int, str]:
        assignments = self._allocate_targets(env)
        proposed: Dict[int, str] = {}
        for aid, state in env.agents.items():
            if state.done:
                proposed[aid] = "STAY"
                continue
            target = assignments[aid]
            proposed[aid] = self._direction(state.position, target)
        return self.monitor.filter_actions(env, proposed)
