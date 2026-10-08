"""EARS v8: route-aware, utility-gated coordination (experimental).

The v6/v7 publication controller and its checkpoints are NOT modified. This
extension uses only observable city state and is a *new* method that must be
validated on untouched geography before publication claims are made.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Any, Dict

import numpy as np

from src.agents.ears_v6 import EARSPolicy
from src.environment.city_twin import Cell, CityTwinEnvironment


@dataclass(frozen=True)
class TopologyGateConfig:
    """Unselected v8 development defaults; tune on train/validation only."""

    enable_benefit_gate: bool = True
    min_net_gain: float = 0.12
    min_goal_distance: int = 3
    energy_reserve_margin: float = 12.0
    idle_cost_per_step: float = 0.10
    max_route_detour: float = 6.0
    require_current_communication: bool = True

    def __post_init__(self) -> None:
        if self.min_goal_distance < 1 or self.energy_reserve_margin < 0:
            raise ValueError("Invalid goal distance or reserve margin")
        if self.max_route_detour < 1 or self.idle_cost_per_step < 0:
            raise ValueError("Invalid detour or idle cost")


class EARSTopoSafePolicy(EARSPolicy):
    """v8 relocation layer built on the unchanged v6 Ant/controller logic.

    Computes graph distances through traversable cells, checks route-plus-return
    energy feasibility, and optionally rejects events with low estimated gain.
    The benefit gate compares observable *proxy utility*, not predicted actual
    hidden target discovery. It is not a learned or ground-truth oracle.
    """

    name = "EARS-TopoSafe"

    def __init__(
        self,
        *args: Any,
        gate_config: TopologyGateConfig | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("strategy_name", self.name)
        super().__init__(*args, **kwargs)
        self.gate_config = gate_config or TopologyGateConfig()
        self.reset_topology_stats()

    def reset_topology_stats(self) -> None:
        self.accepted_relocations = 0
        self.rejected_benefit = 0
        self.rejected_unreachable = 0
        self.rejected_energy = 0
        self.rejected_communication = 0
        self.candidate_evaluations = 0
        self.relocation_planning_seconds = 0.0
        self.route_detours: list[float] = []
        self.route_lengths: list[int] = []
        self._v8_cache: dict[tuple[int, Cell], dict[Cell, int]] = {}

    def reset_episode_state(self) -> None:
        super().reset_episode_state()
        self.reset_topology_stats()

    def _route_distances(self, env: CityTwinEnvironment, source: Cell) -> dict[Cell, int]:
        key = (int(env.steps), source)
        distances = self._v8_cache.get(key)
        if distances is None:
            distances = self._distance_field(env, source)
            self._v8_cache[key] = distances
        return distances

    def _route_to_base(self, env: CityTwinEnvironment) -> dict[Cell, int]:
        """Shortest feasible base route; combines obstacle-aware BFS fields."""
        result: dict[Cell, int] = {}
        for base in env.base_stations:
            for cell, steps in self._route_distances(env, base).items():
                result[cell] = min(steps, result.get(cell, steps))
        return result

    def _local_proxy(
        self, env: CityTwinEnvironment, current: Cell, utility: np.ndarray
    ) -> float:
        """Best feasible next-cell global-utility proxy for Ant continuation."""
        candidates = [current]
        blocked = env.obstacles | env.restricted_zones
        for other in ((current[0] + 1, current[1]),
                      (current[0] - 1, current[1]),
                      (current[0], current[1] + 1),
                      (current[0], current[1] - 1)):
            if env.in_bounds(other) and other not in blocked:
                candidates.append(other)
        return max(float(utility[cell]) for cell in candidates if isfinite(float(utility[cell])))

    def _select_relocation_goals(
        self,
        env: CityTwinEnvironment,
        agents: list[int],
        positions: Dict[int, Cell],
    ) -> None:
        if not agents:
            return
        started = perf_counter()
        utility = self._global_utility(env)
        goal_candidates = [tuple(map(int, cell)) for cell in np.argwhere(np.isfinite(utility))]
        to_base = self._route_to_base(env)
        chosen: list[Cell] = []
        self._v8_cache = {key: distances for key, distances in self._v8_cache.items()
                          if key[0] == int(env.steps)}
        for aid in sorted(agents, key=lambda i: (env.agents[i].battery_level, i)):
            state = env.agents[aid]
            if self.gate_config.require_current_communication and not state.communication_status:
                self.rejected_communication += 1
                continue

            current = state.position
            from_current = self._route_distances(env, current)
            local_value = self._local_proxy(env, current, utility)
            best: tuple[float, Cell, int, float] | None = None
            saw_reachable = False
            saw_energy_feasible = False
            for cell in goal_candidates:
                self.candidate_evaluations += 1
                distance = from_current.get(cell)
                return_distance = to_base.get(cell)
                if distance is None or return_distance is None:
                    continue
                if distance < self.gate_config.min_goal_distance:
                    continue
                direct = max(1, self._manhattan(current, cell))
                detour = float(distance / direct)
                if detour > self.gate_config.max_route_detour:
                    continue
                saw_reachable = True

                # Conservative energy budget: full outbound and return route.
                steps_total = distance + return_distance
                estimated_energy = (
                    steps_total * float(env.move_energy_cost)
                    + steps_total * self.gate_config.idle_cost_per_step
                    + self.gate_config.energy_reserve_margin
                )
                if estimated_energy > float(state.battery_level):
                    continue
                saw_energy_feasible = True
                congestion = self._congestion(cell, aid, positions)
                spacing = 0.0
                if chosen:
                    nearest = min(self._manhattan(cell, goal) for goal in chosen)
                    spacing = max(0.0, float(self.config.goal_spacing - nearest))
                # Preserve v7's coefficients while using true navigable travel cost.
                score = (
                    float(utility[cell])
                    - self.config.global_distance_penalty * distance / max(1.0, env.grid_size)
                    - self.config.global_energy_penalty * (
                        distance * float(env.move_energy_cost)
                        + distance * self.gate_config.idle_cost_per_step
                    ) / 100.0
                    - self.config.global_congestion_penalty * congestion
                    - 0.20 * spacing
                )
                if best is None or (score, tuple(-v for v in cell)) > (
                    best[0], tuple(-v for v in best[1])
                ):
                    best = (score, cell, distance, detour)

            if best is None:
                if not saw_reachable:
                    self.rejected_unreachable += 1
                elif not saw_energy_feasible:
                    self.rejected_energy += 1
                continue

            net_gain = best[0] - local_value
            if self.gate_config.enable_benefit_gate and net_gain < self.gate_config.min_net_gain:
                self.rejected_benefit += 1
                continue
            score, goal, distance, detour = best
            self._relocation_goals[aid] = goal
            self._relocation_until[aid] = int(env.steps + max(1, self.config.relocation_duration))
            self._cooldown_until[aid] = (
                self._relocation_until[aid] + max(1, self.config.relocation_cooldown)
            )
            self.accepted_relocations += 1
            self.route_lengths.append(distance)
            self.route_detours.append(detour)
            chosen.append(goal)
        self.relocation_planning_seconds += perf_counter() - started

    def diagnostics(self) -> dict[str, Any]:
        data = super().diagnostics()
        data.update({
            "ears_v8": True,
            "v8_benefit_gate_enabled": self.gate_config.enable_benefit_gate,
            "v8_accepted_relocations": self.accepted_relocations,
            "v8_rejected_benefit": self.rejected_benefit,
            "v8_rejected_unreachable": self.rejected_unreachable,
            "v8_rejected_energy": self.rejected_energy,
            "v8_rejected_communication": self.rejected_communication,
            "v8_candidate_evaluations": self.candidate_evaluations,
            "v8_planning_seconds": self.relocation_planning_seconds,
            "v8_mean_route_length": float(np.mean(self.route_lengths)) if self.route_lengths else 0.0,
            "v8_mean_detour": float(np.mean(self.route_detours)) if self.route_detours else 0.0,
            "v8_gate_config": vars(self.gate_config).copy(),
        })
        return data
