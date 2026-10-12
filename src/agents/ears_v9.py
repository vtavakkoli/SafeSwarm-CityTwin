"""Experimental EARS v9: marginal sensing gain with persistent route reservations.

No hidden target positions/weights, rollout rewards, or city identities are policy
inputs. v6/v7 and v8 remain frozen. This is a candidate, not a claimed improvement.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np

from src.agents.ears_v6 import EARSPolicy
from src.environment.city_twin import Cell, CityTwinEnvironment


@dataclass(frozen=True)
class InformationRoutingConfig:
    information_weight: float = 2.0
    coverage_weight: float = 0.35
    travel_discount: float = 0.15
    min_route_gain: float = 0.08
    retry_steps: int = 5
    max_route_steps: int = 32
    route_slack: int = 4
    cache_fields: int = 96
    reserve_margin: float = 18.0
    enable_relocation: bool = True

    def __post_init__(self) -> None:
        for name in (
            "information_weight",
            "coverage_weight",
            "travel_discount",
            "min_route_gain",
            "reserve_margin",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("retry_steps", "max_route_steps", "route_slack", "cache_fields"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


class EARSInformationPolicy(EARSPolicy):
    """Coordinate newly sensed area, rather than repeatedly visiting sensed cells.

    Uses the existing shared-map/communication simulation contract and the same
    runtime monitor. Neither this controller nor that monitor is a safety proof.
    """

    name = "EARS-Info-v9"

    def __init__(
        self, *args: Any, routing_config: InformationRoutingConfig | None = None, **kwargs: Any
    ) -> None:
        kwargs.setdefault("strategy_name", self.name)
        super().__init__(*args, **kwargs)
        self.routing_config = routing_config or InformationRoutingConfig()
        self._reset_routing()

    def _reset_routing(self) -> None:
        self._topology_key = None
        self._graph: dict[Cell, tuple[Cell, ...]] = {}
        self._route_cache: OrderedDict[Cell, dict[Cell, int]] = OrderedDict()
        self._base_distances: dict[Cell, int] = {}
        self._return_base: dict[Cell, Cell] = {}
        self._retry_until: dict[int, int] = {}
        self._history_step = -1
        self.accepted_relocations = 0
        self.rejected_relocations = 0
        self.cancelled_relocations = 0
        self.bfs_builds = 0
        self.route_cache_hits = 0
        self.candidate_evaluations = 0
        self.planning_seconds = 0.0
        self._gain = np.empty((0, 0))

    def reset_episode_state(self) -> None:
        super().reset_episode_state()
        self._reset_routing()

    def _ensure_topology(self, env: CityTwinEnvironment) -> None:
        # Cache by topology, not timestep. Invalidate on edits and environment reuse.
        key = (
            id(env),
            env.grid_size,
            frozenset(env.obstacles),
            frozenset(env.restricted_zones),
            frozenset(env.base_stations),
        )
        if key == self._topology_key:
            return
        self._topology_key = key
        blocked = env.obstacles | env.restricted_zones
        free = {(x, y) for x in range(env.grid_size) for y in range(env.grid_size)} - blocked
        self._graph = {
            (x, y): tuple(c for c in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)) if c in free)
            for x, y in sorted(free)
        }
        self._route_cache.clear()
        self._relocation_goals.clear()
        self._relocation_until.clear()
        self._base_distances = {}
        self._return_base = {}
        queue: deque[Cell] = deque()
        for base in sorted(env.base_stations & free):
            self._base_distances[base] = 0
            self._return_base[base] = base
            queue.append(base)
        while queue:
            cell = queue.popleft()
            for other in self._graph[cell]:
                if other not in self._base_distances:
                    self._base_distances[other] = self._base_distances[cell] + 1
                    self._return_base[other] = self._return_base[cell]
                    queue.append(other)

    def _distance_field(self, env: CityTwinEnvironment, goal: Cell) -> dict[Cell, int]:
        self._ensure_topology(env)
        if goal in self._route_cache:
            self.route_cache_hits += 1
            self._route_cache.move_to_end(goal)
            return self._route_cache[goal]
        distances: dict[Cell, int] = {}
        if goal in self._graph:
            distances[goal] = 0
            queue = deque([goal])
            while queue:
                cell = queue.popleft()
                for other in self._graph[cell]:
                    if other not in distances:
                        distances[other] = distances[cell] + 1
                        queue.append(other)
        self.bfs_builds += 1
        self._route_cache[goal] = distances
        if len(self._route_cache) > self.routing_config.cache_fields:
            self._route_cache.popitem(last=False)
        return distances

    @staticmethod
    def _footprint(env: CityTwinEnvironment, cell: Cell) -> list[Cell]:
        r = env.sensor_radius
        return [
            (x, y)
            for x in range(max(0, cell[0] - r), min(env.grid_size, cell[0] + r + 1))
            for y in range(max(0, cell[1] - r), min(env.grid_size, cell[1] + r + 1))
            if abs(x - cell[0]) + abs(y - cell[1]) <= r and (x, y) not in env.obstacles
        ]

    def _information_gain(self, env: CityTwinEnvironment, reserved: set[Cell] | None = None) -> np.ndarray:
        """Normalized count of newly sensed cells; exactly matches sensor geometry."""
        unseen = (env.uncertainty_map >= 0.99).astype(float)
        for cell in env.obstacles | (reserved or set()):
            unseen[cell] = 0.0
        r = env.sensor_radius
        padded = np.pad(unseen, r)
        gain = np.zeros_like(unseen)
        size = env.grid_size
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if abs(dx) + abs(dy) <= r:
                    gain += padded[r + dx : r + dx + size, r + dy : r + dy + size]
        return gain / max(1, 1 + 2 * r * (r + 1))

    @staticmethod
    def _congestion(cell: Cell, agent_id: int, positions: dict[int, Cell]) -> float:
        # Nearby crowding, not a sum that grows with distant robots/team size.
        distances = sorted(
            abs(cell[0] - p[0]) + abs(cell[1] - p[1]) for aid, p in positions.items() if aid != agent_id
        )
        return sum(1.0 / max(0.5, d) for d in distances[:3] if d <= 4)

    def _battery_return_needed(self, env: CityTwinEnvironment, agent_id: int) -> bool:
        distance = self._base_distances.get(env.agents[agent_id].position, float("inf"))
        return env.agents[agent_id].battery_level <= (
            (distance + 1) * env.move_energy_cost + self.routing_config.reserve_margin
        )

    def _event_reason(
        self, env: CityTwinEnvironment, agent_id: int, positions: dict[int, Cell]
    ) -> str | None:
        if not self.routing_config.enable_relocation:
            return None
        if env.steps < self.config.min_steps_before_reallocation:
            return None
        if env.steps < max(self._retry_until.get(agent_id, -1), self._cooldown_until.get(agent_id, -1)):
            return None
        state = env.agents[agent_id]
        if not state.communication_status:
            return None
        history = self._history_for(agent_id)
        if (
            len(history) == history.maxlen
            and len(set(history)) / len(history) <= self.config.stagnation_unique_ratio
        ):
            return "stagnation"
        options = (state.position,) + self._graph.get(state.position, ())
        if max(self._gain[c] for c in options) <= 0:
            return "revisit"
        return None

    def _select_relocation_goals(
        self, env: CityTwinEnvironment, agents: list[int], positions: dict[int, Cell]
    ) -> None:
        if not agents:
            return
        started = perf_counter()
        c = self.routing_config
        reserved: set[Cell] = set()
        for aid, goal in self._relocation_goals.items():
            if aid not in agents and self._relocation_until.get(aid, -1) >= env.steps:
                reserved.update(self._footprint(env, goal))
        coords = np.asarray(list(self._graph), dtype=int)
        for aid in sorted(agents, key=lambda i: (env.agents[i].battery_level, i)):
            state = env.agents[aid]
            # Backoff applies to accepted AND rejected events, including no feasible route.
            self._retry_until[aid] = env.steps + c.retry_steps
            gain = self._information_gain(env, reserved)
            distances = self._distance_field(env, state.position)
            route = np.array([distances.get(tuple(cell), np.inf) for cell in coords])
            home = np.array([self._base_distances.get(tuple(cell), np.inf) for cell in coords])
            remaining = env.max_steps - env.steps
            feasible = (
                (route >= 2)
                & (route <= min(c.max_route_steps, remaining))
                & ((route + home) * env.move_energy_cost + c.reserve_margin <= state.battery_level)
            )
            values = gain[coords[:, 0], coords[:, 1]]
            novelty = 1.0 / (1.0 + env.visit_counts[coords[:, 0], coords[:, 1]])
            scores = (c.information_weight * values + c.coverage_weight * novelty) / (
                1 + c.travel_discount * np.where(np.isfinite(route), route, 0)
            )
            scores[~feasible | (values <= 0)] = -np.inf
            self.candidate_evaluations += int(np.count_nonzero(feasible))
            best = int(np.argmax(scores))
            local = max(
                (c.information_weight * gain[cell] + c.coverage_weight / (1 + env.visit_counts[cell]))
                for cell in (state.position,) + self._graph.get(state.position, ())
            )
            if not np.isfinite(scores[best]) or scores[best] < local + c.min_route_gain:
                self.rejected_relocations += 1
                continue
            goal = tuple(map(int, coords[best]))
            duration = min(remaining, int(route[best]) + c.route_slack)
            self._relocation_goals[aid] = goal
            self._relocation_until[aid] = env.steps + duration - 1
            # Small cooldown is measured from arrival/cancellation, not an obsolete deadline.
            self._cooldown_until[aid] = env.steps + c.retry_steps
            reserved.update(self._footprint(env, goal))
            self.accepted_relocations += 1
        self.planning_seconds += perf_counter() - started

    def _goal_action(
        self, env: CityTwinEnvironment, agent_id: int, goal: Cell, planned: dict[int, Cell]
    ) -> str:
        safe = self.monitor.safe_actions(env, agent_id, planned, count_masked=True)
        if not safe:
            return "STAY"
        current = env.agents[agent_id].position
        distances = self._distance_field(env, goal)
        # Graph progress is primary: accumulated visits cannot reverse the route.
        return max(
            safe,
            key=lambda action: (
                -distances.get(env.next_position(current, action), float("inf")),
                self._gain[env.next_position(current, action)],
                -env.visit_counts[env.next_position(current, action)],
                action,
            ),
        )

    def _information_action(
        self,
        env: CityTwinEnvironment,
        aid: int,
        planned: dict[int, Cell],
        positions: dict[int, Cell],
        reserved: set[Cell],
    ) -> str:
        safe = self.monitor.safe_actions(env, aid, planned, count_masked=True)
        if not safe:
            return "STAY"
        c = self.routing_config
        current = env.agents[aid].position
        footprint_size = max(1, 1 + 2 * env.sensor_radius * (env.sensor_radius + 1))

        def score(action: str) -> tuple[float, str]:
            cell = env.next_position(current, action)
            gain = (
                sum(env.uncertainty_map[p] >= 0.99 and p not in reserved for p in self._footprint(env, cell))
                / footprint_size
            )
            visits = float(env.visit_counts[cell])
            return (
                c.information_weight * gain
                + c.coverage_weight / (1 + visits)
                - 0.10 * min(visits, 5)
                - 0.15 * self._congestion(cell, aid, positions)
                - 0.02 * (action == "STAY"),
                action,
            )

        return max(safe, key=score)

    def act(self, env: CityTwinEnvironment) -> dict[int, str]:
        self._ensure_topology(env)
        self._gain = self._information_gain(env)
        positions = env.get_positions()
        if self._history_step != env.steps:
            for aid, state in env.agents.items():
                self._history_for(aid).append(state.position)
            self._history_step = env.steps
        triggers = []
        for aid, state in env.agents.items():
            goal = self._relocation_goals.get(aid)
            if goal is not None and (
                state.done
                or state.position == goal
                or env.steps > self._relocation_until[aid]
                or self._gain[goal] <= 0
                or self._battery_return_needed(env, aid)
            ):
                del self._relocation_goals[aid]
                self._cooldown_until[aid] = env.steps + self.routing_config.retry_steps
                self.cancelled_relocations += 1
            if state.done or self._battery_return_needed(env, aid) or aid in self._relocation_goals:
                continue
            reason = self._event_reason(env, aid, positions)
            if reason:
                triggers.append(aid)
                self.event_triggers += 1
                self.stagnation_triggers += reason == "stagnation"
                self.revisit_triggers += reason == "revisit"
        self._select_relocation_goals(env, triggers, positions)
        actions: dict[int, str] = {}
        planned: dict[int, Cell] = {}
        reserved: set[Cell] = set()
        for aid in sorted(env.agents):
            state = env.agents[aid]
            if state.done:
                action = "STAY"
            elif self._battery_return_needed(env, aid):
                goal = self._return_base.get(state.position, state.position)
                action = self._goal_action(env, aid, goal, planned)
                self.return_steps += 1
            elif aid in self._relocation_goals:
                action = self._goal_action(env, aid, self._relocation_goals[aid], planned)
                self.relocation_steps += 1
            else:
                action = self._information_action(env, aid, planned, positions, reserved)
                self.ant_steps += 1  # inherited counter means local-option steps for v9
            actions[aid] = action
            planned[aid] = env.next_position(state.position, action)
            if not state.done:
                reserved.update(self._footprint(env, planned[aid]))
        return actions

    def diagnostics(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "v9_config": asdict(self.routing_config),
            "v9_accepted_relocations": self.accepted_relocations,
            "v9_rejected_relocations": self.rejected_relocations,
            "v9_finished_or_cancelled_relocations": self.cancelled_relocations,
            "v9_bfs_builds": self.bfs_builds,
            "v9_route_cache_hits": self.route_cache_hits,
            "v9_candidate_evaluations": self.candidate_evaluations,
            "v9_planning_seconds": self.planning_seconds,
            "v9_local_steps": self.ant_steps,
            "v9_return_steps": self.return_steps,
            "v9_relocation_steps": self.relocation_steps,
        }
