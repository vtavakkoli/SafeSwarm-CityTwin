"""Smart-city digital twin grid environment for fair multi-agent benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np

from src.environment.obstacles import load_real_city_layers

Cell = Tuple[int, int]

MOVE_ACTIONS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "STAY": (0, 0),
}


@dataclass
class AgentState:
    agent_id: int
    position: Cell
    battery_level: float = 100.0
    current_task: str = "explore"
    communication_status: bool = True
    comm_loss_steps: int = 0
    trajectory_history: List[Cell] = field(default_factory=list)
    distance_travelled: int = 0
    done: bool = False


class CityTwinEnvironment:
    """Grid abstraction backed by one immutable city-layer snapshot per episode."""

    def __init__(
        self,
        grid_size: int,
        n_agents: int,
        seed: int = 42,
        place_name: str = "Vienna, Austria",
        *,
        radius_m: int = 1500,
        cache_dir: str | Path = "data/cache",
        allow_network: bool = True,
        force_refresh: bool = False,
        layers: Optional[Mapping[str, Any]] = None,
        max_steps: Optional[int] = None,
        communication_dropout_prob: float = 0.03,
        sensor_radius: int = 1,
    ) -> None:
        if grid_size < 8:
            raise ValueError("grid_size must be at least 8")
        if n_agents < 1:
            raise ValueError("n_agents must be positive")

        self.grid_size = int(grid_size)
        self.n_agents = int(n_agents)
        self.seed = int(seed)
        self.place_name = place_name
        self.radius_m = int(radius_m)
        self.communication_dropout_prob = float(communication_dropout_prob)
        self.sensor_radius = int(sensor_radius)
        self.rng = np.random.default_rng(seed)

        source_layers = dict(layers) if layers is not None else load_real_city_layers(
            place_name=place_name,
            grid_size=grid_size,
            seed=seed,
            radius_m=radius_m,
            cache_dir=cache_dir,
            allow_network=allow_network,
            force_refresh=force_refresh,
        )
        self.obstacles: Set[Cell] = set(source_layers["obstacles"])
        self.restricted_zones: Set[Cell] = set(source_layers["restricted_zones"])
        self.mission_zones: Set[Cell] = set(source_layers["mission_zones"])
        self.base_stations: Set[Cell] = set(source_layers["base_stations"])
        self.priority_cells: Dict[Cell, float] = {
            tuple(cell): float(score)
            for cell, score in dict(source_layers.get("priority_cells", {})).items()
        }
        self.data_metadata: Dict[str, Any] = dict(source_layers.get("metadata", {}))

        self.priority_map = np.zeros((grid_size, grid_size), dtype=float)
        for cell in self.mission_zones:
            self.priority_map[cell] = self.priority_cells.get(cell, 0.5)

        self.visited: Set[Cell] = set()
        self.visit_counts = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.observation_map = np.zeros((grid_size, grid_size), dtype=float)
        self.uncertainty_map = np.ones((grid_size, grid_size), dtype=float)
        self.pheromone_map = np.zeros((grid_size, grid_size), dtype=float)
        self.discovered_missions: Set[Cell] = set()
        self.agents: Dict[int, AgentState] = {}
        self.steps = 0
        self.max_steps = int(max_steps or grid_size * 4)
        self.actual_collisions = 0
        self.actual_restricted_entries = 0
        self.blocked_moves = 0
        self.communication_online_samples = 0
        self.communication_total_samples = 0
        self.reset()

    @property
    def data_source(self) -> str:
        return str(self.data_metadata.get("source", "unknown"))

    @property
    def shared_quality_map(self) -> np.ndarray:
        observed_priority = self.observation_map * (1.0 - self.uncertainty_map)
        return np.clip(observed_priority + 0.25 * self.pheromone_map, 0.0, 2.0)

    @property
    def traversable_cell_count(self) -> int:
        blocked = self.obstacles | self.restricted_zones
        return max(1, self.grid_size * self.grid_size - len(blocked))

    def reset(self) -> Dict[int, AgentState]:
        self.steps = 0
        self.visited = set()
        self.visit_counts.fill(0)
        self.observation_map.fill(0.0)
        self.uncertainty_map.fill(1.0)
        self.pheromone_map.fill(0.0)
        self.discovered_missions = set()
        self.actual_collisions = 0
        self.actual_restricted_entries = 0
        self.blocked_moves = 0
        self.communication_online_samples = 0
        self.communication_total_samples = 0
        self.agents = {}

        base_list = sorted(self.base_stations) or [(0, 0)]
        occupied: Set[Cell] = set()
        for agent_id in range(self.n_agents):
            preferred = base_list[agent_id % len(base_list)]
            position = self._nearest_spawn_cell(preferred, occupied)
            occupied.add(position)
            state = AgentState(agent_id=agent_id, position=position, trajectory_history=[position])
            self.agents[agent_id] = state
            self._record_visit(position)
            self._sense_from(position)
        return self.agents

    def _nearest_spawn_cell(self, preferred: Cell, occupied: Set[Cell]) -> Cell:
        candidates = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in self.obstacles
            and (x, y) not in self.restricted_zones
            and (x, y) not in occupied
        ]
        if not candidates:
            return preferred
        return min(
            candidates,
            key=lambda c: (abs(c[0] - preferred[0]) + abs(c[1] - preferred[1]), c[0], c[1]),
        )

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def next_position(self, pos: Cell, action: str) -> Cell:
        dx, dy = MOVE_ACTIONS.get(action, (0, 0))
        return (pos[0] + dx, pos[1] + dy)

    def cell_to_action(self, current: Cell, candidate: Cell) -> str:
        delta = (candidate[0] - current[0], candidate[1] - current[1])
        for action, move in MOVE_ACTIONS.items():
            if move == delta:
                return action
        return "STAY"

    def valid_free_cell(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and cell not in self.obstacles

    def get_neighbors(self, cell: Cell, *, include_stay: bool = True) -> List[Cell]:
        actions = self.all_candidate_actions() if include_stay else [a for a in MOVE_ACTIONS if a != "STAY"]
        result: List[Cell] = []
        for action in actions:
            candidate = self.next_position(cell, action)
            if self.valid_free_cell(candidate):
                result.append(candidate)
        return result or [cell]

    def get_positions(self) -> Dict[int, Cell]:
        return {aid: state.position for aid, state in self.agents.items()}

    def get_uncertainty(self, cell: Cell) -> float:
        return float(self.uncertainty_map[cell])

    def remaining_missions(self) -> Set[Cell]:
        return self.mission_zones - self.discovered_missions

    def update_communications(self) -> None:
        for agent in self.agents.values():
            if self.rng.random() < self.communication_dropout_prob:
                agent.communication_status = False
                agent.comm_loss_steps += 1
            else:
                agent.communication_status = True
                agent.comm_loss_steps = 0
            self.communication_total_samples += 1
            self.communication_online_samples += int(agent.communication_status)

    def _record_visit(self, cell: Cell) -> None:
        self.visited.add(cell)
        self.visit_counts[cell] += 1

    def _sense_from(self, position: Cell) -> None:
        px, py = position
        for x in range(max(0, px - self.sensor_radius), min(self.grid_size, px + self.sensor_radius + 1)):
            for y in range(max(0, py - self.sensor_radius), min(self.grid_size, py + self.sensor_radius + 1)):
                if abs(x - px) + abs(y - py) > self.sensor_radius:
                    continue
                cell = (x, y)
                if cell in self.obstacles:
                    continue
                self.observation_map[cell] = max(self.observation_map[cell], self.priority_map[cell])
                self.uncertainty_map[cell] *= 0.25
                if cell in self.mission_zones:
                    self.discovered_missions.add(cell)

    def step(self, actions: Dict[int, str]) -> Dict[str, float]:
        self.steps += 1
        self.update_communications()
        self.pheromone_map *= 0.96

        proposed: Dict[int, Cell] = {}
        for aid, state in self.agents.items():
            action = actions.get(aid, "STAY")
            proposed[aid] = self.next_position(state.position, action)

        counts: Dict[Cell, int] = {}
        for candidate in proposed.values():
            counts[candidate] = counts.get(candidate, 0) + 1
        self.actual_collisions += sum(count - 1 for count in counts.values() if count > 1)

        for aid, state in self.agents.items():
            if state.done:
                continue
            action = actions.get(aid, "STAY")
            candidate = proposed[aid]
            old_position = state.position
            if self.valid_free_cell(candidate):
                state.position = candidate
            else:
                self.blocked_moves += 1

            moved = state.position != old_position
            if moved:
                state.distance_travelled += 1
            if state.position in self.restricted_zones and state.position != old_position:
                self.actual_restricted_entries += 1

            energy_cost = 1.5 if moved else 0.5
            state.battery_level = max(0.0, state.battery_level - energy_cost)
            state.trajectory_history.append(state.position)
            self._record_visit(state.position)
            self._sense_from(state.position)
            self.pheromone_map[state.position] += 0.15 + self.priority_map[state.position]

            if state.position in self.mission_zones:
                state.current_task = "mission"
            elif state.battery_level < 25:
                state.current_task = "return_to_base"
            else:
                state.current_task = "explore"
            if state.battery_level <= 0:
                state.done = True

        coverage_ratio = len(self.visited - self.restricted_zones) / self.traversable_cell_count
        target_discovery = len(self.discovered_missions) / max(1, len(self.mission_zones))
        done = self.steps >= self.max_steps or all(a.done for a in self.agents.values())
        return {
            "coverage_ratio": float(np.clip(coverage_ratio, 0.0, 1.0)),
            "target_discovery_rate": float(target_discovery),
            "done": float(done),
        }

    def nearest_base_distance(self, cell: Cell) -> int:
        return min(abs(cell[0] - bx) + abs(cell[1] - by) for bx, by in self.base_stations)

    def all_candidate_actions(self) -> List[str]:
        return list(MOVE_ACTIONS.keys())

    def weighted_target_discovery(self) -> float:
        total = sum(self.priority_cells.get(cell, 0.5) for cell in self.mission_zones)
        discovered = sum(self.priority_cells.get(cell, 0.5) for cell in self.discovered_missions)
        return float(discovered / max(total, 1e-12))

    def redundant_coverage(self) -> float:
        visited = self.visit_counts[self.visit_counts > 0]
        if visited.size == 0:
            return 0.0
        return float(np.mean(visited > 1))

    def energy_consumption(self) -> float:
        return float(sum(100.0 - state.battery_level for state in self.agents.values()))

    def communication_efficiency(self) -> float:
        return float(self.communication_online_samples / max(1, self.communication_total_samples))
