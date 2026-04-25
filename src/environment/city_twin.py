"""Smart-city digital twin grid environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

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
    done: bool = False


class CityTwinEnvironment:
    def __init__(
        self,
        grid_size: int,
        n_agents: int,
        seed: int = 42,
        place_name: str = "San Francisco, California, USA",
    ) -> None:
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.seed = seed
        self.place_name = place_name
        self.rng = np.random.default_rng(seed)

        layers = load_real_city_layers(place_name=place_name, grid_size=grid_size, seed=seed)
        self.obstacles: Set[Cell] = layers["obstacles"]
        self.restricted_zones: Set[Cell] = layers["restricted_zones"]
        self.mission_zones: Set[Cell] = layers["mission_zones"]
        self.base_stations: Set[Cell] = layers["base_stations"]

        self.visited: Set[Cell] = set()
        self.agents: Dict[int, AgentState] = {}
        self.steps = 0
        self.max_steps = grid_size * 4
        self.reset()

    def reset(self) -> Dict[int, AgentState]:
        self.steps = 0
        self.visited = set()
        self.agents = {}
        base_list = list(self.base_stations) or [(0, 0)]

        for agent_id in range(self.n_agents):
            position = base_list[agent_id % len(base_list)]
            state = AgentState(agent_id=agent_id, position=position, trajectory_history=[position])
            self.agents[agent_id] = state
            self.visited.add(position)
        return self.agents

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def next_position(self, pos: Cell, action: str) -> Cell:
        dx, dy = MOVE_ACTIONS.get(action, (0, 0))
        nx, ny = pos[0] + dx, pos[1] + dy
        return (nx, ny)

    def valid_free_cell(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and cell not in self.obstacles

    def update_communications(self, comm_dropout_prob: float = 0.03) -> None:
        for agent in self.agents.values():
            if self.rng.random() < comm_dropout_prob:
                agent.communication_status = False
                agent.comm_loss_steps += 1
            else:
                agent.communication_status = True
                agent.comm_loss_steps = 0

    def step(self, actions: Dict[int, str]) -> Dict[str, float]:
        self.steps += 1
        self.update_communications()

        for aid, action in actions.items():
            state = self.agents[aid]
            if state.done:
                continue
            candidate = self.next_position(state.position, action)
            if self.valid_free_cell(candidate):
                state.position = candidate
            state.battery_level = max(0.0, state.battery_level - (1.5 if action != "STAY" else 0.5))
            state.trajectory_history.append(state.position)
            self.visited.add(state.position)

            if state.position in self.mission_zones:
                state.current_task = "mission"
            if state.battery_level <= 0:
                state.done = True

        coverage_ratio = len(self.visited) / (self.grid_size * self.grid_size)
        done = self.steps >= self.max_steps or all(a.done for a in self.agents.values())
        return {"coverage_ratio": coverage_ratio, "done": float(done)}

    def nearest_base_distance(self, cell: Cell) -> int:
        return min(abs(cell[0] - bx) + abs(cell[1] - by) for bx, by in self.base_stations)

    def all_candidate_actions(self) -> List[str]:
        return list(MOVE_ACTIONS.keys())
