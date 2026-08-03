"""Safety-wrapped bio-inspired policies adapted from BioSwarm-Urban-Monitoring."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.environment.city_twin import Cell, CityTwinEnvironment
from src.safety.runtime_monitor import RuntimeSafetyMonitor


def _congestion(cell: Cell, agent_id: int, positions: Dict[int, Cell]) -> float:
    total = 0.0
    for other_id, other_pos in positions.items():
        if other_id == agent_id:
            continue
        distance = float(np.hypot(cell[0] - other_pos[0], cell[1] - other_pos[1]))
        total += 2.0 if distance < 1e-9 else 1.0 / distance
    return total


def _target_distance(env: CityTwinEnvironment, cell: Cell) -> float:
    targets = env.remaining_missions() or env.mission_zones
    if not targets:
        return 0.0
    return min(float(np.hypot(tx - cell[0], ty - cell[1])) for tx, ty in targets)


class _SafetyWrappedPolicy:
    name = "BioSwarmPolicy"

    def __init__(self, monitor: RuntimeSafetyMonitor | None = None) -> None:
        self.monitor = monitor or RuntimeSafetyMonitor()

    def _propose(self, env: CityTwinEnvironment, agent_id: int, positions: Dict[int, Cell]) -> Cell:
        raise NotImplementedError

    def act(self, env: CityTwinEnvironment) -> Dict[int, str]:
        positions = env.get_positions()
        proposed: Dict[int, str] = {}
        for aid, state in env.agents.items():
            candidate = self._propose(env, aid, positions)
            proposed[aid] = env.cell_to_action(state.position, candidate)
        return self.monitor.filter_actions(env, proposed)


class AntSwarmPolicy(_SafetyWrappedPolicy):
    """Ant-style novelty/pheromone search with anti-clustering."""

    name = "AntSwarmSafe"

    def __init__(
        self,
        monitor: RuntimeSafetyMonitor | None = None,
        explore_weight: float = 1.0,
        revisit_penalty: float = 0.55,
        risk_weight: float = 1.4,
        cluster_penalty: float = 0.85,
    ) -> None:
        super().__init__(monitor)
        self.explore_weight = explore_weight
        self.revisit_penalty = revisit_penalty
        self.risk_weight = risk_weight
        self.cluster_penalty = cluster_penalty

    def _propose(self, env: CityTwinEnvironment, agent_id: int, positions: Dict[int, Cell]) -> Cell:
        current = env.agents[agent_id].position
        best, best_score = current, float("-inf")
        for cell in env.get_neighbors(current):
            visits = float(env.visit_counts[cell])
            novelty = 1.0 / (1.0 + visits)
            priority = float(env.priority_map[cell])
            pheromone = float(env.pheromone_map[cell])
            score = (
                self.explore_weight * novelty
                - self.revisit_penalty * visits
                + self.risk_weight * priority
                + 0.45 * pheromone
                - self.cluster_penalty * _congestion(cell, agent_id, positions)
                - 0.04 * _target_distance(env, cell)
                - 8.0 * float(cell in env.restricted_zones)
            )
            if score > best_score:
                best, best_score = cell, score
        return best


class BeeSwarmPolicy(_SafetyWrappedPolicy):
    """Bee-style scout/employed/onlooker policy with adaptive role allocation."""

    name = "BeeSwarmSafe"

    def __init__(self, monitor: RuntimeSafetyMonitor | None = None) -> None:
        super().__init__(monitor)
        self.stagnation_counter: Dict[int, int] = {}

    @staticmethod
    def _role(env: CityTwinEnvironment, agent_id: int) -> str:
        ratio = agent_id / max(1, len(env.agents))
        if ratio < 0.34:
            return "scout"
        if ratio < 0.67:
            return "employed"
        return "onlooker"

    def _propose(self, env: CityTwinEnvironment, agent_id: int, positions: Dict[int, Cell]) -> Cell:
        current = env.agents[agent_id].position
        neighbors = env.get_neighbors(current)
        role = self._role(env, agent_id)
        best, best_score = current, float("-inf")

        for cell in neighbors:
            priority = float(env.priority_map[cell])
            uncertainty = float(env.uncertainty_map[cell])
            novelty = 1.0 / (1.0 + float(env.visit_counts[cell]))
            shared_quality = float(env.shared_quality_map[cell])
            movement_cost = float(cell != current)
            congestion = _congestion(cell, agent_id, positions)

            if role == "scout":
                role_priority = 0.9 * novelty + 0.8 * uncertainty
                alpha, beta, delta, theta, eta, rho = 0.5, 1.3, 1.1, 1.2, 0.4, 0.65
            elif role == "employed":
                role_priority = 1.2 * priority + 0.45 * shared_quality
                alpha, beta, delta, theta, eta, rho = 1.4, 0.3, 0.35, 1.0, 0.6, 0.75
            else:
                role_priority = 1.0 * shared_quality + 0.7 * priority
                alpha, beta, delta, theta, eta, rho = 1.0, 0.55, 0.45, 1.1, 0.6, 1.0

            score = (
                alpha * priority
                + beta * uncertainty
                + delta * novelty
                + theta * role_priority
                - eta * movement_cost
                - rho * congestion
                - 0.04 * _target_distance(env, cell)
                - 8.0 * float(cell in env.restricted_zones)
            )
            if score > best_score:
                best, best_score = cell, score

        if role == "onlooker":
            previous = self.stagnation_counter.get(agent_id, 0)
            if env.priority_map[best] <= env.priority_map[current]:
                self.stagnation_counter[agent_id] = previous + 1
            else:
                self.stagnation_counter[agent_id] = 0
            if self.stagnation_counter[agent_id] >= 4:
                self.stagnation_counter[agent_id] = 0
                best = min(neighbors, key=lambda c: (env.visit_counts[c], _congestion(c, agent_id, positions)))
        return best


class PSOSwarmPolicy(_SafetyWrappedPolicy):
    """Lightweight particle-swarm movement policy with personal/global targets."""

    name = "PSOSwarmSafe"

    def __init__(self, monitor: RuntimeSafetyMonitor | None = None) -> None:
        super().__init__(monitor)
        self.velocity: Dict[int, np.ndarray] = {}
        self.best_pos: Dict[int, Cell] = {}
        self.best_score: Dict[int, float] = {}

    @staticmethod
    def _utility(env: CityTwinEnvironment, cell: Cell) -> float:
        return float(
            1.5 * env.priority_map[cell]
            + 0.65 * env.uncertainty_map[cell]
            + 0.35 / (1.0 + env.visit_counts[cell])
            + 0.2 * env.pheromone_map[cell]
        )

    def _propose(self, env: CityTwinEnvironment, agent_id: int, positions: Dict[int, Cell]) -> Cell:
        current = env.agents[agent_id].position
        neighbors = env.get_neighbors(current)
        if agent_id not in self.velocity:
            self.velocity[agent_id] = np.zeros(2, dtype=float)
            self.best_pos[agent_id] = current
            self.best_score[agent_id] = self._utility(env, current)

        local_best = max(neighbors, key=lambda c: self._utility(env, c) - 0.7 * _congestion(c, agent_id, positions))
        current_score = self._utility(env, current)
        if current_score > self.best_score[agent_id]:
            self.best_score[agent_id] = current_score
            self.best_pos[agent_id] = current

        personal = np.asarray(self.best_pos[agent_id], dtype=float)
        global_best = np.asarray(local_best, dtype=float)
        now = np.asarray(current, dtype=float)
        self.velocity[agent_id] = (
            0.45 * self.velocity[agent_id]
            + 0.95 * (personal - now)
            + 1.15 * (global_best - now)
        )
        target = now + np.clip(self.velocity[agent_id], -1.0, 1.0)
        candidate = (
            int(np.clip(round(target[0]), 0, env.grid_size - 1)),
            int(np.clip(round(target[1]), 0, env.grid_size - 1)),
        )
        if candidate not in neighbors or candidate in env.restricted_zones:
            candidate = local_best
        return candidate


class UncertaintyAwareBeeAntSwarmPolicy(_SafetyWrappedPolicy):
    """Hybrid bee-ant policy with uncertainty, pheromone, target and safety awareness."""

    name = "UA-HBAS-Safe"

    @staticmethod
    def _role(env: CityTwinEnvironment, agent_id: int) -> str:
        global_uncertainty = float(np.mean(env.uncertainty_map))
        priority_evidence = float(np.mean(env.priority_map > 0.45))
        scout_share = min(0.45, 0.20 + 0.45 * global_uncertainty)
        employed_share = min(0.60, 0.30 + 1.20 * priority_evidence)
        ratio = agent_id / max(1, len(env.agents))
        if ratio < scout_share:
            return "scout"
        if ratio < scout_share + employed_share:
            return "employed"
        return "onlooker"

    def _propose(self, env: CityTwinEnvironment, agent_id: int, positions: Dict[int, Cell]) -> Cell:
        current = env.agents[agent_id].position
        role = self._role(env, agent_id)
        best, best_score = current, float("-inf")

        for cell in env.get_neighbors(current):
            priority = float(env.priority_map[cell])
            uncertainty = float(env.uncertainty_map[cell])
            pheromone = float(env.pheromone_map[cell])
            novelty = 1.0 / (1.0 + float(env.visit_counts[cell]))
            shared_quality = float(env.shared_quality_map[cell])
            movement_cost = float(cell != current)
            congestion = _congestion(cell, agent_id, positions)
            target_dist = _target_distance(env, cell)

            if role == "scout":
                alpha, beta, gamma, delta, theta, kappa, eta, rho = 1.0, 1.5, 0.25, 1.7, 1.0, 0.45, 0.4, 0.45
                role_priority = 0.95 * novelty + 0.95 * uncertainty
            elif role == "employed":
                alpha, beta, gamma, delta, theta, kappa, eta, rho = 2.3, 0.45, 1.5, 1.0, 1.2, 0.75, 0.45, 0.35
                role_priority = 1.25 * priority + 0.85 * pheromone + 0.25 * shared_quality
            else:
                alpha, beta, gamma, delta, theta, kappa, eta, rho = 1.5, 0.7, 1.15, 1.0, 1.2, 1.25, 0.4, 0.5
                role_priority = 1.1 * shared_quality + 0.9 * priority + 0.35 * pheromone

            score = (
                alpha * priority
                + beta * uncertainty
                + gamma * pheromone
                + delta * novelty
                + theta * role_priority
                + kappa * shared_quality
                - eta * movement_cost
                - rho * congestion
                - 0.10 * target_dist
                - 10.0 * float(cell in env.restricted_zones)
            )
            if score > best_score:
                best, best_score = cell, score
        return best
