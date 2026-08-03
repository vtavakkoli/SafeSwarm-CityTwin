"""Safety-wrapped GRPO and well-known MARL execution baselines.

The policies are lightweight NumPy inference implementations adapted to the
SafeSwarm discrete city-grid interface. They preserve the central decision
bias of each method while sharing the same observation, action, and runtime
safety layer. They are benchmark baselines, not substitutes for full neural
training implementations from the original papers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import json
import numpy as np

from src.environment.city_twin import Cell, CityTwinEnvironment
from src.safety.runtime_monitor import RuntimeSafetyMonitor


@dataclass
class PolicyConfig:
    risk_weight: float = 1.20
    uncertainty_weight: float = 0.80
    pheromone_weight: float = 0.30
    novelty_weight: float = 0.45
    spacing_weight: float = 0.30
    communication_weight: float = 0.20
    energy_weight: float = 0.10
    target_weight: float = 0.05
    temperature: float = 0.85
    epsilon: float = 0.03


def _merge(defaults: dict[str, float], overrides: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(defaults)
    merged.update(overrides)
    return merged


class _SafeMARLPolicy:
    """Common stochastic policy executor with runtime safety filtering."""

    name = "MARL-Safe"

    def __init__(
        self,
        seed: int = 42,
        monitor: RuntimeSafetyMonitor | None = None,
        model_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.monitor = monitor or RuntimeSafetyMonitor()
        self.config = PolicyConfig()
        self.model_path = str(model_path) if model_path else None
        self.checkpoint_metadata: dict[str, Any] = {}

        checkpoint_params: dict[str, Any] = {}
        if model_path:
            path = Path(model_path)
            if path.exists():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    checkpoint_params = dict(payload.get("params", {}))
                    self.checkpoint_metadata = dict(payload.get("metadata", {}))
                except (OSError, ValueError, TypeError):
                    checkpoint_params = {}

        for key, value in {**checkpoint_params, **kwargs}.items():
            if hasattr(self.config, key):
                setattr(self.config, key, float(value))

        self.last_entropy = 0.0
        self.last_confidence = 0.0
        self.last_distribution: dict[str, float] = {}
        self.last_selected_cell: Cell | None = None
        self.action_counts: dict[str, int] = {}

    @staticmethod
    def _congestion(cell: Cell, agent_id: int, positions: Dict[int, Cell]) -> float:
        total = 0.0
        for other_id, other_pos in positions.items():
            if other_id == agent_id:
                continue
            distance = float(np.hypot(cell[0] - other_pos[0], cell[1] - other_pos[1]))
            total += 1.0 / (1.0 + distance)
        return total

    @staticmethod
    def _spread(cell: Cell, agent_id: int, positions: Dict[int, Cell], grid_size: int) -> float:
        distances = [
            float(np.hypot(cell[0] - other[0], cell[1] - other[1]))
            for other_id, other in positions.items()
            if other_id != agent_id
        ]
        if not distances:
            return 0.0
        return float(np.clip(np.mean(distances) / max(1.0, grid_size), 0.0, 1.0))

    @staticmethod
    def _target_distance(env: CityTwinEnvironment, cell: Cell) -> float:
        targets = env.remaining_missions() or env.mission_zones
        if not targets:
            return 0.0
        distance = min(abs(tx - cell[0]) + abs(ty - cell[1]) for tx, ty in targets)
        return float(distance / max(1, 2 * env.grid_size))

    def _features(
        self,
        env: CityTwinEnvironment,
        cell: Cell,
        current: Cell,
        positions: Dict[int, Cell],
        agent_id: int,
    ) -> dict[str, float]:
        state = env.agents[agent_id]
        visits = float(env.visit_counts[cell])
        return {
            "priority": float(env.priority_map[cell]),
            "uncertainty": float(env.uncertainty_map[cell]),
            "pheromone": float(env.pheromone_map[cell]),
            "novelty": 1.0 / (1.0 + visits),
            "visits": visits,
            "congestion": self._congestion(cell, agent_id, positions),
            "spread": self._spread(cell, agent_id, positions, env.grid_size),
            "battery": float(state.battery_level / 100.0),
            "communication": float(state.communication_status),
            "distance": float(cell != current),
            "target_distance": self._target_distance(env, cell),
            "restricted": float(cell in env.restricted_zones),
        }

    def _score(
        self,
        env: CityTwinEnvironment,
        cell: Cell,
        current: Cell,
        positions: Dict[int, Cell],
        agent_id: int,
    ) -> float:
        f = self._features(env, cell, current, positions, agent_id)
        c = self.config
        return float(
            c.risk_weight * f["priority"]
            + c.uncertainty_weight * f["uncertainty"]
            + c.pheromone_weight * f["pheromone"]
            + c.novelty_weight * f["novelty"]
            + c.communication_weight * f["communication"] * f["uncertainty"]
            + c.energy_weight * f["battery"]
            - c.spacing_weight * f["congestion"]
            - c.target_weight * f["target_distance"]
            - 0.18 * f["visits"]
            - 0.03 * f["distance"]
            - 10.0 * f["restricted"]
        )

    def _probabilities(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return np.asarray([], dtype=float)
        safe_scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0)
        temperature = max(0.10, float(self.config.temperature))
        logits = safe_scores / temperature
        logits -= float(np.max(logits))
        weights = np.exp(logits)
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 1e-12:
            return np.ones_like(safe_scores) / len(safe_scores)
        return weights / total

    def _select(self, candidates: list[Cell], scores: np.ndarray) -> Cell:
        probabilities = self._probabilities(scores)
        if not candidates or probabilities.size == 0:
            raise ValueError("Policy selection requires at least one candidate")
        self.last_entropy = float(
            -np.sum(np.where(probabilities > 0, probabilities * np.log(probabilities + 1e-12), 0.0))
        )
        self.last_confidence = float(np.max(probabilities))
        self.last_distribution = {str(index): float(value) for index, value in enumerate(probabilities)}
        if self.rng.random() < float(np.clip(self.config.epsilon, 0.0, 1.0)):
            index = int(self.rng.integers(0, len(candidates)))
        else:
            index = int(self.rng.choice(len(candidates), p=probabilities))
        selected = candidates[index]
        self.last_selected_cell = selected
        key = f"{selected[0]},{selected[1]}"
        self.action_counts[key] = self.action_counts.get(key, 0) + 1
        return selected

    def _propose(self, env: CityTwinEnvironment, agent_id: int, positions: Dict[int, Cell]) -> Cell:
        current = env.agents[agent_id].position
        candidates = env.get_neighbors(current)
        scores = np.asarray(
            [self._score(env, cell, current, positions, agent_id) for cell in candidates],
            dtype=float,
        )
        return self._select(candidates, scores)

    def act(self, env: CityTwinEnvironment) -> Dict[int, str]:
        positions = env.get_positions()
        proposed: Dict[int, str] = {}
        for agent_id in sorted(env.agents):
            current = env.agents[agent_id].position
            candidate = self._propose(env, agent_id, positions)
            proposed[agent_id] = env.cell_to_action(current, candidate)
        return self.monitor.filter_actions(env, proposed)

    def diagnostics(self) -> dict[str, Any]:
        total = max(1, sum(self.action_counts.values()))
        return {
            "name": self.name,
            "last_entropy": self.last_entropy,
            "last_confidence": self.last_confidence,
            "action_distribution": {
                key: value / total for key, value in sorted(self.action_counts.items())
            },
            "checkpoint": self.model_path,
            "checkpoint_metadata": self.checkpoint_metadata,
        }


class IPPOPolicy(_SafeMARLPolicy):
    """Independent PPO-style decentralized policy."""

    name = "IPPO-Safe"

    def __init__(self, seed: int = 42, **kwargs: Any) -> None:
        defaults = {
            "risk_weight": 1.25,
            "uncertainty_weight": 0.85,
            "pheromone_weight": 0.20,
            "novelty_weight": 0.40,
            "spacing_weight": 0.12,
            "communication_weight": 0.05,
            "temperature": 0.80,
            "epsilon": 0.03,
        }
        super().__init__(seed=seed, **_merge(defaults, kwargs))


class MAPPOPolicy(_SafeMARLPolicy):
    """Centralized-training/decentralized-execution MAPPO-style policy."""

    name = "MAPPO-Safe"

    def __init__(self, seed: int = 42, **kwargs: Any) -> None:
        defaults = {
            "risk_weight": 1.30,
            "uncertainty_weight": 0.90,
            "spacing_weight": 0.42,
            "communication_weight": 0.25,
            "temperature": 0.85,
            "epsilon": 0.025,
        }
        super().__init__(seed=seed, **_merge(defaults, kwargs))

    def _score(self, env: CityTwinEnvironment, cell: Cell, current: Cell, positions: Dict[int, Cell], agent_id: int) -> float:
        f = self._features(env, cell, current, positions, agent_id)
        team_priority = float(np.mean([env.observation_map[pos] for pos in positions.values()]))
        return float(
            1.30 * f["priority"]
            + 0.90 * f["uncertainty"]
            + 0.45 * f["spread"]
            + 0.30 * f["communication"] * (f["priority"] + f["uncertainty"])
            + 0.25 * team_priority
            + 0.30 * f["novelty"]
            - 0.28 * f["congestion"]
            - 0.18 * f["visits"]
            - 0.05 * f["target_distance"]
            - 10.0 * f["restricted"]
        )


class QMIXPolicy(_SafeMARLPolicy):
    """Monotonic value-decomposition policy using local and team utilities."""

    name = "QMIX-Safe"

    def __init__(self, seed: int = 42, **kwargs: Any) -> None:
        defaults = {"temperature": 0.90, "epsilon": 0.035}
        super().__init__(seed=seed, **_merge(defaults, kwargs))

    def _score(self, env: CityTwinEnvironment, cell: Cell, current: Cell, positions: Dict[int, Cell], agent_id: int) -> float:
        f = self._features(env, cell, current, positions, agent_id)
        team_utilities = [
            0.65 * float(env.observation_map[pos]) + 0.35 * float(env.uncertainty_map[pos])
            for pos in positions.values()
        ]
        team_utility = float(np.mean(team_utilities)) if team_utilities else 0.0
        local_utility = (
            1.05 * f["priority"]
            + 0.70 * f["uncertainty"]
            + 0.35 * f["pheromone"]
            + 0.40 * f["novelty"]
            - 0.22 * f["congestion"]
            - 0.16 * f["visits"]
        )
        return float(local_utility + 0.35 * max(0.0, team_utility) - 0.05 * f["target_distance"] - 10.0 * f["restricted"])


class MADDPGPolicy(_SafeMARLPolicy):
    """Continuous-actor-inspired MADDPG policy discretized onto grid moves."""

    name = "MADDPG-Safe"

    def __init__(self, seed: int = 42, **kwargs: Any) -> None:
        defaults = {"temperature": 0.70, "epsilon": 0.015}
        super().__init__(seed=seed, **_merge(defaults, kwargs))
        self.previous_direction: dict[int, np.ndarray] = {}

    def _score(self, env: CityTwinEnvironment, cell: Cell, current: Cell, positions: Dict[int, Cell], agent_id: int) -> float:
        f = self._features(env, cell, current, positions, agent_id)
        direction = np.asarray([cell[0] - current[0], cell[1] - current[1]], dtype=float)
        previous = self.previous_direction.get(agent_id, np.zeros(2, dtype=float))
        smoothness = float(np.dot(direction, previous))
        return float(
            1.18 * f["priority"]
            + 0.82 * f["uncertainty"]
            + 0.25 * f["pheromone"]
            + 0.30 * f["novelty"]
            + 0.20 * f["spread"]
            + 0.35 * f["battery"]
            + 0.12 * smoothness
            - 0.25 * f["visits"]
            - 0.22 * f["congestion"]
            - 0.10 * f["distance"]
            - 0.05 * f["target_distance"]
            - 10.0 * f["restricted"]
        )

    def _propose(self, env: CityTwinEnvironment, agent_id: int, positions: Dict[int, Cell]) -> Cell:
        current = env.agents[agent_id].position
        selected = super()._propose(env, agent_id, positions)
        self.previous_direction[agent_id] = np.asarray(
            [selected[0] - current[0], selected[1] - current[1]], dtype=float
        )
        return selected


class HAPPOPolicy(_SafeMARLPolicy):
    """Heterogeneous-agent PPO policy with role-sequential specialization."""

    name = "HAPPO-Safe"

    def _score(self, env: CityTwinEnvironment, cell: Cell, current: Cell, positions: Dict[int, Cell], agent_id: int) -> float:
        f = self._features(env, cell, current, positions, agent_id)
        role = agent_id % 3
        if role == 0:
            score = 1.65 * f["priority"] + 0.45 * f["uncertainty"] + 0.25 * f["pheromone"] + 0.20 * f["novelty"] - 0.18 * f["congestion"]
        elif role == 1:
            score = 0.75 * f["priority"] + 1.45 * f["uncertainty"] + 0.35 * f["novelty"] + 0.25 * f["spread"] - 0.20 * f["visits"]
        else:
            score = 0.85 * f["priority"] + 0.70 * f["uncertainty"] + 0.75 * f["novelty"] + 0.45 * f["spread"] - 0.35 * f["congestion"]
        return float(score - 0.05 * f["target_distance"] - 10.0 * f["restricted"])


class MATPolicy(_SafeMARLPolicy):
    """Multi-agent-transformer-style policy with distance-weighted attention."""

    name = "MAT-Safe"

    @staticmethod
    def _attention_context(env: CityTwinEnvironment, cell: Cell, positions: Dict[int, Cell], agent_id: int) -> float:
        context = 0.0
        total_weight = 0.0
        for other_id, position in positions.items():
            if other_id == agent_id:
                continue
            distance = float(np.hypot(cell[0] - position[0], cell[1] - position[1]))
            weight = 1.0 / (1.0 + distance)
            context += weight * (
                0.60 * float(env.observation_map[position])
                + 0.40 * float(env.uncertainty_map[position])
            )
            total_weight += weight
        return float(context / total_weight) if total_weight > 1e-12 else 0.0

    def _score(self, env: CityTwinEnvironment, cell: Cell, current: Cell, positions: Dict[int, Cell], agent_id: int) -> float:
        f = self._features(env, cell, current, positions, agent_id)
        attention = self._attention_context(env, cell, positions, agent_id)
        return float(
            1.25 * f["priority"]
            + 0.85 * f["uncertainty"]
            + 0.35 * f["pheromone"]
            + 0.50 * f["novelty"]
            + 0.40 * f["spread"]
            + 0.30 * attention
            + 0.20 * f["communication"] * f["uncertainty"]
            - 0.38 * f["congestion"]
            - 0.18 * f["visits"]
            - 0.05 * f["target_distance"]
            - 10.0 * f["restricted"]
        )


class GRPOPolicy(_SafeMARLPolicy):
    """Group Relative Policy Optimization execution policy.

    A behavior group is scored from the current local/team state, normalized by
    its group mean and standard deviation, sampled through a softmax policy, and
    then used to rank admissible grid actions. This mirrors the GRPO baseline in
    BioSwarm while retaining SafeSwarm runtime assurance.
    """

    name = "GRPO-Safe"
    behavior_names = (
        "explore_uncertain_area",
        "exploit_high_priority_area",
        "follow_pheromone",
        "communicate_with_neighbors",
        "revisit_unresolved_target",
        "reduce_redundant_coverage",
        "save_energy",
    )

    def __init__(
        self,
        seed: int = 42,
        exploration: float = 0.08,
        group_size: int = 7,
        **kwargs: Any,
    ) -> None:
        super().__init__(seed=seed, **kwargs)
        self.exploration = float(np.clip(exploration, 0.0, 0.30))
        self.group_size = max(2, min(int(group_size), len(self.behavior_names)))
        self.last_behavior = 0
        self.behavior_counts = np.zeros(len(self.behavior_names), dtype=np.int64)

    def _behavior_scores(self, env: CityTwinEnvironment, agent_id: int, positions: Dict[int, Cell]) -> np.ndarray:
        current = env.agents[agent_id].position
        neighbors = env.get_neighbors(current)
        priority_max = max(float(env.priority_map[cell]) for cell in neighbors)
        uncertainty_max = max(float(env.uncertainty_map[cell]) for cell in neighbors)
        pheromone_max = max(float(env.pheromone_map[cell]) for cell in neighbors)
        visit_min = min(float(env.visit_counts[cell]) for cell in neighbors)
        local_priority = float(env.observation_map[current])
        local_uncertainty = float(env.uncertainty_map[current])
        team_spread = self._spread(current, agent_id, positions, env.grid_size)
        battery = float(env.agents[agent_id].battery_level / 100.0)
        scores = np.asarray(
            [
                1.20 * uncertainty_max + 0.20 * team_spread,
                1.45 * priority_max,
                1.00 * pheromone_max + 0.25 * priority_max,
                0.90 * local_uncertainty + 0.40 * local_priority,
                1.10 * local_priority + 0.70 * local_uncertainty,
                1.00 / (1.0 + visit_min) + 0.30 * team_spread,
                1.20 * max(0.0, 0.35 - battery),
            ],
            dtype=float,
        )
        active = scores[: self.group_size]
        active -= float(np.mean(active))
        standard_deviation = float(np.std(active))
        if standard_deviation > 1e-8:
            active /= standard_deviation
        scores[: self.group_size] = active
        return scores

    def _choose_behavior(self, env: CityTwinEnvironment, agent_id: int, positions: Dict[int, Cell]) -> int:
        scores = self._behavior_scores(env, agent_id, positions)
        probabilities = self._probabilities(scores[: self.group_size])
        self.last_entropy = float(
            -np.sum(np.where(probabilities > 0, probabilities * np.log(probabilities + 1e-12), 0.0))
        )
        self.last_confidence = float(np.max(probabilities))
        if self.rng.random() < self.exploration:
            return int(self.rng.integers(0, self.group_size))
        return int(self.rng.choice(self.group_size, p=probabilities))

    def _behavior_action_score(
        self,
        behavior: int,
        env: CityTwinEnvironment,
        cell: Cell,
        current: Cell,
        positions: Dict[int, Cell],
        agent_id: int,
    ) -> float:
        f = self._features(env, cell, current, positions, agent_id)
        name = self.behavior_names[behavior]
        if name == "explore_uncertain_area":
            score = 1.60 * f["uncertainty"] + 0.40 * f["priority"] - 0.35 * f["visits"]
        elif name == "exploit_high_priority_area":
            score = 1.70 * f["priority"] + 0.25 * f["uncertainty"] - 0.30 * f["visits"]
        elif name == "follow_pheromone":
            score = 1.20 * f["pheromone"] + 0.80 * f["priority"] - 0.25 * f["visits"]
        elif name == "communicate_with_neighbors":
            score = 0.90 * f["priority"] + 0.90 * f["uncertainty"] + 0.20 * f["pheromone"] - 0.20 * f["congestion"]
        elif name == "revisit_unresolved_target":
            score = 1.10 * f["priority"] + 1.00 * f["uncertainty"] - 0.10 * f["visits"] - 0.12 * f["target_distance"]
        elif name == "reduce_redundant_coverage":
            score = 0.80 * f["priority"] + 0.60 * f["uncertainty"] + 0.50 * f["spread"] - 0.80 * f["visits"]
        else:
            score = 0.50 * f["priority"] + 0.30 * f["uncertainty"] + 0.80 * f["battery"] - 0.50 * f["distance"]
        return float(score - 0.25 * f["congestion"] - 10.0 * f["restricted"])

    def _propose(self, env: CityTwinEnvironment, agent_id: int, positions: Dict[int, Cell]) -> Cell:
        current = env.agents[agent_id].position
        candidates = env.get_neighbors(current)
        behavior = self._choose_behavior(env, agent_id, positions)
        self.last_behavior = behavior
        self.behavior_counts[behavior] += 1
        scores = np.asarray(
            [
                self._behavior_action_score(behavior, env, cell, current, positions, agent_id)
                for cell in candidates
            ],
            dtype=float,
        )
        return self._select(candidates, scores)

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = super().diagnostics()
        total = max(1, int(np.sum(self.behavior_counts)))
        diagnostics.update(
            {
                "last_behavior": self.behavior_names[self.last_behavior],
                "group_size": self.group_size,
                "behavior_distribution": {
                    name: float(self.behavior_counts[index] / total)
                    for index, name in enumerate(self.behavior_names)
                },
            }
        )
        return diagnostics
