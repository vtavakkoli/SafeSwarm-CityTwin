"""GRPO-Safe with trainable behavior selection and geographical swarm memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.agents.marl_baselines import GRPOPolicy
from src.agents.safe_ppo_core import (
    FEATURE_NAMES,
    VALUE_FEATURE_NAMES,
    PPOResidualMixin,
    feature_index,
    softmax,
)
from src.environment.city_twin import Cell, CityTwinEnvironment


class TrainableGRPOMemoryPolicy(PPOResidualMixin, GRPOPolicy):
    """Safety-masked GRPO with learned high-level behavior and signed search memory.

    The memory is not a map of hidden targets. It contains only sensed evidence,
    uncertainty/frontier state, visit history, pheromone traces and known unsafe
    boundaries. Positive utility propagates into unexplored traversable frontiers;
    revisits, congestion, discovered targets and unsafe boundaries are repulsive.
    """

    name = "GRPO-Safe"

    def __init__(
        self,
        *args: Any,
        memory_decay: float = 0.94,
        memory_weight: float = 0.20,
        frontier_weight: float = 0.35,
        propagation_weight: float = 0.15,
        propagation_steps: int = 2,
        memory_enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        model_path = kwargs.get("model_path")
        super().__init__(*args, **kwargs)
        self.exploration = 0.0
        self.memory_decay = float(np.clip(memory_decay, 0.80, 0.995))
        self.propagation_steps = max(0, int(propagation_steps))
        self.memory_enabled = bool(memory_enabled)
        self.swarm_memory: np.ndarray | None = None
        self.frontier_map: np.ndarray | None = None
        self.memory_updates = 0
        self.memory_peak = 0.0
        self._memory_env_token: int | None = None
        self.behavior_bias = np.zeros(len(self.behavior_names), dtype=float)
        self._pending_behavior: dict[int, dict[str, Any]] = {}

        if model_path:
            self._load_grpo_checkpoint(model_path)
        else:
            # Trainable priors. They are updated by PPO, unlike the v1 fixed bonuses.
            self.residual_weights[feature_index("swarm_memory")] = float(memory_weight)
            self.residual_weights[feature_index("frontier")] = float(frontier_weight)
            self.residual_weights[feature_index("propagation_gradient")] = float(propagation_weight)
            self.residual_weights[feature_index("return_progress")] = 0.25

    def _load_grpo_checkpoint(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            behavior = list(payload.get("behavior_bias", []))
            if len(behavior) == len(self.behavior_bias):
                self.behavior_bias = np.asarray(behavior, dtype=float)
            memory = dict(payload.get("memory", {}))
            if "memory_decay" in memory:
                self.memory_decay = float(np.clip(memory["memory_decay"], 0.80, 0.995))
            if "propagation_steps" in memory:
                self.propagation_steps = max(0, int(memory["propagation_steps"]))
            if "memory_enabled" in memory:
                self.memory_enabled = bool(memory["memory_enabled"])

            # Migrate the old fixed v1 mechanism strengths into trainable weights.
            if str(payload.get("format", "")).endswith("v1"):
                for old_name, feature_name in {
                    "memory_weight": "swarm_memory",
                    "frontier_weight": "frontier",
                    "propagation_weight": "propagation_gradient",
                }.items():
                    if old_name in memory:
                        self.residual_weights[feature_index(feature_name)] = float(memory[old_name])
        except (OSError, ValueError, TypeError):
            return

    @staticmethod
    def _neighbor_mean(values: np.ndarray) -> np.ndarray:
        padded = np.pad(values, 1, mode="constant", constant_values=0.0)
        return (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        ) / 4.0

    def _update_swarm_memory(self, env: CityTwinEnvironment) -> None:
        shape = env.uncertainty_map.shape
        if self.swarm_memory is None or self.swarm_memory.shape != shape:
            self.swarm_memory = np.zeros_like(env.uncertainty_map, dtype=float)

        observed = (env.uncertainty_map < 0.99).astype(float)
        unseen = 1.0 - observed
        frontier = np.clip(self._neighbor_mean(observed) * unseen, 0.0, 1.0)
        observed_priority = env.observation_map * np.clip(1.0 - env.uncertainty_map, 0.0, 1.0)
        priority_halo = self._neighbor_mean(observed_priority) * unseen
        novelty = 1.0 / (1.0 + env.visit_counts.astype(float))
        revisit_penalty = np.clip(env.visit_counts.astype(float) / 3.0, 0.0, 1.0)

        blocked = np.zeros_like(self.swarm_memory, dtype=bool)
        for cell in env.obstacles | env.restricted_zones:
            blocked[cell] = True
        traversable = (~blocked).astype(float)
        unsafe_halo = np.clip(self._neighbor_mean(blocked.astype(float)), 0.0, 1.0)

        occupied = np.zeros_like(self.swarm_memory, dtype=float)
        for state in env.agents.values():
            occupied[state.position] += 1.0
        occupied = np.clip(occupied, 0.0, 1.0)

        evidence = (
            1.20 * frontier
            + 0.80 * priority_halo
            + 0.25 * np.clip(env.pheromone_map, 0.0, 2.0) * novelty
            + 0.15 * unseen * novelty
            - 0.75 * revisit_penalty
            - 0.45 * unsafe_halo
            - 0.30 * occupied
        )
        for cell in env.discovered_missions:
            evidence[cell] -= 1.0

        self.swarm_memory *= self.memory_decay
        self.swarm_memory += evidence
        self.swarm_memory[blocked] = 0.0

        if self.memory_enabled:
            for _ in range(self.propagation_steps):
                positive = np.clip(self.swarm_memory, 0.0, None)
                propagated = self._neighbor_mean(positive) * traversable * unseen
                self.swarm_memory += 0.20 * propagated
                self.swarm_memory[blocked] = 0.0

        raw_peak = float(np.max(np.abs(self.swarm_memory)))
        self.memory_peak = max(self.memory_peak, raw_peak)
        if raw_peak > 1.0:
            self.swarm_memory /= raw_peak
        self.frontier_map = frontier * novelty * traversable
        self.memory_updates += 1

    def reset_episode_state(self) -> None:
        super().reset_episode_state()
        self.swarm_memory = None
        self.frontier_map = None
        self.memory_updates = 0
        self.memory_peak = 0.0
        self._memory_env_token = None
        self._pending_behavior = {}

    def _extra_feature_values(
        self, env: CityTwinEnvironment, cell: Cell, current: Cell
    ) -> tuple[float, float, float]:
        if not self.memory_enabled or self.swarm_memory is None:
            return 0.0, 0.0, 0.0
        memory_value = float(self.swarm_memory[cell])
        frontier_value = 0.0 if self.frontier_map is None else float(self.frontier_map[cell])
        propagation_gradient = float(self.swarm_memory[cell] - self.swarm_memory[current])
        return memory_value, frontier_value, propagation_gradient

    def _observable_behavior_scores(
        self, env: CityTwinEnvironment, agent_id: int, positions: Dict[int, Cell]
    ) -> np.ndarray:
        current = env.agents[agent_id].position
        neighbors = env.get_neighbors(current)
        priority_max = max(float(env.observation_map[cell]) for cell in neighbors)
        uncertainty_max = max(float(env.uncertainty_map[cell]) for cell in neighbors)
        pheromone_max = max(float(env.pheromone_map[cell]) for cell in neighbors)
        visit_min = min(float(env.visit_counts[cell]) for cell in neighbors)
        local_priority = float(env.observation_map[current])
        local_uncertainty = float(env.uncertainty_map[current])
        team_spread = self._spread(current, agent_id, positions, env.grid_size)
        battery = float(env.agents[agent_id].battery_level / 100.0)
        memory_frontier = 0.0
        if self.frontier_map is not None:
            memory_frontier = max(float(self.frontier_map[cell]) for cell in neighbors)
        scores = np.asarray([
            1.10 * uncertainty_max + 0.35 * memory_frontier + 0.20 * team_spread,
            1.25 * priority_max + 0.35 * memory_frontier,
            0.90 * pheromone_max + 0.20 * priority_max,
            0.85 * local_uncertainty + 0.35 * local_priority,
            0.90 * local_priority + 0.55 * local_uncertainty,
            1.00 / (1.0 + visit_min) + 0.35 * team_spread,
            1.60 * max(0.0, 0.45 - battery),
        ], dtype=float)
        active = scores[: self.group_size].copy()
        active -= float(np.mean(active))
        std = float(np.std(active))
        if std > 1e-8:
            active /= std
        scores[: self.group_size] = active
        return scores

    def _choose_behavior(
        self, env: CityTwinEnvironment, agent_id: int, positions: Dict[int, Cell]
    ) -> int:
        base = self._observable_behavior_scores(env, agent_id, positions)[: self.group_size]
        probabilities = softmax(base + self.behavior_bias[: self.group_size], self.config.temperature)
        behavior = int(self.rng.choice(self.group_size, p=probabilities))
        self._pending_behavior[agent_id] = {
            "behavior_base_scores": base.copy(),
            "behavior_index": behavior,
            "behavior_old_probability": float(max(probabilities[behavior], 1e-12)),
        }
        return behavior

    def _behavior_action_score(
        self,
        behavior: int,
        env: CityTwinEnvironment,
        cell: Cell,
        current: Cell,
        positions: Dict[int, Cell],
        agent_id: int,
    ) -> float:
        if self.behavior_names[behavior] == "save_energy":
            battery = float(env.agents[agent_id].battery_level / 100.0)
            progress = float(
                env.nearest_base_distance(current) - env.nearest_base_distance(cell)
            )
            f = self._features(env, cell, current, positions, agent_id)
            return float(
                (2.2 + 1.2 * (1.0 - battery)) * progress
                + 0.8 * float(cell in env.base_stations)
                + 0.20 * f["novelty"]
                - 0.25 * f["congestion"]
            )
        return float(super()._behavior_action_score(
            behavior, env, cell, current, positions, agent_id
        ))

    def act(self, env: CityTwinEnvironment) -> Dict[int, str]:
        env_token = id(env)
        if self._memory_env_token != env_token:
            self.swarm_memory = None
            self.frontier_map = None
            self.memory_updates = 0
            self.memory_peak = 0.0
            self._memory_env_token = env_token
        self._update_swarm_memory(env)
        return super().act(env)

    def _propose(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        positions: Dict[int, Cell],
        candidates: list[Cell] | None = None,
    ) -> Cell:
        current = env.agents[agent_id].position
        candidates = list(candidates) if candidates is not None else env.get_neighbors(current)
        behavior = self._choose_behavior(env, agent_id, positions)
        self.last_behavior = behavior
        self.behavior_counts[behavior] += 1
        base_scores = np.asarray([
            self._behavior_action_score(behavior, env, cell, current, positions, agent_id)
            for cell in candidates
        ], dtype=float)
        features = np.vstack([
            self._feature_vector(env, cell, current, positions, agent_id)
            for cell in candidates
        ])
        selected = self._choose_with_residual(
            env, agent_id, current, candidates, base_scores, features
        )
        pending = self._pending_behavior.pop(agent_id, None)
        if pending is not None and self._pending_decisions:
            self._pending_decisions[-1].update(pending)
        return selected

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "safeswarm-grpo-memory-ppo-v2",
            "strategy": self.name,
            "feature_names": list(FEATURE_NAMES),
            "ppo_residual_weights": self.residual_weights.tolist(),
            "value_feature_names": list(VALUE_FEATURE_NAMES),
            "critic_weights": self.critic_weights.tolist(),
            "behavior_bias": self.behavior_bias.tolist(),
            "params": {key: float(getattr(self.config, key)) for key in vars(self.config)},
            "memory": {
                "memory_decay": self.memory_decay,
                "memory_enabled": self.memory_enabled,
                "propagation_steps": self.propagation_steps,
                "learned_memory_weight": float(self.residual_weights[feature_index("swarm_memory")]),
                "learned_frontier_weight": float(self.residual_weights[feature_index("frontier")]),
                "learned_propagation_weight": float(self.residual_weights[feature_index("propagation_gradient")]),
            },
            "metadata": metadata or {},
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def diagnostics(self) -> dict[str, Any]:
        data = super().diagnostics()
        memory = self.swarm_memory
        data.update({
            "swarm_memory_updates": self.memory_updates,
            "swarm_memory_peak": self.memory_peak,
            "swarm_memory_coverage": 0.0 if memory is None else float(np.mean(np.abs(memory) > 0.05)),
            "memory_decay": self.memory_decay,
            "propagation_steps": self.propagation_steps,
            "memory_enabled": self.memory_enabled,
            "learned_memory_weight": float(self.residual_weights[feature_index("swarm_memory")]),
            "learned_frontier_weight": float(self.residual_weights[feature_index("frontier")]),
            "learned_propagation_weight": float(self.residual_weights[feature_index("propagation_gradient")]),
            "behavior_bias": self.behavior_bias.tolist(),
            "behavior_weight_norm": float(np.linalg.norm(self.behavior_bias)),
        })
        return data
