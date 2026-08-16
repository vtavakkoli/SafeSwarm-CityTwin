"""Trainable PPO residual policies and swarm-memory GRPO execution.

The existing MARL policies remain lightweight interpretable baselines. This module
adds a small clipped-PPO residual over their action scores, plus a GRPO variant with
shared spatial swarm memory and geographic frontier propagation. The residual is
linear and dependency-free (NumPy only), making training auditable and fast enough
for repeated real-city experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

from src.agents.marl_baselines import GRPOPolicy, HAPPOPolicy, IPPOPolicy, MAPPOPolicy
from src.environment.city_twin import Cell, CityTwinEnvironment

FEATURE_NAMES = (
    "priority",
    "uncertainty",
    "pheromone",
    "novelty",
    "spread",
    "battery",
    "communication",
    "congestion",
    "visits",
    "target_distance",
    "distance",
    "swarm_memory",
    "frontier",
)


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    if logits.size == 0:
        return np.asarray([], dtype=float)
    values = np.nan_to_num(logits.astype(float), nan=0.0, posinf=1.0, neginf=-1.0)
    values = values / max(0.10, float(temperature))
    values -= float(np.max(values))
    weights = np.exp(values)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 1e-12:
        return np.ones_like(values) / len(values)
    return weights / total


class PPOResidualMixin:
    """Clipped-PPO residual applied on top of an existing interpretable policy score."""

    residual_l2 = 1e-4

    def __init__(self, *args: Any, model_path: str | Path | None = None, **kwargs: Any) -> None:
        super().__init__(*args, model_path=model_path, **kwargs)
        self.residual_weights = np.zeros(len(FEATURE_NAMES), dtype=float)
        self._pending_decisions: list[dict[str, Any]] = []
        if model_path:
            self._load_trainable_checkpoint(model_path)

    def _load_trainable_checkpoint(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload.get("ppo_residual_weights", [])
            if len(values) == len(FEATURE_NAMES):
                self.residual_weights = np.asarray(values, dtype=float)
        except (OSError, ValueError, TypeError):
            return

    def _extra_feature_values(
        self, env: CityTwinEnvironment, cell: Cell
    ) -> tuple[float, float]:
        return 0.0, 0.0

    def _feature_vector(
        self,
        env: CityTwinEnvironment,
        cell: Cell,
        current: Cell,
        positions: Dict[int, Cell],
        agent_id: int,
    ) -> np.ndarray:
        f = self._features(env, cell, current, positions, agent_id)
        memory_value, frontier_value = self._extra_feature_values(env, cell)
        return np.asarray(
            [
                f["priority"],
                f["uncertainty"],
                f["pheromone"],
                f["novelty"],
                f["spread"],
                f["battery"],
                f["communication"],
                -f["congestion"],
                -min(5.0, f["visits"]) / 5.0,
                -f["target_distance"],
                -f["distance"],
                memory_value,
                frontier_value,
            ],
            dtype=float,
        )

    def reset_episode_state(self) -> None:
        self._pending_decisions = []

    def _choose_with_residual(
        self,
        candidates: list[Cell],
        base_scores: np.ndarray,
        features: np.ndarray,
    ) -> Cell:
        logits = base_scores + features @ self.residual_weights
        probabilities = _softmax(logits, self.config.temperature)
        if self.rng.random() < float(np.clip(self.config.epsilon, 0.0, 1.0)):
            index = int(self.rng.integers(0, len(candidates)))
        else:
            index = int(self.rng.choice(len(candidates), p=probabilities))
        self._pending_decisions.append(
            {
                "features": features.copy(),
                "base_scores": base_scores.copy(),
                "action_index": index,
                "old_probability": float(max(probabilities[index], 1e-12)),
            }
        )
        self.last_entropy = float(
            -np.sum(np.where(probabilities > 0, probabilities * np.log(probabilities + 1e-12), 0.0))
        )
        self.last_confidence = float(np.max(probabilities))
        self.last_distribution = {str(i): float(value) for i, value in enumerate(probabilities)}
        selected = candidates[index]
        self.last_selected_cell = selected
        key = f"{selected[0]},{selected[1]}"
        self.action_counts[key] = self.action_counts.get(key, 0) + 1
        return selected

    def _propose(self, env: CityTwinEnvironment, agent_id: int, positions: Dict[int, Cell]) -> Cell:
        current = env.agents[agent_id].position
        candidates = env.get_neighbors(current)
        base_scores = np.asarray(
            [super()._score(env, cell, current, positions, agent_id) for cell in candidates],
            dtype=float,
        )
        features = np.vstack(
            [self._feature_vector(env, cell, current, positions, agent_id) for cell in candidates]
        )
        return self._choose_with_residual(candidates, base_scores, features)

    def drain_decisions(self) -> list[dict[str, Any]]:
        decisions = self._pending_decisions
        self._pending_decisions = []
        return decisions

    def ppo_update(
        self,
        samples: Iterable[tuple[dict[str, Any], float]],
        *,
        learning_rate: float = 0.03,
        clip_ratio: float = 0.20,
        epochs: int = 4,
        max_grad_norm: float = 2.0,
    ) -> dict[str, float]:
        batch = list(samples)
        if not batch:
            return {
                "loss": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "weight_norm": float(np.linalg.norm(self.residual_weights)),
            }

        advantages = np.asarray([adv for _, adv in batch], dtype=float)
        advantages = (advantages - float(np.mean(advantages))) / (float(np.std(advantages)) + 1e-8)

        approx_kl = 0.0
        clipped = 0
        considered = 0
        loss_value = 0.0
        for _ in range(max(1, epochs)):
            gradients: list[np.ndarray] = []
            losses: list[float] = []
            kls: list[float] = []
            clipped_epoch = 0
            for (decision, _), advantage in zip(batch, advantages):
                features = np.asarray(decision["features"], dtype=float)
                base_scores = np.asarray(decision["base_scores"], dtype=float)
                action = int(decision["action_index"])
                old_probability = float(max(decision["old_probability"], 1e-12))
                logits = base_scores + features @ self.residual_weights
                probabilities = _softmax(logits, self.config.temperature)
                new_probability = float(max(probabilities[action], 1e-12))
                ratio = new_probability / old_probability
                clipped_ratio = float(np.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio))
                unclipped_obj = ratio * float(advantage)
                clipped_obj = clipped_ratio * float(advantage)
                use_clipped = clipped_obj < unclipped_obj
                objective = min(unclipped_obj, clipped_obj)
                losses.append(-objective)

                if not use_clipped:
                    expected = probabilities @ features
                    grad_logp = (features[action] - expected) / max(0.10, float(self.config.temperature))
                    gradients.append(float(advantage) * ratio * grad_logp)
                else:
                    gradients.append(np.zeros_like(self.residual_weights))
                    clipped_epoch += 1
                kls.append(np.log(old_probability) - np.log(new_probability))

            grad = np.mean(gradients, axis=0) - self.residual_l2 * self.residual_weights
            norm = float(np.linalg.norm(grad))
            if norm > max_grad_norm:
                grad *= max_grad_norm / max(norm, 1e-12)
            self.residual_weights += float(learning_rate) * grad
            loss_value = float(np.mean(losses))
            approx_kl = float(np.mean(kls))
            clipped += clipped_epoch
            considered += len(batch)

        return {
            "loss": loss_value,
            "approx_kl": approx_kl,
            "clip_fraction": float(clipped / max(1, considered)),
            "weight_norm": float(np.linalg.norm(self.residual_weights)),
        }

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "safeswarm-ppo-residual-v1",
            "strategy": self.name,
            "feature_names": list(FEATURE_NAMES),
            "ppo_residual_weights": self.residual_weights.tolist(),
            "params": {key: float(getattr(self.config, key)) for key in vars(self.config)},
            "metadata": metadata or {},
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def diagnostics(self) -> dict[str, Any]:
        data = super().diagnostics()
        data.update(
            {
                "ppo_residual_weight_norm": float(np.linalg.norm(self.residual_weights)),
                "ppo_residual_weights": self.residual_weights.tolist(),
            }
        )
        return data


class TrainableIPPOPolicy(PPOResidualMixin, IPPOPolicy):
    name = "IPPO-Safe"


class TrainableMAPPOPolicy(PPOResidualMixin, MAPPOPolicy):
    name = "MAPPO-Safe"


class TrainableHAPPOPolicy(PPOResidualMixin, HAPPOPolicy):
    name = "HAPPO-Safe"


class TrainableGRPOMemoryPolicy(PPOResidualMixin, GRPOPolicy):
    """GRPO-Safe with shared spatial memory, frontier search, and diffusion.

    The memory contains only evidence available to the swarm during execution:
    sensed observations, uncertainty/frontier state, pheromone traces, and visit
    history. It does not introduce new hidden target labels.
    """

    name = "GRPO-Safe"

    def __init__(
        self,
        *args: Any,
        memory_decay: float = 0.965,
        memory_weight: float = 1.10,
        frontier_weight: float = 1.15,
        propagation_weight: float = 0.75,
        propagation_steps: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.memory_decay = float(np.clip(memory_decay, 0.80, 0.999))
        self.memory_weight = float(memory_weight)
        self.frontier_weight = float(frontier_weight)
        self.propagation_weight = float(propagation_weight)
        self.propagation_steps = max(0, int(propagation_steps))
        self.swarm_memory: np.ndarray | None = None
        self.memory_updates = 0
        self.memory_peak = 0.0
        self._memory_env_token: int | None = None
        model_path = kwargs.get("model_path")
        if model_path:
            self._load_memory_checkpoint(model_path)

    def _load_memory_checkpoint(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            memory = dict(payload.get("memory", {}))
            for key in ("memory_decay", "memory_weight", "frontier_weight", "propagation_weight"):
                if key in memory:
                    setattr(self, key, float(memory[key]))
            if "propagation_steps" in memory:
                self.propagation_steps = max(0, int(memory["propagation_steps"]))
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
        if self.swarm_memory is None or self.swarm_memory.shape != env.uncertainty_map.shape:
            self.swarm_memory = np.zeros_like(env.uncertainty_map, dtype=float)

        observed = (env.uncertainty_map < 0.99).astype(float)
        frontier = np.clip(self._neighbor_mean(observed) * (env.uncertainty_map >= 0.99), 0.0, 1.0)
        observed_priority = env.observation_map * (1.0 - env.uncertainty_map)
        novelty = 1.0 / (1.0 + env.visit_counts.astype(float))
        evidence = (
            1.8 * observed_priority
            + 0.55 * frontier
            + 0.25 * np.clip(env.pheromone_map, 0.0, 2.0)
            + 0.15 * observed * novelty
        )
        self.swarm_memory *= self.memory_decay
        self.swarm_memory += evidence

        blocked = np.zeros_like(self.swarm_memory, dtype=bool)
        for cell in env.obstacles | env.restricted_zones:
            blocked[cell] = True
        for _ in range(self.propagation_steps):
            propagated = self._neighbor_mean(self.swarm_memory)
            self.swarm_memory = 0.72 * self.swarm_memory + 0.28 * propagated
            self.swarm_memory[blocked] = 0.0

        peak = float(np.max(self.swarm_memory))
        if peak > 1e-12:
            self.swarm_memory /= max(1.0, peak)
        self.memory_peak = max(self.memory_peak, float(np.max(self.swarm_memory)))
        self.memory_updates += 1

    def reset_episode_state(self) -> None:
        super().reset_episode_state()
        self.swarm_memory = None
        self.memory_updates = 0
        self.memory_peak = 0.0
        self._memory_env_token = None

    def _extra_feature_values(self, env: CityTwinEnvironment, cell: Cell) -> tuple[float, float]:
        memory_value = 0.0 if self.swarm_memory is None else float(self.swarm_memory[cell])
        frontier_value = float(env.uncertainty_map[cell]) / (1.0 + float(env.visit_counts[cell]))
        return memory_value, frontier_value

    def act(self, env: CityTwinEnvironment) -> Dict[int, str]:
        env_token = id(env)
        if self._memory_env_token != env_token:
            self.swarm_memory = None
            self.memory_updates = 0
            self.memory_peak = 0.0
            self._memory_env_token = env_token
        self._update_swarm_memory(env)
        return super().act(env)

    def _propose(self, env: CityTwinEnvironment, agent_id: int, positions: Dict[int, Cell]) -> Cell:
        current = env.agents[agent_id].position
        candidates = env.get_neighbors(current)
        behavior = self._choose_behavior(env, agent_id, positions)
        self.last_behavior = behavior
        self.behavior_counts[behavior] += 1

        base_scores = np.asarray(
            [
                self._behavior_action_score(behavior, env, cell, current, positions, agent_id)
                for cell in candidates
            ],
            dtype=float,
        )
        features = np.vstack(
            [self._feature_vector(env, cell, current, positions, agent_id) for cell in candidates]
        )

        assert self.swarm_memory is not None
        current_memory = float(self.swarm_memory[current])
        memory_scores = np.asarray([float(self.swarm_memory[cell]) for cell in candidates], dtype=float)
        frontier_scores = np.asarray(
            [
                float(env.uncertainty_map[cell]) / (1.0 + float(env.visit_counts[cell]))
                for cell in candidates
            ],
            dtype=float,
        )
        propagation_gradient = memory_scores - current_memory
        enriched_scores = (
            base_scores
            + self.memory_weight * memory_scores
            + self.frontier_weight * frontier_scores
            + self.propagation_weight * propagation_gradient
        )
        return self._choose_with_residual(candidates, enriched_scores, features)

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "safeswarm-grpo-memory-ppo-v1",
            "strategy": self.name,
            "feature_names": list(FEATURE_NAMES),
            "ppo_residual_weights": self.residual_weights.tolist(),
            "params": {key: float(getattr(self.config, key)) for key in vars(self.config)},
            "memory": {
                "memory_decay": self.memory_decay,
                "memory_weight": self.memory_weight,
                "frontier_weight": self.frontier_weight,
                "propagation_weight": self.propagation_weight,
                "propagation_steps": self.propagation_steps,
            },
            "metadata": metadata or {},
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def diagnostics(self) -> dict[str, Any]:
        data = super().diagnostics()
        memory = self.swarm_memory
        data.update(
            {
                "swarm_memory_updates": self.memory_updates,
                "swarm_memory_peak": self.memory_peak,
                "swarm_memory_coverage": 0.0
                if memory is None
                else float(np.mean(memory > 0.05)),
                "memory_decay": self.memory_decay,
                "memory_weight": self.memory_weight,
                "frontier_weight": self.frontier_weight,
                "propagation_weight": self.propagation_weight,
                "propagation_steps": self.propagation_steps,
            }
        )
        return data


TRAINABLE_POLICY_CLASSES = {
    "GRPO-Safe": TrainableGRPOMemoryPolicy,
    "IPPO-Safe": TrainableIPPOPolicy,
    "MAPPO-Safe": TrainableMAPPOPolicy,
    "HAPPO-Safe": TrainableHAPPOPolicy,
}


def checkpoint_path(model_dir: str | Path, strategy: str) -> Path:
    safe_name = strategy.lower().replace("-", "_")
    return Path(model_dir) / f"{safe_name}.json"


def _grpo_ablation(seed: int, path: Path, mode: str) -> TrainableGRPOMemoryPolicy:
    policy = TrainableGRPOMemoryPolicy(seed=seed, model_path=path)
    if mode == "no_memory":
        policy.memory_weight = 0.0
        policy.frontier_weight = 0.0
        policy.propagation_weight = 0.0
        policy.propagation_steps = 0
        policy.residual_weights[-2:] = 0.0
    elif mode == "no_propagation":
        policy.propagation_weight = 0.0
        policy.propagation_steps = 0
    else:
        raise ValueError(f"Unknown GRPO ablation mode: {mode}")
    return policy


def evaluation_factories(seed: int, model_dir: str | Path | None = None) -> dict[str, Any]:
    from src.agents.registry import strategy_factories

    factories = strategy_factories(seed=seed)
    paths: dict[str, Path | None] = {
        strategy: checkpoint_path(model_dir, strategy) if model_dir else None
        for strategy in TRAINABLE_POLICY_CLASSES
    }
    if model_dir:
        missing = [strategy for strategy, path in paths.items() if path is None or not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing trained checkpoints for held-out testing: " + ", ".join(missing)
            )

    for strategy, cls in TRAINABLE_POLICY_CLASSES.items():
        path = paths[strategy]
        factories[strategy] = lambda cls=cls, path=path: cls(seed=seed, model_path=path)

    grpo_path = paths["GRPO-Safe"]
    if grpo_path is not None:
        factories["GRPO-Safe-Ablation-NoMemory"] = (
            lambda path=grpo_path: _grpo_ablation(seed, path, "no_memory")
        )
        factories["GRPO-Safe-Ablation-NoPropagation"] = (
            lambda path=grpo_path: _grpo_ablation(seed, path, "no_propagation")
        )
    return factories
