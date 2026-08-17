"""GRPO-v3: state-conditioned behavior learning plus swarm memory.

v2 exposed only seven global behavior-bias scalars. That is not a learned
controller: the same correction was applied in Vienna, London, low battery,
high uncertainty and target-rich states. v3 adds a compact state-conditioned
behavior matrix and optional teacher distillation while preserving the signed
spatial memory mechanism.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.agents.grpo_memory_v2 import TrainableGRPOMemoryPolicy as _GRPOV2
from src.agents.observable_utils import observable_priority
from src.agents.ppo_v3 import PPOV3Mixin
from src.agents.safe_ppo_core import softmax
from src.environment.city_twin import Cell, CityTwinEnvironment

BEHAVIOR_FEATURE_NAMES = (
    "bias",
    "local_priority",
    "local_uncertainty",
    "local_pheromone",
    "neighbor_priority",
    "neighbor_uncertainty",
    "neighbor_frontier",
    "novelty",
    "team_spread",
    "nearby_agents",
    "battery",
    "base_distance",
    "unexplored_fraction",
)


class TrainableGRPOMemoryPolicy(PPOV3Mixin, _GRPOV2):
    name = "GRPO-Safe"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        model_path = kwargs.get("model_path")
        super().__init__(*args, **kwargs)
        self.behavior_weights = np.zeros(
            (len(BEHAVIOR_FEATURE_NAMES), len(self.behavior_names)), dtype=float
        )
        if model_path:
            self._load_v3_behavior_checkpoint(model_path)

    def _load_v3_behavior_checkpoint(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            weights = np.asarray(payload.get("behavior_weights", []), dtype=float)
            names = list(payload.get("behavior_feature_names", []))
            if weights.shape == self.behavior_weights.shape and (
                not names or names == list(BEHAVIOR_FEATURE_NAMES)
            ):
                self.behavior_weights = weights
        except (OSError, ValueError, TypeError):
            return

    def _behavior_state_features(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        positions: Dict[int, Cell],
    ) -> np.ndarray:
        current = env.agents[agent_id].position
        neighbors = env.get_neighbors(current)
        neighbor_priority = max(observable_priority(env, cell) for cell in neighbors)
        neighbor_uncertainty = max(float(env.uncertainty_map[cell]) for cell in neighbors)
        neighbor_frontier = 0.0
        if self.frontier_map is not None:
            neighbor_frontier = max(float(self.frontier_map[cell]) for cell in neighbors)
        novelty = max(1.0 / (1.0 + float(env.visit_counts[cell])) for cell in neighbors)
        team_spread = self._spread(current, agent_id, positions, env.grid_size)
        nearby = sum(
            1
            for other_id, pos in positions.items()
            if other_id != agent_id
            and float(np.hypot(current[0] - pos[0], current[1] - pos[1])) <= 2.0
        ) / max(1, len(positions) - 1)
        state = env.agents[agent_id]
        base_distance = float(
            env.nearest_base_distance(current) / max(1.0, 2.0 * env.grid_size)
        )
        unexplored = float(np.mean(env.uncertainty_map >= 0.99))
        return np.asarray(
            [
                1.0,
                observable_priority(env, current),
                float(env.uncertainty_map[current]),
                float(np.clip(env.pheromone_map[current], 0.0, 2.0) / 2.0),
                neighbor_priority,
                neighbor_uncertainty,
                neighbor_frontier,
                novelty,
                team_spread,
                float(nearby),
                float(state.battery_level / 100.0),
                base_distance,
                unexplored,
            ],
            dtype=float,
        )

    def _choose_behavior(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        positions: Dict[int, Cell],
    ) -> int:
        base = self._observable_behavior_scores(env, agent_id, positions)[: self.group_size]
        state_features = self._behavior_state_features(env, agent_id, positions)
        logits = (
            base
            + self.behavior_bias[: self.group_size]
            + state_features @ self.behavior_weights[:, : self.group_size]
        )
        probabilities = softmax(logits, self.config.temperature)
        behavior = int(self.rng.choice(self.group_size, p=probabilities))
        self._pending_behavior[agent_id] = {
            "behavior_base_scores": base.copy(),
            "behavior_features": state_features.copy(),
            "behavior_index": behavior,
            "behavior_old_probability": float(max(probabilities[behavior], 1e-12)),
        }
        self.last_entropy = float(
            -np.sum(probabilities * np.log(probabilities + 1e-12))
        )
        self.last_confidence = float(np.max(probabilities))
        return behavior

    def imitation_example(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        positions: Dict[int, Cell],
        candidates: list[Cell],
        target_cell: Cell,
    ) -> dict[str, Any] | None:
        if target_cell not in candidates or not candidates:
            return None
        target_index = candidates.index(target_cell)
        behavior_features = self._behavior_state_features(env, agent_id, positions)
        behavior_base = self._observable_behavior_scores(env, agent_id, positions)[: self.group_size]

        margins: list[float] = []
        action_scores_by_behavior: list[np.ndarray] = []
        for behavior in range(self.group_size):
            scores = np.asarray(
                [
                    self._behavior_action_score(
                        behavior, env, cell, env.agents[agent_id].position, positions, agent_id
                    )
                    for cell in candidates
                ],
                dtype=float,
            )
            action_scores_by_behavior.append(scores)
            margins.append(float(scores[target_index] - np.mean(scores)))
        behavior_index = int(np.argmax(np.asarray(margins)))
        features = np.vstack(
            [
                self._feature_vector(
                    env, cell, env.agents[agent_id].position, positions, agent_id
                )
                for cell in candidates
            ]
        )
        return {
            "features": features,
            "base_scores": action_scores_by_behavior[behavior_index],
            "target_index": int(target_index),
            "behavior_base_scores": behavior_base,
            "behavior_features": behavior_features,
            "behavior_target": int(behavior_index),
        }

    def imitation_update(
        self,
        examples: list[dict[str, Any]],
        *,
        learning_rate: float = 0.01,
        epochs: int = 1,
        label_smoothing: float = 0.08,
        max_grad_norm: float = 0.5,
    ) -> dict[str, float]:
        if not examples:
            return {"imitation_loss": 0.0, "imitation_accuracy": 0.0}
        losses: list[float] = []
        correct = total = 0
        for _ in range(max(1, int(epochs))):
            order = self.rng.permutation(len(examples))
            action_grads: list[np.ndarray] = []
            behavior_bias_grads: list[np.ndarray] = []
            behavior_weight_grads: list[np.ndarray] = []
            for raw_idx in order:
                example = examples[int(raw_idx)]
                features = np.asarray(example["features"], dtype=float)
                base = np.asarray(example["base_scores"], dtype=float)
                target = int(example["target_index"])
                probs = softmax(base + features @ self.residual_weights, self.config.temperature)
                smooth = float(np.clip(label_smoothing, 0.0, 0.3))
                target_probs = np.full(len(probs), smooth / max(1, len(probs)), dtype=float)
                target_probs[target] += 1.0 - smooth
                expected_features = target_probs @ features
                current_features = probs @ features
                action_grads.append(
                    (expected_features - current_features)
                    / max(0.10, float(self.config.temperature))
                )
                losses.append(float(-np.log(max(probs[target], 1e-12))))
                correct += int(np.argmax(probs) == target)
                total += 1

                bbase = np.asarray(example["behavior_base_scores"], dtype=float)
                bfeatures = np.asarray(example["behavior_features"], dtype=float)
                btarget = int(example["behavior_target"])
                blogits = (
                    bbase
                    + self.behavior_bias[: len(bbase)]
                    + bfeatures @ self.behavior_weights[:, : len(bbase)]
                )
                bprobs = softmax(blogits, self.config.temperature)
                btarget_probs = np.full(len(bprobs), smooth / max(1, len(bprobs)), dtype=float)
                btarget_probs[btarget] += 1.0 - smooth
                category_grad = (
                    btarget_probs - bprobs
                ) / max(0.10, float(self.config.temperature))
                bgrad = np.zeros_like(self.behavior_bias)
                bgrad[: len(bbase)] = category_grad
                behavior_bias_grads.append(bgrad)
                wgrad = np.zeros_like(self.behavior_weights)
                wgrad[:, : len(bbase)] = np.outer(bfeatures, category_grad)
                behavior_weight_grads.append(wgrad)

            for parameter, grads in (
                (self.residual_weights, action_grads),
                (self.behavior_bias, behavior_bias_grads),
                (self.behavior_weights, behavior_weight_grads),
            ):
                if not grads:
                    continue
                grad = np.mean(grads, axis=0)
                norm = float(np.linalg.norm(grad))
                if norm > max_grad_norm:
                    grad *= max_grad_norm / max(norm, 1e-12)
                parameter += float(learning_rate) * grad
        return {
            "imitation_loss": float(np.mean(losses)) if losses else 0.0,
            "imitation_accuracy": float(correct / max(1, total)),
            "imitation_examples": float(len(examples)),
        }

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        super().save_checkpoint(path, metadata)
        target = Path(path)
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["format"] = "safeswarm-grpo-memory-ppo-v3"
        payload["behavior_feature_names"] = list(BEHAVIOR_FEATURE_NAMES)
        payload["behavior_weights"] = self.behavior_weights.tolist()
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def diagnostics(self) -> dict[str, Any]:
        data = super().diagnostics()
        data.update(
            {
                "behavior_feature_names": list(BEHAVIOR_FEATURE_NAMES),
                "behavior_state_weight_norm": float(np.linalg.norm(self.behavior_weights)),
            }
        )
        return data
