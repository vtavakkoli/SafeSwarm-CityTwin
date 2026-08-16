"""Safety-aware PPO core used by the trainable SafeSwarm policies.

The actor samples only from actions approved by the runtime monitor. Therefore a
stored PPO decision always corresponds to the action passed to ``env.step``.
A small centralized linear critic supplies a value baseline while execution
remains decentralized and uses only sensed actor features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

from src.agents.marl_baselines import HAPPOPolicy, IPPOPolicy, MAPPOPolicy
from src.environment.city_twin import Cell, CityTwinEnvironment

FEATURE_NAMES = (
    "priority", "uncertainty", "pheromone", "novelty", "spread", "battery",
    "communication", "congestion", "visits", "target_distance", "distance",
    "swarm_memory", "frontier", "propagation_gradient", "return_progress",
)
VALUE_FEATURE_NAMES = (
    "bias", "target_discovery", "coverage", "mean_uncertainty",
    "mean_battery", "redundancy", "step_fraction",
)


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    if values.size == 0:
        return values
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
    values = values / max(0.10, float(temperature))
    values -= float(np.max(values))
    weights = np.exp(values)
    total = float(np.sum(weights))
    if np.isfinite(total) and total > 1e-12:
        return weights / total
    return np.ones_like(values) / len(values)


def feature_index(name: str) -> int:
    return FEATURE_NAMES.index(name)


class PPOResidualMixin:
    residual_l2 = 1e-4
    critic_l2 = 1e-4

    def __init__(self, *args: Any, model_path: str | Path | None = None, **kwargs: Any) -> None:
        super().__init__(*args, model_path=model_path, **kwargs)
        self.config.epsilon = 0.0
        self.residual_weights = np.zeros(len(FEATURE_NAMES), dtype=float)
        self.critic_weights = np.zeros(len(VALUE_FEATURE_NAMES), dtype=float)
        self._pending_decisions: list[dict[str, Any]] = []
        self.forced_fallbacks = 0
        if model_path:
            self._load_trainable_checkpoint(model_path)

    def _load_trainable_checkpoint(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = list(payload.get("ppo_residual_weights", []))
            names = list(payload.get("feature_names", []))
            if names and len(names) == len(values):
                for name, value in zip(names, values):
                    if name in FEATURE_NAMES:
                        self.residual_weights[feature_index(name)] = float(value)
            elif len(values) == len(FEATURE_NAMES):
                self.residual_weights = np.asarray(values, dtype=float)
            critic = list(payload.get("critic_weights", []))
            if len(critic) == len(VALUE_FEATURE_NAMES):
                self.critic_weights = np.asarray(critic, dtype=float)
        except (OSError, ValueError, TypeError):
            return

    def _features(
        self,
        env: CityTwinEnvironment,
        cell: Cell,
        current: Cell,
        positions: Dict[int, Cell],
        agent_id: int,
    ) -> dict[str, float]:
        f = super()._features(env, cell, current, positions, agent_id)
        confidence = float(np.clip(1.0 - env.uncertainty_map[cell], 0.0, 1.0))
        f["priority"] = float(env.observation_map[cell]) * confidence
        f["target_distance"] = 0.0
        return f

    def _extra_feature_values(
        self,
        env: CityTwinEnvironment,
        cell: Cell,
        current: Cell,
    ) -> tuple[float, float, float]:
        return 0.0, 0.0, 0.0

    @staticmethod
    def _return_progress(
        env: CityTwinEnvironment,
        cell: Cell,
        current: Cell,
        battery: float,
    ) -> float:
        pressure = float(np.clip((0.45 - battery) / 0.45, 0.0, 1.0))
        if pressure <= 0.0:
            return 0.0
        progress = env.nearest_base_distance(current) - env.nearest_base_distance(cell)
        return float(np.clip(progress, -1.0, 1.0) * pressure)

    def _feature_vector(
        self,
        env: CityTwinEnvironment,
        cell: Cell,
        current: Cell,
        positions: Dict[int, Cell],
        agent_id: int,
    ) -> np.ndarray:
        f = self._features(env, cell, current, positions, agent_id)
        memory, frontier, propagation = self._extra_feature_values(env, cell, current)
        return np.asarray(
            [
                f["priority"], f["uncertainty"], f["pheromone"], f["novelty"],
                f["spread"], f["battery"], f["communication"], -f["congestion"],
                -min(5.0, f["visits"]) / 5.0, -f["target_distance"], -f["distance"],
                memory, frontier, propagation,
                self._return_progress(env, cell, current, f["battery"]),
            ],
            dtype=float,
        )

    @staticmethod
    def _critic_features(env: CityTwinEnvironment) -> np.ndarray:
        coverage = len(env.visited - env.restricted_zones) / env.traversable_cell_count
        batteries = [state.battery_level / 100.0 for state in env.agents.values()]
        return np.asarray(
            [
                1.0,
                env.weighted_target_discovery(),
                float(np.clip(coverage, 0.0, 1.0)),
                float(np.mean(env.uncertainty_map)),
                float(np.mean(batteries)) if batteries else 0.0,
                env.redundant_coverage(),
                float(env.steps / max(1, env.max_steps)),
            ],
            dtype=float,
        )

    def reset_episode_state(self) -> None:
        self._pending_decisions = []
        self.forced_fallbacks = 0

    def _choose_with_residual(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        current: Cell,
        candidates: list[Cell],
        base_scores: np.ndarray,
        features: np.ndarray,
    ) -> Cell:
        probabilities = softmax(
            base_scores + features @ self.residual_weights,
            self.config.temperature,
        )
        index = int(self.rng.choice(len(candidates), p=probabilities))
        value_features = self._critic_features(env)
        selected = candidates[index]
        self._pending_decisions.append(
            {
                "agent_id": int(agent_id),
                "features": features.copy(),
                "base_scores": base_scores.copy(),
                "action_index": index,
                "old_probability": float(max(probabilities[index], 1e-12)),
                "value_features": value_features,
                "old_value": float(value_features @ self.critic_weights),
                "selected_cell": selected,
                "selected_action": env.cell_to_action(current, selected),
            }
        )
        self.last_entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
        self.last_confidence = float(np.max(probabilities))
        self.last_distribution = {str(i): float(v) for i, v in enumerate(probabilities)}
        self.last_selected_cell = selected
        key = f"{selected[0]},{selected[1]}"
        self.action_counts[key] = self.action_counts.get(key, 0) + 1
        return selected

    def _propose(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        positions: Dict[int, Cell],
        candidates: list[Cell] | None = None,
    ) -> Cell:
        current = env.agents[agent_id].position
        candidates = list(candidates) if candidates is not None else env.get_neighbors(current)
        base_scores = np.asarray(
            [super()._score(env, cell, current, positions, agent_id) for cell in candidates],
            dtype=float,
        )
        features = np.vstack(
            [self._feature_vector(env, cell, current, positions, agent_id) for cell in candidates]
        )
        return self._choose_with_residual(
            env, agent_id, current, candidates, base_scores, features
        )

    def act(self, env: CityTwinEnvironment) -> Dict[int, str]:
        positions = env.get_positions()
        planned: Dict[int, Cell] = {}
        actions: Dict[int, str] = {}
        for agent_id in sorted(env.agents):
            state = env.agents[agent_id]
            if state.done:
                actions[agent_id] = "STAY"
                planned[agent_id] = state.position
                continue
            safe_actions = self.monitor.safe_actions(
                env, agent_id, planned, count_masked=True
            )
            if not safe_actions:
                preferred = "STAY"
                _, violations, _ = self.monitor.is_action_safe(
                    env, agent_id, preferred, planned
                )
                if violations:
                    self.monitor.record_intervention(env, agent_id, preferred, violations)
                action = self.monitor.nearest_safe_action(
                    env, agent_id, planned, preferred
                )
                self.forced_fallbacks += 1
                actions[agent_id] = action
                planned[agent_id] = env.next_position(state.position, action)
                positions[agent_id] = planned[agent_id]
                continue
            candidates = [env.next_position(state.position, action) for action in safe_actions]
            candidate = self._propose(
                env, agent_id, positions, candidates=candidates
            )
            action = env.cell_to_action(state.position, candidate)
            actions[agent_id] = action
            planned[agent_id] = candidate
            positions[agent_id] = candidate
        return actions

    def drain_decisions(self) -> list[dict[str, Any]]:
        decisions = self._pending_decisions
        self._pending_decisions = []
        return decisions

    @staticmethod
    def _parse_sample(sample: tuple[Any, ...]) -> tuple[dict[str, Any], float, float]:
        if len(sample) == 2:
            decision, advantage = sample
            target = float(decision.get("old_value", 0.0)) + float(advantage)
            return decision, float(advantage), target
        if len(sample) == 3:
            decision, advantage, value_target = sample
            return decision, float(advantage), float(value_target)
        raise ValueError("PPO sample must contain decision, advantage, and optional value target")

    def ppo_update(
        self,
        samples: Iterable[tuple[Any, ...]],
        *,
        learning_rate: float = 0.03,
        critic_learning_rate: float | None = None,
        clip_ratio: float = 0.20,
        epochs: int = 4,
        max_grad_norm: float = 2.0,
    ) -> dict[str, float]:
        parsed = [self._parse_sample(sample) for sample in samples]
        if not parsed:
            return self._update_diagnostics(0.0, 0.0, 0.0, 0.0, 0.0)

        advantages = np.asarray([adv for _, adv, _ in parsed], dtype=float)
        std = float(np.std(advantages))
        if std > 1e-8:
            advantages = (advantages - float(np.mean(advantages))) / std
        critic_lr = float(critic_learning_rate or 0.5 * learning_rate)
        policy_loss = value_loss = approx_kl = behavior_kl = 0.0
        clipped = considered = 0

        for _ in range(max(1, epochs)):
            action_grads: list[np.ndarray] = []
            behavior_grads: list[np.ndarray] = []
            critic_grads: list[np.ndarray] = []
            losses: list[float] = []
            value_losses: list[float] = []
            action_kls: list[float] = []
            behavior_kls: list[float] = []
            clipped_epoch = 0

            for (decision, _, target), advantage in zip(parsed, advantages):
                features = np.asarray(decision["features"], dtype=float)
                base = np.asarray(decision["base_scores"], dtype=float)
                action = int(decision["action_index"])
                old_p = float(max(decision["old_probability"], 1e-12))
                probs = softmax(
                    base + features @ self.residual_weights,
                    self.config.temperature,
                )
                new_p = float(max(probs[action], 1e-12))
                ratio = new_p / old_p
                clipped_ratio = float(
                    np.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                )
                unclipped_obj = ratio * float(advantage)
                clipped_obj = clipped_ratio * float(advantage)
                use_clipped = clipped_obj < unclipped_obj
                losses.append(-min(unclipped_obj, clipped_obj))
                if use_clipped:
                    action_grads.append(np.zeros_like(self.residual_weights))
                    clipped_epoch += 1
                else:
                    expected = probs @ features
                    action_grads.append(
                        float(advantage)
                        * ratio
                        * (features[action] - expected)
                        / max(0.10, float(self.config.temperature))
                    )
                action_kls.append(np.log(old_p) - np.log(new_p))

                if "behavior_base_scores" in decision and hasattr(self, "behavior_bias"):
                    bbase = np.asarray(decision["behavior_base_scores"], dtype=float)
                    baction = int(decision["behavior_index"])
                    bold = float(max(decision["behavior_old_probability"], 1e-12))
                    bprobs = softmax(
                        bbase + self.behavior_bias[: len(bbase)],
                        self.config.temperature,
                    )
                    bnew = float(max(bprobs[baction], 1e-12))
                    bratio = bnew / bold
                    bclip = float(
                        np.clip(bratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                    )
                    b_use_clipped = (
                        bclip * float(advantage) < bratio * float(advantage)
                    )
                    bgrad = np.zeros_like(self.behavior_bias)
                    if b_use_clipped:
                        clipped_epoch += 1
                    else:
                        one_hot = np.zeros(len(bbase), dtype=float)
                        one_hot[baction] = 1.0
                        bgrad[: len(bbase)] = (
                            float(advantage)
                            * bratio
                            * (one_hot - bprobs)
                            / max(0.10, float(self.config.temperature))
                        )
                    behavior_grads.append(bgrad)
                    behavior_kls.append(np.log(bold) - np.log(bnew))

                value_features = np.asarray(
                    decision.get(
                        "value_features",
                        np.zeros(len(VALUE_FEATURE_NAMES)),
                    ),
                    dtype=float,
                )
                value = float(value_features @ self.critic_weights)
                error = float(target - value)
                critic_grads.append(error * value_features)
                value_losses.append(0.5 * error * error)

            actor_grad = (
                np.mean(action_grads, axis=0)
                - self.residual_l2 * self.residual_weights
            )
            actor_norm = float(np.linalg.norm(actor_grad))
            if actor_norm > max_grad_norm:
                actor_grad *= max_grad_norm / max(actor_norm, 1e-12)
            self.residual_weights += float(learning_rate) * actor_grad

            if behavior_grads and hasattr(self, "behavior_bias"):
                behavior_grad = (
                    np.mean(behavior_grads, axis=0)
                    - self.residual_l2 * self.behavior_bias
                )
                behavior_norm = float(np.linalg.norm(behavior_grad))
                if behavior_norm > max_grad_norm:
                    behavior_grad *= max_grad_norm / max(behavior_norm, 1e-12)
                self.behavior_bias += float(learning_rate) * behavior_grad

            critic_grad = (
                np.mean(critic_grads, axis=0)
                - self.critic_l2 * self.critic_weights
            )
            critic_norm = float(np.linalg.norm(critic_grad))
            if critic_norm > max_grad_norm:
                critic_grad *= max_grad_norm / max(critic_norm, 1e-12)
            self.critic_weights += critic_lr * critic_grad

            policy_loss = float(np.mean(losses))
            value_loss = float(np.mean(value_losses))
            approx_kl = float(np.mean(action_kls))
            behavior_kl = (
                float(np.mean(behavior_kls)) if behavior_kls else 0.0
            )
            clipped += clipped_epoch
            considered += len(parsed) + len(behavior_grads)

        return self._update_diagnostics(
            policy_loss,
            value_loss,
            approx_kl,
            behavior_kl,
            clipped / max(1, considered),
        )

    def _update_diagnostics(
        self,
        policy_loss: float,
        value_loss: float,
        approx_kl: float,
        behavior_kl: float,
        clip_fraction: float,
    ) -> dict[str, float]:
        return {
            "loss": float(policy_loss + 0.5 * value_loss),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "approx_kl": float(approx_kl),
            "behavior_kl": float(behavior_kl),
            "clip_fraction": float(clip_fraction),
            "weight_norm": float(np.linalg.norm(self.residual_weights)),
            "critic_weight_norm": float(np.linalg.norm(self.critic_weights)),
            "behavior_weight_norm": float(
                np.linalg.norm(getattr(self, "behavior_bias", np.zeros(1)))
            ),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "safeswarm-safe-masked-ppo-v2",
            "strategy": self.name,
            "feature_names": list(FEATURE_NAMES),
            "ppo_residual_weights": self.residual_weights.tolist(),
            "value_feature_names": list(VALUE_FEATURE_NAMES),
            "critic_weights": self.critic_weights.tolist(),
            "params": {
                key: float(getattr(self.config, key))
                for key in vars(self.config)
            },
            "metadata": metadata or {},
        }
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def diagnostics(self) -> dict[str, Any]:
        data = super().diagnostics()
        data.update(
            {
                "ppo_residual_weight_norm": float(
                    np.linalg.norm(self.residual_weights)
                ),
                "critic_weight_norm": float(np.linalg.norm(self.critic_weights)),
                "safety_mask_rejections": int(self.monitor.mask_rejections),
                "forced_fallbacks": int(self.forced_fallbacks),
            }
        )
        return data


class TrainableIPPOPolicy(PPOResidualMixin, IPPOPolicy):
    name = "IPPO-Safe"


class TrainableMAPPOPolicy(PPOResidualMixin, MAPPOPolicy):
    name = "MAPPO-Safe"


class TrainableHAPPOPolicy(PPOResidualMixin, HAPPOPolicy):
    name = "HAPPO-Safe"
