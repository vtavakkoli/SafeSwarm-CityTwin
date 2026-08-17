"""Numerically stable PPO-v3 optimizer for SafeSwarm.

This keeps the repository NumPy-only while importing the training mechanics
that make BioSwarm effective: minibatch updates, entropy regularization,
smaller gradients, KL stopping and state-dependent high-level behavior support.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.agents.marl_baselines import HAPPOPolicy, IPPOPolicy, MAPPOPolicy
from src.agents.observable_utils import observable_target_distance
from src.agents.safe_ppo_core import PPOResidualMixin, softmax
from src.environment.city_twin import Cell, CityTwinEnvironment


class PPOV3Mixin:
    """Drop-in optimizer/actor override layered before ``PPOResidualMixin``."""

    def _features(self, env, cell, current, positions, agent_id):
        features = super()._features(env, cell, current, positions, agent_id)
        features["target_distance"] = float(
            observable_target_distance(env, cell) / max(1.0, 2.0 * env.grid_size)
        )
        return features

    def _choose_with_residual(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        current: Cell,
        candidates: list[Cell],
        base_scores: np.ndarray,
        features: np.ndarray,
    ) -> Cell:
        logits = base_scores + features @ self.residual_weights
        probabilities = softmax(logits, self.config.temperature)
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
        self.last_distribution = {str(i): float(value) for i, value in enumerate(probabilities)}
        self.last_selected_cell = selected
        key = f"{selected[0]},{selected[1]}"
        self.action_counts[key] = self.action_counts.get(key, 0) + 1
        return selected

    @staticmethod
    def _entropy_gradient(features: np.ndarray, probabilities: np.ndarray, temperature: float) -> np.ndarray:
        expected = probabilities @ features
        centered = features - expected
        coeff = -probabilities * (np.log(probabilities + 1e-12) + 1.0)
        return np.sum(coeff[:, None] * centered, axis=0) / max(0.10, float(temperature))

    @staticmethod
    def _categorical_entropy_gradient(probabilities: np.ndarray, temperature: float) -> np.ndarray:
        # Gradient of entropy with respect to additive category logits.
        log_term = np.log(probabilities + 1e-12) + 1.0
        weighted = probabilities * log_term
        return -(weighted - probabilities * float(np.sum(weighted))) / max(0.10, float(temperature))

    def ppo_update(
        self,
        samples: Iterable[tuple[Any, ...]],
        *,
        learning_rate: float = 0.004,
        critic_learning_rate: float | None = None,
        clip_ratio: float = 0.20,
        epochs: int = 6,
        max_grad_norm: float = 0.5,
        minibatch_size: int = 512,
        entropy_coef: float = 0.05,
        target_kl: float = 0.03,
    ) -> dict[str, float]:
        parsed = [self._parse_sample(sample) for sample in samples]
        if not parsed:
            data = self._update_diagnostics(0.0, 0.0, 0.0, 0.0, 0.0)
            data.update({"entropy": 0.0, "behavior_entropy": 0.0, "optimizer_steps": 0.0})
            return data

        advantages = np.asarray([adv for _, adv, _ in parsed], dtype=float)
        std = float(np.std(advantages))
        if std > 1e-8:
            advantages = (advantages - float(np.mean(advantages))) / std
        else:
            advantages = advantages - float(np.mean(advantages))

        critic_lr = float(critic_learning_rate or 0.75 * learning_rate)
        n = len(parsed)
        batch_size = max(32, min(int(minibatch_size), n))
        policy_losses: list[float] = []
        value_losses: list[float] = []
        action_kls: list[float] = []
        behavior_kls: list[float] = []
        entropies: list[float] = []
        behavior_entropies: list[float] = []
        clipped = 0
        considered = 0
        optimizer_steps = 0

        for _ in range(max(1, int(epochs))):
            order = self.rng.permutation(n)
            epoch_kls: list[float] = []
            for start in range(0, n, batch_size):
                indexes = order[start : start + batch_size]
                actor_grads: list[np.ndarray] = []
                critic_grads: list[np.ndarray] = []
                behavior_bias_grads: list[np.ndarray] = []
                behavior_weight_grads: list[np.ndarray] = []

                for idx in indexes:
                    decision, _, target = parsed[int(idx)]
                    advantage = float(advantages[int(idx)])
                    features = np.asarray(decision["features"], dtype=float)
                    base = np.asarray(decision["base_scores"], dtype=float)
                    action = int(decision["action_index"])
                    old_p = float(max(decision["old_probability"], 1e-12))
                    probs = softmax(base + features @ self.residual_weights, self.config.temperature)
                    new_p = float(max(probs[action], 1e-12))
                    ratio = new_p / old_p
                    clipped_ratio = float(np.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio))
                    unclipped_obj = ratio * advantage
                    clipped_obj = clipped_ratio * advantage
                    use_clipped = clipped_obj < unclipped_obj
                    policy_losses.append(-min(unclipped_obj, clipped_obj))
                    expected = probs @ features
                    grad_logp = (features[action] - expected) / max(0.10, float(self.config.temperature))
                    policy_grad = np.zeros_like(self.residual_weights) if use_clipped else advantage * ratio * grad_logp
                    if use_clipped:
                        clipped += 1
                    entropy_grad = self._entropy_gradient(features, probs, self.config.temperature)
                    actor_grads.append(policy_grad + float(entropy_coef) * entropy_grad)
                    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
                    entropies.append(entropy)
                    kl = float(np.log(old_p) - np.log(new_p))
                    action_kls.append(kl)
                    epoch_kls.append(kl)

                    if "behavior_base_scores" in decision and hasattr(self, "behavior_bias"):
                        bbase = np.asarray(decision["behavior_base_scores"], dtype=float)
                        baction = int(decision["behavior_index"])
                        bold = float(max(decision["behavior_old_probability"], 1e-12))
                        bfeatures = np.asarray(decision.get("behavior_features", []), dtype=float)
                        blogits = bbase + self.behavior_bias[: len(bbase)]
                        if bfeatures.size and hasattr(self, "behavior_weights"):
                            blogits = blogits + bfeatures @ self.behavior_weights[:, : len(bbase)]
                        bprobs = softmax(blogits, self.config.temperature)
                        bnew = float(max(bprobs[baction], 1e-12))
                        bratio = bnew / bold
                        bclip = float(np.clip(bratio, 1.0 - clip_ratio, 1.0 + clip_ratio))
                        b_use_clipped = bclip * advantage < bratio * advantage
                        category_grad = np.zeros_like(bprobs)
                        if not b_use_clipped:
                            category_grad[baction] = 1.0
                            category_grad -= bprobs
                            category_grad *= advantage * bratio / max(0.10, float(self.config.temperature))
                        else:
                            clipped += 1
                        category_grad += float(entropy_coef) * self._categorical_entropy_gradient(
                            bprobs, self.config.temperature
                        )
                        bias_grad = np.zeros_like(self.behavior_bias)
                        bias_grad[: len(bbase)] = category_grad
                        behavior_bias_grads.append(bias_grad)
                        if bfeatures.size and hasattr(self, "behavior_weights"):
                            wgrad = np.zeros_like(self.behavior_weights)
                            wgrad[:, : len(bbase)] = np.outer(bfeatures, category_grad)
                            behavior_weight_grads.append(wgrad)
                        behavior_kls.append(float(np.log(bold) - np.log(bnew)))
                        behavior_entropies.append(float(-np.sum(bprobs * np.log(bprobs + 1e-12))))

                    value_features = np.asarray(decision.get("value_features", []), dtype=float)
                    if value_features.size == self.critic_weights.size:
                        value = float(value_features @ self.critic_weights)
                        error = float(target - value)
                        critic_grads.append(error * value_features)
                        value_losses.append(0.5 * error * error)
                    considered += 1

                if actor_grads:
                    grad = np.mean(actor_grads, axis=0) - self.residual_l2 * self.residual_weights
                    norm = float(np.linalg.norm(grad))
                    if norm > max_grad_norm:
                        grad *= max_grad_norm / max(norm, 1e-12)
                    self.residual_weights += float(learning_rate) * grad

                if behavior_bias_grads:
                    grad = np.mean(behavior_bias_grads, axis=0) - self.residual_l2 * self.behavior_bias
                    norm = float(np.linalg.norm(grad))
                    if norm > max_grad_norm:
                        grad *= max_grad_norm / max(norm, 1e-12)
                    self.behavior_bias += float(learning_rate) * grad

                if behavior_weight_grads and hasattr(self, "behavior_weights"):
                    grad = np.mean(behavior_weight_grads, axis=0) - self.residual_l2 * self.behavior_weights
                    norm = float(np.linalg.norm(grad))
                    if norm > max_grad_norm:
                        grad *= max_grad_norm / max(norm, 1e-12)
                    self.behavior_weights += float(learning_rate) * grad

                if critic_grads:
                    grad = np.mean(critic_grads, axis=0) - self.critic_l2 * self.critic_weights
                    norm = float(np.linalg.norm(grad))
                    if norm > max_grad_norm:
                        grad *= max_grad_norm / max(norm, 1e-12)
                    self.critic_weights += critic_lr * grad
                optimizer_steps += 1

            if epoch_kls and abs(float(np.mean(epoch_kls))) > float(target_kl):
                break

        diagnostics = self._update_diagnostics(
            float(np.mean(policy_losses)) if policy_losses else 0.0,
            float(np.mean(value_losses)) if value_losses else 0.0,
            float(np.mean(action_kls)) if action_kls else 0.0,
            float(np.mean(behavior_kls)) if behavior_kls else 0.0,
            float(clipped / max(1, considered)),
        )
        diagnostics.update(
            {
                "entropy": float(np.mean(entropies)) if entropies else 0.0,
                "behavior_entropy": float(np.mean(behavior_entropies)) if behavior_entropies else 0.0,
                "optimizer_steps": float(optimizer_steps),
            }
        )
        return diagnostics


class TrainableIPPOPolicy(PPOV3Mixin, PPOResidualMixin, IPPOPolicy):
    name = "IPPO-Safe"


class TrainableMAPPOPolicy(PPOV3Mixin, PPOResidualMixin, MAPPOPolicy):
    name = "MAPPO-Safe"


class TrainableHAPPOPolicy(PPOV3Mixin, PPOResidualMixin, HAPPOPolicy):
    name = "HAPPO-Safe"
