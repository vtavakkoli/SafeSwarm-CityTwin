"""SafeSwarm PPO-v5 optimizer and observable coordination prior.

v5 keeps the v3 minibatch PPO implementation but fixes three issues exposed by
real held-out testing:

1. checkpoint policies were still sampled stochastically during evaluation,
   causing high-redundancy random walks;
2. IPPO/MAPPO/HAPPO left the global-memory/frontier feature slots at zero;
3. non-GRPO policies had no generic imitation interface for strong teachers.

The new coordination prior is observable-only: it assigns agents to diverse
frontier/evidence goals and exposes progress toward those goals through the
existing residual feature vector.  Training remains stochastic; loaded
checkpoints evaluate deterministically by default.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np

from src.agents.marl_baselines import HAPPOPolicy, IPPOPolicy, MAPPOPolicy
from src.agents.observable_utils import (
    frontier_map,
    observable_guidance_cells,
    observable_search_utility,
    observable_target_distance,
)
from src.agents.safe_ppo_core import FEATURE_NAMES, PPOResidualMixin, softmax
from src.environment.city_twin import Cell, CityTwinEnvironment


def _feature_index(name: str) -> int:
    return FEATURE_NAMES.index(name)


class PPOV3Mixin:
    """Minibatch PPO plus v5 deterministic evaluation/global coordination."""

    def __init__(
        self,
        *args: Any,
        deterministic_eval: bool | None = None,
        coordination_interval: int = 6,
        **kwargs: Any,
    ) -> None:
        model_path = kwargs.get("model_path")
        super().__init__(*args, **kwargs)
        self.deterministic_eval = bool(model_path) if deterministic_eval is None else bool(deterministic_eval)
        self.coordination_interval = max(1, int(coordination_interval))
        self._coordination_goals: dict[int, Cell] = {}
        self._coordination_last_step = -10_000
        self._coordination_refreshes = 0

        # Give every trainable policy a small, auditable anti-redundancy prior.
        # GRPO has its own memory initialisation and therefore keeps its values.
        if not model_path and not hasattr(self, "swarm_memory"):
            self.residual_weights[_feature_index("swarm_memory")] = 0.12
            self.residual_weights[_feature_index("frontier")] = 0.22
            self.residual_weights[_feature_index("propagation_gradient")] = 0.28
            self.residual_weights[_feature_index("visits")] = 0.20

    def reset_episode_state(self) -> None:
        super().reset_episode_state()
        self._coordination_goals = {}
        self._coordination_last_step = -10_000
        self._coordination_refreshes = 0

    def _features(self, env, cell, current, positions, agent_id):
        features = super()._features(env, cell, current, positions, agent_id)
        features["target_distance"] = float(
            observable_target_distance(env, cell) / max(1.0, 2.0 * env.grid_size)
        )
        return features

    @staticmethod
    def _manhattan(a: Cell, b: Cell) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _refresh_coordination_goals(self, env: CityTwinEnvironment) -> None:
        if (
            self._coordination_goals
            and env.steps - self._coordination_last_step < self.coordination_interval
            and all(aid in self._coordination_goals for aid in env.agents if not env.agents[aid].done)
        ):
            return

        active = [aid for aid in sorted(env.agents) if not env.agents[aid].done]
        candidates = observable_guidance_cells(
            env, limit=max(48, len(active) * 12)
        )
        if not candidates:
            self._coordination_goals = {
                aid: env.agents[aid].position for aid in active
            }
            self._coordination_last_step = int(env.steps)
            self._coordination_refreshes += 1
            return

        chosen: list[Cell] = []
        goals: dict[int, Cell] = {}
        min_sep = max(2.0, env.grid_size / max(4.0, np.sqrt(max(1, len(active))) * 2.2))
        # Lower-battery agents select first, favoring reachable goals.
        active.sort(key=lambda aid: (env.agents[aid].battery_level, aid))
        for aid in active:
            current = env.agents[aid].position
            best = max(
                candidates,
                key=lambda cell: (
                    1.20 * observable_search_utility(env, cell)
                    - 0.50 * self._manhattan(current, cell) / max(1.0, env.grid_size)
                    - sum(
                        max(0.0, min_sep - self._manhattan(cell, other)) / min_sep
                        for other in chosen
                    ),
                    -float(env.visit_counts[cell]),
                ),
            )
            goals[aid] = best
            chosen.append(best)
            candidates = [cell for cell in candidates if cell != best] or candidates

        self._coordination_goals = goals
        self._coordination_last_step = int(env.steps)
        self._coordination_refreshes += 1

    def _coordination_values(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        cell: Cell,
        current: Cell,
    ) -> tuple[float, float, float]:
        self._refresh_coordination_goals(env)
        guidance = float(np.tanh(observable_search_utility(env, cell) / 3.0))
        frontier = float(frontier_map(env)[cell])
        goal = self._coordination_goals.get(agent_id, current)
        progress = float(
            np.clip(
                self._manhattan(current, goal) - self._manhattan(cell, goal),
                -1.0,
                1.0,
            )
        )
        return guidance, frontier, progress

    def _feature_vector(
        self,
        env: CityTwinEnvironment,
        cell: Cell,
        current: Cell,
        positions: Dict[int, Cell],
        agent_id: int,
    ) -> np.ndarray:
        f = self._features(env, cell, current, positions, agent_id)
        base_memory, base_frontier, base_propagation = super()._extra_feature_values(
            env, cell, current
        )
        guidance, global_frontier, goal_progress = self._coordination_values(
            env, agent_id, cell, current
        )

        if hasattr(self, "swarm_memory"):
            # GRPO keeps its signed memory while also receiving a weaker team goal.
            memory = float(base_memory) + 0.20 * guidance
            frontier = max(float(base_frontier), global_frontier)
            propagation = float(base_propagation) + 0.35 * goal_progress
        else:
            memory = guidance
            frontier = global_frontier
            propagation = goal_progress

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
        if self.deterministic_eval:
            index = int(np.argmax(probabilities))
        else:
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

    # ---------------------------------------------------------------
    # Generic teacher distillation for IPPO / MAPPO / HAPPO.
    # GRPO overrides both methods with its hierarchical behavior labels.
    # ---------------------------------------------------------------
    def imitation_example(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        positions: Dict[int, Cell],
        candidates: list[Cell],
        target_cell: Cell,
    ) -> dict[str, Any] | None:
        if not candidates or target_cell not in candidates:
            return None
        current = env.agents[agent_id].position
        score_fn = super()._score
        base_scores = np.asarray(
            [score_fn(env, cell, current, positions, agent_id) for cell in candidates],
            dtype=float,
        )
        features = np.vstack(
            [self._feature_vector(env, cell, current, positions, agent_id) for cell in candidates]
        )
        return {
            "features": features,
            "base_scores": base_scores,
            "target_index": int(candidates.index(target_cell)),
        }

    def imitation_update(
        self,
        examples: list[dict[str, Any]],
        *,
        learning_rate: float = 0.008,
        epochs: int = 1,
        label_smoothing: float = 0.06,
        max_grad_norm: float = 0.5,
    ) -> dict[str, float]:
        if not examples:
            return {
                "imitation_loss": 0.0,
                "imitation_accuracy": 0.0,
                "imitation_examples": 0.0,
            }
        losses: list[float] = []
        correct = total = 0
        for _ in range(max(1, int(epochs))):
            order = self.rng.permutation(len(examples))
            grads: list[np.ndarray] = []
            for raw_idx in order:
                example = examples[int(raw_idx)]
                features = np.asarray(example["features"], dtype=float)
                base = np.asarray(example["base_scores"], dtype=float)
                target = int(example["target_index"])
                probs = softmax(base + features @ self.residual_weights, self.config.temperature)
                smooth = float(np.clip(label_smoothing, 0.0, 0.30))
                target_probs = np.full(len(probs), smooth / max(1, len(probs)), dtype=float)
                target_probs[target] += 1.0 - smooth
                grads.append(
                    (target_probs @ features - probs @ features)
                    / max(0.10, float(self.config.temperature))
                )
                losses.append(float(-np.log(max(probs[target], 1e-12))))
                correct += int(np.argmax(probs) == target)
                total += 1
            if grads:
                grad = np.mean(grads, axis=0)
                norm = float(np.linalg.norm(grad))
                if norm > max_grad_norm:
                    grad *= max_grad_norm / max(norm, 1e-12)
                self.residual_weights += float(learning_rate) * grad
        return {
            "imitation_loss": float(np.mean(losses)) if losses else 0.0,
            "imitation_accuracy": float(correct / max(1, total)),
            "imitation_examples": float(len(examples)),
        }

    @staticmethod
    def _entropy_gradient(features: np.ndarray, probabilities: np.ndarray, temperature: float) -> np.ndarray:
        expected = probabilities @ features
        centered = features - expected
        coeff = -probabilities * (np.log(probabilities + 1e-12) + 1.0)
        return np.sum(coeff[:, None] * centered, axis=0) / max(0.10, float(temperature))

    @staticmethod
    def _categorical_entropy_gradient(probabilities: np.ndarray, temperature: float) -> np.ndarray:
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
        advantages = (
            (advantages - float(np.mean(advantages))) / std
            if std > 1e-8 else advantages - float(np.mean(advantages))
        )

        critic_lr = float(critic_learning_rate or 0.75 * learning_rate)
        n = len(parsed)
        batch_size = max(32, min(int(minibatch_size), n))
        policy_losses: list[float] = []
        value_losses: list[float] = []
        action_kls: list[float] = []
        behavior_kls: list[float] = []
        entropies: list[float] = []
        behavior_entropies: list[float] = []
        clipped = considered = optimizer_steps = 0

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
                    actor_grads.append(
                        policy_grad
                        + float(entropy_coef) * self._entropy_gradient(features, probs, self.config.temperature)
                    )
                    entropies.append(float(-np.sum(probs * np.log(probs + 1e-12))))
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
                "deterministic_eval": float(self.deterministic_eval),
                "coordination_refreshes": float(self._coordination_refreshes),
            }
        )
        return diagnostics

    def diagnostics(self) -> dict[str, Any]:
        data = super().diagnostics()
        data.update(
            {
                "deterministic_eval": bool(self.deterministic_eval),
                "coordination_goal_count": len(self._coordination_goals),
                "coordination_refreshes": int(self._coordination_refreshes),
            }
        )
        return data


class TrainableIPPOPolicy(PPOV3Mixin, PPOResidualMixin, IPPOPolicy):
    name = "IPPO-Safe"


class TrainableMAPPOPolicy(PPOV3Mixin, PPOResidualMixin, MAPPOPolicy):
    name = "MAPPO-Safe"


class TrainableHAPPOPolicy(PPOV3Mixin, PPOResidualMixin, HAPPOPolicy):
    name = "HAPPO-Safe"
