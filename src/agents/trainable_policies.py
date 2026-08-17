"""Public registry for trainable SafeSwarm v4 policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.agents.grpo_v3 import TrainableGRPOMemoryPolicy
from src.agents.ppo_v3 import (
    TrainableHAPPOPolicy,
    TrainableIPPOPolicy,
    TrainableMAPPOPolicy,
)
from src.agents.safe_ppo_core import (
    FEATURE_NAMES,
    VALUE_FEATURE_NAMES,
    PPOResidualMixin,
    feature_index,
)
from src.agents.sparx_pattern import SPARXPolicy

TRAINABLE_POLICY_CLASSES = {
    "GRPO-Safe": TrainableGRPOMemoryPolicy,
    "IPPO-Safe": TrainableIPPOPolicy,
    "MAPPO-Safe": TrainableMAPPOPolicy,
    "HAPPO-Safe": TrainableHAPPOPolicy,
}


def checkpoint_path(model_dir: str | Path, strategy: str) -> Path:
    safe_name = strategy.lower().replace("-", "_")
    return Path(model_dir) / f"{safe_name}.json"


def sparx_checkpoint_path(model_dir: str | Path, pattern: str | None = None) -> Path:
    suffix = "" if pattern is None else f"_{pattern}"
    return Path(model_dir) / f"sparx{suffix}_safe.json"


def _grpo_ablation(seed: int, path: Path, mode: str) -> TrainableGRPOMemoryPolicy:
    policy = TrainableGRPOMemoryPolicy(seed=seed, model_path=path)
    if mode == "no_memory":
        policy.memory_enabled = False
        policy.propagation_steps = 0
        for name in ("swarm_memory", "frontier", "propagation_gradient"):
            policy.residual_weights[feature_index(name)] = 0.0
    elif mode == "no_propagation":
        policy.propagation_steps = 0
        policy.residual_weights[feature_index("propagation_gradient")] = 0.0
    elif mode == "no_learned_behavior":
        policy.behavior_bias[:] = 0.0
        policy.behavior_weights[:] = 0.0
    else:
        raise ValueError(f"Unknown GRPO ablation mode: {mode}")
    return policy


def evaluation_factories(seed: int, model_dir: str | Path | None = None) -> dict[str, Any]:
    """Return fixed + trained factories for validation/held-out evaluation.

    PPO checkpoints remain mandatory when ``model_dir`` is supplied. SPARX is
    backward compatible: validation-selected SPARX checkpoints are loaded when
    present, while older v3 checkpoint directories can still be evaluated.
    """

    from src.agents.registry import strategy_factories

    factories = strategy_factories(seed=seed)
    paths: dict[str, Path | None] = {
        strategy: checkpoint_path(model_dir, strategy) if model_dir else None
        for strategy in TRAINABLE_POLICY_CLASSES
    }
    if model_dir:
        missing = [
            strategy for strategy, path in paths.items()
            if path is None or not path.exists()
        ]
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
        factories["GRPO-Safe-Ablation-NoLearnedBehavior"] = (
            lambda path=grpo_path: _grpo_ablation(seed, path, "no_learned_behavior")
        )

    if model_dir:
        selected = sparx_checkpoint_path(model_dir)
        if selected.exists():
            factories["SPARX-Safe"] = (
                lambda path=selected: SPARXPolicy(
                    seed=seed, model_path=path, strategy_name="SPARX-Safe"
                )
            )
        for pattern, label in (("x", "X"), ("plus", "Plus"), ("star", "Star")):
            path = sparx_checkpoint_path(model_dir, pattern)
            if path.exists():
                strategy = f"SPARX-{label}-Safe"
                factories[strategy] = (
                    lambda path=path, strategy=strategy: SPARXPolicy(
                        seed=seed, model_path=path, strategy_name=strategy
                    )
                )
    return factories


__all__ = [
    "FEATURE_NAMES",
    "VALUE_FEATURE_NAMES",
    "PPOResidualMixin",
    "TrainableGRPOMemoryPolicy",
    "TrainableIPPOPolicy",
    "TrainableMAPPOPolicy",
    "TrainableHAPPOPolicy",
    "SPARXPolicy",
    "TRAINABLE_POLICY_CLASSES",
    "checkpoint_path",
    "sparx_checkpoint_path",
    "evaluation_factories",
]
