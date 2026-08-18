"""Public registry for trainable and validation-selected SafeSwarm policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.agents.ears_v6 import (
    EARSNegativePheromonePolicy,
    EARSPolicy,
    HMAPPOEARSPolicy,
)
from src.agents.grpo_v3 import TrainableGRPOMemoryPolicy
from src.agents.ppo_v3 import (
    TrainableHAPPOPolicy,
    TrainableIPPOPolicy,
    TrainableMAPPOPolicy,
)
from src.agents.prism_ant import PRISMAntPolicy
from src.agents.prism_pattern import PRISMPolicy
from src.agents.safe_ppo_core import (
    FEATURE_NAMES,
    VALUE_FEATURE_NAMES,
    PPOResidualMixin,
    feature_index,
)

TRAINABLE_POLICY_CLASSES = {
    "GRPO-Safe": TrainableGRPOMemoryPolicy,
    "IPPO-Safe": TrainableIPPOPolicy,
    "MAPPO-Safe": TrainableMAPPOPolicy,
    "HAPPO-Safe": TrainableHAPPOPolicy,
}


def checkpoint_path(model_dir: str | Path, strategy: str) -> Path:
    safe_name = strategy.lower().replace("-", "_")
    return Path(model_dir) / f"{safe_name}.json"


def prism_checkpoint_path(model_dir: str | Path, pattern: str | None = None) -> Path:
    suffix = "" if pattern is None else f"_{pattern}"
    return Path(model_dir) / f"prism{suffix}_safe.json"


def sparx_checkpoint_path(model_dir: str | Path, pattern: str | None = None) -> Path:
    """Deprecated v4 path helper retained only for checkpoint migration."""
    suffix = "" if pattern is None else f"_{pattern}"
    return Path(model_dir) / f"sparx{suffix}_safe.json"


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


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
    """Return fixed + validation-selected factories for frozen evaluation."""

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
        factories[strategy] = lambda cls=cls, path=path: cls(
            seed=seed, model_path=path, deterministic_eval=True
        )

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
        root = Path(model_dir)
        selected = _first_existing(
            prism_checkpoint_path(root), sparx_checkpoint_path(root)
        )
        if selected is not None:
            factories["PRISM-Safe"] = (
                lambda path=selected: PRISMPolicy(
                    seed=seed, model_path=path, strategy_name="PRISM-Safe"
                )
            )

        for pattern, label in (("x", "X"), ("plus", "Plus"), ("star", "Star")):
            path = _first_existing(
                prism_checkpoint_path(root, pattern),
                sparx_checkpoint_path(root, pattern),
            )
            if path is not None:
                strategy = f"PRISM-{label}-Safe"
                factories[strategy] = (
                    lambda path=path, strategy=strategy: PRISMPolicy(
                        seed=seed, model_path=path, strategy_name=strategy
                    )
                )

        hybrid = root / "prism_ant_safe.json"
        if hybrid.exists():
            factories["PRISM-Ant-Safe"] = (
                lambda path=hybrid: PRISMAntPolicy(
                    seed=seed, model_path=path, strategy_name="PRISM-Ant-Safe"
                )
            )

        ears = root / "ears_safe.json"
        if ears.exists():
            factories["EARS-Safe"] = (
                lambda path=ears: EARSPolicy(
                    seed=seed, model_path=path, strategy_name="EARS-Safe"
                )
            )
        ears_np = root / "ears_np_safe.json"
        if ears_np.exists():
            factories["EARS-NP-Safe"] = (
                lambda path=ears_np: EARSNegativePheromonePolicy(
                    seed=seed, model_path=path, strategy_name="EARS-NP-Safe"
                )
            )
        h_ears = root / "h_mappo_ears_safe.json"
        if h_ears.exists():
            mappo = root / "mappo_safe.json"
            factories["H-MAPPO-EARS-Safe"] = (
                lambda path=h_ears, mappo=mappo: HMAPPOEARSPolicy(
                    seed=seed,
                    model_path=path,
                    mappo_model_path=mappo,
                    strategy_name="H-MAPPO-EARS-Safe",
                )
            )

    for name in list(factories):
        if name.startswith("SPARX"):
            factories.pop(name, None)
    return factories


__all__ = [
    "FEATURE_NAMES",
    "VALUE_FEATURE_NAMES",
    "PPOResidualMixin",
    "TrainableGRPOMemoryPolicy",
    "TrainableIPPOPolicy",
    "TrainableMAPPOPolicy",
    "TrainableHAPPOPolicy",
    "PRISMPolicy",
    "PRISMAntPolicy",
    "EARSPolicy",
    "EARSNegativePheromonePolicy",
    "HMAPPOEARSPolicy",
    "TRAINABLE_POLICY_CLASSES",
    "checkpoint_path",
    "prism_checkpoint_path",
    "sparx_checkpoint_path",
    "evaluation_factories",
]
