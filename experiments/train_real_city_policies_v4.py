"""SafeSwarm v4 PPO-family trainer with robust validation checkpoint gating.

v3 fixed action credit, observability, energy semantics and GRPO optimization.
v4 keeps those learning mechanics but addresses a second-order generalization
failure: model selection on one Amsterdam/west validation domain was a weak
proxy for unseen San Francisco/Paris performance.

Candidate weights are now selected by a multi-city, multi-start-zone robust
validation score. Every genuine validation improvement is archived so the exact
weight trajectory remains auditable. Test metrics are never read here.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.trainable_policies import TRAINABLE_POLICY_CLASSES, checkpoint_path  # noqa: E402
from src.training.geography import (  # noqa: E402
    load_protocol,
    select_cities,
    start_zones_for_split,
    validate_protocol,
)
from src.training.policy_learning import rollout_episode  # noqa: E402
from src.training.validation_selection import validation_selection_stats  # noqa: E402
from experiments import train_real_city_policies_v3 as v3  # noqa: E402


def main() -> None:
    args = v3.parse_args()
    if args.quick:
        args.agents = min(args.agents, 4)
        args.grid_size = min(args.grid_size, 20)
        args.episodes = min(args.episodes, 2)
        args.validation_episodes = min(args.validation_episodes, 2)
        args.validation_repeats = 1
        args.max_steps = min(args.max_steps, 50)
        args.ppo_epochs = min(args.ppo_epochs, 2)
        args.teacher_bootstrap_scenarios = min(args.teacher_bootstrap_scenarios, 1)
        args.early_stop_min_epochs = 1
        args.early_stop_patience = 1

    protocol = load_protocol(args.protocol)
    integrity = validate_protocol(protocol)
    if not all(integrity.values()):
        raise RuntimeError(f"Protocol integrity check failed: {integrity}")

    train_cities = select_cities(protocol, "train")
    train_zones = start_zones_for_split(protocol, "train")
    validation_cities = select_cities(protocol, "validation")
    validation_zones = start_zones_for_split(protocol, "validation")
    if not validation_cities or not validation_zones:
        raise RuntimeError("A disjoint validation split is required")

    if args.quick:
        train_cities = train_cities[:1]
        train_zones = train_zones[:1]
        validation_cities = validation_cities[:1]
        validation_zones = validation_zones[:1]

    prepared_train = v3._prepare(
        train_cities,
        grid_size=args.grid_size,
        seed=args.seed,
        cache_dir=args.cache_dir,
        offline=args.offline,
        require_real_data=args.require_real_data,
        split="training",
    )
    prepared_validation = v3._prepare(
        validation_cities,
        grid_size=args.grid_size,
        seed=args.seed + 70000,
        cache_dir=args.cache_dir,
        offline=args.offline,
        require_real_data=args.require_real_data,
        split="validation",
    )

    zone_order = sorted(train_zones, key=lambda zone: (zone != "center", zone))
    scenarios = [(city, zone) for zone in zone_order for city in train_cities]
    if not scenarios:
        raise RuntimeError("No training scenarios configured")
    epochs = max(1, math.ceil(args.episodes / len(scenarios)))

    output = Path(args.output_root)
    checkpoints = output / "checkpoints"
    candidates = output / "candidates"
    improvements = output / "validation-improvements"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    candidates.mkdir(parents=True, exist_ok=True)
    improvements.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    best_rows: dict[str, dict[str, Any]] = {}
    bootstrap_rows: dict[str, dict[str, float]] = {}
    improvement_manifest: dict[str, list[dict[str, Any]]] = {}

    for strategy_index, (strategy, policy_cls) in enumerate(TRAINABLE_POLICY_CLASSES.items()):
        policy = policy_cls(seed=args.seed + 100 * strategy_index)
        bootstrap = (
            v3._bootstrap_grpo(policy, prepared_train, scenarios, args)
            if strategy == "GRPO-Safe"
            else {"imitation_examples": 0.0, "imitation_loss": 0.0, "imitation_accuracy": 0.0}
        )
        bootstrap_rows[strategy] = bootstrap
        improvement_manifest[strategy] = []

        best_robust = float("-inf")
        best_score = float("-inf")
        best_ci95 = float("nan")
        best_domain_std = float("nan")
        best_worst_domain = float("nan")
        best_epoch = -1
        stale_epochs = 0
        final_path = checkpoint_path(checkpoints, strategy)
        candidate_path = checkpoint_path(candidates, strategy)
        strategy_improvements = improvements / strategy.lower().replace("-", "_")
        strategy_improvements.mkdir(parents=True, exist_ok=True)
        episode_number = 0
        completed_epochs = 0
        last_selection: dict[str, float] = {}

        for epoch in range(epochs):
            completed_epochs = epoch + 1
            batch_samples: list[tuple[dict[str, Any], float, float]] = []
            batch_row_indexes: list[int] = []

            for city, zone in scenarios:
                city_info, layers = prepared_train[city["name"]]
                episode_seed = args.seed + episode_number
                if hasattr(policy, "reset_episode_state"):
                    policy.reset_episode_state()
                env = v3._environment(
                    city_info,
                    layers,
                    zone,
                    grid_size=args.grid_size,
                    agents=args.agents,
                    seed=episode_seed,
                    max_steps=args.max_steps,
                )
                metrics, samples, extra = rollout_episode(
                    env,
                    policy,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    training=True,
                )
                row = metrics.to_dict(
                    strategy=strategy,
                    episode=episode_number,
                    training_epoch=epoch + 1,
                    split="train",
                    city=city_info["name"],
                    start_zone=zone,
                    seed=episode_seed,
                    data_source=env.data_source,
                    agents=args.agents,
                    grid_size=args.grid_size,
                    steps=env.steps,
                )
                from src.evaluation.metrics import episode_operational_score
                row["operational_score"] = episode_operational_score(row)
                row.update(extra)
                records.append(row)
                batch_row_indexes.append(len(records) - 1)
                batch_samples.extend(samples)
                episode_number += 1
                print(
                    f"[train-v4] {strategy} epoch={epoch + 1}/{epochs} city={city_info['name']} "
                    f"zone={zone} score={row['operational_score']:.3f} "
                    f"target={metrics.weighted_target_discovery:.3f} coverage={metrics.coverage_ratio:.3f}",
                    flush=True,
                )

            progress = epoch / max(1, epochs - 1)
            learning_rate = args.learning_rate * (args.lr_decay ** epoch)
            critic_learning_rate = args.critic_learning_rate * (args.lr_decay ** epoch)
            entropy_coef = args.entropy_coef + progress * (
                args.entropy_coef_final - args.entropy_coef
            )
            update = policy.ppo_update(
                batch_samples,
                learning_rate=learning_rate,
                critic_learning_rate=critic_learning_rate,
                clip_ratio=args.clip_ratio,
                epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                entropy_coef=entropy_coef,
                target_kl=args.target_kl,
                max_grad_norm=0.5,
            )
            update.update(
                {
                    "learning_rate": learning_rate,
                    "critic_learning_rate": critic_learning_rate,
                    "entropy_coef": entropy_coef,
                }
            )
            for index in batch_row_indexes:
                records[index].update({f"ppo_{key}": value for key, value in update.items()})

            policy.save_checkpoint(
                candidate_path,
                metadata={
                    "trained_utc": datetime.now(timezone.utc).isoformat(),
                    "protocol": args.protocol,
                    "training_epoch": epoch + 1,
                    "training_cities": [city["name"] for city in train_cities],
                    "training_start_zones": train_zones,
                    "validation_cities": [city["name"] for city in validation_cities],
                    "validation_start_zones": validation_zones,
                    "algorithm": "SafeSwarm PPO-v4: v3 learner + robust multi-domain validation gating",
                    "teacher_bootstrap": bootstrap,
                },
            )

            val_rows, _, _ = v3._validation_scores(
                policy_cls,
                candidate_path,
                prepared_validation,
                validation_cities,
                validation_zones,
                episodes=args.validation_episodes,
                repeats=args.validation_repeats,
                agents=args.agents,
                grid_size=args.grid_size,
                max_steps=args.max_steps,
                seed=args.seed + 80000,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            selection = validation_selection_stats(val_rows)
            last_selection = selection
            for row in val_rows:
                row["training_epoch"] = epoch + 1
                row["validation_mean_score"] = selection["mean_score"]
                row["validation_robust_score"] = selection["robust_score"]
                row["validation_domain_std"] = selection["domain_std"]
                row["validation_worst_domain_score"] = selection["worst_domain_score"]
                validation_records.append(row)

            print(
                f"[validation-v4] {strategy} epoch={epoch + 1}/{epochs} "
                f"mean={selection['mean_score']:.3f} ± {selection['ci95']:.3f} "
                f"robust={selection['robust_score']:.3f} worst-domain={selection['worst_domain_score']:.3f}",
                flush=True,
            )

            if selection["robust_score"] > best_robust + args.early_stop_min_delta:
                best_robust = selection["robust_score"]
                best_score = selection["mean_score"]
                best_ci95 = selection["ci95"]
                best_domain_std = selection["domain_std"]
                best_worst_domain = selection["worst_domain_score"]
                best_epoch = epoch + 1
                stale_epochs = 0
                shutil.copy2(candidate_path, final_path)
                improvement_path = strategy_improvements / f"epoch_{epoch + 1:03d}.json"
                shutil.copy2(candidate_path, improvement_path)
                improvement_manifest[strategy].append(
                    {
                        "epoch": epoch + 1,
                        "checkpoint": str(improvement_path),
                        **selection,
                    }
                )
            else:
                stale_epochs += 1

            if (
                completed_epochs >= args.early_stop_min_epochs
                and stale_epochs >= args.early_stop_patience
            ):
                print(
                    f"[early-stop-v4] {strategy} epoch={completed_epochs}; "
                    f"best_epoch={best_epoch} robust={best_robust:.3f}",
                    flush=True,
                )
                break

        if not final_path.exists():
            shutil.copy2(candidate_path, final_path)
            best_epoch = completed_epochs
            best_robust = last_selection.get("robust_score", float("nan"))
            best_score = last_selection.get("mean_score", float("nan"))
            best_ci95 = last_selection.get("ci95", float("nan"))
            best_domain_std = last_selection.get("domain_std", float("nan"))
            best_worst_domain = last_selection.get("worst_domain_score", float("nan"))

        best_rows[strategy] = {
            "best_validation_score": best_score,
            "best_validation_ci95": best_ci95,
            "best_validation_robust_score": best_robust,
            "best_validation_domain_std": best_domain_std,
            "best_validation_worst_domain_score": best_worst_domain,
            "best_epoch": best_epoch,
            "completed_epochs": completed_epochs,
            "validation_improvement_checkpoints": len(improvement_manifest[strategy]),
            **v3._checkpoint_stats(final_path),
        }

    frame = pd.DataFrame(records)
    validation_frame = pd.DataFrame(validation_records)
    frame.to_csv(output / "training_history.csv", index=False)
    validation_frame.to_csv(output / "validation_history.csv", index=False)

    summaries: list[dict[str, Any]] = []
    for strategy in TRAINABLE_POLICY_CLASSES:
        group = frame[frame["strategy"] == strategy]
        last_epoch = int(group["training_epoch"].max())
        last_batch = group[group["training_epoch"] == last_epoch]
        summaries.append(
            {
                "strategy": strategy,
                "mean_train_score": float(group["operational_score"].mean()),
                "final_train_score": float(last_batch["operational_score"].mean()),
                "mean_target_discovery": float(group["weighted_target_discovery"].mean()),
                "mean_coverage": float(group["coverage_ratio"].mean()),
                "mean_safety_incidents": float(group["actual_safety_incidents"].mean()),
                "mean_safety_interventions": float(group["safety_interventions"].mean()),
                "mean_mask_rejections": float(group["safety_mask_rejections"].mean()),
                "mean_safe_returns": float(group.get("safe_returns", pd.Series([0.0])).mean()),
                "actual_training_episodes": int(len(group)),
                **bootstrap_rows[strategy],
                **best_rows[strategy],
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        "best_validation_robust_score", ascending=False
    )
    summary.to_csv(output / "training_summary.csv", index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "training_version": "SafeSwarm PPO/GRPO v4",
        "arguments": vars(args),
        "protocol_integrity": integrity,
        "checkpoint_directory": str(checkpoints),
        "candidate_directory": str(candidates),
        "validation_improvement_directory": str(improvements),
        "strategies": list(TRAINABLE_POLICY_CLASSES),
        "balanced_scenarios_per_epoch": len(scenarios),
        "checkpoint_selection": (
            "validation-only robust score = mean - 0.5*CI95 - 0.20*domain_std "
            "- 0.10*(mean-worst_domain); each improvement archived; test never consulted"
        ),
        "validation_domains": {
            "cities": [city["name"] for city in validation_cities],
            "start_zones": validation_zones,
        },
        "partial_observability_contract": (
            "primary policies cannot use hidden mission coordinates or priority labels for action selection"
        ),
        "teacher_bootstrap": "GRPO only; observable AntSwarmSafe and UA-HBAS-Safe teachers",
        "credit_assignment": "per-agent difference reward + per-agent GAE with cooperative team component",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    (output / "teacher_bootstrap.json").write_text(
        json.dumps(bootstrap_rows, indent=2, default=str), encoding="utf-8"
    )
    (output / "validation_improvements.json").write_text(
        json.dumps(improvement_manifest, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
