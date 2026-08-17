"""Train SafeSwarm v3 policies with per-agent credit and teacher-bootstrap GRPO.

Key differences from v2:
- fair observable-only policy inputs;
- per-agent difference rewards + per-agent GAE;
- minibatch entropy-regularized PPO with learning-rate decay;
- state-conditioned GRPO behavior policy;
- observable AntSwarm/UA-HBAS teacher warm start for GRPO;
- repeated stochastic validation and early stopping.
"""

from __future__ import annotations

import argparse
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

from src.agents.registry import strategy_factories  # noqa: E402
from src.agents.trainable_policies import TRAINABLE_POLICY_CLASSES, checkpoint_path  # noqa: E402
from src.environment.city_twin import CityTwinEnvironment  # noqa: E402
from src.environment.obstacles import load_real_city_layers  # noqa: E402
from src.evaluation.metrics import episode_operational_score  # noqa: E402
from src.training.geography import (  # noqa: E402
    apply_start_zone,
    load_protocol,
    select_cities,
    start_zones_for_split,
    validate_protocol,
)
from src.training.policy_learning import rollout_episode  # noqa: E402
from src.training.teacher_bootstrap import collect_grpo_teacher_examples  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/real_city_protocol.json")
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument(
        "--episodes",
        type=int,
        default=108,
        help="Requested PPO training episodes/strategy; rounded to complete city×zone epochs.",
    )
    parser.add_argument("--validation-episodes", type=int, default=12)
    parser.add_argument("--validation-repeats", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    parser.add_argument("--learning-rate", type=float, default=0.004)
    parser.add_argument("--critic-learning-rate", type=float, default=0.003)
    parser.add_argument("--lr-decay", type=float, default=0.93)
    parser.add_argument("--clip-ratio", type=float, default=0.20)
    parser.add_argument("--ppo-epochs", type=int, default=6)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--entropy-coef", type=float, default=0.06)
    parser.add_argument("--entropy-coef-final", type=float, default=0.015)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--teacher-bootstrap-scenarios", type=int, default=9)
    parser.add_argument("--teacher-bootstrap-lr", type=float, default=0.01)
    parser.add_argument("--teacher-bootstrap-epochs", type=int, default=1)
    parser.add_argument("--early-stop-min-epochs", type=int, default=6)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.002)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output-root", default="results/train")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--require-real-data", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _prepare(
    cities: list[dict[str, Any]],
    *,
    grid_size: int,
    seed: int,
    cache_dir: str,
    offline: bool,
    require_real_data: bool,
    split: str,
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    prepared: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for offset, city in enumerate(cities):
        layers = load_real_city_layers(
            city["place"],
            grid_size,
            seed + offset,
            radius_m=int(city.get("radius_m", 1400)),
            cache_dir=cache_dir,
            allow_network=not offline,
        )
        metadata = dict(layers.get("metadata", {}))
        if require_real_data and metadata.get("source") != "openstreetmap":
            raise RuntimeError(f"{split} city {city['name']} is not real data: {metadata}")
        prepared[city["name"]] = (city, layers)
    return prepared


def _environment(
    city_info: dict[str, Any],
    layers: dict[str, Any],
    zone: str,
    *,
    grid_size: int,
    agents: int,
    seed: int,
    max_steps: int,
) -> CityTwinEnvironment:
    return CityTwinEnvironment(
        grid_size=grid_size,
        n_agents=agents,
        seed=seed,
        place_name=city_info["place"],
        radius_m=int(city_info.get("radius_m", 1400)),
        layers=apply_start_zone(layers, grid_size, zone),
        max_steps=max_steps,
        allow_network=False,
    )


def _validation_scores(
    policy_cls: Any,
    checkpoint: Path,
    prepared: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    cities: list[dict[str, Any]],
    zones: list[str],
    *,
    episodes: int,
    repeats: int,
    agents: int,
    grid_size: int,
    max_steps: int,
    seed: int,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[dict[str, Any]], float, float]:
    rows: list[dict[str, Any]] = []
    for repeat in range(max(1, repeats)):
        for episode in range(episodes):
            city = cities[episode % len(cities)]
            zone = zones[(episode // len(cities)) % len(zones)]
            city_info, layers = prepared[city["name"]]
            episode_seed = seed + episode + repeat * 10000
            policy = policy_cls(seed=episode_seed, model_path=checkpoint)
            if hasattr(policy, "reset_episode_state"):
                policy.reset_episode_state()
            env = _environment(
                city_info,
                layers,
                zone,
                grid_size=grid_size,
                agents=agents,
                seed=episode_seed,
                max_steps=max_steps,
            )
            metrics, _, extra = rollout_episode(
                env,
                policy,
                gamma=gamma,
                gae_lambda=gae_lambda,
                training=False,
            )
            row = metrics.to_dict(
                strategy=policy.name,
                episode=episode,
                validation_repeat=repeat,
                split="validation",
                city=city_info["name"],
                start_zone=zone,
                seed=episode_seed,
                data_source=env.data_source,
                agents=agents,
                grid_size=grid_size,
                steps=env.steps,
            )
            row["operational_score"] = episode_operational_score(row)
            row.update(extra)
            rows.append(row)
    scores = np.asarray([row["operational_score"] for row in rows], dtype=float)
    mean = float(np.mean(scores)) if scores.size else float("nan")
    ci95 = (
        float(1.96 * np.std(scores, ddof=1) / np.sqrt(scores.size))
        if scores.size > 1 else 0.0
    )
    return rows, mean, ci95


def _checkpoint_stats(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = {
        "final_weight_norm": float(np.linalg.norm(np.asarray(payload.get("ppo_residual_weights", []), dtype=float))),
        "final_critic_norm": float(np.linalg.norm(np.asarray(payload.get("critic_weights", []), dtype=float))),
        "final_behavior_bias_norm": float(np.linalg.norm(np.asarray(payload.get("behavior_bias", []), dtype=float))),
        "final_behavior_state_norm": float(np.linalg.norm(np.asarray(payload.get("behavior_weights", []), dtype=float))),
    }
    return stats


def _bootstrap_grpo(
    policy: Any,
    prepared_train: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    scenarios: list[tuple[dict[str, Any], str]],
    args: argparse.Namespace,
) -> dict[str, float]:
    count = min(max(0, args.teacher_bootstrap_scenarios), len(scenarios))
    if count <= 0 or not hasattr(policy, "imitation_update"):
        return {"imitation_examples": 0.0, "imitation_loss": 0.0, "imitation_accuracy": 0.0}

    teacher_names = ("AntSwarmSafe", "UA-HBAS-Safe")
    examples: list[dict[str, Any]] = []
    for index, (city, zone) in enumerate(scenarios[:count]):
        city_info, layers = prepared_train[city["name"]]
        seed = args.seed + 50000 + index
        env = _environment(
            city_info,
            layers,
            zone,
            grid_size=args.grid_size,
            agents=args.agents,
            seed=seed,
            max_steps=args.max_steps,
        )
        teacher_name = teacher_names[index % len(teacher_names)]
        teacher = strategy_factories(seed=seed)[teacher_name]()
        examples.extend(
            collect_grpo_teacher_examples(policy, teacher, env, max_examples=1500)
        )
        print(
            f"[bootstrap] GRPO teacher={teacher_name} city={city_info['name']} zone={zone} examples={len(examples)}",
            flush=True,
        )

    stats = policy.imitation_update(
        examples,
        learning_rate=args.teacher_bootstrap_lr,
        epochs=args.teacher_bootstrap_epochs,
    )
    return {key: float(value) for key, value in stats.items()}


def main() -> None:
    args = parse_args()
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

    prepared_train = _prepare(
        train_cities,
        grid_size=args.grid_size,
        seed=args.seed,
        cache_dir=args.cache_dir,
        offline=args.offline,
        require_real_data=args.require_real_data,
        split="training",
    )
    prepared_validation = _prepare(
        validation_cities,
        grid_size=args.grid_size,
        seed=args.seed + 70000,
        cache_dir=args.cache_dir,
        offline=args.offline,
        require_real_data=args.require_real_data,
        split="validation",
    )

    # Center first for the teacher warm start, then geographically harder zones.
    zone_order = sorted(train_zones, key=lambda zone: (zone != "center", zone))
    scenarios = [(city, zone) for zone in zone_order for city in train_cities]
    if not scenarios:
        raise RuntimeError("No training scenarios configured")
    epochs = max(1, math.ceil(args.episodes / len(scenarios)))

    output = Path(args.output_root)
    checkpoints = output / "checkpoints"
    candidates = output / "candidates"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    candidates.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    best_rows: dict[str, dict[str, Any]] = {}
    bootstrap_rows: dict[str, dict[str, float]] = {}

    for strategy_index, (strategy, policy_cls) in enumerate(TRAINABLE_POLICY_CLASSES.items()):
        policy = policy_cls(seed=args.seed + 100 * strategy_index)
        bootstrap = (
            _bootstrap_grpo(policy, prepared_train, scenarios, args)
            if strategy == "GRPO-Safe"
            else {"imitation_examples": 0.0, "imitation_loss": 0.0, "imitation_accuracy": 0.0}
        )
        bootstrap_rows[strategy] = bootstrap

        best_score = float("-inf")
        best_ci95 = float("nan")
        best_epoch = -1
        stale_epochs = 0
        final_path = checkpoint_path(checkpoints, strategy)
        candidate_path = checkpoint_path(candidates, strategy)
        episode_number = 0
        completed_epochs = 0

        for epoch in range(epochs):
            completed_epochs = epoch + 1
            batch_samples: list[tuple[dict[str, Any], float, float]] = []
            batch_row_indexes: list[int] = []
            for city, zone in scenarios:
                city_info, layers = prepared_train[city["name"]]
                episode_seed = args.seed + episode_number
                if hasattr(policy, "reset_episode_state"):
                    policy.reset_episode_state()
                env = _environment(
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
                row["operational_score"] = episode_operational_score(row)
                row.update(extra)
                records.append(row)
                batch_row_indexes.append(len(records) - 1)
                batch_samples.extend(samples)
                episode_number += 1
                print(
                    f"[train] {strategy} epoch={epoch + 1}/{epochs} city={city_info['name']} zone={zone} "
                    f"score={row['operational_score']:.3f} target={metrics.weighted_target_discovery:.3f} "
                    f"coverage={metrics.coverage_ratio:.3f} safe_returns={int(extra['safe_returns'])}",
                    flush=True,
                )

            progress = epoch / max(1, epochs - 1)
            learning_rate = args.learning_rate * (args.lr_decay ** epoch)
            critic_learning_rate = args.critic_learning_rate * (args.lr_decay ** epoch)
            entropy_coef = (
                args.entropy_coef
                + progress * (args.entropy_coef_final - args.entropy_coef)
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
                    "algorithm": "SafeSwarm PPO-v3: per-agent credit + minibatch entropy PPO + observable-only inputs",
                    "teacher_bootstrap": bootstrap,
                },
            )
            val_rows, val_score, val_ci95 = _validation_scores(
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
            for row in val_rows:
                row["training_epoch"] = epoch + 1
                validation_records.append(row)
            print(
                f"[validation] {strategy} epoch={epoch + 1}/{epochs} score={val_score:.3f} ± {val_ci95:.3f}",
                flush=True,
            )

            if val_score > best_score + args.early_stop_min_delta:
                best_score = val_score
                best_ci95 = val_ci95
                best_epoch = epoch + 1
                stale_epochs = 0
                shutil.copy2(candidate_path, final_path)
            else:
                stale_epochs += 1

            if (
                completed_epochs >= args.early_stop_min_epochs
                and stale_epochs >= args.early_stop_patience
            ):
                print(
                    f"[early-stop] {strategy} epoch={completed_epochs}; best_epoch={best_epoch} best={best_score:.3f}",
                    flush=True,
                )
                break

        # If min_delta prevented the first save for any reason, preserve candidate.
        if not final_path.exists():
            shutil.copy2(candidate_path, final_path)
            best_epoch = completed_epochs
            best_score = val_score
            best_ci95 = val_ci95

        best_rows[strategy] = {
            "best_validation_score": best_score,
            "best_validation_ci95": best_ci95,
            "best_epoch": best_epoch,
            "completed_epochs": completed_epochs,
            **_checkpoint_stats(final_path),
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
    summary = pd.DataFrame(summaries).sort_values("best_validation_score", ascending=False)
    summary.to_csv(output / "training_summary.csv", index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "protocol_integrity": integrity,
        "checkpoint_directory": str(checkpoints),
        "candidate_directory": str(candidates),
        "strategies": list(TRAINABLE_POLICY_CLASSES),
        "balanced_scenarios_per_epoch": len(scenarios),
        "checkpoint_selection": "highest repeated stochastic validation operational score; test split never consulted",
        "partial_observability_contract": "primary policies cannot use hidden mission coordinates or priority labels for action selection",
        "teacher_bootstrap": "GRPO only; observable AntSwarmSafe and UA-HBAS-Safe teachers",
        "credit_assignment": "per-agent difference reward + per-agent GAE with cooperative team component",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    (output / "teacher_bootstrap.json").write_text(
        json.dumps(bootstrap_rows, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
