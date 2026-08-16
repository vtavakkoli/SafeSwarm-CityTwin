"""Train PPO-family policies on balanced real-city batches and select by validation."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/real_city_protocol.json")
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument(
        "--episodes",
        type=int,
        default=54,
        help="Requested training episodes per strategy; normal runs are rounded up to complete city×zone batches",
    )
    parser.add_argument("--validation-episodes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.985)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--critic-learning-rate", type=float, default=0.0125)
    parser.add_argument("--clip-ratio", type=float, default=0.20)
    parser.add_argument("--ppo-epochs", type=int, default=4)
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
    agents: int,
    grid_size: int,
    max_steps: int,
    seed: int,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[dict[str, Any]], float, float]:
    rows: list[dict[str, Any]] = []
    for episode in range(episodes):
        city = cities[episode % len(cities)]
        zone = zones[(episode // len(cities)) % len(zones)]
        city_info, layers = prepared[city["name"]]
        episode_seed = seed + episode
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
    ci95 = float(1.96 * np.std(scores, ddof=1) / np.sqrt(scores.size)) if scores.size > 1 else 0.0
    return rows, mean, ci95


def _checkpoint_weight_norm(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(np.linalg.norm(np.asarray(payload.get("ppo_residual_weights", []), dtype=float)))


def main() -> None:
    args = parse_args()
    if args.quick:
        args.agents = min(args.agents, 4)
        args.grid_size = min(args.grid_size, 20)
        args.episodes = min(args.episodes, 2)
        args.validation_episodes = min(args.validation_episodes, 2)
        args.max_steps = min(args.max_steps, 50)
        args.ppo_epochs = min(args.ppo_epochs, 2)

    protocol = load_protocol(args.protocol)
    integrity = validate_protocol(protocol)
    if not all(integrity.values()):
        raise RuntimeError(f"Protocol integrity check failed: {integrity}")

    train_cities = select_cities(protocol, "train")
    train_zones = start_zones_for_split(protocol, "train")
    validation_cities = select_cities(protocol, "validation")
    validation_zones = start_zones_for_split(protocol, "validation")
    if not validation_cities or not validation_zones:
        raise RuntimeError("A disjoint validation split is required for checkpoint selection")

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

    scenarios = [(city, zone) for zone in train_zones for city in train_cities]
    if not scenarios:
        raise RuntimeError("No training city/start-zone scenarios configured")
    epochs = max(1, math.ceil(args.episodes / len(scenarios)))
    actual_episodes = epochs * len(scenarios)

    output = Path(args.output_root)
    checkpoints = output / "checkpoints"
    candidates = output / "candidates"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    candidates.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    best_rows: dict[str, dict[str, Any]] = {}

    for strategy_index, (strategy, policy_cls) in enumerate(TRAINABLE_POLICY_CLASSES.items()):
        policy = policy_cls(seed=args.seed + 100 * strategy_index)
        best_score = float("-inf")
        best_ci95 = float("nan")
        best_epoch = -1
        final_path = checkpoint_path(checkpoints, strategy)
        candidate_path = checkpoint_path(candidates, strategy)
        episode_number = 0

        for epoch in range(epochs):
            batch_samples: list[tuple[dict[str, Any], float, float]] = []
            batch_row_indexes: list[int] = []
            for city, zone in scenarios:
                city_info, layers = prepared_train[city["name"]]
                # Paired environment seeds across strategies make algorithm comparisons less noisy.
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
                    f"[train] {strategy} epoch={epoch + 1}/{epochs} city={city_info['name']} "
                    f"zone={zone} score={row['operational_score']:.3f} target={metrics.weighted_target_discovery:.3f} "
                    f"coverage={metrics.coverage_ratio:.3f} masked={int(extra['safety_mask_rejections'])}",
                    flush=True,
                )

            update = policy.ppo_update(
                batch_samples,
                learning_rate=args.learning_rate,
                critic_learning_rate=args.critic_learning_rate,
                clip_ratio=args.clip_ratio,
                epochs=args.ppo_epochs,
            )
            for index in batch_row_indexes:
                records[index].update({f"ppo_{key}": value for key, value in update.items()})

            policy.save_checkpoint(
                candidate_path,
                metadata={
                    "trained_utc": datetime.now(timezone.utc).isoformat(),
                    "protocol": args.protocol,
                    "training_split": "train",
                    "validation_split": "validation",
                    "training_epoch": epoch + 1,
                    "training_cities": [city["name"] for city in train_cities],
                    "training_start_zones": train_zones,
                    "algorithm": "safe-action-masked clipped PPO + centralized linear critic + GAE",
                },
            )
            val_rows, val_score, val_ci95 = _validation_scores(
                policy_cls,
                candidate_path,
                prepared_validation,
                validation_cities,
                validation_zones,
                episodes=args.validation_episodes,
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
            if val_score > best_score:
                best_score = val_score
                best_ci95 = val_ci95
                best_epoch = epoch + 1
                shutil.copy2(candidate_path, final_path)

        best_rows[strategy] = {
            "best_validation_score": best_score,
            "best_validation_ci95": best_ci95,
            "best_epoch": best_epoch,
            "final_weight_norm": _checkpoint_weight_norm(final_path),
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
        summaries.append({
            "strategy": strategy,
            "mean_train_score": float(group["operational_score"].mean()),
            "final_train_score": float(last_batch["operational_score"].mean()),
            "mean_target_discovery": float(group["weighted_target_discovery"].mean()),
            "mean_coverage": float(group["coverage_ratio"].mean()),
            "mean_safety_incidents": float(group["actual_safety_incidents"].mean()),
            "mean_safety_interventions": float(group["safety_interventions"].mean()),
            "mean_mask_rejections": float(group["safety_mask_rejections"].mean()),
            "actual_training_episodes": int(len(group)),
            **best_rows[strategy],
        })
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
        "actual_training_episodes_per_strategy": actual_episodes,
        "checkpoint_selection": "highest mean validation operational score; test split is never consulted",
        "note": "Training updates occur only after a complete city×start-zone batch. Environment seeds are paired across strategies.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
