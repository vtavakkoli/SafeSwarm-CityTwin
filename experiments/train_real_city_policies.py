"""Train PPO-family residual policies on the real-city TRAIN split only."""

from __future__ import annotations

import argparse
import json
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
from src.evaluation.metrics import EpisodeMetrics, episode_operational_score  # noqa: E402
from src.safety.runtime_monitor import RuntimeSafetyMonitor  # noqa: E402
from src.training.geography import (  # noqa: E402
    apply_start_zone,
    load_protocol,
    select_cities,
    start_zones_for_split,
    validate_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/real_city_protocol.json")
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument(
        "--episodes",
        type=int,
        default=18,
        help="Training episodes per strategy (two passes over every train city/start-zone pair)",
    )
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.985)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--clip-ratio", type=float, default=0.20)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output-root", default="results/train")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--require-real-data", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _episode_metrics(env: CityTwinEnvironment, monitor: RuntimeSafetyMonitor) -> EpisodeMetrics:
    actual_incidents = env.actual_collisions + env.actual_restricted_entries
    battery_failures = sum(state.battery_level <= 0 for state in env.agents.values())
    coverage = len(env.visited - env.restricted_zones) / env.traversable_cell_count
    return EpisodeMetrics(
        mission_success_rate=len(env.discovered_missions) / max(1, len(env.mission_zones)),
        weighted_target_discovery=env.weighted_target_discovery(),
        coverage_ratio=float(np.clip(coverage, 0.0, 1.0)),
        number_of_safety_violations=int(monitor.safety_violations),
        safety_interventions=int(monitor.intervention_count),
        actual_safety_incidents=int(actual_incidents),
        collision_count=int(env.actual_collisions),
        restricted_zone_entries=int(env.actual_restricted_entries),
        battery_failures=int(battery_failures),
        energy_consumption=env.energy_consumption(),
        redundant_coverage=env.redundant_coverage(),
        communication_efficiency=env.communication_efficiency(),
        distance_travelled=sum(state.distance_travelled for state in env.agents.values()),
        runtime_seconds=0.0,
        runtime_overhead=0.0,
        blocked_moves=int(env.blocked_moves),
    )


def _dense_reward(
    env: CityTwinEnvironment,
    previous: dict[str, float],
) -> tuple[float, dict[str, float]]:
    current = {
        "target": env.weighted_target_discovery(),
        "coverage": len(env.visited - env.restricted_zones) / env.traversable_cell_count,
        "incidents": float(env.actual_collisions + env.actual_restricted_entries),
        "energy": env.energy_consumption() / max(1.0, 100.0 * env.n_agents),
        "redundancy": env.redundant_coverage(),
    }
    reward = (
        2.2 * (current["target"] - previous["target"])
        + 0.9 * (current["coverage"] - previous["coverage"])
        - 1.5 * (current["incidents"] - previous["incidents"])
        - 0.12 * max(0.0, current["energy"] - previous["energy"])
        - 0.08 * max(0.0, current["redundancy"] - previous["redundancy"])
    )
    return float(reward), current


def _discounted_returns(step_rewards: list[float], gamma: float) -> list[float]:
    result = [0.0] * len(step_rewards)
    running = 0.0
    for index in range(len(step_rewards) - 1, -1, -1):
        running = step_rewards[index] + gamma * running
        result[index] = running
    return result


def _train_episode(
    env: CityTwinEnvironment,
    policy: Any,
    *,
    gamma: float,
    learning_rate: float,
    clip_ratio: float,
    ppo_epochs: int,
) -> tuple[EpisodeMetrics, dict[str, float]]:
    policy.monitor = RuntimeSafetyMonitor()
    step_decisions: list[list[dict[str, Any]]] = []
    rewards: list[float] = []
    previous = {
        "target": 0.0,
        "coverage": 0.0,
        "incidents": 0.0,
        "energy": 0.0,
        "redundancy": 0.0,
    }

    while True:
        actions = policy.act(env)
        decisions = policy.drain_decisions()
        info = env.step(actions)
        reward, previous = _dense_reward(env, previous)
        step_decisions.append(decisions)
        rewards.append(reward)
        if info["done"] > 0:
            break

    returns = _discounted_returns(rewards, gamma)
    samples: list[tuple[dict[str, Any], float]] = []
    for decisions, advantage in zip(step_decisions, returns):
        samples.extend((decision, advantage) for decision in decisions)

    update = policy.ppo_update(
        samples,
        learning_rate=learning_rate,
        clip_ratio=clip_ratio,
        epochs=ppo_epochs,
    )
    return _episode_metrics(env, policy.monitor), update


def main() -> None:
    args = parse_args()
    if args.quick:
        args.agents = min(args.agents, 4)
        args.grid_size = min(args.grid_size, 20)
        args.episodes = min(args.episodes, 2)
        args.max_steps = min(args.max_steps, 50)
        args.ppo_epochs = min(args.ppo_epochs, 2)

    protocol = load_protocol(args.protocol)
    integrity = validate_protocol(protocol)
    if not all(integrity.values()):
        raise RuntimeError(f"Protocol integrity check failed: {integrity}")

    train_cities = select_cities(protocol, "train")
    train_zones = start_zones_for_split(protocol, "train")
    prepared: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for city in train_cities:
        layers = load_real_city_layers(
            city["place"],
            args.grid_size,
            args.seed,
            radius_m=int(city.get("radius_m", 1400)),
            cache_dir=args.cache_dir,
            allow_network=not args.offline,
        )
        metadata = dict(layers.get("metadata", {}))
        if args.require_real_data and metadata.get("source") != "openstreetmap":
            raise RuntimeError(f"Training city {city['name']} is not real data: {metadata}")
        prepared[city["name"]] = (city, layers)

    output = Path(args.output_root)
    checkpoints = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for strategy, policy_cls in TRAINABLE_POLICY_CLASSES.items():
        policy = policy_cls(seed=args.seed)
        for episode in range(args.episodes):
            city = train_cities[episode % len(train_cities)]
            city_info, layers = prepared[city["name"]]
            zone = train_zones[(episode // len(train_cities)) % len(train_zones)]
            episode_seed = args.seed + 1000 * list(TRAINABLE_POLICY_CLASSES).index(strategy) + episode
            zoned_layers = apply_start_zone(layers, args.grid_size, zone)
            if hasattr(policy, "reset_episode_state"):
                policy.reset_episode_state()
            env = CityTwinEnvironment(
                grid_size=args.grid_size,
                n_agents=args.agents,
                seed=episode_seed,
                place_name=city_info["place"],
                radius_m=int(city_info.get("radius_m", 1400)),
                layers=zoned_layers,
                max_steps=args.max_steps,
                allow_network=False,
            )
            metrics, update = _train_episode(
                env,
                policy,
                gamma=args.gamma,
                learning_rate=args.learning_rate,
                clip_ratio=args.clip_ratio,
                ppo_epochs=args.ppo_epochs,
            )
            row = metrics.to_dict(
                strategy=strategy,
                episode=episode,
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
            row.update({f"ppo_{key}": value for key, value in update.items()})
            diag = policy.diagnostics()
            row["swarm_memory_coverage"] = float(diag.get("swarm_memory_coverage", 0.0))
            row["swarm_memory_peak"] = float(diag.get("swarm_memory_peak", 0.0))
            records.append(row)
            print(
                f"[train] {strategy} episode={episode + 1}/{args.episodes} "
                f"city={city_info['name']} zone={zone} score={row['operational_score']:.3f} "
                f"target={metrics.weighted_target_discovery:.3f} coverage={metrics.coverage_ratio:.3f}",
                flush=True,
            )

        policy.save_checkpoint(
            checkpoint_path(checkpoints, strategy),
            metadata={
                "trained_utc": datetime.now(timezone.utc).isoformat(),
                "protocol": args.protocol,
                "training_split": "train",
                "training_cities": [city["name"] for city in train_cities],
                "training_start_zones": train_zones,
                "episodes": args.episodes,
                "seed": args.seed,
                "algorithm": "clipped PPO residual over interpretable SafeSwarm action scores",
            },
        )

    frame = pd.DataFrame(records)
    frame.to_csv(output / "training_history.csv", index=False)
    summary = (
        frame.groupby("strategy", as_index=False)
        .agg(
            mean_train_score=("operational_score", "mean"),
            final_train_score=("operational_score", "last"),
            mean_target_discovery=("weighted_target_discovery", "mean"),
            mean_coverage=("coverage_ratio", "mean"),
            mean_safety_incidents=("actual_safety_incidents", "mean"),
            final_weight_norm=("ppo_weight_norm", "last"),
        )
        .sort_values("mean_train_score", ascending=False)
    )
    summary.to_csv(output / "training_summary.csv", index=False)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "protocol_integrity": integrity,
        "checkpoint_directory": str(checkpoints),
        "strategies": list(TRAINABLE_POLICY_CLASSES),
        "note": "No validation or test-city metric is used by the optimizer.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
