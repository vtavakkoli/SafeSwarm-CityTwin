"""Select EARS v6 controllers using train + validation only.

The v5 real result showed that PRISM-Ant already matched/exceeded AntSwarm's
SWAP discovery but lost on movement efficiency and redundant coverage.  v6
therefore keeps Ant local search as the default behavior and searches only the
parameters controlling *when* global intervention happens and how expensive
that intervention may be.

No held-out test or SWAP score is read by this script.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import train_real_city_policies_v3 as v3  # noqa: E402
from experiments.run_city_benchmark import run_episode  # noqa: E402
from src.agents.ears_v6 import (  # noqa: E402
    EARSConfig,
    EARSNegativePheromonePolicy,
    EARSPolicy,
    HMAPPOEARSConfig,
    HMAPPOEARSPolicy,
)
from src.evaluation.metrics import episode_operational_score  # noqa: E402
from src.training.geography import (  # noqa: E402
    load_protocol,
    select_cities,
    start_zones_for_split,
    validate_protocol,
)
from src.training.validation_selection import validation_selection_stats  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/real_city_protocol.json")
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--train-episodes", type=int, default=6)
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--validation-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=6262)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output-root", default="results/train")
    parser.add_argument("--model-dir", default="results/train/checkpoints")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--require-real-data", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _domains(
    cities: list[dict[str, Any]], zones: list[str], requested: int
) -> list[tuple[dict[str, Any], str]]:
    base = [(city, zone) for zone in zones for city in cities]
    if not base:
        return []
    return [base[index % len(base)] for index in range(max(1, requested))]


def _evaluate(
    policy_factory: Callable[[int], Any],
    prepared: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    domains: list[tuple[dict[str, Any], str]],
    args: argparse.Namespace,
    *,
    seed: int,
    split: str,
    strategy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode, (city, zone) in enumerate(domains):
        city_info, layers = prepared[city["name"]]
        episode_seed = seed + episode
        env = v3._environment(
            city_info,
            layers,
            zone,
            grid_size=args.grid_size,
            agents=args.agents,
            seed=episode_seed,
            max_steps=args.max_steps,
        )
        policy = policy_factory(episode_seed)
        metrics = run_episode(env, policy)
        row = metrics.to_dict(
            strategy=strategy,
            episode=episode,
            split=split,
            city=city_info["name"],
            start_zone=zone,
            seed=episode_seed,
            data_source=env.data_source,
            agents=env.n_agents,
            grid_size=env.grid_size,
            steps=env.steps,
        )
        row["operational_score"] = episode_operational_score(row)
        diagnostics = policy.diagnostics() if hasattr(policy, "diagnostics") else {}
        for key in (
            "ears_event_triggers",
            "ears_ant_fraction",
            "ears_relocation_fraction",
            "ears_negative_escape_fraction",
            "ears_negative_pheromone_mean",
            "h_mappo_ears_mappo_fraction",
        ):
            row[key] = diagnostics.get(key, 0.0)
        rows.append(row)
    return rows


def _ears_candidates() -> list[tuple[str, EARSConfig]]:
    return [
        (
            "conservative",
            EARSConfig(
                stagnation_unique_ratio=0.38,
                local_revisit_trigger=0.62,
                congestion_trigger=1.28,
                relocation_duration=6,
                relocation_cooldown=9,
                global_distance_penalty=0.24,
                global_energy_penalty=0.32,
            ),
        ),
        ("balanced", EARSConfig()),
        (
            "responsive",
            EARSConfig(
                stagnation_unique_ratio=0.53,
                local_revisit_trigger=0.49,
                congestion_trigger=1.02,
                relocation_duration=8,
                relocation_cooldown=6,
                global_distance_penalty=0.15,
                global_energy_penalty=0.20,
            ),
        ),
        (
            "short-hop",
            EARSConfig(
                stagnation_unique_ratio=0.47,
                local_revisit_trigger=0.53,
                relocation_duration=5,
                relocation_cooldown=7,
                global_distance_penalty=0.30,
                global_energy_penalty=0.38,
                goal_spacing=4,
            ),
        ),
    ]


def _np_candidates(base: EARSConfig) -> list[tuple[str, EARSConfig]]:
    rows: list[tuple[str, EARSConfig]] = []
    for label, weight, decay, diffusion in (
        ("light", 0.45, 0.93, 0.10),
        ("balanced", 0.72, 0.92, 0.13),
        ("strong", 0.95, 0.90, 0.16),
    ):
        values = asdict(base)
        values.update(
            {
                "negative_pheromone_weight": weight,
                "negative_decay": decay,
                "negative_diffusion": diffusion,
                "negative_visit_deposit": 0.24 + 0.05 * weight,
                "negative_congestion_deposit": 0.16 + 0.05 * weight,
            }
        )
        rows.append((label, EARSConfig(**values)))
    return rows


def _h_candidates() -> list[tuple[str, HMAPPOEARSConfig]]:
    return [
        (
            "ant-dominant",
            HMAPPOEARSConfig(
                mappo_uncertainty_min=0.48,
                mappo_uncertainty_max=0.82,
                mappo_max_local_revisit=0.30,
                evidence_priority_trigger=0.16,
            ),
        ),
        ("balanced", HMAPPOEARSConfig()),
        (
            "mappo-open",
            HMAPPOEARSConfig(
                mappo_uncertainty_min=0.28,
                mappo_uncertainty_max=0.94,
                mappo_max_local_revisit=0.50,
                evidence_priority_trigger=0.24,
            ),
        ),
    ]


def _select_two_stage(
    candidates: list[tuple[str, Any]],
    make_policy: Callable[[Any, int], Any],
    prepared_train: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    prepared_validation: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    train_domains: list[tuple[dict[str, Any], str]],
    validation_domains: list[tuple[dict[str, Any], str]],
    args: argparse.Namespace,
    *,
    strategy: str,
    seed: int,
) -> tuple[Any, dict[str, float], list[dict[str, Any]]]:
    history: list[dict[str, Any]] = []
    training_scores: list[tuple[float, str, Any]] = []
    for index, (label, config) in enumerate(candidates):
        rows = _evaluate(
            lambda episode_seed, config=config: make_policy(config, episode_seed),
            prepared_train,
            train_domains,
            args,
            seed=seed + index * 1000,
            split="train",
            strategy=strategy,
        )
        train_score = float(np.mean([row["operational_score"] for row in rows]))
        training_scores.append((train_score, label, config))
        history.append(
            {
                "strategy": strategy,
                "candidate": label,
                "train_score": train_score,
                "validation_mean_score": np.nan,
                "validation_robust_score": np.nan,
                "selected": 0,
            }
        )
    training_scores.sort(key=lambda item: item[0], reverse=True)
    shortlist = training_scores[: min(2, len(training_scores))]

    validated: list[tuple[float, str, Any, dict[str, float]]] = []
    for offset, (train_score, label, config) in enumerate(shortlist):
        rows = _evaluate(
            lambda episode_seed, config=config: make_policy(config, episode_seed),
            prepared_validation,
            validation_domains,
            args,
            seed=seed + 50000 + offset * 1000,
            split="validation",
            strategy=strategy,
        )
        stats = validation_selection_stats(rows)
        validated.append((stats["robust_score"], label, config, stats))
        for record in history:
            if record["candidate"] == label:
                record.update(
                    {
                        "validation_mean_score": stats["mean_score"],
                        "validation_ci95": stats["ci95"],
                        "validation_domain_std": stats["domain_std"],
                        "validation_worst_domain_score": stats["worst_domain_score"],
                        "validation_robust_score": stats["robust_score"],
                    }
                )
                break
    validated.sort(key=lambda item: item[0], reverse=True)
    _, label, config, stats = validated[0]
    for record in history:
        if record["candidate"] == label:
            record["selected"] = 1
    return config, dict(stats), history


def main() -> None:
    args = parse_args()
    if args.quick:
        args.agents = min(args.agents, 4)
        args.grid_size = min(args.grid_size, 20)
        args.max_steps = min(args.max_steps, 50)
        args.train_episodes = 1
        args.validation_episodes = min(args.validation_episodes, 2)
        args.validation_repeats = 1

    protocol = load_protocol(args.protocol)
    integrity = validate_protocol(protocol)
    if not all(integrity.values()):
        raise RuntimeError(f"Protocol integrity check failed: {integrity}")

    train_cities = select_cities(protocol, "train")
    train_zones = start_zones_for_split(protocol, "train")
    validation_cities = select_cities(protocol, "validation")
    validation_zones = start_zones_for_split(protocol, "validation")
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
        split="EARS training",
    )
    prepared_validation = v3._prepare(
        validation_cities,
        grid_size=args.grid_size,
        seed=args.seed + 70000,
        cache_dir=args.cache_dir,
        offline=args.offline,
        require_real_data=args.require_real_data,
        split="EARS validation",
    )
    train_domains = _domains(train_cities, train_zones, args.train_episodes)
    validation_domains = _domains(
        validation_cities,
        validation_zones,
        args.validation_episodes * args.validation_repeats,
    )
    if not train_domains or not validation_domains:
        raise RuntimeError("EARS requires disjoint train and validation domains")

    output = Path(args.output_root)
    checkpoints = Path(args.model_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    base_config, base_stats, base_history = _select_two_stage(
        _ears_candidates(),
        lambda config, seed: EARSPolicy(seed=seed, config=config),
        prepared_train,
        prepared_validation,
        train_domains,
        validation_domains,
        args,
        strategy="EARS-Safe",
        seed=args.seed,
    )
    EARSPolicy(seed=args.seed, config=base_config).save_checkpoint(
        checkpoints / "ears_safe.json",
        {
            "selected_utc": datetime.now(timezone.utc).isoformat(),
            "validation": base_stats,
            "selection_rule": "train shortlist + validation robust score; test/SWAP never consulted",
        },
    )

    np_config, np_stats, np_history = _select_two_stage(
        _np_candidates(base_config),
        lambda config, seed: EARSNegativePheromonePolicy(seed=seed, config=config),
        prepared_train,
        prepared_validation,
        train_domains,
        validation_domains,
        args,
        strategy="EARS-NP-Safe",
        seed=args.seed + 100000,
    )
    EARSNegativePheromonePolicy(seed=args.seed, config=np_config).save_checkpoint(
        checkpoints / "ears_np_safe.json",
        {
            "selected_utc": datetime.now(timezone.utc).isoformat(),
            "validation": np_stats,
            "selection_rule": "train shortlist + validation robust score; test/SWAP never consulted",
        },
    )

    mappo_path = checkpoints / "mappo_safe.json"
    if not mappo_path.exists():
        raise FileNotFoundError(f"H-MAPPO-EARS requires trained MAPPO checkpoint: {mappo_path}")

    h_config, h_stats, h_history = _select_two_stage(
        _h_candidates(),
        lambda config, seed: HMAPPOEARSPolicy(
            seed=seed,
            config=np_config,
            hierarchical_config=config,
            mappo_model_path=mappo_path,
        ),
        prepared_train,
        prepared_validation,
        train_domains,
        validation_domains,
        args,
        strategy="H-MAPPO-EARS-Safe",
        seed=args.seed + 200000,
    )
    HMAPPOEARSPolicy(
        seed=args.seed,
        config=np_config,
        hierarchical_config=h_config,
        mappo_model_path=mappo_path,
    ).save_checkpoint(
        checkpoints / "h_mappo_ears_safe.json",
        {
            "selected_utc": datetime.now(timezone.utc).isoformat(),
            "validation": h_stats,
            "selection_rule": "train shortlist + validation robust score; test/SWAP never consulted",
        },
    )

    history = pd.DataFrame(base_history + np_history + h_history)
    history.to_csv(output / "ears_candidate_history.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "strategy": "EARS-Safe",
                "validation_mean_score": base_stats["mean_score"],
                "validation_robust_score": base_stats["robust_score"],
                "validation_worst_domain_score": base_stats["worst_domain_score"],
                "negative_pheromone_weight": base_config.negative_pheromone_weight,
            },
            {
                "strategy": "EARS-NP-Safe",
                "validation_mean_score": np_stats["mean_score"],
                "validation_robust_score": np_stats["robust_score"],
                "validation_worst_domain_score": np_stats["worst_domain_score"],
                "negative_pheromone_weight": np_config.negative_pheromone_weight,
            },
            {
                "strategy": "H-MAPPO-EARS-Safe",
                "validation_mean_score": h_stats["mean_score"],
                "validation_robust_score": h_stats["robust_score"],
                "validation_worst_domain_score": h_stats["worst_domain_score"],
                "negative_pheromone_weight": np_config.negative_pheromone_weight,
            },
        ]
    ).sort_values("validation_robust_score", ascending=False)
    summary.to_csv(output / "ears_summary.csv", index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "version": "SafeSwarm v6 EARS",
        "algorithms": {
            "EARS-Safe": "Ant default + event-triggered global reallocation",
            "EARS-NP-Safe": "EARS + repulsive negative-pheromone exclusion halo",
            "H-MAPPO-EARS-Safe": "MAPPO-assisted hierarchical option controller over EARS primitives",
        },
        "ant_reference_unchanged": True,
        "selection_rule": "train shortlist then validation robust score; held-out test and SWAP never consulted",
        "protocol_integrity": integrity,
        "training_cities": [city["name"] for city in train_cities],
        "validation_cities": [city["name"] for city in validation_cities],
        "checkpoints": {
            "EARS-Safe": str(checkpoints / "ears_safe.json"),
            "EARS-NP-Safe": str(checkpoints / "ears_np_safe.json"),
            "H-MAPPO-EARS-Safe": str(checkpoints / "h_mappo_ears_safe.json"),
        },
        "source_mappo_checkpoint": str(mappo_path),
        "arguments": vars(args),
    }
    (output / "ears_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
