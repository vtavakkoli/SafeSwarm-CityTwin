"""Tune SPARX probability-map weights and select X/+ /Star by validation.

The search controller is optimized on training cities only. At each round a
small paired parameter search chooses a training candidate, then the candidate
is evaluated on the disjoint multi-domain validation split. Weights are saved
only when the validation robust score improves. The final SPARX-Safe pattern is
selected from X, Plus and Star using validation only; test cities are never read.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
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

from experiments.run_city_benchmark import run_episode  # noqa: E402
from experiments import train_real_city_policies_v3 as v3  # noqa: E402
from src.agents.sparx_pattern import (  # noqa: E402
    PATTERN_MODES,
    SPARXConfig,
    SPARXPolicy,
    SPARX_PARAMETER_BOUNDS,
    SPARX_TUNABLE_FIELDS,
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
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--train-scenarios-per-candidate", type=int, default=4)
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--validation-repeats", type=int, default=2)
    parser.add_argument("--mutation-scale", type=float, default=0.16)
    parser.add_argument("--validation-min-delta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output-root", default="results/train")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--require-real-data", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _config_from_dict(values: dict[str, float]) -> SPARXConfig:
    clipped: dict[str, float] = {}
    defaults = asdict(SPARXConfig())
    for name, default in defaults.items():
        low, high = SPARX_PARAMETER_BOUNDS[name]
        clipped[name] = float(np.clip(values.get(name, default), low, high))
    return SPARXConfig(**clipped)


def _mutate(
    config: SPARXConfig,
    rng: np.random.Generator,
    scale: float,
    direction: float,
    round_index: int,
) -> SPARXConfig:
    values = asdict(config)
    # Cover every field deterministically while still perturbing several coupled
    # search terms together. Plus/minus candidates use the same selected fields.
    count = min(5, len(SPARX_TUNABLE_FIELDS))
    start = (round_index * count) % len(SPARX_TUNABLE_FIELDS)
    names = [SPARX_TUNABLE_FIELDS[(start + index) % len(SPARX_TUNABLE_FIELDS)] for index in range(count)]
    jitter = rng.uniform(0.55, 1.20, size=count)
    for index, name in enumerate(names):
        low, high = SPARX_PARAMETER_BOUNDS[name]
        current = float(values[name])
        span = high - low
        if current <= 1e-9:
            proposal = current + direction * span * scale * 0.20 * float(jitter[index])
        else:
            proposal = current * float(np.exp(direction * scale * float(jitter[index])))
        values[name] = float(np.clip(proposal, low, high))
    return _config_from_dict(values)


def _episode_row(
    policy: SPARXPolicy,
    env: Any,
    *,
    split: str,
    city: str,
    zone: str,
    seed: int,
    episode: int,
    pattern: str,
) -> dict[str, Any]:
    metrics = run_episode(env, policy)
    row = metrics.to_dict(
        strategy=policy.name,
        episode=episode,
        split=split,
        city=city,
        start_zone=zone,
        seed=seed,
        data_source=env.data_source,
        agents=env.n_agents,
        grid_size=env.grid_size,
        steps=env.steps,
        sparx_pattern=pattern,
    )
    row["operational_score"] = episode_operational_score(row)
    diagnostics = policy.diagnostics()
    for key in (
        "sparx_memory_coverage",
        "sparx_memory_peak",
        "sparx_assignment_refreshes",
        "sparx_goal_switches",
        "sparx_probability_entropy",
        "sparx_probability_peak",
        "sparx_region_count",
        "safety_mask_rejections",
        "forced_fallbacks",
    ):
        row[key] = diagnostics.get(key, 0.0)
    return row


def _evaluate(
    config: SPARXConfig,
    pattern: str,
    prepared: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    scenarios: list[tuple[dict[str, Any], str]],
    *,
    agents: int,
    grid_size: int,
    max_steps: int,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode, (city, zone) in enumerate(scenarios):
        city_info, layers = prepared[city["name"]]
        episode_seed = seed + episode
        env = v3._environment(
            city_info,
            layers,
            zone,
            grid_size=grid_size,
            agents=agents,
            seed=episode_seed,
            max_steps=max_steps,
        )
        policy = SPARXPolicy(
            seed=episode_seed,
            pattern_mode=pattern,
            strategy_name=f"SPARX-{pattern.title()}-Safe",
            **asdict(config),
        )
        rows.append(
            _episode_row(
                policy,
                env,
                split=split,
                city=city_info["name"],
                zone=zone,
                seed=episode_seed,
                episode=episode,
                pattern=pattern,
            )
        )
    return rows


def _validation_scenarios(
    cities: list[dict[str, Any]],
    zones: list[str],
    episodes: int,
    repeats: int,
) -> list[tuple[dict[str, Any], str]]:
    domains = [(city, zone) for zone in zones for city in cities]
    if not domains:
        return []
    requested = max(1, episodes) * max(1, repeats)
    return [domains[index % len(domains)] for index in range(requested)]


def _checkpoint_name(pattern: str) -> str:
    return f"sparx_{pattern}_safe.json"


def main() -> None:
    args = parse_args()
    if args.quick:
        args.agents = min(args.agents, 4)
        args.grid_size = min(args.grid_size, 20)
        args.max_steps = min(args.max_steps, 50)
        args.rounds = min(args.rounds, 2)
        args.train_scenarios_per_candidate = 1
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
        split="SPARX training",
    )
    prepared_validation = v3._prepare(
        validation_cities,
        grid_size=args.grid_size,
        seed=args.seed + 70000,
        cache_dir=args.cache_dir,
        offline=args.offline,
        require_real_data=args.require_real_data,
        split="SPARX validation",
    )

    train_domains = [(city, zone) for zone in train_zones for city in train_cities]
    validation_domains = _validation_scenarios(
        validation_cities,
        validation_zones,
        args.validation_episodes,
        args.validation_repeats,
    )
    if not train_domains or not validation_domains:
        raise RuntimeError("SPARX requires training and validation domains")

    output = Path(args.output_root)
    checkpoints = output / "checkpoints"
    improvements = output / "sparx-validation-improvements"
    checkpoints.mkdir(parents=True, exist_ok=True)
    improvements.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    history: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    best_checkpoint_by_pattern: dict[str, Path] = {}

    for pattern_index, pattern in enumerate(PATTERN_MODES):
        current = SPARXConfig()
        best_validation = float("-inf")
        best_stats: dict[str, float] = {}
        best_round = -1
        best_training_score = float("-inf")
        pattern_dir = improvements / pattern
        pattern_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoints / _checkpoint_name(pattern)

        for round_index in range(max(1, args.rounds)):
            count = max(1, min(args.train_scenarios_per_candidate, len(train_domains)))
            offset = (round_index * count + pattern_index) % len(train_domains)
            paired_domains = [train_domains[(offset + index) % len(train_domains)] for index in range(count)]
            plus = _mutate(current, rng, args.mutation_scale, +1.0, round_index)
            minus = _mutate(current, rng, args.mutation_scale, -1.0, round_index)
            candidates = [("incumbent", current), ("plus", plus), ("minus", minus)]

            training_results: list[tuple[float, str, SPARXConfig, list[dict[str, Any]]]] = []
            for candidate_index, (label, config) in enumerate(candidates):
                rows = _evaluate(
                    config,
                    pattern,
                    prepared_train,
                    paired_domains,
                    agents=args.agents,
                    grid_size=args.grid_size,
                    max_steps=args.max_steps,
                    seed=args.seed + 100000 * pattern_index + 1000 * round_index + 100 * candidate_index,
                    split="train",
                )
                score = float(np.mean([row["operational_score"] for row in rows]))
                training_results.append((score, label, config, rows))

            training_results.sort(key=lambda item: item[0], reverse=True)
            train_score, chosen_label, current, chosen_rows = training_results[0]
            best_training_score = max(best_training_score, train_score)

            validation_rows = _evaluate(
                current,
                pattern,
                prepared_validation,
                validation_domains,
                agents=args.agents,
                grid_size=args.grid_size,
                max_steps=args.max_steps,
                seed=args.seed + 500000 + 10000 * pattern_index + 100 * round_index,
                split="validation",
            )
            stats = validation_selection_stats(validation_rows)
            improved = stats["robust_score"] > best_validation + args.validation_min_delta

            if improved:
                best_validation = stats["robust_score"]
                best_stats = dict(stats)
                best_round = round_index + 1
                policy = SPARXPolicy(
                    seed=args.seed,
                    pattern_mode=pattern,
                    strategy_name=f"SPARX-{pattern.title()}-Safe",
                    **asdict(current),
                )
                metadata = {
                    "saved_utc": datetime.now(timezone.utc).isoformat(),
                    "protocol": args.protocol,
                    "pattern": pattern,
                    "tuning_round": round_index + 1,
                    "train_score": train_score,
                    "validation": stats,
                    "selection_rule": "validation robust score improvement only; test never consulted",
                }
                policy.save_checkpoint(checkpoint, metadata)
                improvement_path = pattern_dir / f"round_{round_index + 1:03d}.json"
                shutil.copy2(checkpoint, improvement_path)

            history.append(
                {
                    "pattern": pattern,
                    "round": round_index + 1,
                    "chosen_candidate": chosen_label,
                    "train_score": train_score,
                    "validation_mean_score": stats["mean_score"],
                    "validation_ci95": stats["ci95"],
                    "validation_domain_std": stats["domain_std"],
                    "validation_worst_domain_score": stats["worst_domain_score"],
                    "validation_robust_score": stats["robust_score"],
                    "saved_validation_improvement": int(improved),
                    **{f"weight_{key}": value for key, value in asdict(current).items()},
                }
            )
            print(
                f"[sparx] pattern={pattern} round={round_index + 1}/{args.rounds} "
                f"train={train_score:.3f} validation={stats['mean_score']:.3f} "
                f"robust={stats['robust_score']:.3f} saved={'yes' if improved else 'no'}",
                flush=True,
            )

        if not checkpoint.exists():
            policy = SPARXPolicy(
                seed=args.seed,
                pattern_mode=pattern,
                strategy_name=f"SPARX-{pattern.title()}-Safe",
                **asdict(current),
            )
            policy.save_checkpoint(
                checkpoint,
                {
                    "saved_utc": datetime.now(timezone.utc).isoformat(),
                    "pattern": pattern,
                    "selection_rule": "fallback final candidate; test never consulted",
                },
            )
            best_validation = float("nan")
            best_stats = {}
            best_round = args.rounds

        best_checkpoint_by_pattern[pattern] = checkpoint
        summaries.append(
            {
                "pattern": pattern,
                "checkpoint": str(checkpoint),
                "best_round": best_round,
                "best_training_score": best_training_score,
                "best_validation_robust_score": best_validation,
                "best_validation_score": best_stats.get("mean_score", float("nan")),
                "best_validation_ci95": best_stats.get("ci95", float("nan")),
                "best_validation_domain_std": best_stats.get("domain_std", float("nan")),
                "best_validation_worst_domain_score": best_stats.get("worst_domain_score", float("nan")),
            }
        )

    summary = pd.DataFrame(summaries).sort_values(
        "best_validation_robust_score", ascending=False
    ).reset_index(drop=True)
    selected_pattern = str(summary.iloc[0]["pattern"])
    selected_source = best_checkpoint_by_pattern[selected_pattern]
    selected_policy = SPARXPolicy(
        seed=args.seed,
        model_path=selected_source,
        strategy_name="SPARX-Safe",
    )
    selected_path = checkpoints / "sparx_safe.json"
    selected_policy.save_checkpoint(
        selected_path,
        {
            **selected_policy.checkpoint_metadata,
            "selected_pattern": selected_pattern,
            "selected_utc": datetime.now(timezone.utc).isoformat(),
            "pattern_selection": "highest validation robust score across X, Plus and Star; test never consulted",
        },
    )

    pd.DataFrame(history).to_csv(output / "sparx_tuning_history.csv", index=False)
    summary.to_csv(output / "sparx_pattern_summary.csv", index=False)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "SPARX: Swarm Probability-map Allocation & Region eXploration",
        "patterns": list(PATTERN_MODES),
        "selected_pattern": selected_pattern,
        "selected_checkpoint": str(selected_path),
        "protocol_integrity": integrity,
        "training_cities": [city["name"] for city in train_cities],
        "training_start_zones": train_zones,
        "validation_cities": [city["name"] for city in validation_cities],
        "validation_start_zones": validation_zones,
        "selection_rule": (
            "training-driven paired parameter search; checkpoint saved only on validation robust-score increase; "
            "pattern selected on validation; test never consulted"
        ),
        "probability_map_semantics": (
            "normalized observable search utility, not calibrated hidden-target probability"
        ),
        "arguments": vars(args),
    }
    (output / "sparx_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"[sparx] selected pattern={selected_pattern} checkpoint={selected_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
