"""Tune PRISM X/+ /Star and PRISM-Ant using training + validation only.

PRISM replaces the v4 SPARX name.  Pattern weights are trained by paired
parameter search on training cities and archived only when the multi-domain
validation robust score improves.  After the best PRISM pattern is selected,
PRISM-Ant fusion strengths are compared on validation.  Test and SWAP views are
never consulted.
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

from experiments import train_real_city_policies_v3 as v3  # noqa: E402
from experiments.run_city_benchmark import run_episode  # noqa: E402
from src.agents.prism_ant import PRISMAntConfig, PRISMAntPolicy  # noqa: E402
from src.agents.prism_pattern import (  # noqa: E402
    PATTERN_MODES,
    PRISMConfig,
    PRISMPolicy,
    PRISM_PARAMETER_BOUNDS,
    PRISM_TUNABLE_FIELDS,
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


def _config(values: dict[str, float]) -> PRISMConfig:
    defaults = asdict(PRISMConfig())
    defaults.update(values)
    for name, (low, high) in PRISM_PARAMETER_BOUNDS.items():
        defaults[name] = float(np.clip(defaults[name], low, high))
    return PRISMConfig(**defaults)


def _mutate(
    config: PRISMConfig,
    rng: np.random.Generator,
    scale: float,
    direction: float,
    round_index: int,
) -> PRISMConfig:
    values = asdict(config)
    count = min(5, len(PRISM_TUNABLE_FIELDS))
    start = (round_index * count) % len(PRISM_TUNABLE_FIELDS)
    for offset in range(count):
        name = PRISM_TUNABLE_FIELDS[(start + offset) % len(PRISM_TUNABLE_FIELDS)]
        low, high = PRISM_PARAMETER_BOUNDS[name]
        current = float(values[name])
        jitter = float(rng.uniform(0.55, 1.20))
        if abs(current) <= 1e-12:
            proposal = current + direction * (high - low) * scale * 0.20 * jitter
        else:
            proposal = current * float(np.exp(direction * scale * jitter))
        values[name] = float(np.clip(proposal, low, high))
    return _config(values)


def _episode_row(
    policy: Any,
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
    )
    row["prism_pattern"] = pattern
    row["operational_score"] = episode_operational_score(row)
    return row


def _domains(
    cities: list[dict[str, Any]],
    zones: list[str],
    requested: int,
) -> list[tuple[dict[str, Any], str]]:
    base = [(city, zone) for zone in zones for city in cities]
    if not base:
        return []
    return [base[index % len(base)] for index in range(max(1, requested))]


def _evaluate_prism(
    config: PRISMConfig,
    pattern: str,
    prepared: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    domains: list[tuple[dict[str, Any], str]],
    args: argparse.Namespace,
    *,
    seed: int,
    split: str,
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
        policy = PRISMPolicy(
            seed=episode_seed,
            pattern_mode=pattern,
            strategy_name=f"PRISM-{pattern.title()}-Safe",
            **asdict(config),
        )
        rows.append(
            _episode_row(
                policy, env, split=split, city=city_info["name"], zone=zone,
                seed=episode_seed, episode=episode, pattern=pattern,
            )
        )
    return rows


def _evaluate_hybrid(
    prism_config: PRISMConfig,
    hybrid_config: PRISMAntConfig,
    pattern: str,
    prepared: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    domains: list[tuple[dict[str, Any], str]],
    args: argparse.Namespace,
    *,
    seed: int,
    split: str,
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
        policy = PRISMAntPolicy(
            seed=episode_seed,
            pattern_mode=pattern,
            strategy_name="PRISM-Ant-Safe",
            hybrid_config=hybrid_config,
            **asdict(prism_config),
        )
        rows.append(
            _episode_row(
                policy, env, split=split, city=city_info["name"], zone=zone,
                seed=episode_seed, episode=episode, pattern=pattern,
            )
        )
    return rows


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
        split="PRISM training",
    )
    prepared_validation = v3._prepare(
        validation_cities,
        grid_size=args.grid_size,
        seed=args.seed + 70000,
        cache_dir=args.cache_dir,
        offline=args.offline,
        require_real_data=args.require_real_data,
        split="PRISM validation",
    )

    train_domains = [(city, zone) for zone in train_zones for city in train_cities]
    validation_domains = _domains(
        validation_cities,
        validation_zones,
        args.validation_episodes * args.validation_repeats,
    )
    if not train_domains or not validation_domains:
        raise RuntimeError("PRISM requires train and validation domains")

    output = Path(args.output_root)
    checkpoints = output / "checkpoints"
    improvements = output / "prism-validation-improvements"
    checkpoints.mkdir(parents=True, exist_ok=True)
    improvements.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    history: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    best_paths: dict[str, Path] = {}

    for pattern_index, pattern in enumerate(PATTERN_MODES):
        current = PRISMConfig()
        best_robust = float("-inf")
        best_stats: dict[str, float] = {}
        best_round = -1
        best_train = float("-inf")
        checkpoint = checkpoints / f"prism_{pattern}_safe.json"
        pattern_dir = improvements / pattern
        pattern_dir.mkdir(parents=True, exist_ok=True)

        for round_index in range(max(1, args.rounds)):
            count = max(1, min(args.train_scenarios_per_candidate, len(train_domains)))
            offset = (round_index * count + pattern_index) % len(train_domains)
            paired = [train_domains[(offset + i) % len(train_domains)] for i in range(count)]
            candidates = [
                ("incumbent", current),
                ("plus", _mutate(current, rng, args.mutation_scale, +1.0, round_index)),
                ("minus", _mutate(current, rng, args.mutation_scale, -1.0, round_index)),
            ]
            scored: list[tuple[float, str, PRISMConfig]] = []
            for candidate_index, (label, config) in enumerate(candidates):
                rows = _evaluate_prism(
                    config, pattern, prepared_train, paired, args,
                    seed=args.seed + pattern_index * 100000 + round_index * 1000 + candidate_index * 100,
                    split="train",
                )
                scored.append((float(np.mean([r["operational_score"] for r in rows])), label, config))
            scored.sort(key=lambda item: item[0], reverse=True)
            train_score, chosen, current = scored[0]
            best_train = max(best_train, train_score)

            val_rows = _evaluate_prism(
                current, pattern, prepared_validation, validation_domains, args,
                seed=args.seed + 500000 + pattern_index * 10000 + round_index * 100,
                split="validation",
            )
            stats = validation_selection_stats(val_rows)
            improved = stats["robust_score"] > best_robust + args.validation_min_delta
            if improved:
                best_robust = stats["robust_score"]
                best_stats = dict(stats)
                best_round = round_index + 1
                policy = PRISMPolicy(
                    seed=args.seed,
                    pattern_mode=pattern,
                    strategy_name=f"PRISM-{pattern.title()}-Safe",
                    **asdict(current),
                )
                policy.save_checkpoint(
                    checkpoint,
                    {
                        "saved_utc": datetime.now(timezone.utc).isoformat(),
                        "pattern": pattern,
                        "tuning_round": round_index + 1,
                        "train_score": train_score,
                        "validation": stats,
                        "selection_rule": "validation robust-score improvement only; test/SWAP never consulted",
                    },
                )
                shutil.copy2(checkpoint, pattern_dir / f"round_{round_index + 1:03d}.json")

            history.append(
                {
                    "pattern": pattern,
                    "round": round_index + 1,
                    "chosen_candidate": chosen,
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
                f"[prism] pattern={pattern} round={round_index + 1}/{args.rounds} "
                f"train={train_score:.3f} validation={stats['mean_score']:.3f} "
                f"robust={stats['robust_score']:.3f} saved={'yes' if improved else 'no'}",
                flush=True,
            )

        if not checkpoint.exists():
            PRISMPolicy(seed=args.seed, pattern_mode=pattern, **asdict(current)).save_checkpoint(
                checkpoint, {"pattern": pattern, "selection_rule": "fallback; test/SWAP never consulted"}
            )
        best_paths[pattern] = checkpoint
        summaries.append(
            {
                "pattern": pattern,
                "checkpoint": str(checkpoint),
                "best_round": best_round,
                "best_training_score": best_train,
                "best_validation_robust_score": best_robust,
                "best_validation_score": best_stats.get("mean_score", float("nan")),
                "best_validation_ci95": best_stats.get("ci95", float("nan")),
                "best_validation_domain_std": best_stats.get("domain_std", float("nan")),
                "best_validation_worst_domain_score": best_stats.get("worst_domain_score", float("nan")),
            }
        )

    pattern_summary = pd.DataFrame(summaries).sort_values(
        "best_validation_robust_score", ascending=False
    ).reset_index(drop=True)
    selected_pattern = str(pattern_summary.iloc[0]["pattern"])
    selected_source = best_paths[selected_pattern]
    selected = PRISMPolicy(
        seed=args.seed, model_path=selected_source, strategy_name="PRISM-Safe"
    )
    selected.save_checkpoint(
        checkpoints / "prism_safe.json",
        {
            **selected.checkpoint_metadata,
            "selected_pattern": selected_pattern,
            "selected_utc": datetime.now(timezone.utc).isoformat(),
            "selection_rule": "best pattern by validation robust score; test/SWAP never consulted",
        },
    )

    # Hybrid search: preserve the selected PRISM global controller and select the
    # Ant/PRISM fusion strength only from validation.  These presets deliberately
    # span PRISM-heavy through Ant-heavy behavior.
    selected_payload = json.loads(selected_source.read_text(encoding="utf-8"))
    selected_config = _config(dict(selected_payload.get("config", {})))
    blends = [0.45, 0.60, 0.72, 0.82]
    if args.quick:
        blends = [0.55, 0.72]
    hybrid_rows: list[dict[str, Any]] = []
    hybrid_candidates: list[tuple[float, PRISMAntConfig, dict[str, float]]] = []
    for index, blend in enumerate(blends):
        config = PRISMAntConfig(
            ant_blend=blend,
            ant_revisit_penalty=0.68 + 0.30 * blend,
            ant_congestion_penalty=0.62 + 0.18 * blend,
        )
        rows = _evaluate_hybrid(
            selected_config, config, selected_pattern,
            prepared_validation, validation_domains, args,
            seed=args.seed + 900000 + index * 1000,
            split="validation",
        )
        stats = validation_selection_stats(rows)
        hybrid_candidates.append((stats["robust_score"], config, stats))
        hybrid_rows.append(
            {
                "ant_blend": blend,
                "ant_revisit_penalty": config.ant_revisit_penalty,
                "ant_congestion_penalty": config.ant_congestion_penalty,
                **{f"validation_{key}": value for key, value in stats.items()},
            }
        )
        print(
            f"[prism-ant] blend={blend:.2f} validation={stats['mean_score']:.3f} "
            f"robust={stats['robust_score']:.3f}",
            flush=True,
        )
    hybrid_candidates.sort(key=lambda item: item[0], reverse=True)
    _, best_hybrid, best_hybrid_stats = hybrid_candidates[0]
    PRISMAntPolicy(
        seed=args.seed,
        pattern_mode=selected_pattern,
        hybrid_config=best_hybrid,
        strategy_name="PRISM-Ant-Safe",
        **asdict(selected_config),
    ).save_checkpoint(
        checkpoints / "prism_ant_safe.json",
        {
            "selected_pattern": selected_pattern,
            "validation": best_hybrid_stats,
            "selection_rule": "PRISM-Ant fusion selected on validation only; test/SWAP never consulted",
            "source_prism_checkpoint": str(selected_source),
        },
    )

    pd.DataFrame(history).to_csv(output / "prism_tuning_history.csv", index=False)
    pattern_summary.to_csv(output / "prism_pattern_summary.csv", index=False)
    pd.DataFrame(hybrid_rows).sort_values(
        "validation_robust_score", ascending=False
    ).to_csv(output / "prism_ant_summary.csv", index=False)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "PRISM: Probability-guided Region-Integrated Search with Memory",
        "hybrid_algorithm": "PRISM-Ant: PRISM global allocation + AntSwarm local search",
        "patterns": list(PATTERN_MODES),
        "selected_pattern": selected_pattern,
        "selected_checkpoint": str(checkpoints / "prism_safe.json"),
        "hybrid_checkpoint": str(checkpoints / "prism_ant_safe.json"),
        "hybrid_ant_blend": best_hybrid.ant_blend,
        "protocol_integrity": integrity,
        "training_cities": [city["name"] for city in train_cities],
        "training_start_zones": train_zones,
        "validation_cities": [city["name"] for city in validation_cities],
        "validation_start_zones": validation_zones,
        "selection_rule": (
            "training-driven paired PRISM parameter search; weights, pattern, and hybrid "
            "fusion selected only by validation robust score; test/SWAP never consulted"
        ),
        "probability_map_semantics": "normalized observable search utility, not calibrated hidden-target probability",
        "arguments": vars(args),
    }
    (output / "prism_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
