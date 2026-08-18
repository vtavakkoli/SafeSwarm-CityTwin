"""Validation-gated v5 coordination/distillation upgrade for PPO-family policies.

This stage starts from the robust v4 checkpoints, adds the v5 observable global
frontier/goal coordinator and distills safe actions from AntSwarmSafe and
UA-HBAS-Safe on *training cities only*.  A checkpoint is replaced only when the
multi-domain validation robust score improves.  Test and SWAP datasets are never
consulted.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import train_real_city_policies_v3 as v3  # noqa: E402
from src.agents.registry import strategy_factories  # noqa: E402
from src.agents.trainable_policies import (  # noqa: E402
    TRAINABLE_POLICY_CLASSES,
    checkpoint_path,
)
from src.training.geography import (  # noqa: E402
    load_protocol,
    select_cities,
    start_zones_for_split,
    validate_protocol,
)
from src.training.teacher_bootstrap import collect_teacher_examples  # noqa: E402
from src.training.validation_selection import validation_selection_stats  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/real_city_protocol.json")
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=424)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--teacher-scenarios", type=int, default=9)
    parser.add_argument("--teacher-examples-per-scenario", type=int, default=1400)
    parser.add_argument("--imitation-learning-rate", type=float, default=0.008)
    parser.add_argument("--imitation-epochs", type=int, default=1)
    parser.add_argument("--validation-episodes", type=int, default=12)
    parser.add_argument("--validation-repeats", type=int, default=2)
    parser.add_argument("--validation-min-delta", type=float, default=0.001)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--model-dir", default="results/train/checkpoints")
    parser.add_argument("--output-root", default="results/train")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--require-real-data", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _validation(
    policy_cls: Any,
    checkpoint: Path,
    prepared_validation: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    validation_cities: list[dict[str, Any]],
    validation_zones: list[str],
    args: argparse.Namespace,
    seed_offset: int,
) -> dict[str, float]:
    rows, _, _ = v3._validation_scores(
        policy_cls,
        checkpoint,
        prepared_validation,
        validation_cities,
        validation_zones,
        episodes=args.validation_episodes,
        repeats=args.validation_repeats,
        agents=args.agents,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        seed=args.seed + 80000 + seed_offset,
        gamma=0.99,
        gae_lambda=0.97,
    )
    return validation_selection_stats(rows)


def _collect_examples(
    policy: Any,
    prepared_train: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    scenarios: list[tuple[dict[str, Any], str]],
    args: argparse.Namespace,
    strategy_index: int,
    round_index: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    examples: list[dict[str, Any]] = []
    teacher_counts = {"AntSwarmSafe": 0, "UA-HBAS-Safe": 0}
    count = min(max(1, args.teacher_scenarios), len(scenarios))
    teachers = ("AntSwarmSafe", "UA-HBAS-Safe")
    for index in range(count):
        city, zone = scenarios[(index + round_index) % len(scenarios)]
        city_info, layers = prepared_train[city["name"]]
        seed = args.seed + 100000 + 10000 * strategy_index + 1000 * round_index + index
        env = v3._environment(
            city_info,
            layers,
            zone,
            grid_size=args.grid_size,
            agents=args.agents,
            seed=seed,
            max_steps=args.max_steps,
        )
        teacher_name = teachers[index % len(teachers)]
        teacher = strategy_factories(seed=seed)[teacher_name]()
        batch = collect_teacher_examples(
            policy,
            teacher,
            env,
            max_examples=args.teacher_examples_per_scenario,
        )
        examples.extend(batch)
        teacher_counts[teacher_name] += len(batch)
        print(
            f"[v5-distill] {policy.name} teacher={teacher_name} city={city_info['name']} "
            f"zone={zone} examples={len(batch)} total={len(examples)}",
            flush=True,
        )
    return examples, teacher_counts


def main() -> None:
    args = parse_args()
    if args.quick:
        args.agents = min(args.agents, 4)
        args.grid_size = min(args.grid_size, 20)
        args.max_steps = min(args.max_steps, 50)
        args.rounds = 1
        args.teacher_scenarios = 1
        args.teacher_examples_per_scenario = min(args.teacher_examples_per_scenario, 300)
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
        split="v5 coordination training",
    )
    prepared_validation = v3._prepare(
        validation_cities,
        grid_size=args.grid_size,
        seed=args.seed + 70000,
        cache_dir=args.cache_dir,
        offline=args.offline,
        require_real_data=args.require_real_data,
        split="v5 coordination validation",
    )
    scenarios = [
        (city, zone)
        for zone in sorted(train_zones, key=lambda value: (value != "center", value))
        for city in train_cities
    ]

    model_dir = Path(args.model_dir)
    output = Path(args.output_root)
    candidates = output / "coordination-candidates"
    improvements = output / "coordination-improvements"
    candidates.mkdir(parents=True, exist_ok=True)
    improvements.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    strategy_summary: list[dict[str, Any]] = []

    for strategy_index, (strategy, policy_cls) in enumerate(TRAINABLE_POLICY_CLASSES.items()):
        final_path = checkpoint_path(model_dir, strategy)
        if not final_path.exists():
            raise FileNotFoundError(f"Missing base checkpoint for v5 upgrade: {final_path}")

        baseline = _validation(
            policy_cls, final_path, prepared_validation, validation_cities,
            validation_zones, args, strategy_index * 1000,
        )
        best = dict(baseline)
        accepted = 0
        total_examples = 0
        last_imitation = {
            "imitation_loss": 0.0,
            "imitation_accuracy": 0.0,
            "imitation_examples": 0.0,
        }
        teacher_totals = {"AntSwarmSafe": 0, "UA-HBAS-Safe": 0}
        strategy_dir = improvements / strategy.lower().replace("-", "_")
        strategy_dir.mkdir(parents=True, exist_ok=True)

        for round_index in range(max(1, args.rounds)):
            # Reload the current validation-selected weight; rejected candidates
            # never leak into later rounds.
            policy = policy_cls(
                seed=args.seed + strategy_index * 100 + round_index,
                model_path=final_path,
                deterministic_eval=False,
            )
            examples, teacher_counts = _collect_examples(
                policy, prepared_train, scenarios, args, strategy_index, round_index
            )
            for key, value in teacher_counts.items():
                teacher_totals[key] += value
            total_examples += len(examples)
            if not examples:
                break

            last_imitation = policy.imitation_update(
                examples,
                learning_rate=args.imitation_learning_rate,
                epochs=args.imitation_epochs,
            )
            candidate = candidates / f"{strategy.lower().replace('-', '_')}_r{round_index + 1}.json"
            policy.save_checkpoint(
                candidate,
                metadata={
                    "v5_coordination_upgrade": True,
                    "upgrade_round": round_index + 1,
                    "teacher_counts": teacher_counts,
                    "imitation": last_imitation,
                    "selection_rule": "validation robust score only; test/SWAP never consulted",
                },
            )
            selection = _validation(
                policy_cls, candidate, prepared_validation, validation_cities,
                validation_zones, args, strategy_index * 1000 + (round_index + 1) * 100,
            )
            improved = selection["robust_score"] > best["robust_score"] + args.validation_min_delta
            records.append(
                {
                    "strategy": strategy,
                    "round": round_index + 1,
                    "examples": len(examples),
                    "imitation_loss": float(last_imitation.get("imitation_loss", 0.0)),
                    "imitation_accuracy": float(last_imitation.get("imitation_accuracy", 0.0)),
                    "baseline_robust_score": float(best["robust_score"]),
                    "candidate_mean_score": float(selection["mean_score"]),
                    "candidate_robust_score": float(selection["robust_score"]),
                    "candidate_worst_domain_score": float(selection["worst_domain_score"]),
                    "accepted": int(improved),
                }
            )
            print(
                f"[v5-validation] {strategy} round={round_index + 1} "
                f"robust={selection['robust_score']:.3f} baseline={best['robust_score']:.3f} "
                f"accepted={'yes' if improved else 'no'}",
                flush=True,
            )
            if improved:
                accepted += 1
                best = dict(selection)
                shutil.copy2(candidate, final_path)
                shutil.copy2(candidate, strategy_dir / f"round_{round_index + 1:02d}.json")

        strategy_summary.append(
            {
                "strategy": strategy,
                "baseline_validation_score": baseline["mean_score"],
                "baseline_validation_robust_score": baseline["robust_score"],
                "final_validation_score": best["mean_score"],
                "final_validation_robust_score": best["robust_score"],
                "final_validation_worst_domain_score": best["worst_domain_score"],
                "accepted_upgrades": accepted,
                "teacher_examples": total_examples,
                "ant_teacher_examples": teacher_totals["AntSwarmSafe"],
                "uahbas_teacher_examples": teacher_totals["UA-HBAS-Safe"],
                "last_imitation_loss": float(last_imitation.get("imitation_loss", 0.0)),
                "last_imitation_accuracy": float(last_imitation.get("imitation_accuracy", 0.0)),
            }
        )

    pd.DataFrame(records).to_csv(output / "coordination_upgrade_history.csv", index=False)
    summary = pd.DataFrame(strategy_summary).sort_values(
        "final_validation_robust_score", ascending=False
    )
    summary.to_csv(output / "coordination_upgrade_summary.csv", index=False)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "training_version": "SafeSwarm PPO/GRPO coordination v5",
        "arguments": vars(args),
        "protocol_integrity": integrity,
        "teachers": ["AntSwarmSafe", "UA-HBAS-Safe"],
        "students": list(TRAINABLE_POLICY_CLASSES),
        "upgrade": (
            "observable global frontier/goal coordination + teacher action distillation; "
            "candidate promoted only on robust validation improvement"
        ),
        "model_selection_excludes": ["test", "SWAP"],
    }
    (output / "coordination_upgrade_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
