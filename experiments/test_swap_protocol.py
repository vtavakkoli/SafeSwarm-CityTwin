"""Evaluate frozen policies on SWAP seed-indexed alternate real-city target views.

SWAP (Seeded World Alternate Protocol) changes the held-out mission set for each
seed while preserving the cached OSM city geometry. It is an anti-overfitting
stress test only: no SWAP metric may select a checkpoint, PRISM pattern, or
PRISM-Ant fusion strength.
"""

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

from experiments.run_city_benchmark import run_episode  # noqa: E402
from src.agents.trainable_policies import evaluation_factories  # noqa: E402
from src.environment.city_twin import CityTwinEnvironment  # noqa: E402
from src.environment.obstacles import OSM_ATTRIBUTION, load_real_city_layers  # noqa: E402
from src.evaluation.metrics import episode_operational_score, rank_algorithms  # noqa: E402
from src.training.geography import (  # noqa: E402
    apply_start_zone,
    load_protocol,
    select_cities,
    start_zones_for_split,
    validate_protocol,
)
from src.training.swap_protocol import seeded_mission_view, swap_seed_series  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/real_city_protocol.json")
    parser.add_argument("--model-dir", default="results/train/checkpoints")
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=51042, help="Episode stochasticity seed")
    parser.add_argument("--dataset-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--dataset-seed-base", type=int, default=2042)
    parser.add_argument("--dataset-seed-count", type=int, default=3)
    parser.add_argument("--target-fraction", type=float, default=0.72)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output-root", default="results/swap-test")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--require-real-data", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _ci95(values: pd.Series) -> float:
    n = int(values.count())
    if n <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1) / np.sqrt(n))


def main() -> None:
    args = parse_args()
    dataset_seeds = list(args.dataset_seeds or swap_seed_series(
        args.dataset_seed_base, args.dataset_seed_count
    ))
    if args.quick:
        args.agents = min(args.agents, 4)
        args.grid_size = min(args.grid_size, 20)
        args.episodes = min(args.episodes, 2)
        args.max_steps = min(args.max_steps, 50)
        dataset_seeds = dataset_seeds[:1]

    protocol = load_protocol(args.protocol)
    integrity = validate_protocol(protocol)
    if not all(integrity.values()):
        raise RuntimeError(f"Protocol integrity check failed: {integrity}")
    cities = select_cities(protocol, "test")
    zones = start_zones_for_split(protocol, "test")
    if args.quick:
        cities = cities[:1]
        zones = zones[:1]

    records: list[dict[str, Any]] = []
    dataset_manifest: list[dict[str, Any]] = []
    signatures: set[str] = set()

    for city_index, city in enumerate(cities):
        base_layers = load_real_city_layers(
            city["place"], args.grid_size, args.seed + city_index,
            radius_m=int(city.get("radius_m", 1400)), cache_dir=args.cache_dir,
            allow_network=not args.offline,
        )
        metadata = dict(base_layers.get("metadata", {}))
        if args.require_real_data and metadata.get("source") != "openstreetmap":
            raise RuntimeError(f"SWAP city {city['name']} is not real OSM data: {metadata}")

        for dataset_seed in dataset_seeds:
            swap_layers = seeded_mission_view(
                base_layers, dataset_seed, target_fraction=args.target_fraction
            )
            swap_meta = dict(swap_layers.get("metadata", {}))
            signature = str(swap_meta["swap_signature"])
            signatures.add(f"{city['name']}:{signature}")
            dataset_manifest.append(
                {
                    "city": city["name"],
                    "place": city["place"],
                    "data_source": swap_meta.get("source"),
                    "swap_seed": int(dataset_seed),
                    "swap_signature": signature,
                    "dataset_changed": bool(swap_meta["swap_dataset_changed"]),
                    "original_target_count": int(swap_meta["swap_original_target_count"]),
                    "target_count": int(swap_meta["swap_target_count"]),
                    "target_fraction": float(swap_meta["swap_target_fraction"]),
                }
            )

            for episode in range(args.episodes):
                zone = zones[episode % len(zones)]
                episode_seed = args.seed + episode + city_index * 10000 + dataset_seed * 100
                layers = apply_start_zone(swap_layers, args.grid_size, zone)
                for strategy, factory in evaluation_factories(episode_seed, args.model_dir).items():
                    env = CityTwinEnvironment(
                        grid_size=args.grid_size, n_agents=args.agents, seed=episode_seed,
                        place_name=city["place"], radius_m=int(city.get("radius_m", 1400)),
                        layers=layers, max_steps=args.max_steps, allow_network=False,
                    )
                    policy = factory()
                    metrics = run_episode(env, policy)
                    row = metrics.to_dict(
                        strategy=strategy, episode=episode, split="swap-test",
                        city=city["name"], place=city["place"], start_zone=zone,
                        data_source=env.data_source, agents=args.agents,
                        grid_size=args.grid_size, steps=env.steps, seed=episode_seed,
                    )
                    row.update(
                        {
                            "swap_seed": int(dataset_seed),
                            "swap_signature": signature,
                            "swap_target_count": int(swap_meta["swap_target_count"]),
                            "swap_original_target_count": int(swap_meta["swap_original_target_count"]),
                        }
                    )
                    row["operational_score"] = episode_operational_score(row)
                    records.append(row)
                    print(
                        f"[SWAP] city={city['name']} dataset-seed={dataset_seed} "
                        f"episode={episode + 1}/{args.episodes} strategy={strategy} "
                        f"score={row['operational_score']:.3f}", flush=True,
                    )

    output = Path(args.output_root)
    tables = output / "tables"
    output.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    _, overall = rank_algorithms(frame)
    overall_ci = frame.groupby("strategy")["operational_score"].apply(_ci95)
    overall = overall.merge(
        overall_ci.rename("operational_score_ci95"),
        left_on="strategy", right_index=True, how="left"
    ).sort_values("operational_score", ascending=False).reset_index(drop=True)
    overall["rank"] = np.arange(1, len(overall) + 1)

    seed_rows: list[dict[str, Any]] = []
    for (swap_seed, strategy), group in frame.groupby(["swap_seed", "strategy"]):
        seed_rows.append(
            {
                "swap_seed": int(swap_seed), "strategy": strategy,
                "operational_score": float(group["operational_score"].mean()),
                "operational_score_ci95": _ci95(group["operational_score"]),
                "weighted_target_discovery": float(group["weighted_target_discovery"].mean()),
                "coverage_ratio": float(group["coverage_ratio"].mean()),
                "redundant_coverage": float(group["redundant_coverage"].mean()),
            }
        )
    seed_ranking = pd.DataFrame(seed_rows).sort_values(
        ["swap_seed", "operational_score"], ascending=[True, False]
    )
    seed_ranking["seed_rank"] = seed_ranking.groupby("swap_seed").cumcount() + 1

    frame.to_csv(tables / "episode_results.csv", index=False)
    seed_ranking.to_csv(tables / "seed_ranking.csv", index=False)
    overall.to_csv(tables / "overall_ranking.csv", index=False)
    dataset_frame = pd.DataFrame(dataset_manifest)
    dataset_frame.to_csv(tables / "dataset_manifest.csv", index=False)

    expected_views = len(cities) * len(dataset_seeds)
    dataset_changed_pass = bool(dataset_frame["dataset_changed"].all())
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "SWAP: Seeded World Alternate Protocol",
        "arguments": {**vars(args), "dataset_seeds": dataset_seeds},
        "protocol_integrity": integrity,
        "data_attribution": OSM_ATTRIBUTION,
        "physical_city_geometry": "cached OSM geometry fixed per city",
        "dataset_change": "priority-stratified hidden mission target subset changes by SWAP seed",
        "dataset_changed_pass": dataset_changed_pass,
        "unique_dataset_views": int(len(signatures)),
        "expected_dataset_views": int(expected_views),
        "unique_signatures_pass": bool(len(signatures) == expected_views),
        "used_for_model_selection": False,
        "model_selection_exclusion": "SWAP runs occur only after all PPO/GRPO/PRISM/PRISM-Ant selection is frozen",
        "winner": overall.iloc[0].to_dict() if not overall.empty else {},
        "datasets": dataset_manifest,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
