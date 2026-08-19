"""Run frozen EARS mechanism ablations on publication held-out cities.

No result from this script can modify a checkpoint.  Every policy uses the
validation-selected EARS checkpoint; ablations only disable one mechanism after
selection so the experiment measures causal contribution rather than retuning.
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
from src.agents.bio_swarm_agents import AntSwarmPolicy  # noqa: E402
from src.agents.ears_ablations_v7 import EARSAblationPolicy  # noqa: E402
from src.agents.ears_v6 import EARSPolicy  # noqa: E402
from src.environment.city_twin import CityTwinEnvironment  # noqa: E402
from src.environment.obstacles import OSM_ATTRIBUTION, load_real_city_layers  # noqa: E402
from src.evaluation.metrics import episode_operational_score  # noqa: E402
from src.training.geography import (  # noqa: E402
    apply_start_zone,
    load_protocol,
    select_cities,
    start_zones_for_split,
    validate_protocol,
)

METRICS = [
    "operational_score",
    "mission_success_rate",
    "weighted_target_discovery",
    "coverage_ratio",
    "energy_consumption",
    "redundant_coverage",
    "communication_efficiency",
    "distance_travelled",
    "runtime_seconds",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", default="configs/publication_protocol_v7.json")
    p.add_argument("--model-dir", default="results/train/checkpoints")
    p.add_argument("--agents", type=int, default=8)
    p.add_argument("--grid-size", type=int, default=40)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=160)
    p.add_argument("--seed", type=int, default=71042)
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--output-root", default="results/publication/ablations")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--require-real-data", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def _mean_std(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(by, keys))
        row["n"] = int(len(group))
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        for name in (
            "ears_event_triggers",
            "ears_stagnation_triggers",
            "ears_revisit_triggers",
            "ears_congestion_triggers",
            "ears_ant_fraction",
            "ears_relocation_fraction",
        ):
            row[f"{name}_mean"] = float(group[name].mean()) if name in group else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.quick:
        args.agents = min(args.agents, 4)
        args.grid_size = min(args.grid_size, 20)
        args.episodes = min(args.episodes, 2)
        args.max_steps = min(args.max_steps, 50)

    protocol = load_protocol(args.protocol)
    integrity = validate_protocol(protocol)
    if not all(integrity.values()):
        raise RuntimeError(f"Protocol integrity check failed: {integrity}")
    cities = select_cities(protocol, "test")
    zones = start_zones_for_split(protocol, "test")
    if args.quick:
        cities = cities[:2]
        zones = zones[:1]

    checkpoint = Path(args.model_dir) / "ears_safe.json"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Frozen EARS checkpoint is required: {checkpoint}")

    records: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    ablations = [
        "stagnation_only",
        "revisit_only",
        "congestion_only",
        "no_energy_battery",
    ]

    for city_index, city in enumerate(cities):
        layers = load_real_city_layers(
            city["place"], args.grid_size, args.seed + city_index,
            radius_m=int(city.get("radius_m", 1400)), cache_dir=args.cache_dir,
            allow_network=not args.offline,
        )
        meta = dict(layers.get("metadata", {}))
        meta.update({"city": city["name"], "split": "publication-test"})
        if args.require_real_data and meta.get("source") != "openstreetmap":
            raise RuntimeError(f"Ablation city {city['name']} is not real OSM data: {meta}")
        metadata.append(meta)

        for episode in range(args.episodes):
            zone = zones[episode % len(zones)]
            episode_seed = args.seed + city_index * 10000 + episode
            zoned = apply_start_zone(layers, args.grid_size, zone)
            factories: dict[str, Any] = {
                # AntSwarmPolicy is deterministic and has no seed constructor;
                # environment stochasticity remains paired through episode_seed.
                "AntSwarmSafe": lambda: AntSwarmPolicy(),
                "EARS-Safe": lambda seed=episode_seed: EARSPolicy(
                    seed=seed, model_path=checkpoint, strategy_name="EARS-Safe"
                ),
            }
            for ablation in ablations:
                factories[f"EARS-Ablation-{ablation}"] = (
                    lambda ablation=ablation, seed=episode_seed: EARSAblationPolicy(
                        seed=seed, model_path=checkpoint, ablation=ablation
                    )
                )

            for strategy, factory in factories.items():
                env = CityTwinEnvironment(
                    grid_size=args.grid_size, n_agents=args.agents, seed=episode_seed,
                    place_name=city["place"], radius_m=int(city.get("radius_m", 1400)),
                    layers=zoned, max_steps=args.max_steps, allow_network=False,
                )
                policy = factory()
                metrics = run_episode(env, policy)
                row = metrics.to_dict(
                    strategy=policy.name if strategy.startswith("EARS-Ablation") else strategy,
                    episode=episode, split="publication-ablation", city=city["name"],
                    place=city["place"], start_zone=zone, data_source=env.data_source,
                    agents=args.agents, grid_size=args.grid_size, steps=env.steps,
                    seed=episode_seed,
                )
                row["operational_score"] = episode_operational_score(row)
                diagnostics = policy.diagnostics() if hasattr(policy, "diagnostics") else {}
                for name in (
                    "ears_event_triggers",
                    "ears_stagnation_triggers",
                    "ears_revisit_triggers",
                    "ears_congestion_triggers",
                    "ears_ant_fraction",
                    "ears_relocation_fraction",
                    "ears_negative_escape_fraction",
                    "ears_return_fraction",
                ):
                    row[name] = diagnostics.get(name, 0.0)
                row["ears_ablation"] = diagnostics.get("ears_ablation", "full" if strategy == "EARS-Safe" else "baseline")
                records.append(row)
                print(
                    f"[ablation-v7] city={city['name']} episode={episode + 1}/{args.episodes} "
                    f"strategy={row['strategy']} score={row['operational_score']:.3f} "
                    f"triggers={row['ears_event_triggers']}", flush=True,
                )

    output = Path(args.output_root)
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(tables / "episode_results.csv", index=False)
    city_summary = _mean_std(frame, ["city", "strategy"])
    overall_summary = _mean_std(frame, ["strategy"])
    city_summary.to_csv(tables / "city_mean_std.csv", index=False)
    overall_summary.to_csv(tables / "overall_mean_std.csv", index=False)

    trigger_cols = [
        "ears_event_triggers", "ears_stagnation_triggers", "ears_revisit_triggers",
        "ears_congestion_triggers", "ears_ant_fraction", "ears_relocation_fraction",
    ]
    trigger_summary = (
        frame.groupby(["city", "strategy"], as_index=False)[trigger_cols]
        .mean(numeric_only=True)
        .sort_values(["city", "strategy"])
    )
    trigger_summary.to_csv(tables / "trigger_summary.csv", index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": args.protocol,
        "protocol_integrity": integrity,
        "checkpoint": str(checkpoint),
        "selection_frozen": True,
        "used_for_model_selection": False,
        "ablation_contract": (
            "all variants load the same validation-selected EARS checkpoint; "
            "only the named post-selection mechanism is disabled"
        ),
        "cities": metadata,
        "data_attribution": OSM_ATTRIBUTION,
        "arguments": vars(args),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
