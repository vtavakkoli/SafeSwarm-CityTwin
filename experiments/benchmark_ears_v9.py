"""Select EARS v9 on validation geography, freeze it, then evaluate unchanged.

Outputs are exclusive, raw episodes are flushed immediately, and selection can
retain v7. No promised improvement, test-set tuning, or automatic confirmation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_ears_v8 import MEASURES, PAIR_KEYS, topology_metrics  # noqa: E402
from experiments.run_city_benchmark import run_episode  # noqa: E402
from src.agents.bio_swarm_agents import AntSwarmPolicy  # noqa: E402
from src.agents.ears_v6 import EARSPolicy  # noqa: E402
from src.agents.ears_v9 import EARSInformationPolicy, InformationRoutingConfig  # noqa: E402
from src.environment.city_twin import CityTwinEnvironment  # noqa: E402
from src.environment.obstacles import load_real_city_layers  # noqa: E402
from src.evaluation.metrics import episode_operational_score  # noqa: E402
from src.training.geography import apply_start_zone, load_protocol, select_cities, start_zones_for_split  # noqa: E402

BASELINE = "EARS-Safe-v7"
CANDIDATES = {
    "EARS-Info-v9": InformationRoutingConfig(),
    "EARS-Info-Local-v9": InformationRoutingConfig(enable_relocation=False),
    "EARS-Balanced-v9": InformationRoutingConfig(information_weight=1.0, coverage_weight=1.5),
    "EARS-Balanced-Local-v9": InformationRoutingConfig(
        information_weight=1.0, coverage_weight=1.5, enable_relocation=False
    ),
}
# Declared before validation. These are engineering promotion gates, NOT a
# statistical significance claim from only two validation cities.
PROMOTION = {
    "min_mean_delta": 0.002,
    "min_stratum_delta": -0.005,
    "max_runtime_ratio": 2.0,
    "min_discovery_delta": 0.0,
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "select", "development", "confirm"), default="select")
    p.add_argument("--model-dir", default="results/train/checkpoints")
    p.add_argument("--selection", help="Frozen selected_model.json from a completed real-data selection")
    p.add_argument("--output-root", help="New, empty directory; defaults to a unique timestamped directory")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--agents-grid", default="4,8,16,32")
    p.add_argument("--dropout-grid", default="0,0.03,0.15,0.30")
    p.add_argument("--grid-size", type=int, default=40)
    p.add_argument(
        "--episodes", type=int, default=4, help="At least four for selection: all four validation starts"
    )
    p.add_argument("--max-steps", type=int, default=160)
    p.add_argument("--seed", type=int, default=91042)
    p.add_argument(
        "--offline", action="store_true", help="Use cached real maps, or synthetic maps in smoke mode only"
    )
    p.add_argument("--require-real-data", action="store_true")
    return p


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprint():
    # Includes simulator, scoring, safety, data transforms and the whole policy
    # dependency graph. A code edit after selection requires a new selection.
    paths = sorted((ROOT / "src").rglob("*.py")) + [
        ROOT / "experiments/run_city_benchmark.py",
        Path(__file__),
        ROOT / "experiments/benchmark_ears_v8.py",
    ]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def layer_digest(layers):
    data = {
        key: sorted(layers[key])
        for key in ("obstacles", "restricted_zones", "base_stations", "mission_zones")
    }
    data["priority_cells"] = sorted((list(cell), value) for cell, value in layers["priority_cells"].items())
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def paired_comparisons(frame):
    rows = []
    for (agents, dropout), group in frame.groupby(["agents", "dropout"]):
        pivot = group.pivot(index=list(PAIR_KEYS), columns="strategy", values="operational_score")
        if pivot.isna().any().any():
            raise ValueError("Incomplete paired results")
        for method in pivot.columns:
            if method == "AntSwarmSafe":
                continue
            baselines = ("AntSwarmSafe",) if method == BASELINE else (BASELINE, "AntSwarmSafe")
            for baseline in baselines:
                delta = pivot[method] - pivot[baseline]
                means = delta.groupby(level="city").mean().to_numpy()
                observed = float(means.mean())
                null = np.array(
                    [np.mean(means * signs) for signs in itertools.product((-1, 1), repeat=len(means))]
                )
                rows.append(
                    {
                        "method": method,
                        "baseline": baseline,
                        "agents": agents,
                        "dropout": dropout,
                        "city_count": len(means),
                        "paired_episodes": len(delta),
                        "citymean_delta": observed,
                        "city_signflip_p": float(np.mean(abs(null) >= abs(observed) - 1e-12)),
                        "cities_improved": int(np.count_nonzero(means > 0)),
                    }
                )
    return pd.DataFrame(rows)


def choose_candidate(frame):
    """Equal weight per city/setting; failures cannot be hidden by aggregation."""
    if frame.duplicated([*PAIR_KEYS, "strategy"]).any():
        raise ValueError("Duplicate paired results")
    required = {BASELINE, "AntSwarmSafe", *CANDIDATES}
    pivot = frame.pivot(index=list(PAIR_KEYS), columns="strategy", values="operational_score")
    if set(pivot.columns) != required or pivot.isna().any().any():
        raise ValueError("Selection requires every candidate on every paired episode")
    group = frame.groupby(["city", "agents", "dropout", "strategy"])[
        ["operational_score", "weighted_target_discovery", "runtime_seconds"]
    ].mean()
    base = group.xs(BASELINE, level="strategy")
    ant = group.xs("AntSwarmSafe", level="strategy")
    decisions = []
    for name in CANDIDATES:
        candidate = group.xs(name, level="strategy")
        delta = candidate.operational_score - base.operational_score
        discovery = candidate.weighted_target_discovery - base.weighted_target_discovery
        runtime_ratio = float(candidate.runtime_seconds.mean() / max(1e-9, base.runtime_seconds.mean()))
        raw = frame[frame.strategy == name]
        incidents = int(raw[["actual_safety_incidents", "battery_failures", "blocked_moves"]].sum().sum())
        reasons = []
        if delta.mean() < PROMOTION["min_mean_delta"]:
            reasons.append("mean score gain below promotion margin")
        if delta.min() < PROMOTION["min_stratum_delta"]:
            reasons.append("city/setting regression exceeds limit")
        if candidate.operational_score.mean() < ant.operational_score.mean():
            reasons.append("mean score below Ant")
        if discovery.mean() < PROMOTION["min_discovery_delta"] - 1e-12:
            reasons.append("mean discovery regresses")
        if runtime_ratio > PROMOTION["max_runtime_ratio"]:
            reasons.append("runtime exceeds limit")
        if incidents:
            reasons.append("physical incident, battery failure or invalid move")
        decisions.append(
            {
                "method": name,
                "eligible": not reasons,
                "reasons": reasons,
                "mean_delta": float(delta.mean()),
                "worst_stratum_delta": float(delta.min()),
                "discovery_delta": float(discovery.mean()),
                "runtime_ratio": runtime_ratio,
                "incident_count": incidents,
            }
        )
    eligible = [item for item in decisions if item["eligible"]]
    selected = (
        max(eligible, key=lambda item: (item["mean_delta"], item["method"]))["method"]
        if eligible
        else BASELINE
    )
    return selected, decisions


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def run(args):
    robots_grid = sorted(set(int(x) for x in args.agents_grid.split(",")))
    dropout_grid = sorted(set(float(x) for x in args.dropout_grid.split(",")))
    if min(robots_grid) < 1 or any(not np.isfinite(x) or not 0 <= x <= 1 for x in dropout_grid):
        raise ValueError("Invalid robot/dropout grid")
    if args.grid_size < 8 or args.max_steps < 1 or args.episodes < 1:
        raise ValueError("Invalid grid size, episode count or horizon")
    if args.mode == "select" and (args.episodes < 4 or args.episodes % 4):
        raise ValueError(
            "Selection needs balanced coverage of all four validation starts; use a multiple of 4"
        )
    if args.mode in ("development", "confirm") and not args.selection:
        raise ValueError("Freeze a real-data selection before development/confirmation evaluation")
    if args.mode in ("smoke", "select") and args.selection:
        raise ValueError("--selection is only for development/confirm")
    protocol_path = ROOT / (
        "configs/confirmation_protocol_v9.json"
        if args.mode == "confirm"
        else "configs/publication_protocol_v7.json"
    )
    protocol = load_protocol(protocol_path)
    split = "validation" if args.mode in ("select", "smoke") else "test"
    cities = select_cities(protocol, split)
    zones = start_zones_for_split(protocol, split)
    if args.mode == "smoke":
        cities = cities[:1]
    checkpoint = Path(args.model_dir) / "ears_safe.json"
    if not checkpoint.is_file() and args.mode != "smoke":
        raise FileNotFoundError(f"Frozen v7 checkpoint missing: {checkpoint}")
    checkpoint_hash = digest(checkpoint) if checkpoint.is_file() else None
    sources = source_fingerprint()
    configs = CANDIDATES.copy()
    frozen = None
    if args.selection:
        frozen = json.loads(Path(args.selection).read_text())
        if frozen.get("format") != "ears-v9-selection-v1" or not frozen.get("real_data"):
            raise ValueError("Not a real-data v9 selection artifact")
        if frozen["source_sha256"] != sources or frozen["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError(
                "Code/checkpoint changed since selection; rerun selection, do not silently retune"
            )
        if args.grid_size != frozen["grid_size"] or args.max_steps != frozen["max_steps"]:
            raise ValueError("Grid size/horizon must match selection")
        if not set(robots_grid).issubset(frozen["agents_grid"]) or not set(dropout_grid).issubset(
            frozen["dropout_grid"]
        ):
            raise ValueError("Robot/dropout settings must have been included in selection")
        configs = (
            {frozen["selected_method"]: InformationRoutingConfig(**frozen["routing_config"])}
            if frozen["routing_config"] is not None
            else {}
        )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output_root or f"results/v9/{args.mode}-{stamp}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite results: {output}")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {output.resolve()}", flush=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    methods = ["AntSwarmSafe", BASELINE, *configs]
    manifest = {
        "status": "running",
        "mode": args.mode,
        "started_utc": stamp,
        "arguments": vars(args),
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": digest(protocol_path),
        "source_sha256": sources,
        "checkpoint_sha256": checkpoint_hash,
        "git_commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "method_names": methods,
        "candidate_configs": {k: asdict(v) for k, v in configs.items()},
        "promotion_rules": PROMOTION,
        "city_layers": [],
        "selected_model_sha256": digest(args.selection) if args.selection else None,
        "expected_rows": len(cities) * len(robots_grid) * len(dropout_grid) * args.episodes * len(methods),
        "completed_rows": 0,
        "limitations": [
            "Shared-map OSM grid simulation, not physical/distributed robots",
            "Development cities have already been inspected; not held-out evidence",
            "Validation promotion is descriptive, not proof of generalization",
            "Legacy OSM caches may lack explicit target provenance",
        ],
    }
    atomic_json(output / "manifest.json", manifest)
    records = []
    fields = [
        "strategy",
        "episode",
        "city",
        "start_zone",
        "agents",
        "dropout",
        "steps",
        "data_source",
        "mission_success_rate",
        "weighted_target_discovery",
        "coverage_ratio",
        "number_of_safety_violations",
        "safety_interventions",
        "actual_safety_incidents",
        "collision_count",
        "restricted_zone_entries",
        "battery_failures",
        "energy_consumption",
        "redundant_coverage",
        "communication_efficiency",
        "distance_travelled",
        "runtime_seconds",
        "runtime_overhead",
        "blocked_moves",
        "operational_score",
    ]
    try:
        with (
            (output / "episodes.csv").open("x", newline="") as raw,
            (output / "diagnostics.jsonl").open("x") as diagnostics,
        ):
            writer = csv.DictWriter(raw, fieldnames=fields)
            writer.writeheader()
            raw.flush()
            for ci, city in enumerate(cities):
                layers = load_real_city_layers(
                    city["place"],
                    args.grid_size,
                    args.seed + ci,
                    radius_m=city.get("radius_m", 1400),
                    cache_dir=args.cache_dir,
                    allow_network=not args.offline,
                )
                source = layers["metadata"].get("source")
                if (args.mode != "smoke" or args.require_real_data) and source != "openstreetmap":
                    raise ValueError(f"Real OSM data required for {city['name']}: {layers['metadata']}")
                for zone in sorted(set(zones[: min(len(zones), args.episodes)])):
                    zoned = apply_start_zone(layers, args.grid_size, zone)
                    quality = topology_metrics(zoned, args.grid_size)
                    manifest["city_layers"].append(
                        {
                            "city": city["name"],
                            "zone": zone,
                            "sha256": layer_digest(zoned),
                            "source": source,
                            **quality,
                        }
                    )
                    # Validate the ACTUAL start-zone bases, not the original map bases.
                    if args.mode == "confirm" and (
                        quality["original_targets"] < 6 or quality["reachable_mission_fraction"] < 1
                    ):
                        raise ValueError(f"Confirmation data-quality failure: {city['name']} / {zone}")
                atomic_json(output / "manifest.json", manifest)
                for robots in robots_grid:
                    for dropout in dropout_grid:
                        for episode in range(args.episodes):
                            zone = zones[episode % len(zones)]
                            episode_seed = args.seed + ci * 10000 + episode
                            zoned = apply_start_zone(layers, args.grid_size, zone)
                            # Rotate order deterministically to reduce consistent warmup/timing bias.
                            offset = episode % len(methods)
                            for method in methods[offset:] + methods[:offset]:
                                env = CityTwinEnvironment(
                                    grid_size=args.grid_size,
                                    n_agents=robots,
                                    seed=episode_seed,
                                    layers=zoned,
                                    max_steps=args.max_steps,
                                    communication_dropout_prob=dropout,
                                    allow_network=False,
                                )
                                kwargs = {
                                    "seed": episode_seed,
                                    "model_path": checkpoint if checkpoint.is_file() else None,
                                    "strategy_name": method,
                                }
                                if method == "AntSwarmSafe":
                                    policy = AntSwarmPolicy()
                                elif method == BASELINE:
                                    policy = EARSPolicy(**kwargs)
                                else:
                                    policy = EARSInformationPolicy(**kwargs, routing_config=configs[method])
                                row = run_episode(env, policy).to_dict(
                                    strategy=method,
                                    episode=episode,
                                    city=city["name"],
                                    start_zone=zone,
                                    agents=robots,
                                    dropout=dropout,
                                    steps=env.steps,
                                    data_source=source,
                                )
                                row["operational_score"] = episode_operational_score(row)
                                writer.writerow(row)
                                raw.flush()
                                detail = policy.diagnostics() if hasattr(policy, "diagnostics") else {}
                                diagnostics.write(
                                    json.dumps(
                                        {
                                            "pair": {k: row[k] for k in PAIR_KEYS},
                                            "strategy": method,
                                            "diagnostics": detail,
                                        }
                                    )
                                    + "\n"
                                )
                                diagnostics.flush()
                                records.append(row)
                                manifest["completed_rows"] = len(records)
                                if len(records) % len(methods) == 0:
                                    atomic_json(output / "manifest.json", manifest)
                                print(
                                    f"{len(records)}/{manifest['expected_rows']} {city['name']} n={robots} "
                                    f"dropout={dropout} ep={episode} {method} "
                                    f"J={row['operational_score']:.5f}",
                                    flush=True,
                                )
        frame = pd.DataFrame(records)
        city_means = frame.groupby(["city", "agents", "dropout", "strategy"], as_index=False)[
            list(MEASURES)
        ].mean()
        summary = city_means.groupby(["agents", "dropout", "strategy"], as_index=False)[list(MEASURES)].mean()
        city_means.to_csv(output / "city_means.csv", index=False)
        summary.to_csv(output / "summary.csv", index=False)
        comparisons = paired_comparisons(frame)
        comparisons.to_csv(output / "paired_city_tests.csv", index=False)
        if args.mode == "select":
            selected, decisions = choose_candidate(frame)
            artifact = {
                "format": "ears-v9-selection-v1",
                "real_data": True,
                "selection_split": "validation",
                "selected_method": selected,
                "routing_config": asdict(configs[selected]) if selected in configs else None,
                "source_sha256": sources,
                "checkpoint_sha256": checkpoint_hash,
                "protocol_sha256": digest(protocol_path),
                "grid_size": args.grid_size,
                "max_steps": args.max_steps,
                "agents_grid": robots_grid,
                "dropout_grid": dropout_grid,
                "episodes_sha256": digest(output / "episodes.csv"),
                "promotion_rules": PROMOTION,
                "decisions": decisions,
            }
            atomic_json(output / "selected_model.json", artifact)
            manifest["selected_method"] = selected
            print(f"Selected: {selected}. Frozen artifact: {output / 'selected_model.json'}", flush=True)
        manifest["status"] = "completed"
        manifest["finished_utc"] = datetime.now(UTC).isoformat()
        manifest["episodes_sha256"] = digest(output / "episodes.csv")
        atomic_json(output / "manifest.json", manifest)
        (output / "index.html").write_text(
            "<!doctype html><meta charset='utf-8'><title>EARS v9 evaluation</title>"
            "<style>body{font:15px system-ui;margin:2rem}td,th{padding:6px}"
            "table{border-collapse:collapse}</style>"
            f"<h1>EARS v9 — {args.mode}</h1><p>Complete: {len(frame)} episodes. "
            "Scores do not imply statistical superiority. "
            "Raw data and provenance are saved alongside this report.</p>"
            + summary.round(5).to_html(index=False)
            + "<h2>Paired city comparisons (unadjusted)</h2>"
            + comparisons.round(5).to_html(index=False),
            encoding="utf-8",
        )
        print(f"Completed. Report: {(output / 'index.html').resolve()}", flush=True)
    except BaseException as exc:
        manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            completed_rows=len(records),
            error=f"{type(exc).__name__}: {exc}",
        )
        atomic_json(output / "manifest.json", manifest)
        raise
    return output


if __name__ == "__main__":
    run(parser().parse_args())
