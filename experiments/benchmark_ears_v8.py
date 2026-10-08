"""Compare frozen Ant, EARS v7, route-only v8 and gated v8 fairly.

Development mode uses the existing eight-city diagnostics. --confirm uses an
independent, preregistered geographic split and insists on real OSM data.
No test result here may tune an algorithm or overwrite a v7 publication output.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_city_benchmark import run_episode  # noqa: E402
from src.agents.bio_swarm_agents import AntSwarmPolicy  # noqa: E402
from src.agents.ears_v6 import EARSPolicy  # noqa: E402
from src.agents.ears_v8 import EARSTopoSafePolicy, TopologyGateConfig  # noqa: E402
from src.environment.city_twin import CityTwinEnvironment  # noqa: E402
from src.environment.obstacles import load_real_city_layers  # noqa: E402
from src.evaluation.metrics import episode_operational_score  # noqa: E402
from src.training.geography import (  # noqa: E402
    apply_start_zone, load_protocol, select_cities, start_zones_for_split, validate_protocol,
)

METHODS = ("AntSwarmSafe", "EARS-Safe-v7", "EARS-TopoSafe", "EARS-TopoSafe-Gated")
MEASURES = ("operational_score", "weighted_target_discovery", "coverage_ratio",
            "energy_consumption", "distance_travelled", "redundant_coverage",
            "actual_safety_incidents", "runtime_seconds", "communication_efficiency")
PAIR_KEYS = ("city", "start_zone", "episode", "agents", "dropout")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", default="configs/publication_protocol_v7.json")
    p.add_argument("--confirm", action="store_true",
                   help="Use exclusively preregistered confirmatory cities with real OSM data")
    p.add_argument("--model-dir", default="results/train/checkpoints")
    p.add_argument("--output-root", default="results/v8/development")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--agents-grid", default="8", help="Comma-separated robot counts, e.g. 4,8,16")
    p.add_argument("--dropout-grid", default="0.03", help="Comma-separated communication loss probabilities")
    p.add_argument("--grid-size", type=int, default=40)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=160)
    p.add_argument("--seed", type=int, default=81042)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--require-real-data", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p


def _graph_distances(grid: int, start: tuple[int, int], blocked: set) -> dict:
    from collections import deque
    distances = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nxt[0] < grid and 0 <= nxt[1] < grid and nxt not in blocked and nxt not in distances:
                distances[nxt] = distances[(x, y)] + 1
                queue.append(nxt)
    return distances


def topology_metrics(layers: dict, grid_size: int) -> dict[str, float]:
    """Evaluator-only OSM topology diagnostics; NEVER passed as policy input."""
    blocked = set(layers["obstacles"]) | set(layers["restricted_zones"])
    free = {(x, y) for x in range(grid_size) for y in range(grid_size)} - blocked
    if not free:
        raise ValueError("Map has no traversable cells")
    unexplored = set(free)
    components = []
    while unexplored:
        component = set(_graph_distances(grid_size, min(unexplored), blocked))
        components.append(len(component))
        unexplored -= component
    distances = {}
    for base in layers["base_stations"]:
        for cell, length in _graph_distances(grid_size, base, blocked).items():
            distances[cell] = min(length, distances.get(cell, length))
    mission_cells = set(layers["mission_zones"])
    reachable = mission_cells & set(distances)
    return {
        "components": len(components),
        "largest_component_fraction": max(components) / len(free),
        "reachable_mission_fraction": len(reachable) / max(1, len(mission_cells)),
        "reachable_free_fraction": len(distances) / len(free),
        "original_targets": len(mission_cells),
    }


def _paired_statistics(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """City-aware exact sign flips plus hierarchical bootstrap; one city is one unit."""
    out = []
    rng = np.random.default_rng(seed)
    for (robots, dropout), group in frame.groupby(["agents", "dropout"], sort=True):
        for method in METHODS[1:]:
            pivot = group.pivot(index=list(PAIR_KEYS), columns="strategy", values="operational_score")
            if not {METHODS[0], method}.issubset(pivot.columns):
                raise ValueError("Missing matched baseline/candidate records")
            if pivot[[METHODS[0], method]].isna().any().any():
                raise ValueError("Incomplete paired episodes")
            delta = (pivot[method] - pivot[METHODS[0]]).rename("delta").reset_index()
            groups = {str(city): g["delta"].to_numpy(float) for city, g in delta.groupby("city")}
            cities = sorted(groups)
            city_means = np.array([groups[c].mean() for c in cities], dtype=float)
            observed = float(city_means.mean())
            # Exact city sign-flips are meaningful only with independent city units.
            if len(cities) <= 16:
                null = [np.mean(np.array(signs) * city_means)
                        for signs in itertools.product((-1, 1), repeat=len(cities))]
                p_value = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
            else:
                signs = rng.choice((-1., 1.), size=(20000, len(cities)))
                p_value = float((1 + np.count_nonzero(np.abs(signs @ city_means / len(cities)) >= abs(observed))) / 20001)
            boot = []
            for _ in range(2000):
                sample_cities = rng.choice(cities, size=len(cities), replace=True)
                boot.append(np.mean([rng.choice(groups[c], size=len(groups[c]), replace=True).mean()
                                     for c in sample_cities]))
            loo = ([float(np.mean(np.delete(city_means, i))) for i in range(len(cities))]
                   if len(cities) > 1 else [observed])
            out.append({
                "method": method, "baseline": METHODS[0], "agents": robots, "dropout": dropout,
                "city_count": len(cities), "paired_episodes": len(delta),
                "citymean_delta": observed, "city_signflip_p": p_value,
                "hierarchical_ci_low": float(np.quantile(boot, 0.025)),
                "hierarchical_ci_high": float(np.quantile(boot, 0.975)),
                "leave_one_city_out_min": min(loo), "leave_one_city_out_max": max(loo),
                "cities_improved": int(np.count_nonzero(city_means > 0)),
            })
    return pd.DataFrame(out)


def run(args: argparse.Namespace) -> None:
    if args.confirm:
        if args.offline or args.quick:
            raise ValueError("--confirm requires full, online-or-cached real OSM validation")
        if args.protocol != parser().get_default("protocol"):
            raise ValueError("--confirm uses only the fixed v8 confirmation protocol")
        args.protocol = "configs/confirmation_protocol_v8.json"
        args.require_real_data = True
        if args.output_root == parser().get_default("output_root"):
            args.output_root = "results/v8/confirmation"
    agents_grid = sorted(set(int(n) for n in args.agents_grid.split(",")))
    dropout_grid = sorted(set(float(x) for x in args.dropout_grid.split(",")))
    if min(agents_grid) < 1 or any(not 0 <= x <= 1 for x in dropout_grid):
        raise ValueError("Invalid agents/dropout grid")
    if args.quick:
        agents_grid = agents_grid[:1]
        dropout_grid = dropout_grid[:1]
        args.grid_size = min(20, args.grid_size)
        args.episodes = min(2, args.episodes)
        args.max_steps = min(50, args.max_steps)

    protocol = load_protocol(args.protocol)
    integrity = validate_protocol(protocol)
    if not all(integrity.values()):
        raise ValueError(f"Geography split invalid: {integrity}")
    cities = select_cities(protocol, "test")
    zones = start_zones_for_split(protocol, "test")
    if args.quick:
        cities, zones = cities[:2], zones[:1]

    checkpoint = Path(args.model_dir) / "ears_safe.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Frozen v6 EARS checkpoint missing: {checkpoint}")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    map_rows = []
    for ci, city in enumerate(cities):
        layers = load_real_city_layers(
            city["place"], args.grid_size, args.seed + ci,
            radius_m=int(city.get("radius_m", 1400)),
            cache_dir=args.cache_dir, allow_network=not args.offline,
        )
        meta = dict(layers.get("metadata", {}))
        if args.require_real_data and meta.get("source") != "openstreetmap":
            raise ValueError(f"Non-OSM geography rejected for {city['name']}: {meta}")
        tm = topology_metrics(layers, args.grid_size)
        map_rows.append({"city": city["name"], "data_source": meta.get("source"), **tm})
        if args.confirm and (tm["reachable_mission_fraction"] < 1 or tm["original_targets"] < 6):
            raise ValueError(f"Invalid confirmatory topology/mission data in {city['name']}: {tm}")
        for robots in agents_grid:
            for dropout in dropout_grid:
                for episode in range(args.episodes):
                    zone = zones[episode % len(zones)]
                    episode_seed = args.seed + ci * 10000 + episode
                    zoned = apply_start_zone(layers, args.grid_size, zone)
                    factories = {
                        "AntSwarmSafe": lambda: AntSwarmPolicy(),
                        "EARS-Safe-v7": lambda: EARSPolicy(
                            seed=episode_seed, model_path=checkpoint, strategy_name="EARS-Safe-v7"),
                        "EARS-TopoSafe": lambda: EARSTopoSafePolicy(
                            seed=episode_seed, model_path=checkpoint,
                            strategy_name="EARS-TopoSafe",
                            gate_config=TopologyGateConfig(enable_benefit_gate=False)),
                        "EARS-TopoSafe-Gated": lambda: EARSTopoSafePolicy(
                            seed=episode_seed, model_path=checkpoint,
                            strategy_name="EARS-TopoSafe-Gated"),
                    }
                    for method, factory in factories.items():
                        env = CityTwinEnvironment(
                            grid_size=args.grid_size, n_agents=robots, seed=episode_seed,
                            place_name=city["place"], radius_m=int(city.get("radius_m", 1400)),
                            layers=zoned, max_steps=args.max_steps, allow_network=False,
                            communication_dropout_prob=dropout,
                        )
                        policy = factory()
                        metrics = run_episode(env, policy)
                        row = metrics.to_dict(
                            strategy=method, episode=episode, city=city["name"],
                            start_zone=zone, agents=robots, dropout=dropout,
                            steps=env.steps, data_source=env.data_source,
                        )
                        row["operational_score"] = episode_operational_score(row)
                        if hasattr(policy, "diagnostics"):
                            diagnostics = policy.diagnostics()
                            for key, value in diagnostics.items():
                                if key.startswith("v8_") and isinstance(value, (int, float, bool)):
                                    row[key] = value
                        records.append(row)
                        print(f"{city['name']} robots={robots} dropout={dropout} "
                              f"episode={episode} {method} J={row['operational_score']:.4f}", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(output / "episodes.csv", index=False)
    pd.DataFrame(map_rows).to_csv(output / "topology.csv", index=False)
    summary = (df.groupby(["agents", "dropout", "city", "strategy"], as_index=False)
               [list(MEASURES)].mean(numeric_only=True))
    summary.to_csv(output / "city_means.csv", index=False)
    overall = (summary.groupby(["agents", "dropout", "strategy"], as_index=False)
               [list(MEASURES)].mean(numeric_only=True))
    overall.to_csv(output / "summary.csv", index=False)
    paired = _paired_statistics(df, args.seed)
    paired.to_csv(output / "paired_city_tests.csv", index=False)
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "confirmatory" if args.confirm else "development",
        "v7_publication_results_preserved": True,
        "checkpoint_sha256": checkpoint_hash, "protocol": args.protocol,
        "protocol_sha256": hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest(),
        "arguments": vars(args), "geographic_integrity": integrity,
        "cities": [city["name"] for city in cities],
        "dataset_quality": map_rows, "method_names": list(METHODS),
        "limitations": ["OSM-derived grid simulation, not physical robots",
                        "Communication dropout is simulated; delayed or Byzantine messaging is not"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>EARS v8 — reproducible comparison</title>"
        "<style>body{font:16px system-ui;margin:3rem auto;max-width:1150px;line-height:1.55}"
        "table{border-collapse:collapse;font-size:13px;width:100%}"
        "th,td{border:1px solid #ccd;padding:8px;text-align:left}"
        "th{background:#eef3f7}h1,h2{color:#183047}</style></head><body>"
        "<h1>EARS v8: geographically paired algorithm comparison</h1>"
        "<p>Mode: " + escape(manifest["status"]) +
        ". Results are simulations; no statistical improvement is presumed.</p>"
        "<h2>Eight-city development or new-city confirmation</h2>"
        + overall.round(5).to_html(index=False, escape=True) +
        "<h2>Independent-city randomization and hierarchical uncertainty</h2>"
        + paired.round(5).to_html(index=False, escape=True) +
        "<h2>Topology data-quality summary</h2>"
        + pd.DataFrame(map_rows).round(5).to_html(index=False, escape=True) +
        "<p>All episodes, manifests and raw diagnostics are available alongside this file.</p>"
        "</body></html>"
    )
    (output / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    run(parser().parse_args())
