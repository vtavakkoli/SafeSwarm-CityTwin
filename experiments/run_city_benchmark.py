"""Benchmark SafeSwarm and BioSwarm algorithms on shared real-city snapshots."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.registry import strategy_factories  # noqa: E402
from src.environment.city_twin import CityTwinEnvironment  # noqa: E402
from src.environment.obstacles import OSM_ATTRIBUTION, load_real_city_layers  # noqa: E402
from src.evaluation.metrics import EpisodeMetrics, rank_algorithms  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/real_cities.json")
    parser.add_argument("--cities", nargs="*", help="Optional city names from the config")
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output-root", default="results/real_city_benchmark")
    parser.add_argument("--offline", action="store_true", help="Use deterministic synthetic city layers")
    parser.add_argument("--require-real-data", action="store_true", help="Fail if any city falls back to synthetic data")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Small CI/development benchmark")
    return parser.parse_args()


def load_city_config(path: str | Path, selected: list[str] | None = None) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cities = list(payload["cities"])
    if selected:
        wanted = {name.casefold() for name in selected}
        cities = [city for city in cities if city["name"].casefold() in wanted]
        missing = wanted - {city["name"].casefold() for city in cities}
        if missing:
            raise ValueError(f"Unknown city names: {sorted(missing)}")
    if not cities:
        raise ValueError("No cities selected")
    return cities


def run_episode(env: CityTwinEnvironment, policy: object) -> EpisodeMetrics:
    start = time.perf_counter()
    info: dict[str, float] = {"coverage_ratio": 0.0}
    while True:
        actions = policy.act(env)  # type: ignore[attr-defined]
        info = env.step(actions)
        if info["done"] > 0:
            break
    elapsed = time.perf_counter() - start
    monitor = getattr(policy, "monitor", None)

    actual_incidents = env.actual_collisions + env.actual_restricted_entries
    battery_failures = sum(state.battery_level <= 0 for state in env.agents.values())
    return EpisodeMetrics(
        mission_success_rate=len(env.discovered_missions) / max(1, len(env.mission_zones)),
        weighted_target_discovery=env.weighted_target_discovery(),
        coverage_ratio=float(info["coverage_ratio"]),
        number_of_safety_violations=0 if monitor is None else int(monitor.safety_violations),
        safety_interventions=0 if monitor is None else int(monitor.intervention_count),
        actual_safety_incidents=int(actual_incidents),
        collision_count=int(env.actual_collisions),
        restricted_zone_entries=int(env.actual_restricted_entries),
        battery_failures=int(battery_failures),
        energy_consumption=env.energy_consumption(),
        redundant_coverage=env.redundant_coverage(),
        communication_efficiency=env.communication_efficiency(),
        distance_travelled=sum(state.distance_travelled for state in env.agents.values()),
        runtime_seconds=float(elapsed),
        runtime_overhead=float(elapsed / max(1, env.steps)),
        blocked_moves=int(env.blocked_moves),
    )


def _save_plot(frame: pd.DataFrame, value: str, ylabel: str, path: Path, ascending: bool = False) -> None:
    ordered = frame.sort_values(value, ascending=ascending)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    ax.bar(ordered["strategy"], ordered[value])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Strategy")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_html_report(
    records: pd.DataFrame,
    city_summary: pd.DataFrame,
    overall: pd.DataFrame,
    city_metadata: list[dict[str, Any]],
    args: argparse.Namespace,
    output: Path,
) -> None:
    winner = overall.iloc[0]
    all_real = all(meta.get("source") == "openstreetmap" for meta in city_metadata)
    source_badges = "".join(
        f'<span class="badge {"real" if meta.get("source") == "openstreetmap" else "fallback"}">'
        f'{escape(str(meta.get("place_name")))} · {escape(str(meta.get("source")))}</span>'
        for meta in city_metadata
    )
    overall_table = overall.round(4).to_html(index=False, classes="data", border=0)
    city_table = city_summary.round(4).to_html(index=False, classes="data", border=0)
    source_table = pd.DataFrame(city_metadata).fillna("").to_html(index=False, classes="data", border=0)

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SafeSwarm Real-City Algorithm Benchmark</title>
<style>
:root{{--ink:#172033;--muted:#667085;--line:#dfe5ef;--accent:#3448c5;--good:#087830;--warn:#9a6700;--bg:#f4f7fb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,system-ui,Segoe UI,Arial,sans-serif}}
header{{padding:38px 5vw;background:linear-gradient(120deg,#111827,#3448c5);color:white}} header h1{{margin:0 0 8px;font-size:32px}}
main{{max-width:1500px;margin:-18px auto 40px;padding:0 24px}} section,.card{{background:white;border:1px solid var(--line);border-radius:14px;box-shadow:0 5px 18px #1018280b}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin-bottom:18px}} .card{{padding:18px}}
.card strong{{display:block;font-size:24px}} section{{padding:22px;margin:18px 0;overflow:auto}} h2{{margin-top:0}}
.badge{{display:inline-block;padding:6px 10px;border-radius:999px;margin:4px 6px 4px 0;font-size:12px;font-weight:700}} .real{{background:#e7f8ed;color:var(--good)}} .fallback{{background:#fff4d6;color:var(--warn)}}
table.data{{border-collapse:collapse;width:100%;font-size:13px}} .data th,.data td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} .data th:first-child,.data td:first-child{{text-align:left}} .data th{{position:sticky;top:0;background:#f8fafc}}
.figure-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}} .figure-grid img{{width:100%;border:1px solid var(--line);border-radius:10px}}
code{{background:#eef2ff;padding:2px 5px;border-radius:5px}} .muted{{color:var(--muted)}}
</style></head><body>
<header><h1>SafeSwarm Real-City Algorithm Benchmark</h1><p>Fair, safety-aware comparison on cached city snapshots</p></header>
<main>
<div class="cards">
<div class="card"><span class="muted">Best overall</span><strong>{escape(str(winner['strategy']))}</strong><span>score {winner['operational_score']:.3f}</span></div>
<div class="card"><span class="muted">Cities</span><strong>{len(city_metadata)}</strong><span>{'all real OSM snapshots' if all_real else 'includes labelled fallback data'}</span></div>
<div class="card"><span class="muted">Algorithms</span><strong>{records['strategy'].nunique()}</strong><span>{len(records)} episode runs</span></div>
<div class="card"><span class="muted">Experiment</span><strong>{args.agents} agents</strong><span>{args.grid_size}×{args.grid_size}, {args.max_steps} steps</span></div>
</div>
<section><h2>Data provenance</h2><p>{source_badges}</p><p class="muted">Real snapshots are downloaded with OSMnx, cached as compact grid layers, and attributed to {OSM_ATTRIBUTION}. Synthetic fallback is never presented as real data.</p>{source_table}</section>
<section><h2>Overall ranking</h2><p>The operational score weights priority-target discovery (35%), coverage (20%), safety (20%), energy (10%), coordination (10%), and communication availability (5%). Runtime is reported separately because it is hardware dependent.</p>{overall_table}</section>
<section><h2>Figures</h2><div class="figure-grid"><img src="figures/overall_score.png" alt="Overall score"><img src="figures/target_discovery.png" alt="Target discovery"><img src="figures/safety_incidents.png" alt="Safety incidents"></div></section>
<section><h2>Per-city results</h2>{city_table}</section>
<section><h2>Reproduction</h2><p><code>docker compose up --build benchmark-real-cities</code></p><p><code>docker compose up --build benchmark-offline</code> runs the deterministic CI-sized fallback benchmark.</p></section>
</main></body></html>"""
    output.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.quick:
        args.agents = min(args.agents, 4)
        args.grid_size = min(args.grid_size, 20)
        args.episodes = min(args.episodes, 2)
        args.max_steps = min(args.max_steps, 40)

    output = Path(args.output_root)
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    cities = load_city_config(args.config, args.cities)
    records: list[dict[str, Any]] = []
    city_metadata: list[dict[str, Any]] = []

    for city in cities:
        layers = load_real_city_layers(
            city["place"],
            args.grid_size,
            args.seed,
            radius_m=int(city.get("radius_m", 1400)),
            cache_dir=args.cache_dir,
            allow_network=not args.offline,
            force_refresh=args.force_refresh,
        )
        metadata = dict(layers.get("metadata", {}))
        metadata["city"] = city["name"]
        city_metadata.append(metadata)
        if args.require_real_data and metadata.get("source") != "openstreetmap":
            raise RuntimeError(f"{city['name']} did not load real data: {metadata.get('fallback_reason')}")

        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            for strategy, factory in strategy_factories(seed=episode_seed).items():
                env = CityTwinEnvironment(
                    grid_size=args.grid_size,
                    n_agents=args.agents,
                    seed=episode_seed,
                    place_name=city["place"],
                    radius_m=int(city.get("radius_m", 1400)),
                    layers=layers,
                    max_steps=args.max_steps,
                    allow_network=False,
                )
                policy = factory()
                metrics = run_episode(env, policy)
                records.append(
                    metrics.to_dict(
                        strategy=strategy,
                        episode=episode,
                        city=city["name"],
                        place=city["place"],
                        data_source=env.data_source,
                        feature_count=int(env.data_metadata.get("feature_count", 0)),
                        agents=args.agents,
                        grid_size=args.grid_size,
                        steps=env.steps,
                        seed=episode_seed,
                    )
                )
                print(
                    f"[{city['name']}] episode={episode + 1}/{args.episodes} strategy={strategy} "
                    f"targets={metrics.weighted_target_discovery:.3f} coverage={metrics.coverage_ratio:.3f} "
                    f"incidents={metrics.actual_safety_incidents}",
                    flush=True,
                )

    frame = pd.DataFrame(records)
    city_summary, overall = rank_algorithms(frame)
    frame.to_csv(tables / "episode_results.csv", index=False)
    city_summary.to_csv(tables / "city_ranking.csv", index=False)
    overall.to_csv(tables / "overall_ranking.csv", index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "cities": city_metadata,
        "winner": overall.iloc[0].to_dict(),
        "data_attribution": OSM_ATTRIBUTION,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    _save_plot(overall, "operational_score", "Operational score", figures / "overall_score.png")
    _save_plot(overall, "weighted_target_discovery", "Weighted target discovery", figures / "target_discovery.png")
    _save_plot(overall, "actual_safety_incidents", "Actual safety incidents", figures / "safety_incidents.png", ascending=True)
    build_html_report(frame, city_summary, overall, city_metadata, args, output / "report.html")
    print(f"Benchmark report: {output / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
