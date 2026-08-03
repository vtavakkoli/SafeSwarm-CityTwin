"""Backward-compatible single-city benchmark entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_city_benchmark import _save_plot, build_html_report, run_episode
from src.agents.registry import strategy_factories
from src.environment.city_twin import CityTwinEnvironment
from src.environment.obstacles import load_real_city_layers
from src.evaluation.metrics import rank_algorithms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--place", default="Vienna, Austria")
    parser.add_argument("--radius-m", type=int, default=1400)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output-root", default="results/single_city_benchmark")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_root)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    layers = load_real_city_layers(
        args.place,
        args.grid_size,
        args.seed,
        radius_m=args.radius_m,
        cache_dir=args.cache_dir,
        allow_network=not args.offline,
    )
    records = []
    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        for strategy, factory in strategy_factories(episode_seed).items():
            env = CityTwinEnvironment(
                grid_size=args.grid_size,
                n_agents=args.agents,
                seed=episode_seed,
                place_name=args.place,
                radius_m=args.radius_m,
                layers=layers,
                max_steps=args.max_steps,
                allow_network=False,
            )
            metrics = run_episode(env, factory())
            records.append(
                metrics.to_dict(
                    strategy,
                    episode,
                    city=args.place,
                    place=args.place,
                    data_source=env.data_source,
                    feature_count=int(env.data_metadata.get("feature_count", 0)),
                    agents=args.agents,
                    grid_size=args.grid_size,
                    steps=env.steps,
                    seed=episode_seed,
                )
            )

    frame = pd.DataFrame(records)
    city_summary, overall = rank_algorithms(frame)
    frame.to_csv(tables / "episode_results.csv", index=False)
    city_summary.to_csv(tables / "city_ranking.csv", index=False)
    overall.to_csv(tables / "overall_ranking.csv", index=False)
    _save_plot(overall, "operational_score", "Operational score", figures / "overall_score.png")
    _save_plot(overall, "weighted_target_discovery", "Weighted target discovery", figures / "target_discovery.png")
    _save_plot(overall, "actual_safety_incidents", "Actual safety incidents", figures / "safety_incidents.png", ascending=True)
    build_html_report(frame, city_summary, overall, [dict(layers["metadata"], city=args.place)], args, output / "report.html")
    (output / "manifest.json").write_text(
        json.dumps({"arguments": vars(args), "data": layers["metadata"]}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Single-city benchmark report: {output / 'report.html'}")


if __name__ == "__main__":
    main()
