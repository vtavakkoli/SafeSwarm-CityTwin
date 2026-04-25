"""Run safety feasibility experiments across agent strategies."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.agents.greedy_agent import GreedyAgentPolicy
from src.agents.random_agent import RandomAgentPolicy
from src.agents.safe_swarm_agent import SafeSwarmAgentPolicy
from src.agents.safety_filtered_agent import SafetyFilteredAgentPolicy
from src.environment.city_twin import CityTwinEnvironment
from src.evaluation.metrics import EpisodeMetrics
from src.safety.runtime_monitor import RuntimeSafetyMonitor
from src.visualization.plots import plot_bar, plot_trajectories_with_restricted_zones


def run_episode(env: CityTwinEnvironment, policy, monitor: RuntimeSafetyMonitor | None = None):
    env.reset()
    step_count = 0
    t0 = time.perf_counter()

    while True:
        actions = policy.act(env)
        env_info = env.step(actions)
        step_count += 1
        if env_info["done"] > 0:
            break

    elapsed = time.perf_counter() - t0

    reached_missions = sum(1 for a in env.agents.values() if a.current_task == "mission")
    mission_success_rate = reached_missions / max(1, env.n_agents)

    m = EpisodeMetrics(
        mission_success_rate=mission_success_rate,
        number_of_safety_violations=0 if monitor is None else monitor.safety_violations,
        collision_count=0 if monitor is None else monitor.collision_count,
        restricted_zone_entries=0 if monitor is None else monitor.restricted_zone_entries,
        battery_failures=sum(1 for a in env.agents.values() if a.battery_level <= 0)
        if monitor is None
        else monitor.battery_failures,
        coverage_ratio=env_info["coverage_ratio"],
        runtime_overhead=elapsed / max(step_count, 1),
    )
    trajectories = {aid: s.trajectory_history for aid, s in env.agents.items()}
    return m, trajectories


def generate_report(df: pd.DataFrame, out_report: Path, args: argparse.Namespace) -> None:
    summary = df.groupby("strategy", as_index=False).mean(numeric_only=True)
    lines = [
        "# Safety Feasibility Report",
        "",
        "## Experimental setup",
        f"- Date: auto-generated",
        f"- Grid size: {args.grid_size}",
        f"- Agents: {args.agents}",
        f"- Episodes: {args.episodes}",
        f"- Seed: {args.seed}",
        f"- City data source: OpenStreetMap place='{args.place}' (fallback synthetic if unavailable)",
        "",
        "## Key findings",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"- **{row['strategy']}**: success={row['mission_success_rate']:.3f}, "
            f"violations={row['number_of_safety_violations']:.2f}, "
            f"coverage={row['coverage_ratio']:.3f}, overhead={row['runtime_overhead']:.6f}s/step"
        )

    lines.extend(
        [
            "",
            "## IEEE-feasibility interpretation",
            "The results support feasibility when safety filtering lowers violation rates while maintaining competitive mission success and bounded runtime overhead.",
        ]
    )
    out_report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--place", type=str, default="San Francisco, California, USA")
    args = parser.parse_args()

    out_tables = Path("results/tables")
    out_figs = Path("results/figures")
    out_reports = Path("results/reports")
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    records = []
    stored_env = None
    stored_trajectories = None

    strategies = {
        "RandomAgent": lambda: (RandomAgentPolicy(seed=args.seed), None),
        "GreedyAgent": lambda: (GreedyAgentPolicy(), None),
        "SafetyFilteredAgent": lambda: (SafetyFilteredAgentPolicy(monitor=RuntimeSafetyMonitor()), None),
        "SafeSwarmAgent": lambda: (SafeSwarmAgentPolicy(monitor=RuntimeSafetyMonitor()), None),
    }

    for strategy_name, factory in strategies.items():
        for episode in range(args.episodes):
            env = CityTwinEnvironment(
                grid_size=args.grid_size,
                n_agents=args.agents,
                seed=args.seed + episode,
                place_name=args.place,
            )
            policy, _ = factory()
            monitor = getattr(policy, "monitor", None)
            metrics, trajectories = run_episode(env, policy, monitor)
            records.append(metrics.to_dict(strategy=strategy_name, episode=episode))
            if strategy_name == "SafeSwarmAgent" and episode == args.episodes - 1:
                stored_env = env
                stored_trajectories = trajectories

    df = pd.DataFrame(records)
    csv_path = out_tables / "safety_experiment_results.csv"
    df.to_csv(csv_path, index=False)

    plot_bar(df, "number_of_safety_violations", out_figs / "safety_violations_comparison.png", "Violations")
    plot_bar(df, "mission_success_rate", out_figs / "mission_success_comparison.png", "Mission success rate")
    plot_bar(df, "runtime_overhead", out_figs / "runtime_overhead.png", "Runtime overhead (s/step)")

    if stored_env is not None and stored_trajectories is not None:
        plot_trajectories_with_restricted_zones(
            stored_env,
            stored_trajectories,
            out_figs / "trajectories_with_restricted_zones.png",
        )

    generate_report(df, out_reports / "safety_feasibility_report.md", args)


if __name__ == "__main__":
    main()
