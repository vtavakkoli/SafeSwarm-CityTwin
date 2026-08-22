"""Visual demo for the four leading SafeSwarm/EARS policies on real city maps.

This script is intentionally post-selection. It loads the frozen v6 checkpoints
for H-MAPPO-EARS-Safe, EARS-Safe and EARS-NP-Safe, keeps AntSwarmSafe as the
fixed interpretable controller, and replays all four policies on the same cached
OpenStreetMap-derived city layers, start zone and random seed.

For every strategy/city pair it writes:
  * animation.gif          -- animated trajectories and target discoveries;
  * trajectory_map.png     -- publication-quality static trajectory map;
  * visit_heatmap.png      -- revisit/coverage heat map;
  * final_snapshot.png     -- final state with detected-target overlay;
  * detection_events.csv   -- discovery chronology;
  * summary.json           -- run metrics + policy diagnostics.

It also writes an HTML comparison dashboard for each city and a root index.
Hidden mission markers are visualized only after policy decisions are executed;
they are never passed to the policy.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.bio_swarm_agents import AntSwarmPolicy  # noqa: E402
from src.agents.ears_v6 import (  # noqa: E402
    EARSNegativePheromonePolicy,
    EARSPolicy,
    HMAPPOEARSPolicy,
)
from src.environment.city_twin import CityTwinEnvironment  # noqa: E402
from src.environment.obstacles import OSM_ATTRIBUTION, load_real_city_layers  # noqa: E402
from src.evaluation.metrics import episode_operational_score  # noqa: E402
from src.training.geography import (  # noqa: E402
    apply_start_zone,
    load_protocol,
    select_cities,
    start_zones_for_split,
)

TOP_STRATEGIES = (
    "H-MAPPO-EARS-Safe",
    "EARS-Safe",
    "EARS-NP-Safe",
    "AntSwarmSafe",
)

# Publication-report values are only a dashboard fallback. When the frozen
# overall_ranking.csv exists, the demo reads that file instead.
PUBLICATION_REFERENCE: dict[str, dict[str, float]] = {
    "H-MAPPO-EARS-Safe": {
        "operational_score": 0.7845,
        "weighted_target_discovery": 0.9160,
        "coverage_ratio": 0.5417,
        "runtime_seconds": 5.49,
    },
    "EARS-Safe": {
        "operational_score": 0.7805,
        "weighted_target_discovery": 0.9123,
        "coverage_ratio": 0.5139,
        "runtime_seconds": 2.50,
    },
    "EARS-NP-Safe": {
        "operational_score": 0.7797,
        "weighted_target_discovery": 0.9008,
        "coverage_ratio": 0.5454,
        "runtime_seconds": 2.55,
    },
    "AntSwarmSafe": {
        "operational_score": 0.7754,
        "weighted_target_discovery": 0.8979,
        "coverage_ratio": 0.5056,
        "runtime_seconds": 1.81,
    },
}

STRATEGY_ACCENTS = {
    "H-MAPPO-EARS-Safe": "#6d5dfc",
    "EARS-Safe": "#ea6a22",
    "EARS-NP-Safe": "#d04b95",
    "AntSwarmSafe": "#238636",
}

AGENT_COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#ca8a04",
    "#475569",
    "#db2777",
    "#4f46e5",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", default="configs/publication_protocol_v7.json")
    p.add_argument("--ranking", default="results/publication/test/tables/overall_ranking.csv")
    p.add_argument("--model-dir", default="results/train/checkpoints")
    p.add_argument("--cities", default="San Francisco")
    p.add_argument("--all-test-cities", action="store_true")
    p.add_argument("--start-zone", default=None)
    p.add_argument("--agents", type=int, default=8)
    p.add_argument("--grid-size", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=160)
    p.add_argument("--seed", type=int, default=1042)
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--output-root", default="results/demo/top4-real-city")
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--require-real-data", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _resolve_cities(protocol: dict[str, Any], requested: str, all_test: bool) -> list[dict[str, Any]]:
    test_cities = select_cities(protocol, "test")
    if all_test:
        return test_cities
    names = [name.strip() for name in requested.split(",") if name.strip()]
    if not names:
        names = [str(test_cities[0]["name"])]
    by_name = {str(city["name"]).casefold(): city for city in test_cities}
    selected: list[dict[str, Any]] = []
    for name in names:
        city = by_name.get(name.casefold())
        if city is None:
            available = ", ".join(str(item["name"]) for item in test_cities)
            raise ValueError(f"Unknown test city {name!r}. Available: {available}")
        selected.append(city)
    return selected


def _checkpoint_paths(model_dir: str | Path) -> dict[str, list[Path]]:
    root = Path(model_dir)
    return {
        "H-MAPPO-EARS-Safe": [root / "h_mappo_ears_safe.json", root / "mappo_safe.json"],
        "EARS-Safe": [root / "ears_safe.json"],
        "EARS-NP-Safe": [root / "ears_np_safe.json"],
        "AntSwarmSafe": [],
    }


def _policy_factories(seed: int, model_dir: str | Path) -> dict[str, Callable[[], object]]:
    paths = _checkpoint_paths(model_dir)
    missing = [str(path) for required in paths.values() for path in required if not path.exists()]
    if missing:
        message = "\n  - ".join(missing)
        raise FileNotFoundError(
            "The top-4 real-city demo requires the frozen trained checkpoints. "
            "Run `docker compose up --build publication` first, or mount an existing results directory.\n"
            f"Missing:\n  - {message}"
        )

    h_ears, mappo = paths["H-MAPPO-EARS-Safe"]
    ears = paths["EARS-Safe"][0]
    ears_np = paths["EARS-NP-Safe"][0]
    return {
        "H-MAPPO-EARS-Safe": lambda: HMAPPOEARSPolicy(
            seed=seed,
            model_path=h_ears,
            mappo_model_path=mappo,
            strategy_name="H-MAPPO-EARS-Safe",
        ),
        "EARS-Safe": lambda: EARSPolicy(
            seed=seed, model_path=ears, strategy_name="EARS-Safe"
        ),
        "EARS-NP-Safe": lambda: EARSNegativePheromonePolicy(
            seed=seed, model_path=ears_np, strategy_name="EARS-NP-Safe"
        ),
        "AntSwarmSafe": AntSwarmPolicy,
    }


def _reference_metrics(path: Path) -> dict[str, dict[str, float]]:
    result = {name: dict(values) for name, values in PUBLICATION_REFERENCE.items()}
    if not path.exists():
        return result
    frame = pd.read_csv(path)
    if frame.empty or "strategy" not in frame.columns:
        return result
    metric_columns = (
        "operational_score",
        "weighted_target_discovery",
        "coverage_ratio",
        "runtime_seconds",
    )
    for strategy in TOP_STRATEGIES:
        rows = frame[frame["strategy"].astype(str) == strategy]
        if rows.empty:
            continue
        row = rows.iloc[0]
        for column in metric_columns:
            if column in row.index and pd.notna(row[column]):
                result[strategy][column] = float(row[column])
    return result


def _base_canvas(env: CityTwinEnvironment) -> np.ndarray:
    canvas = np.zeros((env.grid_size, env.grid_size), dtype=int)
    for cell in env.obstacles:
        canvas[cell] = 1
    for cell in env.restricted_zones:
        canvas[cell] = 2
    for cell in env.base_stations:
        canvas[cell] = 3
    return canvas.T


def _draw_city_background(ax: Any, env: CityTwinEnvironment) -> None:
    cmap = ListedColormap(["#f8fafc", "#334155", "#fecaca", "#86efac"])
    ax.imshow(
        _base_canvas(env),
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=3,
        interpolation="nearest",
        alpha=0.98,
    )
    ax.set_xticks(np.arange(-0.5, env.grid_size, 5), minor=True)
    ax.set_yticks(np.arange(-0.5, env.grid_size, 5), minor=True)
    ax.grid(which="minor", color="#cbd5e1", linewidth=0.28, alpha=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)


def _metric_row(env: CityTwinEnvironment, info: dict[str, Any], policy: object) -> dict[str, Any]:
    collisions = int(getattr(env, "actual_collisions", 0))
    restricted = int(getattr(env, "actual_restricted_entries", 0))
    row: dict[str, Any] = {
        "weighted_target_discovery": float(env.weighted_target_discovery()),
        "coverage_ratio": float(info["coverage_ratio"]),
        "actual_safety_incidents": collisions + restricted,
        "energy_consumption": float(env.energy_consumption()),
        "redundant_coverage": float(env.redundant_coverage()),
        "communication_efficiency": float(env.communication_efficiency()),
        "distance_travelled": int(sum(state.distance_travelled for state in env.agents.values())),
        "steps": int(env.steps),
        "agents": int(len(env.agents)),
        "targets_detected": int(len(env.discovered_missions)),
        "targets_total": int(len(env.mission_zones)),
    }
    row["mission_success_rate"] = row["targets_detected"] / max(1, row["targets_total"])
    row["operational_score"] = episode_operational_score(row)
    diagnostics = policy.diagnostics() if hasattr(policy, "diagnostics") else {}
    if diagnostics:
        row["diagnostics"] = diagnostics
    return row


def _save_trajectory_map(
    env: CityTwinEnvironment,
    trajectories: dict[int, list[tuple[int, int]]],
    detections: list[dict[str, Any]],
    metrics: dict[str, Any],
    path: Path,
    strategy: str,
    city_name: str,
    zone: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 8.6), dpi=190)
    _draw_city_background(ax, env)
    for aid, trajectory in sorted(trajectories.items()):
        if not trajectory:
            continue
        xs = [cell[0] for cell in trajectory]
        ys = [cell[1] for cell in trajectory]
        color = AGENT_COLORS[aid % len(AGENT_COLORS)]
        ax.plot(xs, ys, linewidth=1.65, alpha=0.84, color=color, label=f"robot {aid}")
        ax.scatter(xs[:1], ys[:1], s=42, marker="o", facecolor="white", edgecolor=color, linewidth=1.5)
        ax.scatter(xs[-1:], ys[-1:], s=45, marker="o", color=color, edgecolor="white", linewidth=0.8)
    if env.mission_zones:
        ax.scatter(
            [cell[0] for cell in env.mission_zones],
            [cell[1] for cell in env.mission_zones],
            marker="x",
            s=34,
            linewidths=1.25,
            color="#64748b",
            alpha=0.32,
            label="mission target (evaluation overlay)",
        )
    if env.discovered_missions:
        ax.scatter(
            [cell[0] for cell in env.discovered_missions],
            [cell[1] for cell in env.discovered_missions],
            marker="*",
            s=95,
            color="#f59e0b",
            edgecolor="#78350f",
            linewidth=0.4,
            label="detected target",
            zorder=8,
        )
    for index, event in enumerate(detections, start=1):
        ax.annotate(
            str(index),
            (event["x"], event["y"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.5,
            color="#7c2d12",
        )

    accent = STRATEGY_ACCENTS[strategy]
    ax.set_title(f"{strategy}\n{city_name} · {zone.replace('_', ' ')}", fontsize=13, fontweight="bold", color=accent)
    ax.text(
        0.015,
        0.015,
        f"score {metrics['operational_score']:.4f}  ·  discovery {metrics['weighted_target_discovery']:.3f}  ·  coverage {metrics['coverage_ratio']:.3f}\n"
        f"distance {metrics['distance_travelled']}  ·  energy {metrics['energy_consumption']:.1f}  ·  incidents {metrics['actual_safety_incidents']}",
        transform=ax.transAxes,
        fontsize=8.2,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=accent, alpha=0.92),
    )
    ax.set_xlabel("OSM-derived grid x")
    ax.set_ylabel("OSM-derived grid y")
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.legend(loc="upper right", fontsize=6.4, ncol=2, framealpha=0.92)
    fig.text(0.5, 0.008, OSM_ATTRIBUTION, ha="center", fontsize=7, color="#64748b")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_heatmap(
    env: CityTwinEnvironment,
    path: Path,
    strategy: str,
    city_name: str,
    zone: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 7.8), dpi=190)
    visits = env.visit_counts.astype(float).T
    blocked = np.zeros_like(visits, dtype=bool)
    for x, y in env.obstacles | env.restricted_zones:
        blocked[y, x] = True
    masked = np.ma.masked_where(blocked, visits)
    heat = LinearSegmentedColormap.from_list(
        "safe_swarm_heat", ["#f8fafc", "#bae6fd", "#38bdf8", "#2563eb", "#312e81"]
    )
    _draw_city_background(ax, env)
    image = ax.imshow(masked, origin="lower", cmap=heat, interpolation="nearest", alpha=0.82)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("visit count")
    if env.discovered_missions:
        ax.scatter(
            [cell[0] for cell in env.discovered_missions],
            [cell[1] for cell in env.discovered_missions],
            marker="*",
            s=85,
            color="#f59e0b",
            edgecolor="#78350f",
            linewidth=0.5,
            label="detected target",
        )
        ax.legend(loc="upper right", fontsize=7)
    ax.set_title(
        f"Visit intensity · {strategy}\n{city_name} · {zone.replace('_', ' ')}",
        fontsize=12.5,
        fontweight="bold",
        color=STRATEGY_ACCENTS[strategy],
    )
    ax.set_xlabel("OSM-derived grid x")
    ax.set_ylabel("OSM-derived grid y")
    fig.text(0.5, 0.008, OSM_ATTRIBUTION, ha="center", fontsize=7, color="#64748b")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_snapshot(
    env: CityTwinEnvironment,
    path: Path,
    strategy: str,
    city_name: str,
    zone: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), dpi=190)
    _draw_city_background(ax, env)
    if env.mission_zones:
        hidden = env.mission_zones - env.discovered_missions
        if hidden:
            ax.scatter(
                [cell[0] for cell in hidden],
                [cell[1] for cell in hidden],
                marker="x",
                s=34,
                color="#94a3b8",
                alpha=0.34,
                label="undetected target (evaluation overlay)",
            )
    if env.discovered_missions:
        ax.scatter(
            [cell[0] for cell in env.discovered_missions],
            [cell[1] for cell in env.discovered_missions],
            marker="*",
            s=100,
            color="#f59e0b",
            edgecolor="#78350f",
            linewidth=0.6,
            label="detected target",
        )
    positions = env.get_positions()
    for aid in sorted(positions):
        x, y = positions[aid]
        color = AGENT_COLORS[aid % len(AGENT_COLORS)]
        ax.scatter([x], [y], s=70, color=color, edgecolor="white", linewidth=0.9, zorder=10)
        ax.annotate(str(aid), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7, fontweight="bold")
    ax.set_title(
        f"Final city-twin state · {strategy}\n{city_name}",
        fontsize=12.5,
        fontweight="bold",
        color=STRATEGY_ACCENTS[strategy],
    )
    ax.set_xlabel("OSM-derived grid x")
    ax.set_ylabel("OSM-derived grid y")
    ax.legend(loc="upper right", fontsize=7)
    fig.text(0.5, 0.008, OSM_ATTRIBUTION, ha="center", fontsize=7, color="#64748b")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_animation(
    env: CityTwinEnvironment,
    frames: list[dict[str, Any]],
    path: Path,
    strategy: str,
    city_name: str,
    zone: str,
    fps: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.7, 7.7), dpi=120)
    accent = STRATEGY_ACCENTS[strategy]

    def draw(index: int) -> None:
        frame = frames[index]
        ax.clear()
        _draw_city_background(ax, env)
        discovered = set(frame["discovered"])
        hidden = env.mission_zones - discovered
        if hidden:
            ax.scatter(
                [c[0] for c in hidden],
                [c[1] for c in hidden],
                marker="x",
                s=26,
                color="#94a3b8",
                alpha=0.22,
                label="undetected target (evaluation overlay)",
            )
        if discovered:
            ax.scatter(
                [c[0] for c in discovered],
                [c[1] for c in discovered],
                marker="*",
                s=80,
                color="#f59e0b",
                edgecolor="#78350f",
                linewidth=0.5,
                label="detected target",
                zorder=8,
            )

        trails = frame["trails"]
        for aid in sorted(trails):
            trail = trails[aid]
            if not trail:
                continue
            color = AGENT_COLORS[aid % len(AGENT_COLORS)]
            xs = [c[0] for c in trail]
            ys = [c[1] for c in trail]
            ax.plot(xs, ys, linewidth=1.2, color=color, alpha=0.62)
            ax.scatter([xs[-1]], [ys[-1]], s=48, color=color, edgecolor="white", linewidth=0.8, zorder=10)
            ax.annotate(str(aid), (xs[-1], ys[-1]), xytext=(3, 3), textcoords="offset points", fontsize=6.5, fontweight="bold")

        ratio = len(discovered) / max(1, len(env.mission_zones))
        ax.set_title(
            f"{strategy} · {city_name}\nstep {frame['step']}  ·  detected {len(discovered)}/{len(env.mission_zones)} ({ratio:.0%})",
            fontsize=11.5,
            fontweight="bold",
            color=accent,
        )
        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(-0.5, env.grid_size - 0.5)
        ax.set_xlabel("OSM-derived grid x")
        ax.set_ylabel("OSM-derived grid y")
        ax.legend(loc="upper right", fontsize=6.2, framealpha=0.90)
        ax.text(
            0.015,
            0.015,
            f"start: {zone.replace('_', ' ')} · seed {frame['seed']}",
            transform=ax.transAxes,
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=accent, alpha=0.90),
        )

    animation = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / max(1, fps), repeat=True)
    animation.save(path, writer=PillowWriter(fps=max(1, fps)))
    plt.close(fig)


def _run_strategy(
    *,
    city: dict[str, Any],
    base_layers: dict[str, Any],
    zone: str,
    strategy: str,
    factory: Callable[[], object],
    args: argparse.Namespace,
    output: Path,
) -> dict[str, Any]:
    zoned = apply_start_zone(base_layers, args.grid_size, zone)
    env = CityTwinEnvironment(
        grid_size=args.grid_size,
        n_agents=args.agents,
        seed=args.seed,
        place_name=city["place"],
        radius_m=int(city.get("radius_m", 1400)),
        layers=zoned,
        max_steps=args.max_steps,
        allow_network=False,
    )
    policy = factory()
    output.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    known = set(env.discovered_missions)

    def capture() -> dict[str, Any]:
        return {
            "step": int(env.steps),
            "seed": int(args.seed),
            "discovered": sorted(env.discovered_missions),
            "trails": {aid: list(state.trajectory_history) for aid, state in env.agents.items()},
        }

    frames.append(capture())
    info: dict[str, Any] = {"coverage_ratio": 0.0, "done": 0.0}
    while True:
        actions = policy.act(env)
        info = env.step(actions)
        new = sorted(set(env.discovered_missions) - known)
        for cell in new:
            detections.append({"step": int(env.steps), "x": int(cell[0]), "y": int(cell[1])})
        known = set(env.discovered_missions)
        if env.steps % max(1, args.frame_stride) == 0 or info["done"] > 0 or new:
            frames.append(capture())
        if info["done"] > 0:
            break

    metrics = _metric_row(env, info, policy)
    trajectories = {aid: list(state.trajectory_history) for aid, state in env.agents.items()}
    _save_trajectory_map(
        env, trajectories, detections, metrics, output / "trajectory_map.png", strategy, str(city["name"]), zone
    )
    _save_heatmap(env, output / "visit_heatmap.png", strategy, str(city["name"]), zone)
    _save_snapshot(env, output / "final_snapshot.png", strategy, str(city["name"]), zone)
    _save_animation(env, frames, output / "animation.gif", strategy, str(city["name"]), zone, args.fps)
    pd.DataFrame(detections, columns=["step", "x", "y"]).to_csv(output / "detection_events.csv", index=False)

    summary = {
        "strategy": strategy,
        "city": city,
        "start_zone": zone,
        "seed": int(args.seed),
        "data_source": env.data_source,
        "data_attribution": OSM_ATTRIBUTION,
        "evaluation_overlay_notice": (
            "mission-target markers in images/GIF are post-evaluation overlays; policies receive observable evidence only"
        ),
        "metrics": metrics,
        "detections": detections,
        "artifacts": {
            "animation": "animation.gif",
            "trajectory_map": "trajectory_map.png",
            "visit_heatmap": "visit_heatmap.png",
            "final_snapshot": "final_snapshot.png",
            "detection_events": "detection_events.csv",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _metric_cell(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _city_dashboard(
    city: dict[str, Any],
    zone: str,
    summaries: list[dict[str, Any]],
    references: dict[str, dict[str, float]],
    output: Path,
) -> None:
    cards: list[str] = []
    table_rows: list[str] = []
    for summary in summaries:
        strategy = str(summary["strategy"])
        slug = _slug(strategy)
        metric = summary["metrics"]
        ref = references[strategy]
        accent = STRATEGY_ACCENTS[strategy]
        cards.append(
            f"""
            <article class='card' style='--accent:{accent}'>
              <div class='card-head'><div><div class='eyebrow'>Frozen policy</div><h2>{html.escape(strategy)}</h2></div>
              <div class='score'>{metric['operational_score']:.4f}<span>demo score</span></div></div>
              <div class='metrics'>
                <div><strong>{metric['weighted_target_discovery']:.3f}</strong><span>discovery</span></div>
                <div><strong>{metric['coverage_ratio']:.3f}</strong><span>coverage</span></div>
                <div><strong>{metric['energy_consumption']:.1f}</strong><span>energy</span></div>
                <div><strong>{metric['distance_travelled']}</strong><span>distance</span></div>
              </div>
              <img class='animation' src='{slug}/animation.gif' alt='{html.escape(strategy)} animation'>
              <div class='image-grid'>
                <a href='{slug}/trajectory_map.png'><img src='{slug}/trajectory_map.png' alt='trajectory map'></a>
                <a href='{slug}/visit_heatmap.png'><img src='{slug}/visit_heatmap.png' alt='visit heatmap'></a>
              </div>
              <div class='links'><a href='{slug}/final_snapshot.png'>Final snapshot</a><a href='{slug}/summary.json'>JSON metrics</a><a href='{slug}/detection_events.csv'>Detections CSV</a></div>
            </article>
            """
        )
        table_rows.append(
            "<tr>"
            f"<td><span class='dot' style='background:{accent}'></span>{html.escape(strategy)}</td>"
            f"<td>{_metric_cell(ref.get('operational_score'))}</td>"
            f"<td>{_metric_cell(ref.get('weighted_target_discovery'))}</td>"
            f"<td>{_metric_cell(ref.get('coverage_ratio'))}</td>"
            f"<td>{_metric_cell(ref.get('runtime_seconds'), 2)}</td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>SafeSwarm top-4 · {html.escape(str(city['name']))}</title>
<style>
:root{{--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--paper:#f8fafc;--panel:#fff}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(180deg,#eef6ff 0,#f8fafc 360px);color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
.wrap{{width:min(1440px,94vw);margin:0 auto;padding:42px 0 70px}} .hero{{display:grid;grid-template-columns:1.55fr .85fr;gap:28px;align-items:end;margin-bottom:30px}}
.kicker{{font-size:12px;letter-spacing:.13em;text-transform:uppercase;font-weight:800;color:#2563eb}} h1{{font-size:clamp(30px,4vw,54px);line-height:1.02;margin:7px 0 12px;letter-spacing:-.035em}} .lead{{font-size:17px;color:#475569;max-width:820px}}
.meta{{background:rgba(255,255,255,.86);border:1px solid #dbeafe;border-radius:18px;padding:18px 20px;box-shadow:0 14px 45px rgba(15,23,42,.08)}} .meta strong{{display:block;font-size:19px}} .meta span{{color:var(--muted)}}
.reference{{background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:18px 20px;margin-bottom:28px;box-shadow:0 12px 35px rgba(15,23,42,.05)}} .reference h2{{margin:0 0 10px;font-size:18px}} table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}} th,td{{padding:10px 12px;border-top:1px solid var(--line);text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}} .dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}} .card{{background:var(--panel);border:1px solid var(--line);border-top:4px solid var(--accent);border-radius:22px;padding:18px;box-shadow:0 16px 40px rgba(15,23,42,.07)}} .card-head{{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}} .eyebrow{{color:var(--accent);font-size:11px;letter-spacing:.09em;text-transform:uppercase;font-weight:800}} .card h2{{margin:2px 0 0;font-size:22px}} .score{{font-size:27px;font-weight:800;color:var(--accent);text-align:right;line-height:1}} .score span{{display:block;margin-top:6px;font-size:10px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:15px 0}} .metrics div{{background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:9px;text-align:center}} .metrics strong{{display:block;font-size:16px}} .metrics span{{font-size:10px;color:var(--muted);text-transform:uppercase}}
.animation{{width:100%;display:block;border-radius:15px;border:1px solid var(--line);background:white}} .image-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}} .image-grid img{{display:block;width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:12px;border:1px solid var(--line)}} .links{{display:flex;gap:12px;flex-wrap:wrap;margin-top:12px}} a{{color:#2563eb;text-decoration:none;font-weight:700;font-size:13px}} a:hover{{text-decoration:underline}}
.notice{{margin:26px 0 0;padding:15px 17px;border-radius:14px;background:#fffbeb;border:1px solid #fde68a;color:#92400e}} footer{{color:var(--muted);font-size:12px;margin-top:28px;text-align:center}}
@media(max-width:900px){{.hero,.grid{{grid-template-columns:1fr}} .reference{{overflow-x:auto}}}} @media(max-width:620px){{.metrics{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main class='wrap'>
<section class='hero'><div><div class='kicker'>SafeSwarm-CityTwin · real-city visual demo</div><h1>{html.escape(str(city['name']))}</h1><p class='lead'>Side-by-side replay of the four leading frozen policies on the same OpenStreetMap-derived city twin, start geometry and seed. Each card includes the full animation, trajectory map and visit heatmap.</p></div><div class='meta'><strong>{html.escape(str(city['place']))}</strong><span>Start zone: {html.escape(zone.replace('_',' '))}<br>Common seed: {summaries[0]['seed']}<br>Data: OpenStreetMap-derived</span></div></section>
<section class='reference'><h2>Publication benchmark reference</h2><table><thead><tr><th>Strategy</th><th>Operational score</th><th>Discovery</th><th>Coverage</th><th>Runtime (s)</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></section>
<section class='grid'>{''.join(cards)}</section>
<div class='notice'><strong>Scientific boundary:</strong> hidden mission markers are post-evaluation overlays for visualization only. Policy decisions use the repository's observable-only state contract.</div>
<footer>{html.escape(OSM_ATTRIBUTION)}</footer>
</main></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")


def _root_dashboard(city_outputs: list[tuple[dict[str, Any], Path]], output_root: Path) -> None:
    links = []
    for city, path in city_outputs:
        rel = path.relative_to(output_root).as_posix()
        links.append(f"<a class='city' href='{rel}/index.html'><strong>{html.escape(str(city['name']))}</strong><span>{html.escape(str(city['place']))}</span></a>")
    page = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>SafeSwarm real-city top-4 demo</title><style>body{{margin:0;background:#f8fafc;color:#0f172a;font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}main{{width:min(1000px,92vw);margin:0 auto;padding:60px 0}}.k{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#2563eb;font-weight:800}}h1{{font-size:46px;line-height:1.03;letter-spacing:-.035em;margin:8px 0 12px}}p{{color:#64748b;max-width:760px}}.cities{{display:grid;grid-template-columns:repeat(2,1fr);gap:15px;margin-top:30px}}.city{{display:block;background:white;border:1px solid #e2e8f0;border-radius:17px;padding:20px;text-decoration:none;color:#0f172a;box-shadow:0 12px 32px rgba(15,23,42,.06)}}.city:hover{{border-color:#93c5fd;transform:translateY(-1px)}}.city strong{{display:block;font-size:20px}}.city span{{color:#64748b;font-size:13px}}@media(max-width:680px){{.cities{{grid-template-columns:1fr}}}}</style></head><body><main><div class='k'>SafeSwarm-CityTwin</div><h1>Top-4 real-city policy demo</h1><p>Frozen H-MAPPO-EARS, EARS, EARS-NP and AntSwarmSafe policies replayed on identical real-map-derived city twins. Open a city to compare animations, heat maps and static trajectories.</p><div class='cities'>{''.join(links)}</div></main></body></html>"""
    (output_root / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.quick:
        args.agents = min(args.agents, 4)
        args.grid_size = min(args.grid_size, 20)
        args.max_steps = min(args.max_steps, 50)
        args.frame_stride = max(args.frame_stride, 3)
        args.fps = min(args.fps, 6)

    protocol = load_protocol(args.protocol)
    cities = _resolve_cities(protocol, args.cities, args.all_test_cities)
    zones = start_zones_for_split(protocol, "test")
    zone = args.start_zone or zones[0]
    if zone not in zones:
        raise ValueError(f"Demo start zone {zone!r} must be one of publication test zones: {zones}")

    factories = _policy_factories(args.seed, args.model_dir)
    references = _reference_metrics(Path(args.ranking))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    city_outputs: list[tuple[dict[str, Any], Path]] = []

    for city in cities:
        layers = load_real_city_layers(
            city["place"],
            args.grid_size,
            args.seed,
            radius_m=int(city.get("radius_m", 1400)),
            cache_dir=args.cache_dir,
            allow_network=not args.offline,
        )
        metadata = dict(layers.get("metadata", {}))
        if args.require_real_data and metadata.get("source") != "openstreetmap":
            raise RuntimeError(f"Demo city {city['name']} is not real OpenStreetMap data: {metadata}")

        city_output = output_root / _slug(str(city["name"]))
        city_output.mkdir(parents=True, exist_ok=True)
        summaries: list[dict[str, Any]] = []
        for strategy in TOP_STRATEGIES:
            print(f"[demo] {city['name']} · {strategy}", flush=True)
            summary = _run_strategy(
                city=city,
                base_layers=layers,
                zone=zone,
                strategy=strategy,
                factory=factories[strategy],
                args=args,
                output=city_output / _slug(strategy),
            )
            summaries.append(summary)

        pd.DataFrame(
            [
                {"strategy": item["strategy"], **{k: v for k, v in item["metrics"].items() if k != "diagnostics"}}
                for item in summaries
            ]
        ).to_csv(city_output / "demo_metrics.csv", index=False)
        _city_dashboard(city, zone, summaries, references, city_output)
        city_outputs.append((city, city_output))

    _root_dashboard(city_outputs, output_root)
    print(f"[demo] dashboard: {output_root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
