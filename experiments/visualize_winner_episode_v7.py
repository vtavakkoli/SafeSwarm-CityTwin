"""Create a reproducible qualitative episode for the frozen publication winner.

Outputs an animated GIF, final visit heat map, static agent trajectory map,
detection-event CSV, and JSON summary. Hidden mission cells are shown only in
these post-evaluation figures and are never exposed to the policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.trainable_policies import evaluation_factories  # noqa: E402
from src.environment.city_twin import CityTwinEnvironment  # noqa: E402
from src.environment.obstacles import OSM_ATTRIBUTION, load_real_city_layers  # noqa: E402
from src.evaluation.metrics import episode_operational_score  # noqa: E402
from src.training.geography import apply_start_zone, load_protocol, select_cities, start_zones_for_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", default="configs/publication_protocol_v7.json")
    p.add_argument("--ranking", default="results/publication/test/tables/overall_ranking.csv")
    p.add_argument("--model-dir", default="results/train/checkpoints")
    p.add_argument("--city", default="San Francisco")
    p.add_argument("--start-zone", default=None)
    p.add_argument("--agents", type=int, default=8)
    p.add_argument("--grid-size", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=160)
    p.add_argument("--seed", type=int, default=1042)
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--output-root", default="results/publication/visualization")
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--require-real-data", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def _background(env: CityTwinEnvironment) -> np.ndarray:
    canvas = np.zeros((env.grid_size, env.grid_size), dtype=int)
    for cell in env.obstacles:
        canvas[cell] = 1
    for cell in env.restricted_zones:
        canvas[cell] = 2
    for cell in env.base_stations:
        canvas[cell] = 3
    return canvas.T


def _policy_name(ranking_path: Path) -> str:
    if ranking_path.exists():
        ranking = pd.read_csv(ranking_path)
        if not ranking.empty:
            return str(ranking.sort_values("rank").iloc[0]["strategy"])
    return "EARS-Safe"


def _save_static_map(
    env: CityTwinEnvironment,
    trajectories: dict[int, list[tuple[int, int]]],
    detections: list[dict[str, Any]],
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), dpi=170)
    cmap = ListedColormap(["white", "#343a40", "#ef9a9a", "#4caf50"])
    ax.imshow(_background(env), origin="lower", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    for aid, trajectory in sorted(trajectories.items()):
        if not trajectory:
            continue
        x = [cell[0] for cell in trajectory]
        y = [cell[1] for cell in trajectory]
        ax.plot(x, y, linewidth=1.4, alpha=0.82, label=f"agent {aid}")
        ax.scatter([x[-1]], [y[-1]], s=18)
    if env.mission_zones:
        mx = [cell[0] for cell in env.mission_zones]
        my = [cell[1] for cell in env.mission_zones]
        ax.scatter(mx, my, marker="x", s=48, linewidths=1.8, label="mission targets (evaluation only)")
    for index, event in enumerate(detections, start=1):
        x, y = event["x"], event["y"]
        ax.annotate(str(index), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_heatmap(env: CityTwinEnvironment, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), dpi=170)
    image = ax.imshow(env.visit_counts.T, origin="lower", interpolation="nearest")
    fig.colorbar(image, ax=ax, label="visit count")
    if env.discovered_missions:
        x = [cell[0] for cell in env.discovered_missions]
        y = [cell[1] for cell in env.discovered_missions]
        ax.scatter(x, y, marker="x", s=45, linewidths=1.5, label="detected target")
        ax.legend(loc="upper right")
    ax.set_title(title)
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_animation(
    env: CityTwinEnvironment,
    frames: list[dict[str, Any]],
    path: Path,
    fps: int,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=120)
    cmap = ListedColormap(["white", "#343a40", "#ef9a9a", "#4caf50"])
    background = _background(env)

    def draw(index: int) -> None:
        frame = frames[index]
        ax.clear()
        ax.imshow(background, origin="lower", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        discovered = frame["discovered"]
        if env.mission_zones:
            hidden = env.mission_zones - discovered
            if hidden:
                ax.scatter(
                    [c[0] for c in hidden], [c[1] for c in hidden],
                    marker="x", s=30, alpha=0.22, label="undetected target (evaluation overlay)",
                )
        if discovered:
            ax.scatter(
                [c[0] for c in discovered], [c[1] for c in discovered],
                marker="*", s=75, label="detected target",
            )
        positions = frame["positions"]
        ax.scatter(
            [positions[aid][0] for aid in sorted(positions)],
            [positions[aid][1] for aid in sorted(positions)],
            s=38,
        )
        for aid, cell in positions.items():
            ax.annotate(str(aid), cell, xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.set_title(
            f"{title}\nstep {frame['step']} · detected {len(discovered)}/{len(env.mission_zones)}"
        )
        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(-0.5, env.grid_size - 0.5)
        ax.set_xlabel("grid x")
        ax.set_ylabel("grid y")
        ax.legend(loc="upper right", fontsize=7)

    animation = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / max(1, fps), repeat=False)
    animation.save(path, writer=PillowWriter(fps=max(1, fps)))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.quick:
        args.agents = min(args.agents, 4)
        args.grid_size = min(args.grid_size, 20)
        args.max_steps = min(args.max_steps, 50)
        args.frame_stride = max(2, args.frame_stride)

    protocol = load_protocol(args.protocol)
    test_cities = select_cities(protocol, "test")
    city = next((item for item in test_cities if item["name"].casefold() == args.city.casefold()), None)
    if city is None:
        city = test_cities[0]
    zones = start_zones_for_split(protocol, "test")
    zone = args.start_zone or zones[0]
    winner = _policy_name(Path(args.ranking))

    layers = load_real_city_layers(
        city["place"], args.grid_size, args.seed,
        radius_m=int(city.get("radius_m", 1400)), cache_dir=args.cache_dir,
        allow_network=not args.offline,
    )
    metadata = dict(layers.get("metadata", {}))
    if args.require_real_data and metadata.get("source") != "openstreetmap":
        raise RuntimeError(f"Visualization city {city['name']} is not real OSM data: {metadata}")
    zoned = apply_start_zone(layers, args.grid_size, zone)
    env = CityTwinEnvironment(
        grid_size=args.grid_size, n_agents=args.agents, seed=args.seed,
        place_name=city["place"], radius_m=int(city.get("radius_m", 1400)),
        layers=zoned, max_steps=args.max_steps, allow_network=False,
    )
    factories = evaluation_factories(args.seed, args.model_dir)
    if winner not in factories:
        raise KeyError(f"Winner {winner!r} is not available in evaluation factories")
    policy = factories[winner]()

    frames: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    known = set(env.discovered_missions)
    frames.append({"step": 0, "positions": dict(env.get_positions()), "discovered": set(known)})
    while True:
        actions = policy.act(env)
        info = env.step(actions)
        new = sorted(set(env.discovered_missions) - known)
        for cell in new:
            detections.append({"step": int(env.steps), "x": int(cell[0]), "y": int(cell[1])})
        known = set(env.discovered_missions)
        if env.steps % max(1, args.frame_stride) == 0 or info["done"] > 0 or new:
            frames.append(
                {"step": int(env.steps), "positions": dict(env.get_positions()), "discovered": set(known)}
            )
        if info["done"] > 0:
            break

    trajectories = {aid: list(state.trajectory_history) for aid, state in env.agents.items()}
    diagnostics = policy.diagnostics() if hasattr(policy, "diagnostics") else {}
    row = {
        "weighted_target_discovery": env.weighted_target_discovery(),
        "coverage_ratio": float(info["coverage_ratio"]),
        "actual_safety_incidents": int(env.actual_collisions + env.actual_restricted_entries),
        "energy_consumption": env.energy_consumption(),
        "redundant_coverage": env.redundant_coverage(),
        "communication_efficiency": env.communication_efficiency(),
        "distance_travelled": int(sum(state.distance_travelled for state in env.agents.values())),
        "steps": int(env.steps),
        "agents": int(args.agents),
    }
    row["operational_score"] = episode_operational_score(row)
    row["mission_success_rate"] = len(env.discovered_missions) / max(1, len(env.mission_zones))

    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    title = f"{winner} · {city['name']} · {zone}"
    _save_static_map(env, trajectories, detections, output / "winner_trajectory_map.png", title)
    _save_heatmap(env, output / "winner_visit_heatmap.png", title)
    _save_animation(env, frames, output / "winner_episode.gif", args.fps, title)
    pd.DataFrame(detections).to_csv(output / "detection_events.csv", index=False)

    summary = {
        "winner": winner,
        "city": city,
        "start_zone": zone,
        "seed": args.seed,
        "data_source": env.data_source,
        "data_attribution": OSM_ATTRIBUTION,
        "evaluation_overlay_notice": (
            "mission-target markers in the figures are post-evaluation overlays; the policy receives observable evidence only"
        ),
        "metrics": row,
        "diagnostics": diagnostics,
        "detections": detections,
        "artifacts": {
            "animation": "winner_episode.gif",
            "heatmap": "winner_visit_heatmap.png",
            "trajectory_map": "winner_trajectory_map.png",
            "detection_events": "detection_events.csv",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    html = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Winner episode</title><style>body{{font:15px/1.5 system-ui;margin:30px;max-width:1100px}}img{{max-width:100%;border:1px solid #ddd;border-radius:10px;margin:10px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}</style></head><body><h1>{winner} · {city['name']}</h1><p>Frozen post-selection qualitative episode. Target markers are evaluation overlays only.</p><h2>Animation</h2><img src='winner_episode.gif'><div class='grid'><div><h2>Visit heat map</h2><img src='winner_visit_heatmap.png'></div><div><h2>Static trajectory map</h2><img src='winner_trajectory_map.png'></div></div><pre>{json.dumps(row, indent=2)}</pre></body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
