"""Evaluate v4 trained and fixed policies on held-out real cities/start zones."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from html import escape
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/real_city_protocol.json")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--model-dir", default="results/train/checkpoints")
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output-root", default="results/test")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--require-real-data", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _ci95(values: pd.Series) -> float:
    n = int(values.count())
    if n <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1) / np.sqrt(n))


def _mean_column(frame: pd.DataFrame, name: str) -> float:
    return float(frame[name].mean()) if name in frame and not frame.empty else 0.0


def _strategy_note(overall: pd.DataFrame, strategy: str) -> str:
    frame = overall[overall["strategy"] == strategy]
    if frame.empty:
        return f"{strategy} was not present."
    row = frame.iloc[0]
    return (
        f"{strategy}: rank {int(row['rank'])}, score {row['operational_score']:.3f} "
        f"± {row['operational_score_ci95']:.3f} (95% CI), "
        f"discovery {row['weighted_target_discovery']:.3f}, coverage {row['coverage_ratio']:.3f}, "
        f"redundancy {row['redundant_coverage']:.3f}."
    )


def _report(
    overall: pd.DataFrame,
    records: pd.DataFrame,
    city_metadata: list[dict[str, Any]],
    integrity: dict[str, bool],
    sparx_manifest: dict[str, Any],
    args: argparse.Namespace,
    path: Path,
) -> None:
    winner = overall.iloc[0]
    ranking = overall.round(4).to_html(index=False, classes="data", border=0)
    sources = pd.DataFrame(city_metadata).fillna("").to_html(index=False, classes="data", border=0)

    grpo_note = _strategy_note(overall, "GRPO-Safe")
    grpo_rows = records[records["strategy"] == "GRPO-Safe"]
    mechanism_note = (
        "Mean GRPO diagnostics: "
        f"masked candidates={_mean_column(grpo_rows, 'safety_mask_rejections'):.1f}, "
        f"forced fallbacks={_mean_column(grpo_rows, 'forced_fallbacks'):.2f}, "
        f"safe returns={_mean_column(grpo_rows, 'safe_returns'):.2f}, "
        f"memory coverage={_mean_column(grpo_rows, 'swarm_memory_coverage'):.3f}."
    )
    ablations = overall[overall["strategy"].str.startswith("GRPO-Safe-Ablation")]
    ablation_note = ""
    if not ablations.empty:
        parts = [
            f"{r['strategy'].replace('GRPO-Safe-Ablation-', '')}={r['operational_score']:.3f}"
            for _, r in ablations.sort_values("strategy").iterrows()
        ]
        ablation_note = "GRPO mechanism ablations: " + ", ".join(parts) + "."

    selected_pattern = str(sparx_manifest.get("selected_pattern", "not available"))
    sparx_note = _strategy_note(overall, "SPARX-Safe")
    pattern_notes = " ".join(
        _strategy_note(overall, f"SPARX-{label}-Safe")
        for label in ("X", "Plus", "Star")
        if not overall[overall["strategy"] == f"SPARX-{label}-Safe"].empty
    )
    sparx_rows = records[records["strategy"] == "SPARX-Safe"]
    sparx_diag = (
        "Mean SPARX diagnostics: "
        f"memory coverage={_mean_column(sparx_rows, 'sparx_memory_coverage'):.3f}, "
        f"probability entropy={_mean_column(sparx_rows, 'sparx_probability_entropy'):.3f}, "
        f"region count={_mean_column(sparx_rows, 'sparx_region_count'):.1f}, "
        f"goal switches={_mean_column(sparx_rows, 'sparx_goal_switches'):.1f}."
    )

    checks = " · ".join(
        f"{escape(k)}={'PASS' if v else 'FAIL'}" for k, v in integrity.items()
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SafeSwarm v4 Held-out Real-City Test</title>
<style>body{{font:15px/1.5 Inter,system-ui,sans-serif;margin:0;background:#f6f8fb;color:#172033}}header{{padding:36px 5vw;background:#172033;color:white}}main{{max-width:1500px;margin:auto;padding:22px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}.card,section{{background:white;border:1px solid #dde3ec;border-radius:13px;padding:18px;margin:14px 0}}.card strong{{display:block;font-size:24px}}table.data{{border-collapse:collapse;width:100%;font-size:12px}}.data th,.data td{{padding:8px;border-bottom:1px solid #e5e9f0;text-align:right}}.data th:first-child,.data td:first-child{{text-align:left}}.good{{color:#087830;font-weight:700}}code{{background:#eef2ff;padding:2px 5px;border-radius:5px}}</style></head><body>
<header><h1>SafeSwarm v4 · Held-out Real-City Test</h1><p>Observable-only policies · robust validation selection · unseen city + unseen starts</p></header><main><div class="cards">
<div class="card">Winner<strong>{escape(str(winner['strategy']))}</strong>score {winner['operational_score']:.3f}</div><div class="card">SPARX selection<strong>{escape(selected_pattern)}</strong>chosen on validation only</div><div class="card">Runs<strong>{len(records)}</strong>{records['strategy'].nunique()} algorithms</div><div class="card">Integrity<strong class="good">{'PASS' if all(integrity.values()) else 'FAIL'}</strong>{checks}</div></div>
<section><h2>Fair observation contract</h2><p>All primary strategies choose actions from runtime-observable evidence only. Hidden mission coordinates and priority labels are reserved for scoring.</p></section>
<section><h2>SPARX pattern-memory hypothesis</h2><p>{escape(sparx_note)}</p><p>{escape(sparx_diag)}</p><p>{escape(pattern_notes)}</p><p><strong>Important:</strong> the selected X/Plus/Star pattern and all SPARX weights were chosen using training + validation only. The held-out X/Plus/Star rows are mechanism analysis, not model selection.</p></section>
<section><h2>GRPO-Safe</h2><p>{escape(grpo_note)}</p><p>{escape(mechanism_note)}</p><p>{escape(ablation_note)}</p></section>
<section><h2>Held-out ranking</h2>{ranking}</section><section><h2>Real-data provenance</h2><p>Attribution: {escape(OSM_ATTRIBUTION)}</p>{sources}</section><section><h2>Reproduce</h2><p><code>docker compose up --build pipeline</code></p></section></main></body></html>"""
    path.write_text(html, encoding="utf-8")


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
    cities = select_cities(protocol, args.split)
    zones = start_zones_for_split(protocol, args.split)

    prepared: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    metadata_records: list[dict[str, Any]] = []
    for city in cities:
        layers = load_real_city_layers(
            city["place"], args.grid_size, args.seed,
            radius_m=int(city.get("radius_m", 1400)),
            cache_dir=args.cache_dir, allow_network=not args.offline,
        )
        metadata = dict(layers.get("metadata", {}))
        metadata.update({"city": city["name"], "split": args.split})
        if args.require_real_data and metadata.get("source") != "openstreetmap":
            raise RuntimeError(f"Test city {city['name']} is not real data: {metadata}")
        prepared[city["name"]] = (city, layers)
        metadata_records.append(metadata)

    sparx_manifest_path = Path(args.model_dir).parent / "sparx_manifest.json"
    sparx_manifest = (
        json.loads(sparx_manifest_path.read_text(encoding="utf-8"))
        if sparx_manifest_path.exists()
        else {}
    )

    records: list[dict[str, Any]] = []
    for city in cities:
        city_info, layers = prepared[city["name"]]
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            zone = zones[episode % len(zones)]
            zoned_layers = apply_start_zone(layers, args.grid_size, zone)
            for strategy, factory in evaluation_factories(episode_seed, args.model_dir).items():
                env = CityTwinEnvironment(
                    grid_size=args.grid_size, n_agents=args.agents, seed=episode_seed,
                    place_name=city_info["place"], radius_m=int(city_info.get("radius_m", 1400)),
                    layers=zoned_layers, max_steps=args.max_steps, allow_network=False,
                )
                policy = factory()
                metrics = run_episode(env, policy)
                row = metrics.to_dict(
                    strategy=strategy, episode=episode, split=args.split,
                    city=city_info["name"], place=city_info["place"], start_zone=zone,
                    data_source=env.data_source, agents=args.agents, grid_size=args.grid_size,
                    steps=env.steps, seed=episode_seed,
                )
                row["operational_score"] = episode_operational_score(row)
                diagnostics = policy.diagnostics() if hasattr(policy, "diagnostics") else {}
                monitor = getattr(policy, "monitor", None)
                row.update(
                    {
                        "checkpoint": diagnostics.get("checkpoint"),
                        "ppo_residual_weight_norm": diagnostics.get("ppo_residual_weight_norm", 0.0),
                        "critic_weight_norm": diagnostics.get("critic_weight_norm", 0.0),
                        "behavior_weight_norm": diagnostics.get("behavior_weight_norm", 0.0),
                        "behavior_state_weight_norm": diagnostics.get("behavior_state_weight_norm", 0.0),
                        "safety_mask_rejections": diagnostics.get("safety_mask_rejections", 0),
                        "forced_fallbacks": diagnostics.get("forced_fallbacks", 0),
                        "return_guard_interventions": 0 if monitor is None else int(getattr(monitor, "return_guard_interventions", 0)),
                        "safe_returns": int(env.safe_return_count),
                        "swarm_memory_coverage": diagnostics.get("swarm_memory_coverage", 0.0),
                        "swarm_memory_peak": diagnostics.get("swarm_memory_peak", 0.0),
                        "learned_memory_weight": diagnostics.get("learned_memory_weight", 0.0),
                        "learned_frontier_weight": diagnostics.get("learned_frontier_weight", 0.0),
                        "learned_propagation_weight": diagnostics.get("learned_propagation_weight", 0.0),
                        "sparx_pattern": diagnostics.get("sparx_pattern", ""),
                        "sparx_memory_coverage": diagnostics.get("sparx_memory_coverage", 0.0),
                        "sparx_memory_peak": diagnostics.get("sparx_memory_peak", 0.0),
                        "sparx_assignment_refreshes": diagnostics.get("sparx_assignment_refreshes", 0),
                        "sparx_goal_switches": diagnostics.get("sparx_goal_switches", 0),
                        "sparx_probability_entropy": diagnostics.get("sparx_probability_entropy", 0.0),
                        "sparx_probability_peak": diagnostics.get("sparx_probability_peak", 0.0),
                        "sparx_region_count": diagnostics.get("sparx_region_count", 0),
                    }
                )
                records.append(row)
                print(
                    f"[{args.split}] {city_info['name']} episode={episode + 1}/{args.episodes} "
                    f"zone={zone} strategy={strategy} score={row['operational_score']:.3f}", flush=True,
                )

    output = Path(args.output_root)
    tables = output / "tables"
    output.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    city_summary, overall = rank_algorithms(frame)
    score_ci = frame.groupby("strategy")["operational_score"].apply(_ci95).rename("operational_score_ci95")
    overall = overall.merge(score_ci, left_on="strategy", right_index=True, how="left").sort_values("operational_score", ascending=False).reset_index(drop=True)
    overall["rank"] = np.arange(1, len(overall) + 1)
    frame.to_csv(tables / "episode_results.csv", index=False)
    city_summary.to_csv(tables / "city_ranking.csv", index=False)
    overall.to_csv(tables / "overall_ranking.csv", index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "protocol_integrity": integrity,
        "cities": metadata_records,
        "winner": overall.iloc[0].to_dict(),
        "test_is_held_out": args.split == "test",
        "data_attribution": OSM_ATTRIBUTION,
        "training_version": "SafeSwarm PPO/GRPO/SPARX v4",
        "observation_contract": "observable-only primary policies; hidden target labels reserved for evaluation",
        "checkpoint_selection": "validation-only; held-out test never selects PPO/GRPO/SPARX weights or pattern",
        "grpo_ablations": ["NoMemory", "NoPropagation", "NoLearnedBehavior"],
        "sparx_algorithm": "SPARX: Swarm Probability-map Allocation & Region eXploration",
        "sparx_selected_pattern": sparx_manifest.get("selected_pattern"),
        "sparx_patterns_reported": ["X", "Plus", "Star"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    _report(overall, frame, metadata_records, integrity, sparx_manifest, args, output / "report.html")


if __name__ == "__main__":
    main()
