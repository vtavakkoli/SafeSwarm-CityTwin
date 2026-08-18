"""Evaluate frozen SafeSwarm v5 policies on held-out real cities/start zones."""

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


def _mean(frame: pd.DataFrame, name: str) -> float:
    return float(frame[name].mean()) if name in frame and not frame.empty else 0.0


def _strategy_note(overall: pd.DataFrame, strategy: str) -> str:
    rows = overall[overall["strategy"] == strategy]
    if rows.empty:
        return f"{strategy} was not present."
    row = rows.iloc[0]
    return (
        f"{strategy}: rank {int(row['rank'])}, score {row['operational_score']:.3f} "
        f"± {row['operational_score_ci95']:.3f}, discovery {row['weighted_target_discovery']:.3f}, "
        f"coverage {row['coverage_ratio']:.3f}, redundancy {row['redundant_coverage']:.3f}."
    )


def _report(
    overall: pd.DataFrame,
    records: pd.DataFrame,
    metadata: list[dict[str, Any]],
    integrity: dict[str, bool],
    prism_manifest: dict[str, Any],
    args: argparse.Namespace,
    path: Path,
) -> None:
    winner = overall.iloc[0]
    selected = str(prism_manifest.get("selected_pattern", "not available"))
    ranking = overall.round(4).to_html(index=False, classes="data", border=0)
    sources = pd.DataFrame(metadata).fillna("").to_html(index=False, classes="data", border=0)
    prism_rows = records[records["strategy"] == "PRISM-Safe"]
    hybrid_rows = records[records["strategy"] == "PRISM-Ant-Safe"]
    checks = " · ".join(f"{escape(k)}={'PASS' if v else 'FAIL'}" for k, v in integrity.items())
    prism_diag = (
        f"memory coverage={_mean(prism_rows, 'prism_memory_coverage'):.3f}, "
        f"probability entropy={_mean(prism_rows, 'prism_probability_entropy'):.3f}, "
        f"regions={_mean(prism_rows, 'prism_region_count'):.1f}."
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SafeSwarm v5 Held-out Test</title>
<style>body{{font:15px/1.5 Inter,system-ui,sans-serif;margin:0;background:#f6f8fb;color:#172033}}header{{padding:36px 5vw;background:#172033;color:white}}main{{max-width:1500px;margin:auto;padding:22px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}.card,section{{background:white;border:1px solid #dde3ec;border-radius:13px;padding:18px;margin:14px 0}}.card strong{{display:block;font-size:24px}}table.data{{border-collapse:collapse;width:100%;font-size:12px}}.data th,.data td{{padding:8px;border-bottom:1px solid #e5e9f0;text-align:right}}.data th:first-child,.data td:first-child{{text-align:left}}.good{{color:#087830;font-weight:700}}code{{background:#eef2ff;padding:2px 5px;border-radius:5px}}</style></head><body>
<header><h1>SafeSwarm v5 · Held-out Real-City Test</h1><p>PRISM + PRISM-Ant · deterministic learned-policy inference · unseen cities/starts</p></header><main>
<div class="cards"><div class="card">Winner<strong>{escape(str(winner['strategy']))}</strong>score {winner['operational_score']:.3f}</div><div class="card">PRISM pattern<strong>{escape(selected)}</strong>validation selected</div><div class="card">PRISM-Ant<strong>{'present' if not hybrid_rows.empty else 'missing'}</strong>hybrid frozen before test</div><div class="card">Integrity<strong class="good">{'PASS' if all(integrity.values()) else 'FAIL'}</strong>{checks}</div></div>
<section><h2>Scientific contract</h2><p>All primary policies act on runtime-observable evidence only. PPO/GRPO weights, PRISM pattern/weights and PRISM-Ant fusion parameters are selected on training + validation only. This held-out test cannot change them. SWAP is a separate post-selection stress test.</p></section>
<section><h2>PRISM / PRISM-Ant</h2><p>{escape(_strategy_note(overall, 'PRISM-Safe'))}</p><p>{escape(_strategy_note(overall, 'PRISM-Ant-Safe'))}</p><p>{escape(prism_diag)}</p><p>Pattern variants: {escape(' '.join(_strategy_note(overall, f'PRISM-{label}-Safe') for label in ('X','Plus','Star') if not overall[overall['strategy']==f'PRISM-{label}-Safe'].empty))}</p></section>
<section><h2>Learned baselines</h2><p>{escape(_strategy_note(overall, 'IPPO-Safe'))}</p><p>{escape(_strategy_note(overall, 'MAPPO-Safe'))}</p><p>{escape(_strategy_note(overall, 'HAPPO-Safe'))}</p><p>{escape(_strategy_note(overall, 'GRPO-Safe'))}</p></section>
<section><h2>Held-out ranking</h2>{ranking}</section><section><h2>Real-data provenance</h2><p>{escape(OSM_ATTRIBUTION)}</p>{sources}</section></main></body></html>"""
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
    if args.quick:
        cities = cities[:1]
        zones = zones[:1]

    prepared: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    metadata: list[dict[str, Any]] = []
    for city in cities:
        layers = load_real_city_layers(
            city["place"], args.grid_size, args.seed,
            radius_m=int(city.get("radius_m", 1400)),
            cache_dir=args.cache_dir, allow_network=not args.offline,
        )
        meta = dict(layers.get("metadata", {}))
        meta.update({"city": city["name"], "split": args.split})
        if args.require_real_data and meta.get("source") != "openstreetmap":
            raise RuntimeError(f"Held-out city {city['name']} is not real data: {meta}")
        prepared[city["name"]] = (city, layers)
        metadata.append(meta)

    prism_manifest_path = Path(args.model_dir).parent / "prism_manifest.json"
    prism_manifest = (
        json.loads(prism_manifest_path.read_text(encoding="utf-8"))
        if prism_manifest_path.exists() else {}
    )

    records: list[dict[str, Any]] = []
    for city in cities:
        city_info, layers = prepared[city["name"]]
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            zone = zones[episode % len(zones)]
            zoned = apply_start_zone(layers, args.grid_size, zone)
            for strategy, factory in evaluation_factories(episode_seed, args.model_dir).items():
                env = CityTwinEnvironment(
                    grid_size=args.grid_size, n_agents=args.agents, seed=episode_seed,
                    place_name=city_info["place"], radius_m=int(city_info.get("radius_m", 1400)),
                    layers=zoned, max_steps=args.max_steps, allow_network=False,
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
                        "deterministic_eval": diagnostics.get("deterministic_eval", False),
                        "coordination_refreshes": diagnostics.get("coordination_refreshes", 0),
                        "ppo_residual_weight_norm": diagnostics.get("ppo_residual_weight_norm", 0.0),
                        "critic_weight_norm": diagnostics.get("critic_weight_norm", 0.0),
                        "behavior_state_weight_norm": diagnostics.get("behavior_state_weight_norm", 0.0),
                        "safety_mask_rejections": diagnostics.get("safety_mask_rejections", 0),
                        "forced_fallbacks": diagnostics.get("forced_fallbacks", 0),
                        "return_guard_interventions": 0 if monitor is None else int(getattr(monitor, "return_guard_interventions", 0)),
                        "safe_returns": int(env.safe_return_count),
                        "swarm_memory_coverage": diagnostics.get("swarm_memory_coverage", 0.0),
                        "prism_pattern": diagnostics.get("prism_pattern", ""),
                        "prism_memory_coverage": diagnostics.get("prism_memory_coverage", 0.0),
                        "prism_probability_entropy": diagnostics.get("prism_probability_entropy", 0.0),
                        "prism_region_count": diagnostics.get("prism_region_count", 0),
                        "prism_ant_blend": diagnostics.get("prism_ant_blend", 0.0),
                    }
                )
                records.append(row)
                print(
                    f"[{args.split}-v5] {city_info['name']} episode={episode + 1}/{args.episodes} "
                    f"zone={zone} strategy={strategy} score={row['operational_score']:.3f}",
                    flush=True,
                )

    output = Path(args.output_root)
    tables = output / "tables"
    output.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    city_summary, overall = rank_algorithms(frame)
    ci = frame.groupby("strategy")["operational_score"].apply(_ci95).rename("operational_score_ci95")
    overall = overall.merge(ci, left_on="strategy", right_index=True, how="left")
    overall = overall.sort_values("operational_score", ascending=False).reset_index(drop=True)
    overall["rank"] = np.arange(1, len(overall) + 1)
    frame.to_csv(tables / "episode_results.csv", index=False)
    city_summary.to_csv(tables / "city_ranking.csv", index=False)
    overall.to_csv(tables / "overall_ranking.csv", index=False)

    names = set(overall["strategy"])
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "protocol_integrity": integrity,
        "cities": metadata,
        "winner": overall.iloc[0].to_dict(),
        "test_is_held_out": args.split == "test",
        "data_attribution": OSM_ATTRIBUTION,
        "training_version": "SafeSwarm PPO/GRPO/PRISM v5",
        "observation_contract": "observable-only primary policies; hidden target labels reserved for evaluation",
        "checkpoint_selection": "validation-only; held-out test and SWAP never select weights, PRISM pattern, or hybrid fusion",
        "deterministic_learned_evaluation": True,
        "grpo_ablations": ["NoMemory", "NoPropagation", "NoLearnedBehavior"],
        "prism_algorithm": "PRISM: Probability-guided Region-Integrated Search with Memory",
        "prism_ant_algorithm": "PRISM-Ant: PRISM global allocation + AntSwarm local search",
        "prism_selected_pattern": prism_manifest.get("selected_pattern"),
        "prism_patterns_reported": ["X", "Plus", "Star"],
        "obsolete_sparx_names_present": any(name.startswith("SPARX") for name in names),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    _report(overall, frame, metadata, integrity, prism_manifest, args, output / "report.html")


if __name__ == "__main__":
    main()
