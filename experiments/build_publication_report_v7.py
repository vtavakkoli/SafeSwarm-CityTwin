"""Build the final SafeSwarm v7 publication-validation report."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="results/publication")
    p.add_argument("--output", default="results/publication/report.html")
    return p.parse_args()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "<p>Not available.</p>"
    return frame.round(digits).to_html(index=False, classes="data", border=0)


def _metric_row(frame: pd.DataFrame, metric: str) -> pd.Series | None:
    rows = frame[frame["metric"] == metric] if not frame.empty and "metric" in frame else pd.DataFrame()
    return None if rows.empty else rows.iloc[0]


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    test = _read(root / "test" / "tables" / "overall_ranking.csv")
    city = _read(root / "test" / "tables" / "city_ranking.csv")
    swap = _read(root / "swap-test" / "tables" / "overall_ranking.csv")
    paired = _read(root / "statistics" / "tables" / "paired_test.csv")
    paired_swap = _read(root / "statistics" / "tables" / "paired_swap.csv")
    mean_std = _read(root / "statistics" / "tables" / "test_citymean_mean_std.csv")
    ablations = _read(root / "ablations" / "tables" / "overall_mean_std.csv")
    triggers = _read(root / "ablations" / "tables" / "trigger_summary.csv")
    sensitivity = _read(root / "sensitivity" / "tables" / "sensitivity_summary.csv")
    test_manifest_path = root / "test" / "manifest.json"
    test_manifest = json.loads(test_manifest_path.read_text(encoding="utf-8")) if test_manifest_path.exists() else {}
    vis_summary_path = root / "visualization" / "summary.json"
    vis_summary = json.loads(vis_summary_path.read_text(encoding="utf-8")) if vis_summary_path.exists() else {}

    if test.empty:
        raise FileNotFoundError("publication held-out ranking is required before building the report")
    winner = test.sort_values("rank").iloc[0]
    ant = test[test["strategy"] == "AntSwarmSafe"]
    ears = test[test["strategy"] == "EARS-Safe"]
    op = _metric_row(paired, "operational_score")
    swap_op = _metric_row(paired_swap, "operational_score")
    city_count = int(winner.get("cities_evaluated", test_manifest.get("publication_test_city_count", 0)))

    gate_checks = {
        "at_least_6_heldout_cities": city_count >= 6,
        "paired_operational_ci_excludes_zero": bool(
            op is not None
            and (float(op["paired_bootstrap_ci95_low"]) > 0.0 or float(op["paired_bootstrap_ci95_high"]) < 0.0)
        ),
        "hierarchical_ci_excludes_zero": bool(
            op is not None
            and (float(op["hierarchical_bootstrap_ci95_low"]) > 0.0 or float(op["hierarchical_bootstrap_ci95_high"]) < 0.0)
        ),
        "paired_randomization_p_lt_0_05": bool(
            op is not None and float(op["paired_permutation_pvalue_two_sided"]) < 0.05
        ),
        "swap_statistics_available": swap_op is not None,
        "mechanism_ablations_available": not ablations.empty,
        "score_sensitivity_available": not sensitivity.empty,
        "winner_visualization_available": bool(vis_summary),
    }
    evidence_gate = all(gate_checks.values())

    method_rows = test[test["strategy"].isin([
        "EARS-Safe", "EARS-NP-Safe", "H-MAPPO-EARS-Safe", "AntSwarmSafe",
        "PRISM-Ant-Safe", "MAPPO-Safe", "HAPPO-Safe", "IPPO-Safe",
    ])]
    if not mean_std.empty:
        method_rows = method_rows.merge(mean_std, on="strategy", how="left")

    comparisons = []
    if not ears.empty and not ant.empty:
        e = ears.iloc[0]
        a = ant.iloc[0]
        for metric in (
            "operational_score", "weighted_target_discovery", "coverage_ratio",
            "energy_consumption", "redundant_coverage", "distance_travelled",
        ):
            comparisons.append(
                {
                    "metric": metric,
                    "EARS-Safe": float(e[metric]),
                    "AntSwarmSafe": float(a[metric]),
                    "EARS-minus-Ant": float(e[metric] - a[metric]),
                }
            )
    comparison_table = pd.DataFrame(comparisons)

    checks_html = "".join(
        f"<li class={'pass' if value else 'fail'}>{escape(name.replace('_', ' '))}: "
        f"<strong>{'PASS' if value else 'INCOMPLETE'}</strong></li>"
        for name, value in gate_checks.items()
    )
    op_text = "paired statistics unavailable"
    if op is not None:
        op_text = (
            f"Δ={float(op['paired_delta_mean']):.4f}, paired bootstrap 95% CI "
            f"[{float(op['paired_bootstrap_ci95_low']):.4f}, {float(op['paired_bootstrap_ci95_high']):.4f}], "
            f"hierarchical 95% CI [{float(op['hierarchical_bootstrap_ci95_low']):.4f}, "
            f"{float(op['hierarchical_bootstrap_ci95_high']):.4f}], "
            f"paired randomization p={float(op['paired_permutation_pvalue_two_sided']):.5f}."
        )
    sens_text = "sensitivity unavailable"
    if not sensitivity.empty:
        s = sensitivity.iloc[0]
        sens_text = (
            f"EARS beats Ant in {100 * float(s['candidate_beats_baseline_fraction']):.1f}% of "
            f"weight perturbations; 95% margin range "
            f"[{float(s['margin_p02_5']):.4f}, {float(s['margin_p97_5']):.4f}]."
        )

    html = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>SafeSwarm v7 Publication Validation</title><style>:root{{--ink:#172033;--muted:#667085;--line:#dfe5ef;--bg:#f4f7fb;--good:#087830;--bad:#a11}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,system-ui,sans-serif}}header{{padding:38px 5vw;background:linear-gradient(120deg,#111827,#3448c5);color:white}}main{{max-width:1500px;margin:-18px auto 40px;padding:0 24px}}section,.card{{background:white;border:1px solid var(--line);border-radius:14px;box-shadow:0 5px 18px #1018280b}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-bottom:18px}}.card{{padding:18px}}.card strong{{display:block;font-size:24px}}section{{padding:22px;margin:18px 0;overflow:auto}}table.data{{border-collapse:collapse;width:100%;font-size:12px}}.data th,.data td{{padding:8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}.data th:first-child,.data td:first-child{{text-align:left}}.pass{{color:var(--good)}}.fail{{color:var(--bad)}}img{{max-width:100%;border:1px solid var(--line);border-radius:10px}}.visuals{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}code{{background:#eef2ff;padding:2px 5px;border-radius:5px}}</style></head><body>
<header><h1>SafeSwarm CityTwin v7 · Publication Validation</h1><p>Frozen EARS models · 8-city held-out generalization · SWAP · paired/hierarchical statistics · ablations · score sensitivity · qualitative episode</p></header><main>
<div class='cards'><div class='card'>Held-out winner<strong>{escape(str(winner['strategy']))}</strong>score {float(winner['operational_score']):.3f}</div><div class='card'>Held-out cities<strong>{city_count}</strong>post-selection only</div><div class='card'>Evidence gate<strong>{'PASS' if evidence_gate else 'INCOMPLETE'}</strong>reporting check, not model selection</div><div class='card'>SWAP<strong>{'available' if not swap.empty else 'missing'}</strong>alternate hidden targets</div></div>
<section><h2>Scientific contract</h2><p>All checkpoints and EARS trigger/relocation parameters are frozen from the original v6 train/validation protocol before this suite runs. The eight publication test cities, SWAP views, ablations, statistics, sensitivity scenarios and visualization are evaluation-only and cannot update a model.</p><ul>{checks_html}</ul></section>
<section><h2>Headline EARS vs Ant result</h2><p>{escape(op_text)}</p>{_table(comparison_table)}</section>
<section><h2>Mean ± standard deviation across held-out city means</h2><p>These columns make between-city variability explicit instead of presenting only a pooled confidence interval.</p>{_table(method_rows)}</section>
<section><h2>All held-out city results</h2>{_table(city)}</section>
<section><h2>SWAP ranking</h2><p>Alternate hidden mission subsets are generated only after all model selection is frozen.</p>{_table(swap)}</section>
<section><h2>Paired statistical analysis</h2>{_table(paired)}</section>
<section><h2>SWAP paired statistical analysis</h2>{_table(paired_swap)}</section>
<section><h2>EARS mechanism ablations</h2><p>Full EARS is compared with stagnation-only, revisit-only, congestion-only, and no-energy/battery-aware relocation variants using the same frozen checkpoint.</p>{_table(ablations)}<h3>Trigger frequency by city</h3>{_table(triggers)}</section>
<section><h2>Operational-score sensitivity</h2><p>{escape(sens_text)}</p>{_table(sensitivity)}</section>
<section><h2>Winner episode: what the swarm actually does</h2><p>Mission markers in these figures are post-evaluation overlays only; the policy still acts on observable evidence.</p><div class='visuals'><div><h3>Animation</h3><img src='visualization/winner_episode.gif'></div><div><h3>Visit heat map</h3><img src='visualization/winner_visit_heatmap.png'></div><div><h3>Static trajectories + detections</h3><img src='visualization/winner_trajectory_map.png'></div></div></section>
<section><h2>Publication interpretation</h2><p>A strong claim requires more than rank 1: the paired and hierarchical intervals should support a positive EARS-vs-Ant operational delta, the raw discovery/redundancy/energy metrics should tell the same mechanism story, the result should persist on SWAP and multiple cities, and reasonable operational-score weight changes should not reverse it systematically.</p></section>
<section><h2>Reproduce</h2><p><code>docker compose up --build publication</code></p></section>
</main></body></html>"""
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
