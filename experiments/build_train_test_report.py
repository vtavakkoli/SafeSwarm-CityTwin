"""Build a publication-oriented SafeSwarm v4 train/validation/test report."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", default="results/train")
    parser.add_argument("--test-root", default="results/test")
    parser.add_argument("--output", default="results/report.html")
    return parser.parse_args()


def _one(frame: pd.DataFrame, strategy: str) -> pd.Series | None:
    rows = frame[frame["strategy"] == strategy]
    return None if rows.empty else rows.iloc[0]


def main() -> None:
    args = parse_args()
    train_root = Path(args.train_root)
    test_root = Path(args.test_root)
    train = pd.read_csv(train_root / "training_summary.csv")
    test = pd.read_csv(test_root / "tables" / "overall_ranking.csv")
    train_manifest = json.loads((train_root / "manifest.json").read_text(encoding="utf-8"))
    test_manifest = json.loads((test_root / "manifest.json").read_text(encoding="utf-8"))
    sparx_manifest_path = train_root / "sparx_manifest.json"
    sparx_manifest = json.loads(sparx_manifest_path.read_text(encoding="utf-8")) if sparx_manifest_path.exists() else {}
    sparx_summary_path = train_root / "sparx_pattern_summary.csv"
    sparx_summary = pd.read_csv(sparx_summary_path) if sparx_summary_path.exists() else pd.DataFrame()

    winner = test.iloc[0]
    grpo = _one(test, "GRPO-Safe")
    sparx = _one(test, "SPARX-Safe")
    ippo = _one(test, "IPPO-Safe")
    mappo = _one(test, "MAPPO-Safe")
    selected_pattern = str(sparx_manifest.get("selected_pattern", "n/a"))

    train_table = train.round(4).to_html(index=False, classes="data", border=0)
    test_table = test.round(4).to_html(index=False, classes="data", border=0)
    sparx_table = (
        sparx_summary.round(4).to_html(index=False, classes="data", border=0)
        if not sparx_summary.empty else "<p>No SPARX tuning summary found.</p>"
    )
    protocol_pass = all(test_manifest["protocol_integrity"].values())
    observation_contract = escape(str(test_manifest.get("observation_contract", "not recorded")))
    credit = escape(str(train_manifest.get("credit_assignment", "not recorded")))
    selection = escape(str(train_manifest.get("checkpoint_selection", "not recorded")))

    def card(row: pd.Series | None, name: str) -> str:
        if row is None:
            return f'<div class="card">{escape(name)}<strong>n/a</strong></div>'
        return (
            f'<div class="card">{escape(name)}<strong>rank {int(row["rank"])}</strong>'
            f'score {float(row["operational_score"]):.3f}</div>'
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SafeSwarm CityTwin v4 Report</title>
<style>:root{{--ink:#172033;--muted:#667085;--line:#dfe5ef;--bg:#f4f7fb}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,system-ui,sans-serif}}header{{padding:38px 5vw;background:linear-gradient(120deg,#111827,#3448c5);color:white}}main{{max-width:1500px;margin:-18px auto 40px;padding:0 24px}}section,.card{{background:white;border:1px solid var(--line);border-radius:14px;box-shadow:0 5px 18px #1018280b}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-bottom:18px}}.card{{padding:18px}}.card strong{{display:block;font-size:24px}}section{{padding:22px;margin:18px 0;overflow:auto}}table.data{{border-collapse:collapse;width:100%;font-size:13px}}.data th,.data td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}.data th:first-child,.data td:first-child{{text-align:left}}code{{background:#eef2ff;padding:2px 5px;border-radius:5px}}</style></head><body>
<header><h1>SafeSwarm CityTwin v4 · Train → Robust Validate → Pattern Select → Held-out Test</h1><p>SPARX shared probability memory + region allocation + X / + / ★ search</p></header>
<main><div class="cards">
<div class="card">Held-out winner<strong>{escape(str(winner['strategy']))}</strong>score {float(winner['operational_score']):.3f}</div>
<div class="card">SPARX pattern<strong>{escape(selected_pattern)}</strong>selected before test</div>
{card(sparx, 'SPARX-Safe')}{card(ippo, 'IPPO-Safe')}{card(mappo, 'MAPPO-Safe')}{card(grpo, 'GRPO-Safe')}
<div class="card">Protocol<strong>{'PASS' if protocol_pass else 'FAIL'}</strong>city + validation/test geography</div></div>
<section><h2>What v4 fixes</h2><p>v3 showed a large validation-to-test collapse for PPO-family checkpoints. v4 broadens validation to multiple unseen cities/start zones and promotes weights only when a conservative robust validation score improves. It also adds SPARX to explicitly allocate agents across probability-mass regions rather than relying only on local action preferences.</p><p><strong>Observation fairness:</strong> {observation_contract}.</p><p><strong>Training credit:</strong> {credit}.</p><p><strong>Checkpoint selection:</strong> {selection}.</p></section>
<section><h2>SPARX pattern selection</h2><p><strong>SPARX</strong> means <em>Swarm Probability-map Allocation &amp; Region eXploration</em>. X, Plus and Star are tuned independently. The selected SPARX-Safe pattern comes from validation only; the held-out pattern rows are retained only for mechanism analysis.</p>{sparx_table}</section>
<section><h2>PPO/GRPO training and robust validation</h2>{train_table}</section>
<section><h2>Held-out ranking</h2>{test_table}</section>
<section><h2>Scientific interpretation</h2><p>Neither SPARX nor GRPO is forced to win. The report always surfaces the actual held-out winner. Compare discovery, coverage, redundancy, energy and confidence intervals—not only the aggregate operational score.</p></section>
<section><h2>Reproduction</h2><p><code>docker compose up --build prepare-data</code></p><p><code>docker compose up --build train</code></p><p><code>docker compose up --build train-sparx</code></p><p><code>docker compose up --build test</code></p><p><code>docker compose up --build pipeline</code></p></section>
</main></body></html>"""
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
