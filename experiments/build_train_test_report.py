"""Build one publication-oriented HTML report joining training and held-out testing."""

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


def main() -> None:
    args = parse_args()
    train_root = Path(args.train_root)
    test_root = Path(args.test_root)
    train = pd.read_csv(train_root / "training_summary.csv")
    test = pd.read_csv(test_root / "tables" / "overall_ranking.csv")
    test_manifest = json.loads((test_root / "manifest.json").read_text(encoding="utf-8"))
    winner = test.iloc[0]
    grpo = test[test["strategy"] == "GRPO-Safe"]
    grpo_rank = int(grpo.iloc[0]["rank"]) if not grpo.empty else -1
    grpo_score = float(grpo.iloc[0]["operational_score"]) if not grpo.empty else float("nan")
    train_table = train.round(4).to_html(index=False, classes="data", border=0)
    test_table = test.round(4).to_html(index=False, classes="data", border=0)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SafeSwarm CityTwin Train/Test Report</title>
<style>
:root{{--ink:#172033;--muted:#667085;--line:#dfe5ef;--bg:#f4f7fb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,system-ui,sans-serif}}
header{{padding:38px 5vw;background:linear-gradient(120deg,#111827,#3448c5);color:white}}
main{{max-width:1500px;margin:-18px auto 40px;padding:0 24px}} section,.card{{background:white;border:1px solid var(--line);border-radius:14px;box-shadow:0 5px 18px #1018280b}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-bottom:18px}} .card{{padding:18px}} .card strong{{display:block;font-size:24px}}
section{{padding:22px;margin:18px 0;overflow:auto}} table.data{{border-collapse:collapse;width:100%;font-size:13px}} .data th,.data td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right}}
.data th:first-child,.data td:first-child{{text-align:left}} code{{background:#eef2ff;padding:2px 5px;border-radius:5px}}
</style></head><body>
<header><h1>SafeSwarm CityTwin · Train → Held-out Test</h1><p>Real-city policy learning with geographic generalization</p></header>
<main><div class="cards">
<div class="card">Held-out winner<strong>{escape(str(winner['strategy']))}</strong>score {float(winner['operational_score']):.3f}</div>
<div class="card">GRPO-Safe<strong>rank {grpo_rank}</strong>score {grpo_score:.3f}</div>
<div class="card">Train strategies<strong>{len(train)}</strong>PPO-family checkpoints</div>
<div class="card">Protocol<strong>{'PASS' if all(test_manifest['protocol_integrity'].values()) else 'FAIL'}</strong>city + start-zone separation</div>
</div>
<section><h2>Experimental contract</h2><p><strong>Training</strong> uses only configured train cities and train start zones. Checkpoints are written before testing. <strong>Testing</strong> uses held-out city geography and start zones that never appear during training. Test metrics do not select checkpoints.</p><p>The GRPO-Safe hypothesis is evaluated, not enforced: swarm memory and geographic propagation are enabled, but the same held-out operational score ranks every strategy.</p></section>
<section><h2>Training summary</h2>{train_table}</section><section><h2>Held-out test ranking</h2>{test_table}</section>
<section><h2>Reproduction</h2><p><code>docker compose up --build prepare-data</code></p><p><code>docker compose up --build train</code></p><p><code>docker compose up --build test</code></p><p><code>docker compose up --build pipeline</code> executes the complete chain and writes this combined report.</p></section>
</main></body></html>"""
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
