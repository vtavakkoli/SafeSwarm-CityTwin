"""Build a publication-oriented SafeSwarm v3 train/validation/test report."""

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
    train_manifest = json.loads((train_root / "manifest.json").read_text(encoding="utf-8"))
    test_manifest = json.loads((test_root / "manifest.json").read_text(encoding="utf-8"))
    winner = test.iloc[0]
    grpo = test[test["strategy"] == "GRPO-Safe"]
    grpo_rank = int(grpo.iloc[0]["rank"]) if not grpo.empty else -1
    grpo_score = float(grpo.iloc[0]["operational_score"]) if not grpo.empty else float("nan")
    grpo_train = train[train["strategy"] == "GRPO-Safe"]
    best_validation = float(grpo_train.iloc[0]["best_validation_score"]) if not grpo_train.empty else float("nan")
    best_epoch = int(grpo_train.iloc[0]["best_epoch"]) if not grpo_train.empty else -1
    train_table = train.round(4).to_html(index=False, classes="data", border=0)
    test_table = test.round(4).to_html(index=False, classes="data", border=0)
    protocol_pass = all(test_manifest["protocol_integrity"].values())
    observation_contract = escape(str(test_manifest.get("observation_contract", "not recorded")))
    credit = escape(str(train_manifest.get("credit_assignment", "not recorded")))
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SafeSwarm CityTwin v3 Report</title>
<style>:root{{--ink:#172033;--muted:#667085;--line:#dfe5ef;--bg:#f4f7fb}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,system-ui,sans-serif}}header{{padding:38px 5vw;background:linear-gradient(120deg,#111827,#3448c5);color:white}}main{{max-width:1500px;margin:-18px auto 40px;padding:0 24px}}section,.card{{background:white;border:1px solid var(--line);border-radius:14px;box-shadow:0 5px 18px #1018280b}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-bottom:18px}}.card{{padding:18px}}.card strong{{display:block;font-size:24px}}section{{padding:22px;margin:18px 0;overflow:auto}}table.data{{border-collapse:collapse;width:100%;font-size:13px}}.data th,.data td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right}}.data th:first-child,.data td:first-child{{text-align:left}}code{{background:#eef2ff;padding:2px 5px;border-radius:5px}}</style></head><body>
<header><h1>SafeSwarm CityTwin v3 · Train → Validate → Held-out Test</h1><p>Observable-only real-city policy learning with geographic generalization</p></header>
<main><div class="cards"><div class="card">Held-out winner<strong>{escape(str(winner['strategy']))}</strong>score {float(winner['operational_score']):.3f}</div><div class="card">GRPO-Safe<strong>rank {grpo_rank}</strong>test {grpo_score:.3f}</div><div class="card">GRPO validation<strong>{best_validation:.3f}</strong>best epoch {best_epoch}</div><div class="card">Protocol<strong>{'PASS' if protocol_pass else 'FAIL'}</strong>city + start-zone separation</div></div>
<section><h2>Experimental contract</h2><p><strong>Observation fairness:</strong> {observation_contract}.</p><p><strong>Training credit:</strong> {credit}.</p><p>Training uses Vienna/London/Berlin, checkpoint selection uses Amsterdam only, and frozen checkpoints are tested on San Francisco/Paris with unseen starting zones. Test metrics never select a model.</p></section>
<section><h2>Why v3 must be rerun</h2><p>The v3 observation contract, energy/return semantics, per-agent credit assignment, minibatch PPO optimizer, GRPO state-conditioned behavior policy, teacher bootstrap, and third GRPO ablation materially differ from v2. Numerical v2 rankings are therefore not valid v3 results.</p></section>
<section><h2>Training and validation summary</h2>{train_table}</section><section><h2>Held-out ranking</h2>{test_table}</section>
<section><h2>Interpretation</h2><p>GRPO-Safe is a hypothesis, not a forced winner. Compare the full checkpoint with <code>NoMemory</code>, <code>NoPropagation</code>, and <code>NoLearnedBehavior</code>. A mechanism claim is strongest when the full model improves consistently over the corresponding ablations and confidence intervals are reported.</p></section>
<section><h2>Reproduction</h2><p><code>docker compose up --build prepare-data</code></p><p><code>docker compose up --build train</code></p><p><code>docker compose up --build test</code></p><p><code>docker compose up --build pipeline</code></p></section>
</main></body></html>"""
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
