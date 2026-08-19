"""Build the publication-oriented SafeSwarm v6 train/test/SWAP report."""

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
    parser.add_argument("--swap-root", default="results/swap-test")
    parser.add_argument("--output", default="results/report.html")
    return parser.parse_args()


def _one(frame: pd.DataFrame, strategy: str) -> pd.Series | None:
    rows = frame[frame["strategy"] == strategy]
    return None if rows.empty else rows.iloc[0]


def _card(row: pd.Series | None, name: str) -> str:
    if row is None:
        return f'<div class="card">{escape(name)}<strong>n/a</strong></div>'
    return (
        f'<div class="card">{escape(name)}<strong>rank {int(row["rank"])}</strong>'
        f'score {float(row["operational_score"]):.3f}</div>'
    )


def main() -> None:
    args = parse_args()
    train_root = Path(args.train_root)
    test_root = Path(args.test_root)
    swap_root = Path(args.swap_root)
    train = pd.read_csv(train_root / "training_summary.csv")
    test = pd.read_csv(test_root / "tables" / "overall_ranking.csv")
    prism_summary = pd.read_csv(train_root / "prism_pattern_summary.csv")
    hybrid_summary = pd.read_csv(train_root / "prism_ant_summary.csv")
    upgrade_summary = pd.read_csv(train_root / "coordination_upgrade_summary.csv")
    ears_summary_path = train_root / "ears_summary.csv"
    ears_summary = pd.read_csv(ears_summary_path) if ears_summary_path.exists() else pd.DataFrame()
    test_manifest = json.loads((test_root / "manifest.json").read_text(encoding="utf-8"))
    prism_manifest = json.loads((train_root / "prism_manifest.json").read_text(encoding="utf-8"))
    ears_manifest_path = train_root / "ears_manifest.json"
    ears_manifest = json.loads(ears_manifest_path.read_text(encoding="utf-8")) if ears_manifest_path.exists() else {}
    swap_manifest_path = swap_root / "manifest.json"
    swap_manifest = json.loads(swap_manifest_path.read_text(encoding="utf-8")) if swap_manifest_path.exists() else {}
    swap = (
        pd.read_csv(swap_root / "tables" / "overall_ranking.csv")
        if (swap_root / "tables" / "overall_ranking.csv").exists() else pd.DataFrame()
    )

    winner = test.iloc[0]
    selected_pattern = str(prism_manifest.get("selected_pattern", "n/a"))
    protocol_pass = all(test_manifest["protocol_integrity"].values())
    train_table = train.round(4).to_html(index=False, classes="data", border=0)
    test_table = test.round(4).to_html(index=False, classes="data", border=0)
    prism_table = prism_summary.round(4).to_html(index=False, classes="data", border=0)
    hybrid_table = hybrid_summary.round(4).to_html(index=False, classes="data", border=0)
    upgrade_table = upgrade_summary.round(4).to_html(index=False, classes="data", border=0)
    ears_table = (
        ears_summary.round(4).to_html(index=False, classes="data", border=0)
        if not ears_summary.empty else "<p>EARS selection summary not available.</p>"
    )
    swap_table = (
        swap.round(4).to_html(index=False, classes="data", border=0)
        if not swap.empty else "<p>SWAP results not available.</p>"
    )
    swap_views = int(swap_manifest.get("unique_dataset_views", 0))
    ears_rule = escape(str(ears_manifest.get("selection_rule", "not recorded")))

    html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SafeSwarm CityTwin v6 Report</title><style>:root{{--ink:#172033;--line:#dfe5ef;--bg:#f4f7fb}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,system-ui,sans-serif}}header{{padding:38px 5vw;background:linear-gradient(120deg,#111827,#3448c5);color:white}}main{{max-width:1500px;margin:-18px auto 40px;padding:0 24px}}section,.card{{background:white;border:1px solid var(--line);border-radius:14px;box-shadow:0 5px 18px #1018280b}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-bottom:18px}}.card{{padding:18px}}.card strong{{display:block;font-size:24px}}section{{padding:22px;margin:18px 0;overflow:auto}}table.data{{border-collapse:collapse;width:100%;font-size:13px}}.data th,.data td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}.data th:first-child,.data td:first-child{{text-align:left}}code{{background:#eef2ff;padding:2px 5px;border-radius:5px}}</style></head><body>
<header><h1>SafeSwarm CityTwin v6 · Ant → Event Trigger → Reallocate → Held-out → SWAP</h1><p>EARS keeps Ant local search by default and invokes global intelligence only when observable progress degrades.</p></header><main>
<div class="cards"><div class="card">Held-out winner<strong>{escape(str(winner['strategy']))}</strong>score {float(winner['operational_score']):.3f}</div>{_card(_one(test,'AntSwarmSafe'),'AntSwarmSafe')}{_card(_one(test,'EARS-Safe'),'EARS-Safe')}{_card(_one(test,'EARS-NP-Safe'),'EARS-NP-Safe')}{_card(_one(test,'H-MAPPO-EARS-Safe'),'H-MAPPO-EARS-Safe')}{_card(_one(test,'PRISM-Ant-Safe'),'PRISM-Ant-Safe')}<div class="card">SWAP views<strong>{swap_views}</strong>evaluation only</div><div class="card">Protocol<strong>{'PASS' if protocol_pass else 'FAIL'}</strong>no test selection</div></div>
<section><h2>What v6 tests</h2><p>The v5 real result showed that PRISM-Ant already had enough discovery but paid too much in redundant coverage, distance and energy. v6 therefore leaves <strong>AntSwarmSafe unchanged</strong> and asks a narrower question: can Ant be improved by invoking global planning only after observable stagnation, revisit pressure or clustering?</p><p><strong>EARS-Safe</strong> uses event-triggered global relocation with explicit distance/energy penalties. <strong>EARS-NP-Safe</strong> adds a repulsive negative-pheromone exclusion halo. <strong>H-MAPPO-EARS-Safe</strong> adds the trained MAPPO checkpoint as one explicit high-level option rather than forcing MAPPO to rediscover Ant's low-level heuristic.</p><p><strong>Selection:</strong> {ears_rule}</p></section>
<section><h2>EARS validation selection</h2>{ears_table}</section>
<section><h2>PRISM pattern selection</h2><p>PRISM remains as the v5 global-search comparison; selected pattern: <strong>{escape(selected_pattern)}</strong>.</p>{prism_table}</section>
<section><h2>PRISM-Ant validation fusion</h2>{hybrid_table}</section>
<section><h2>PPO/GRPO coordination upgrade</h2>{upgrade_table}</section>
<section><h2>Base training summary</h2>{train_table}</section>
<section><h2>Primary held-out ranking</h2>{test_table}</section>
<section><h2>SWAP anti-overfitting stress test</h2><p>SWAP changes hidden mission subsets by dataset seed while keeping physical OSM city geometry fixed. EARS parameters are already frozen before these runs.</p>{swap_table}</section>
<section><h2>Scientific boundary</h2><p>The repository does not tune EARS to the primary test or SWAP ranking and does not require it to beat AntSwarm. A credible v6 result requires the frozen EARS variants to improve Ant consistently across held-out cities and multiple SWAP seeds, especially in redundancy, energy and distance.</p></section>
<section><h2>Reproduce</h2><p><code>docker compose up --build pipeline</code></p><p>Or run <code>prepare-data</code>, <code>train</code>, <code>upgrade-ppo</code>, <code>train-prism</code>, <code>train-ears</code>, <code>test</code>, and <code>test-swap</code> separately.</p></section>
</main></body></html>"""
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
