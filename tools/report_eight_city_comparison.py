"""Complete-ledger-only analysis for the eight-city, all-method comparison."""

from __future__ import annotations

# ruff: noqa: E402 -- matplotlib backend and repository path precede imports.
import fcntl
import html
import json
import sqlite3
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.report_publication_v9 import city_test, holm
from tools.eight_city_comparison import KEYS, METHODS, atomic_json, digest, read_ledger, verify
from tools.mechanism_probe_v9 import FULL

METRICS = [
    "operational_score",
    "weighted_target_discovery",
    "coverage_ratio",
    "energy_consumption",
    "runtime_seconds",
    "policy_seconds",
    "runtime_overhead",
    "completed_all_targets",
    "capped_completion_steps",
    "actual_safety_incidents",
    "battery_failures",
    "blocked_moves",
]


def latex_table(frame, smoke=False):
    """Standalone table fragment without an optional pandas/Jinja2 dependency."""
    escapes = {"\\": r"\textbackslash{}", "_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#"}

    def cell(value):
        value = f"{value:.4f}" if isinstance(value, float) else str(value)
        return "".join(escapes.get(char, char) for char in value)

    caption = (
        "SYNTHETIC SOFTWARE CHECK; NOT PAPER RESULTS."
        if smoke
        else "All-method comparison at the original setting; equal-city means."
    )
    lines = [
        r"% Requires booktabs and graphicx. Synthetic smoke outputs are not paper evidence.",
        r"\begin{table*}[t]\centering",
        r"\caption{" + caption + "}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrrrr}\toprule",
        r"Method & $J$ & Discovery & Coverage & Energy & Runtime (s) & Completion & Incidents \\",
        r"\midrule",
    ]
    lines += [" & ".join(cell(v) for v in row) + r" \\" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines + [r"\bottomrule\end{tabular}}\end{table*}", ""])


def analyze(frame, plan):
    if frame.duplicated(KEYS).any() or set(frame.strategy) != set(METHODS):
        raise ValueError("Duplicate rows or missing methods")
    if not np.isfinite(frame[METRICS]).all().all():
        raise ValueError("Nonfinite comparison metrics")
    pair = [k for k in KEYS if k != "strategy"]
    grouped = frame.groupby(pair)
    if (grouped.strategy.nunique() != len(METHODS)).any():
        raise ValueError("Incomplete matched-method blocks")
    if (grouped.communication_draws_sha256.nunique() != 1).any():
        raise ValueError("Different communication schedule within paired scenario")
    views = pd.DataFrame(plan["swap_views"])
    eligible = set(views.loc[views.city_eligible_changed_swap, "city"])
    # Separate panels: do not merge stopping rules, histories, or target variants.
    subsets = {panel: g.copy() for panel, g in frame.groupby("panel")}
    original = frame[(frame.panel == "mission_stop") & (frame.agents == 8) & (frame.dropout == 0.03)]
    if plan["smoke_only"]:
        original = frame[frame.panel == "mission_stop"]
    subsets["original_setting"] = original
    subsets["swap_changed"] = frame[(frame.panel == "swap") & frame.city.isin(eligible)]
    summaries, city_tables, tests, strata = [], [], [], []
    for panel, group in subsets.items():
        if group.empty:
            continue
        city = group.groupby(["city", "strategy"])[METRICS].mean().reset_index()
        city.insert(0, "analysis_panel", panel)
        city_tables.append(city)
        summary = city.groupby("strategy")[METRICS].mean().reset_index()
        summary.insert(0, "analysis_panel", panel)
        summary["cities"] = group.city.nunique()
        summary["episodes_per_method"] = group.groupby("strategy").size().reindex(summary.strategy).to_numpy()
        summary["rank_J"] = summary.operational_score.rank(method="min", ascending=False).astype(int)
        summaries.append(summary)
        for metric in METRICS:
            wide = city.pivot(index="city", columns="strategy", values=metric)
            for control in METHODS:
                if control == FULL:
                    continue
                result = city_test(wide[FULL] - wide[control])
                primary = (
                    panel == "original_setting" and metric == "operational_score" and control == "EARS-Safe"
                )
                tests.append(
                    dict(analysis_panel=panel, metric=metric, comparator=control, primary=primary, **result)
                )
        setting = group.groupby(["city", "agents", "dropout", "strategy"])[METRICS].mean()
        wide = setting["operational_score"].unstack("strategy")
        for control in METHODS:
            if control != FULL:
                d = (wide[FULL] - wide[control]).rename("delta_J").reset_index()
                d["analysis_panel"], d["comparator"] = panel, control
                strata.append(d)
    tests = pd.DataFrame(tests)
    tests["p_holm_secondary"] = np.nan
    secondary = ~tests.primary
    tests.loc[secondary, "p_holm_secondary"] = holm(tests.loc[secondary, "p_two_sided"])
    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(city_tables, ignore_index=True),
        tests,
        pd.concat(strata, ignore_index=True),
    )


def _report(root):
    plan = verify(root, execution=False)
    if not (root / "episodes.sqlite").is_file():
        raise ValueError("No experiment journal; preflight is not completed research")
    with sqlite3.connect(f"file:{root / 'episodes.sqlite'}?mode=ro", uri=True) as db:
        rows, seen = read_ledger(db, plan)
    if len(seen) != plan["expected_rows"]:
        raise ValueError(
            f"Incomplete study: {len(seen)}/{plan['expected_rows']}; no publication ranking generated"
        )
    frame = pd.DataFrame([{k: v for k, v in r.items() if k != "diagnostics"} for r in rows])
    summary, city, tests, strata = analyze(frame, plan)
    out = root / "report"
    out.mkdir(exist_ok=True)
    frame.to_csv(root / "episodes.csv", index=False)
    with (root / "diagnostics.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps({k: row[k] for k in [*KEYS, "diagnostics"]}, allow_nan=False) + "\n")
    for name, data in [
        ("summary", summary),
        ("city_means", city),
        ("contrasts", tests),
        ("setting_deltas", strata),
    ]:
        data.to_csv(out / f"{name}.csv", index=False)
    views = pd.DataFrame(plan["swap_views"])
    views.to_csv(out / "swap_validity.csv", index=False)
    pd.DataFrame(plan["method_manifest"]).to_csv(out / "method_manifest.csv", index=False)
    primary = tests[tests.primary].iloc[0]
    selected = summary[summary.analysis_panel == "original_setting"].sort_values("rank_J")
    display = selected[
        [
            "strategy",
            "operational_score",
            "weighted_target_discovery",
            "coverage_ratio",
            "energy_consumption",
            "runtime_seconds",
            "completed_all_targets",
            "actual_safety_incidents",
        ]
    ]
    (out / "all_methods_original_setting.tex").write_text(latex_table(display, plan["smoke_only"]))
    # Always show every method; unfiltered controls remain visibly identified.
    colors = [
        "#007D77" if m == FULL else "#C69437" if m in ["RandomAgent", "GreedyAgent"] else "#758DA7"
        for m in selected.strategy
    ]
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(9, 9), layout="constrained")
    if plan["smoke_only"]:
        fig.suptitle("SYNTHETIC SOFTWARE CHECK — NOT PAPER RESULTS", color="#BA544D", fontsize=11)
    ax.barh(selected.strategy, selected.operational_score, color=colors)
    ax.invert_yaxis()
    ax.set(
        xlabel="Mean operational score (equal city weight)", title="Original setting: all 30 implementations"
    )
    fig.savefig(out / "all_methods.pdf")
    fig.savefig(out / "all_methods.png", dpi=150)
    plt.close(fig)
    c = city[city.analysis_panel == "original_setting"].pivot(
        index="city", columns="strategy", values="operational_score"
    )
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    if plan["smoke_only"]:
        fig.suptitle("SYNTHETIC SOFTWARE CHECK — NOT PAPER RESULTS", color="#BA544D", fontsize=11)
    delta = c[FULL] - c["EARS-Safe"]
    ax.barh(delta.index, delta, color=["#007D77" if v >= 0 else "#BA544D" for v in delta])
    ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel="EARS-Info-v9 minus EARS-Safe score", title="Primary city-level contrast")
    fig.savefig(out / "city_contrast.pdf")
    fig.savefig(out / "city_contrast.png", dpi=150)
    plt.close(fig)
    label = (
        "SYNTHETIC SOFTWARE CHECK — NOT PAPER RESULTS"
        if plan["smoke_only"]
        else "Exploratory eight-city comparison"
    )
    flags = [
        "All 26 historical implementations and four information-routing variants rerun together.",
        "Original-setting primary test: 8 robots, dropout 0.03, mission stopping; "
        "all other comparisons secondary.",
        "Fixed-budget still permits all-agent termination; "
        "energy is not equal-duration energy in every episode.",
        "Dropout draws are paired by step and robot, but shared maps remain available during disconnection.",
        "Native safety/return implementations differ; "
        "RandomAgent and GreedyAgent are unfiltered reference controls.",
        "QMIX, MADDPG and MAT names denote lightweight policy-style adaptations, "
        "not full published implementations.",
        "The eight city identities are reused; "
        "new checkpoints or snapshots do not create untouched confirmation.",
        "Unchanged or repeated SWAP views are retained in the all-city descriptive panel "
        "and excluded from swap_changed inference.",
        "No retuning, method omission, target projection, "
        "synthetic real-map fallback, or automatic promotion occurs.",
    ]
    header = (
        f"<h1>{html.escape(label)}</h1>"
        f"<p>{len(rows):,} episodes; {len(plan['cities'])} cities; 30 methods.</p>"
    )
    body = "<ul>" + "".join(f"<li>{html.escape(f)}</li>" for f in flags) + "</ul>"
    body += (
        f"<h2>Primary comparison</h2><p>Mean difference {primary.delta:+.6f}; "
        f"exact city p={primary.p_two_sided:.6f}. Descriptive bootstrap interval "
        f"[{primary.city_bootstrap_low:+.6f}, {primary.city_bootstrap_high:+.6f}].</p>"
    )
    body += (
        '<img src="all_methods.png" alt="All 30 method scores">'
        '<img src="city_contrast.png" alt="Eight city differences">'
    )
    body += "<h2>Original-setting comparison</h2>" + display.round(5).to_html(index=False)
    body += "<h2>All analysis panels</h2>" + summary.round(5).to_html(index=False)
    body += "<h2>Implementation scope</h2>" + pd.DataFrame(plan["method_manifest"]).to_html(index=False)
    body += "<h2>Map diagnostics</h2>" + pd.DataFrame(plan["map_audit"]).to_html(index=False)
    body += "<h2>SWAP validity</h2>" + views.to_html(index=False)
    (out / "index.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>Eight-city comparison</title>'
        "<style>body{font:15px/1.5 system-ui;margin:32px;color:#172033}"
        "table{border-collapse:collapse;font-size:12px}td,th{padding:5px;border:1px solid #ccd3da}"
        "img{max-width:100%;max-height:1000px}h2{margin-top:32px}</style></head><body>"
        + header
        + body
        + "</body></html>"
    )
    atomic_json(
        out / "assessment.json",
        dict(
            status="complete",
            smoke_only=plan["smoke_only"],
            independent_confirmation=False,
            primary_delta=float(primary.delta),
            primary_p=float(primary.p_two_sided),
            primary_city_count=int(primary.cities),
            primary_nominal_p_below_05=bool(primary.p_two_sided < 0.05),
            publication_acceptance_assessed=False,
            limitations=flags,
            changed_swap_cities=sorted(set(views.loc[views.city_eligible_changed_swap, "city"])),
        ),
    )
    atomic_json(
        root / "completion.json",
        dict(
            status="completed",
            rows=len(rows),
            expected_rows=plan["expected_rows"],
            smoke_only=plan["smoke_only"],
            plan_sha256=digest(root / "frozen/plan.json"),
            episodes_sha256=digest(root / "episodes.csv"),
            journal_sha256=digest(root / "episodes.sqlite"),
            report_hashes={p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()},
        ),
    )
    atomic_json(
        root / "progress.json", dict(status="completed", rows=len(rows), expected_rows=plan["expected_rows"])
    )
    archive = root.parent / (root.name + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.name != "failure.json":
                z.write(p, root.name + "/" + str(p.relative_to(root)))
    print(f"Complete: {len(rows)} episodes. Report: {out / 'index.html'}. Archive: {archive}", flush=True)


def report(root, locked=False):
    root = Path(root).resolve()
    if locked:
        return _report(root)
    with (root.parent / (root.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _report(root)
