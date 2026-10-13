"""Audited tables, statistical comparisons and publication report from completed runs."""
# Long manuscript paragraphs and equations are intentionally kept readable.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import html
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import reportlab
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.publication_ears_v9 import (  # noqa: E402
    ABLATION,
    BASELINE,
    CANDIDATE,
    EVIDENCE,
    KEYS,
    digest,
    export,
    jobs,
    key,
    ledger,
    read_json,
    read_rows,
    verify,
)
from src.evaluation.metrics import episode_operational_score  # noqa: E402

METRICS = [
    "operational_score",
    "weighted_target_discovery",
    "coverage_ratio",
    "energy_consumption",
    "distance_travelled",
    "runtime_seconds",
    "completed_all_targets",
    "capped_completion_steps",
    "actual_safety_incidents",
    "battery_failures",
]
LABELS = {
    "operational_score": "J",
    "weighted_target_discovery": "Discovery",
    "coverage_ratio": "Coverage",
    "energy_consumption": "Energy",
    "runtime_seconds": "Time (s)",
}


def report_styles():
    fonts = Path(reportlab.__file__).parent / "fonts"
    for name, filename in (("Vera", "Vera.ttf"), ("VeraBold", "VeraBd.ttf")):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "VeraBold" if style.name.startswith(("Heading", "Title")) else "Vera"
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 13
    styles["BodyText"].spaceAfter = 8
    styles["Heading1"].fontSize = 15
    styles["Heading1"].leading = 19
    styles["Heading1"].textColor = colors.HexColor("#193c5b")
    styles["Title"].fontSize = 19
    styles["Title"].leading = 24
    return styles


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p))))
    result = np.empty(len(p))
    result[order] = np.minimum(1, adjusted)
    return result


def city_test(values):
    x = np.asarray(values, dtype=float)
    observed = float(x.mean())
    null = np.asarray([np.mean(x * signs) for signs in itertools.product((-1, 1), repeat=len(x))])
    p = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    rng = np.random.default_rng(91827)
    bootstrap = rng.choice(x, size=(20000, len(x)), replace=True).mean(axis=1)
    lo, hi = np.quantile(bootstrap, [0.025, 0.975])
    return dict(
        delta=observed,
        p_two_sided=p,
        city_bootstrap_low=float(lo),
        city_bootstrap_high=float(hi),
        cities=len(x),
        cities_positive=int((x > 0).sum()),
    )


def analyze(frame):
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate episode keys")
    if not np.isfinite(frame[METRICS].to_numpy()).all():
        raise ValueError("Nonfinite results")
    if not np.allclose(
        frame.apply(episode_operational_score, axis=1), frame.operational_score, atol=1e-12, rtol=0
    ):
        raise ValueError("Score audit failed")
    pairkeys = [k for k in KEYS if k != "strategy"]
    statistics = []
    for panel, group in frame.groupby("panel"):
        for metric in METRICS:
            pivot = group.pivot(index=pairkeys, columns="strategy", values=metric)
            if pivot.isna().any().any():
                raise ValueError("Incomplete matched comparison")
            for comparator in (BASELINE, "AntSwarmSafe", ABLATION):
                deltas = (pivot[CANDIDATE] - pivot[comparator]).groupby(level="city").mean()
                primary = panel == "mission_stop" and metric == "operational_score" and comparator == BASELINE
                statistics.append(
                    dict(
                        panel=panel,
                        metric=metric,
                        candidate=CANDIDATE,
                        comparator=comparator,
                        primary=primary,
                        paired_episodes=len(pivot),
                        **city_test(deltas),
                    )
                )
    stats = pd.DataFrame(statistics)
    secondary = ~stats.primary
    stats["p_holm"] = np.nan
    stats.loc[secondary, "p_holm"] = holm(stats.loc[secondary, "p_two_sided"])
    city = frame.groupby(["panel", "city", "agents", "dropout", "strategy"], as_index=False)[METRICS].mean()
    summary = city.groupby(["panel", "strategy"], as_index=False)[METRICS].mean()
    strata = city[city.panel == "mission_stop"].pivot(
        index=["city", "agents", "dropout"], columns="strategy", values="operational_score"
    )
    strata["Info_minus_v7"] = strata[CANDIDATE] - strata[BASELINE]
    return city, summary, stats, strata.reset_index()


def sensitivity(frame):
    rows = []
    f = frame[frame.panel == "mission_stop"].copy()
    components = np.column_stack(
        [
            f.weighted_target_discovery,
            f.coverage_ratio,
            1 - np.minimum(1, f.actual_safety_incidents / (f.steps * f.agents).clip(lower=1)),
            1 - np.minimum(1, f.energy_consumption / (100 * f.agents)),
            1 - np.minimum(1, f.redundant_coverage),
            f.communication_efficiency,
        ]
    )
    weights = {
        "original": [0.35, 0.20, 0.20, 0.10, 0.10, 0.05],
        "discovery_emphasis": [0.45, 0.10, 0.20, 0.10, 0.10, 0.05],
        "coverage_emphasis": [0.25, 0.30, 0.20, 0.10, 0.10, 0.05],
    }
    for name, w in weights.items():
        f["weighted_score"] = components @ np.asarray(w)
        means = f.groupby(["city", "strategy"])["weighted_score"].mean().unstack()
        rows.append(
            dict(
                weighting=name,
                weights=json.dumps(w),
                Info_minus_v7=float((means[CANDIDATE] - means[BASELINE]).mean()),
            )
        )
    return pd.DataFrame(rows)


def md_table(frame):
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def latex_table(frame):
    """Portable booktabs table without pandas' optional Jinja2 dependency."""
    escapes = {"\\": r"\textbackslash{}", "_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#"}

    def cell(value):
        text = f"{value:.5f}" if isinstance(value, float) else str(value)
        return "".join(escapes.get(char, char) for char in text)

    lines = [
        r"% Requires \usepackage{booktabs}",
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{Equal-city mean outcomes of the frozen EARS study.}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        " & ".join(cell(c) for c in frame.columns) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(cell(v) for v in row) + r" \\" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])


def pdf_table(frame):
    styles = report_styles()
    data = [[Paragraph(html.escape(str(c)), styles["BodyText"]) for c in frame.columns]]
    for row in frame.itertuples(index=False, name=None):
        data.append(
            [
                Paragraph(
                    html.escape(f"{v:.4f}" if isinstance(v, float) else str(v)),
                    styles["BodyText"],
                )
                for v in row
            ]
        )
    widths = [1.65 * inch] + [(6.8 * inch - 1.65 * inch) / (len(frame.columns) - 1)] * (
        len(frame.columns) - 1
    )
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#304b68")),
                ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#dce2e8")),
            ]
        )
    )
    return table


def write_report(out, title, sections):
    out.mkdir(parents=True, exist_ok=True)
    styles = report_styles()
    story = [Paragraph(html.escape(title), styles["Title"]), Spacer(1, 12)]
    markdown, web = ["# " + title], ["<h1>" + html.escape(title) + "</h1>"]
    for heading, items in sections:
        section_story = [Paragraph(html.escape(heading), styles["Heading1"])]
        markdown.append("\n## " + heading)
        web.append("<h2>" + html.escape(heading) + "</h2>")
        for item in items:
            if isinstance(item, pd.DataFrame):
                table = pdf_table(item)
                section_story += [table, Spacer(1, 12)]
                markdown.append(md_table(item))
                web.append(item.to_html(index=False, float_format=lambda x: f"{x:.6f}"))
            elif isinstance(item, Path):
                section_story += [Image(str(item), width=6.8 * inch, height=3.5 * inch), Spacer(1, 10)]
                markdown.append(f"![Results]({item.name})")
                web.append(f'<img alt="Results" src="{item.name}" style="width:100%">')
            else:
                section_story.append(Paragraph(html.escape(item), styles["BodyText"]))
                markdown.append(item)
                web.append("<p>" + html.escape(item) + "</p>")

        story += (
            [KeepTogether(section_story)]
            if heading == "Run identity"
            else [KeepTogether(section_story[:2]), *section_story[2:]]
        )

    def footer(canvas, doc):
        canvas.setFont("Vera", 7)
        canvas.setFillColor(colors.HexColor("#526678"))
        canvas.drawString(42, 24, "SafeSwarm-CityTwin | Reproducible research report")
        canvas.drawRightString(570, 24, str(doc.page))

    SimpleDocTemplate(
        str(out / "report.pdf"),
        pagesize=(612, 792),
        rightMargin=42,
        leftMargin=42,
        topMargin=38,
        bottomMargin=42,
        title=title,
        author="SafeSwarm-CityTwin",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    (out / "report.md").write_text("\n\n".join(markdown) + "\n")
    (out / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>"
        + html.escape(title)
        + "</title><style>body{max-width:1050px;margin:40px auto;font:16px/1.5 system-ui;color:#18364b}"
        "table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ccd6df}"
        "th{background:#e8eef4}h1,h2{line-height:1.2}</style>" + "\n".join(web)
    )


def historical_sections():
    selection = read_json(EVIDENCE / "selected_model.json")
    if digest(EVIDENCE / "episodes.csv") != selection["episodes_sha256"]:
        raise ValueError("Historical evidence hash mismatch")
    f = pd.read_csv(EVIDENCE / "episodes.csv")
    expected_methods = {
        BASELINE,
        "AntSwarmSafe",
        "EARS-Info-v9",
        "EARS-Info-Local-v9",
        "EARS-Balanced-v9",
        "EARS-Balanced-Local-v9",
    }
    if len(f) != 768 or set(f.strategy) != expected_methods:
        raise ValueError("Unexpected historical evidence dimensions")
    if not np.allclose(f.apply(episode_operational_score, axis=1), f.operational_score, atol=1e-12, rtol=0):
        raise ValueError("Historical score audit failed")
    means = (
        f.groupby("strategy")[
            [
                "operational_score",
                "weighted_target_discovery",
                "coverage_ratio",
                "energy_consumption",
                "runtime_seconds",
            ]
        ]
        .mean()
        .reset_index()
    )
    info = means.set_index("strategy").loc[CANDIDATE]
    base = means.set_index("strategy").loc[BASELINE]
    decision = next(d for d in selection["decisions"] if d["method"] == CANDIDATE)
    return [
        (
            "Evidence status",
            [
                "This report audits completed validation evidence and specifies the final evaluation workflow. "
                "Independent six-city confirmation has not been supplied or executed for this report. "
                "It is not evidence that the candidate is ready for an unconditional superiority claim.",
                "The original selection retained EARS-Safe-v7. EARS-Info-v9 is frozen separately as a research "
                "candidate because of its average validation gains. This post-validation choice does not reverse "
                "the failed promotion decision. All four v9 variants failed at least one declared gate.",
            ],
        ),
        (
            "Completed validation results",
            [
                "The supplied run contains 768 matched episodes: Amsterdam and Prague, four start zones, "
                "4/8/16/32 agents, dropout probabilities 0/0.03/0.15/0.30, and six methods. "
                "Each method has 128 episodes. These cities were used for selection, so these are descriptive results.",
                means.rename(columns={"strategy": "Method", **LABELS}),
                f"Relative to v7, Info-v9 increased J by {info.operational_score - base.operational_score:.6f} "
                f"and weighted discovery by {100 * (info.weighted_target_discovery - base.weighted_target_discovery):.3f} percentage points. "
                f"Energy decreased by {100 * (1 - info.energy_consumption / base.energy_consumption):.2f}% and runtime by "
                f"{100 * (1 - info.runtime_seconds / base.runtime_seconds):.2f}% on the reported machine. Coverage decreased "
                f"by {100 * (base.coverage_ratio - info.coverage_ratio):.3f} percentage points.",
                f"The worst city/setting J change was {decision['worst_stratum_delta']:.6f}, below the -0.005 "
                "regression limit. Zero physical incidents and battery failures were recorded in this sample; "
                "this does not establish a universal safety guarantee.",
            ],
        ),
        (
            "Interpretation and paper claim",
            [
                "A defensible current statement is: In two validation cities, information-aware routing improved "
                "mean discovery, operational score, energy consumption and measured runtime, while reducing "
                "coverage at termination and failing the prespecified robustness gate. Independent confirmation remains pending.",
                "At larger team sizes, both methods can finish all targets. Earlier termination gives the faster "
                "method less time to accumulate area coverage. The fixed-budget panel tests this explanation while "
                "retaining the original mission-stop score as the primary endpoint.",
                "The controller is a calibrated, event-driven policy, not a newly trained neural network. "
                "The simulation uses shared maps and a communication dropout abstraction. Do not describe it as "
                "a validated physical deployment or a fully decentralized communication protocol.",
            ],
        ),
        (
            "Frozen final evaluation",
            [
                "The default complete study uses Zurich, Madrid, Lisbon, Stockholm, Helsinki and Copenhagen, "
                "with 20 paired episodes per setting, alternating north-east and south-west starts. "
                "These geographies are independent only if they have not been inspected for tuning in another run.",
                "Mission-stop: 6 cities x 16 agent/dropout settings x 20 episodes x 4 methods = 7,680 rows. "
                "Fixed-budget: the same 7,680 rows with only mission-completion stopping disabled. "
                "SWAP: 6 cities x 3 target-subset seeds x 20 episodes x 4 methods at 8 agents and 0.03 dropout = 1,440 rows. "
                "Total: 16,800 episodes. Four methods are AntSwarmSafe, EARS-Safe-v7, EARS-Info-v9, and its no-relocation ablation.",
                "Primary endpoint: the equal-city mean Info-v9 minus v7 operational-score difference under mission-stop, "
                "tested two-sided by exact city sign flipping at alpha 0.05. Secondary panel/metric/comparator tests "
                "are corrected together with Holm's method. Six cities provide limited inferential resolution; "
                "city bootstrap intervals are approximate and the exchangeability assumption must be considered.",
                "Completion is reported as full-target success probability and first completion step capped at the horizon "
                "for failures. Fixed-budget still allows all-agent termination and retains all safety rules. "
                "Original, discovery-emphasis and coverage-emphasis weights are reported separately; the primary weights never change.",
            ],
        ),
        (
            "Reproduction and provenance",
            [
                "Use docker compose run --rm --build ears-v9-publication to freeze the existing checkpoint, run or "
                "resume all evaluation panels, and create the final report. Output is under results/publication-v9/. "
                "Use ears-v9-train only for an explicitly new controller calibration; this creates a different model lineage. "
                "The detailed commands and recovery procedure are in docs/EARS_V9_PUBLICATION.md.",
                "Frozen artifacts include the exact controller, normalized source hashes, immutable map snapshots, "
                "evaluation settings and historical selection. SQLite commits each completed episode. Final deliverables "
                "include raw episodes, diagnostics, city summaries, paired tests, score sensitivity, figures, LaTeX tables, "
                "and reports in PDF, HTML and Markdown.",
                "Historical checkpoint SHA-256: " + selection["checkpoint_sha256"],
                "Historical episode SHA-256: " + selection["episodes_sha256"],
                "Historical maps were labeled OpenStreetMap but lack explicit target provenance. The final pipeline "
                "requires verified OpenStreetMap targets, refreshes legacy snapshots online and rejects fallback data. "
                "The historic values above remain unchanged.",
            ],
        ),
        (
            "Before submission",
            [
                "Run the entire frozen study, retain null results and regressions, and report every prespecified panel. "
                "If the reserved cities were previously inspected, designate a genuinely untouched protocol before running. "
                "Use the generated report to update the manuscript's abstract, tables and limitations. Do not merge "
                "validation and test outcomes into one claimed held-out performance number.",
                "Archive the checkpoint, study manifest, map snapshots, raw journal/CSV and exact code revision. "
                "Check the target venue's formatting and reporting requirements, OSM attribution and data distribution "
                "obligations, and include a literature/novelty assessment. This technical report is not a completed "
                "venue-specific manuscript or an acceptance prediction.",
            ],
        ),
    ]


def report(root):
    root = Path(root)
    spec = verify(root)
    db = ledger(root)
    try:
        rows = read_rows(db, spec)
    finally:
        db.close()
    expected = {key(j) for j in jobs(spec)}
    if {key(r) for r in rows} != expected:
        raise ValueError(
            f"Final report requires the full study: {len(rows)}/{len(expected)} rows. Resume test."
        )
    frame = export(root, rows)
    city, summary, stats, strata = analyze(frame)
    weights = sensitivity(frame)
    out = root / "report"
    out.mkdir(exist_ok=True)
    for name, f in [
        ("city_means", city),
        ("summary", summary),
        ("paired_tests", stats),
        ("stratum_deltas", strata),
        ("score_sensitivity", weights),
    ]:
        f.to_csv(out / f"{name}.csv", index=False)
    # Standalone table requires booktabs, as noted in the generated file.
    paper_table = summary[["panel", "strategy", *LABELS]].rename(
        columns={"panel": "Panel", "strategy": "Method", **LABELS}
    )
    (out / "paper_table.tex").write_text(latex_table(paper_table))
    main = stats[stats.primary].iloc[0]
    exploratory_subset = bool(spec.get("exploratory_subset", False))
    projected_maps = [
        m["city"] for m in spec["maps"]
        if m.get("metadata", {}).get("target_projection_method")
    ]
    sparse_exclusions = spec.get("sparse_city_exclusions", [])
    amended_geography = bool(projected_maps)
    confirmation_protocol_completed = bool(
        not spec["smoke_only"] and not exploratory_subset and not amended_geography
    )
    superiority = bool(
        confirmation_protocol_completed and main.delta > 0 and main.p_two_sided < 0.05
    )
    if spec["smoke_only"]:
        status = "SMOKE TEST ONLY - SYNTHETIC DATA"
        verdict = "Synthetic smoke results verify software execution only. No scientific conclusion is permitted."
    elif exploratory_subset or amended_geography:
        status = "EXPLORATORY - AMENDED GEOGRAPHIC PROTOCOL; NOT SIX-CITY CONFIRMATION"
        verdict = (
            "This is an exploratory geographic amendment, NOT the original six-city "
            "independent confirmation study. No confirmatory superiority claim is permitted. "
            f"Evaluated {len(spec['cities'])} cities; excluded={spec.get('excluded_cities', [])}; "
            f"Low-target omissions={sparse_exclusions}; "
            f"OSM grid-projection cities={projected_maps}. "
            "All inferential p-values and uncertainty intervals are descriptive."
        )
    else:
        status = "Completed frozen research study"
        verdict = (
            "The primary test supports an average J advantage on these six cities."
            if superiority else "The primary test does not establish an average J advantage."
        )
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    percity = (
        city[city.panel == "mission_stop"].groupby(["city", "strategy"]).operational_score.mean().unstack()
    )
    (percity[CANDIDATE] - percity[BASELINE]).plot.barh(ax=axes[0], color="#236b8e")
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_xlabel("Info-v9 minus v7: mean J")
    axes[0].set_title("Primary endpoint by city")
    axes[0].set_ylabel("")
    heat = strata.groupby(["agents", "dropout"]).Info_minus_v7.mean().unstack()
    limit = max(float(np.abs(heat.to_numpy()).max()), 0.001)
    im = axes[1].imshow(heat, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    axes[1].set_xticks(range(len(heat.columns)), [str(x) for x in heat.columns])
    axes[1].set_yticks(range(len(heat.index)), [str(x) for x in heat.index])
    axes[1].set(xlabel="Dropout probability", ylabel="Agents", title="Mean J difference by setting")
    fig.colorbar(im, ax=axes[1], shrink=0.7)
    fig.savefig(out / "score_deltas.png", dpi=180)
    fig.savefig(out / "score_deltas.pdf")
    plt.close(fig)
    sections = [
        (
            "Status and primary finding",
            [
                status,
                verdict,
                (
                    "Explicit pre-evaluation sparse-city exclusions: "
                    + (json.dumps(sparse_exclusions, sort_keys=True) if sparse_exclusions else "none")
                    + ". Original mission counts have not been increased and no synthetic maps used."
                ),
                (
                    f"Evaluated cities: {', '.join(c['name'] for c in spec['cities'])}. "
                    f"Excluded: {', '.join(spec.get('excluded_cities', [])) or 'none'}. "
                    f"Distinct city units: {len(spec['cities'])}. "
                    f"Smallest possible exact two-sided city sign-flip p-value: "
                    f"{2 / (2 ** len(spec['cities'])):.6f}. "
                    (
                        f"With {len(spec['cities'])} city units, alpha=0.05 cannot be "
                        "reached by this test."
                        if len(spec["cities"]) <= 5 else
                        "The exact city sign-flip test is interpreted under its assumptions."
                    )
                ),
                f"Completed {len(frame):,} of {spec['expected_rows']:,} episodes. "
                f"Primary mean J difference = {main.delta:.6f}; two-sided city sign-flip p = {main.p_two_sided:.6f}; "
                f"approximate 95% city-bootstrap interval [{main.city_bootstrap_low:.6f}, {main.city_bootstrap_high:.6f}]. "
                f"Worst city/setting difference = {strata.Info_minus_v7.min():.6f}; "
                f"{int((strata.Info_minus_v7 < -0.005).sum())} of {len(strata)} strata breach the original -0.005 limit.",
                "An average advantage does not imply superiority at every setting. The original failed validation "
                "promotion remains part of the record. Geographic independence requires that no test-city results "
                "were used for tuning. The final checkpoint "
                + ("matches" if spec["same_checkpoint_as_historical_validation"] else "DOES NOT match")
                + " the historical validation checkpoint.",
            ],
        )
    ]
    for panel in spec["panels"]:
        table = summary[summary.panel == panel][["strategy", *LABELS]].rename(
            columns={"strategy": "Method", **LABELS}
        )
        sections.append((panel.replace("_", " ").title(), [table]))
    sections += [
        (
            "Completion and observed safety",
            [
                summary[["panel", "strategy", "completed_all_targets", "capped_completion_steps"]].rename(
                    columns={
                        "panel": "Panel",
                        "strategy": "Method",
                        "completed_all_targets": "Full completion",
                        "capped_completion_steps": "Capped steps",
                    }
                ),
                f"Across the complete study, physical incidents total {int(frame.actual_safety_incidents.sum())}, "
                f"battery failures total {int(frame.battery_failures.sum())}, and blocked moves total {int(frame.blocked_moves.sum())}. "
                "These are simulation observations, not a formal safety proof.",
                "Failed episodes receive the full horizon in capped completion time, including episodes in which "
                "all agents terminate early. Fixed-budget means a common horizon ceiling, not guaranteed continued "
                "motion after all agents stop.",
            ],
        ),
        (
            "Paired comparisons and sensitivity",
            [
                out / "score_deltas.png",
                "paired_tests.csv contains all prespecified tests and Holm-adjusted secondary p-values. "
                "The primary test uses cities as the independent unit; seeds and configurations are not "
                "treated as independent cities. Bootstrap intervals are descriptive and not multiplicity-adjusted.",
                weights[["weighting", "Info_minus_v7"]],
                "Score sensitivity changes only the reported exploratory weights, never the primary outcome. "
                "Runtime comparisons apply to the recorded hardware, dependencies and load; inspect per-step "
                "runtime in episodes.csv because mission-stop episode lengths differ.",
            ],
        ),
    ]
    history = historical_sections()[2:]
    if exploratory_subset or amended_geography:
        # Historical six-city *planned study* prose must not masquerade as
        # the completed reduced-city protocol. Keep the historical failures.
        history = [(h, items) for h, items in history
                   if h not in ("Frozen final evaluation", "Reproduction and provenance", "Before submission")]
    sections += history
    sections.append(
        (
            "Run identity",
            [
                "Frozen study SHA-256: " + digest(root / "frozen/study.json"),
                "Raw episode SHA-256: " + digest(root / "episodes.csv"),
                "Checkpoint SHA-256: " + spec["checkpoint_sha256"],
                "See frozen/study.json for exact source hashes, OSM attribution and timestamps, snapshots, "
                "controller settings and pairing seeds. Source hashes normalize CRLF to LF; checkpoint and data hashes use exact bytes.",
            ],
        )
    )
    # Completed reports must not reuse text that says independent confirmation is pending.
    sections = [
        (
            h,
            [
                i.replace(
                    "Independent confirmation remains pending.",
                    (
                        "This smoke run is not independent confirmation."
                        if spec["smoke_only"] else (
                            "This exploratory amended-geography analysis is not independent "
                            "six-city confirmation."
                            if exploratory_subset or amended_geography
                            else "The confirmation results for this study are reported above."
                        )
                    ),
                )
                if isinstance(i, str)
                else i
                for i in items
            ],
        )
        for h, items in sections
    ]
    write_report(out, "EARS v9 - " + status, sections)
    (out / "conclusion.json").write_text(
        json.dumps(
            dict(
                smoke_only=spec["smoke_only"],
                full_matrix_completed=True,
                exploratory_subset=exploratory_subset,
                excluded_cities=spec.get("excluded_cities", []),
                included_cities=[c["name"] for c in spec["cities"]],
                evaluated_city_count=len(spec["cities"]),
                sparse_city_exclusions=sparse_exclusions,
                skip_sparse_cities_enabled=bool(spec.get("skip_sparse_cities", False)),
                amended_osm_geography=amended_geography,
                amended_osm_cities=projected_maps,
                original_six_city_confirmation_completed=confirmation_protocol_completed,
                smallest_attainable_signflip_p=2 / (2 ** len(spec["cities"])),
                primary_average_advantage_supported=superiority,
                primary_delta=float(main.delta),
                primary_p=float(main.p_two_sided),
                worst_stratum_delta=float(strata.Info_minus_v7.min()),
                universal_superiority_claim_permitted=False,
            ),
            indent=2,
        )
    )
    print(f"Final report: {out / 'report.pdf'}", flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation-only", action="store_true")
    p.add_argument("--output-root", default="results/publication-v9")
    args = p.parse_args()
    if args.validation_only:
        write_report(
            Path(args.output_root),
            "EARS v9 - Validation evidence and publication protocol",
            historical_sections(),
        )
    else:
        report(Path(args.output_root))
