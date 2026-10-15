"""Publication evidence report from a validated, complete all-method ledger."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import reportlab
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)


def write_report(root, plan, frame, summary, city, tests):
    out = Path(root) / "report"
    smoke = plan["smoke_only"]
    metrics = [
        "operational_score",
        "weighted_target_discovery",
        "coverage_ratio",
        "energy_consumption",
        "runtime_seconds",
        "completed_all_targets",
        "actual_safety_incidents",
    ]
    city_sd = city.groupby(["analysis_panel", "strategy"])[metrics].std(ddof=1).add_suffix("_city_sd")
    city_sd.to_csv(out / "between_city_sd.csv")
    setting_city = frame.groupby(["panel", "city", "agents", "dropout", "strategy"])[metrics].mean()
    setting_city.to_csv(out / "setting_city_means.csv")
    setting = setting_city.groupby(["panel", "agents", "dropout", "strategy"]).mean()
    setting.to_csv(out / "setting_means.csv")
    primary = tests[tests.primary].iloc[0]
    provenance = plan.get("candidate_provenance", {})
    training = provenance.get("training")
    scope = "SYNTHETIC SOFTWARE CHECK - NOT PAPER RESULTS" if smoke else "Exploratory eight-city evaluation"
    narrative = (
        f"The complete evaluation contains {len(frame):,} episodes across {len(plan['cities'])} cities "
        f"and {len(plan['methods'])} implementations. At the original comparison setting, "
        f"EARS-Info-v9 minus EARS-Safe has an equal-city mean operational-score difference of "
        f"{primary.delta:+.6f} (two-sided exact city sign-flip p={primary.p_two_sided:.6f}; "
        f"descriptive 95% city-bootstrap interval [{primary.city_bootstrap_low:+.6f}, "
        f"{primary.city_bootstrap_high:+.6f}]). These outcomes are conditional on the frozen models "
        "and reused geography. They do not establish independent geographic confirmation."
    )
    limitations = [
        "The eight test city identities have been used previously; all conclusions are exploratory.",
        "One training replicate per learned method: uncertainty over training seeds is not estimated.",
        "Matched evaluation does not imply equal training compute or equal model capacity. "
        "Training commands, selection histories, and elapsed stage times accompany the package.",
        "QMIX, MADDPG and MAT are lightweight repository policy-style adaptations, "
        "not faithful reproductions of the published full algorithms.",
        "Native safety and return mechanisms differ; RandomAgent and GreedyAgent are unfiltered controls.",
        "Known grid geometry and shared observed maps remain available. Communication dropout "
        "does not model a fully decentralized communication system.",
        "Fixed-budget disables mission stopping, but all-agent termination can still end episodes early.",
        "Unchanged or duplicate SWAP target views remain descriptive; changed-view inference "
        "only includes eligible cities listed in swap_validity.csv.",
        "There are only eight geographic units. Bootstrap intervals are descriptive; "
        "secondary comparisons use conservative Holm adjustment across the full secondary family.",
        "Zero measured incidents do not prove physical safety. Complete experiments do not "
        "automatically make an IEEE manuscript submission-ready.",
    ]
    if smoke:
        limitations.insert(0, "All figures and measurements in this report are synthetic software checks.")
    stage_rows = []
    if training:
        for name, stage in training["stages"].items():
            stage_rows.append(
                dict(stage=name, elapsed_seconds=stage["seconds"], command=" ".join(stage["command"]))
            )
        pd.DataFrame(stage_rows).to_csv(out / "training_budget.csv", index=False)
    text = [f"# {scope}", "", "## Results", "", narrative, "", "## Interpretation and limitations", ""]
    text += ["- " + item for item in limitations]
    text += [
        "",
        "## Reproducibility",
        "",
        "Raw episodes, paired contrasts, city means, setting means, all-method tables, frozen "
        "sources, models, maps and training provenance accompany this report. "
        "Consult index.html for the interactive-document view and CSV files for full precision.",
        "",
    ]
    (out / "publication_report.md").write_text("\n".join(text), encoding="utf-8")
    # This is a measured results section, not an invented complete article.
    tex = [
        "% Generated from the complete frozen ledger; review before manuscript integration.",
        r"\subsection{All-method evaluation}",
        scope.replace("_", r"\_"),
        "",
        narrative.replace("%", r"\%"),
        "",
        r"\input{all_methods_original_setting.tex}",
        "",
        "% See publication_report.md for limitations that must accompany these results.",
        "",
    ]
    (out / "results_section.tex").write_text("\n".join(tex), encoding="utf-8")

    fonts = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont("ReportSans", str(fonts / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("ReportSans-Bold", str(fonts / "VeraBd.ttf")))
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        if hasattr(style, "fontName"):
            style.fontName = "ReportSans-Bold" if "Bold" in style.fontName else "ReportSans"
    styles.add(ParagraphStyle("CellSmall", fontName="ReportSans", fontSize=7, leading=8.5))
    styles.add(ParagraphStyle("ReportBody", parent=styles["BodyText"], fontSize=9, leading=13, spaceAfter=8))
    styles["Title"].textColor = colors.HexColor("#173B54")
    styles["Heading1"].textColor = colors.HexColor("#173B54")
    story = []

    def para(value, style="ReportBody"):
        return Paragraph(html.escape(str(value)), styles[style])

    def table(data, widths):
        cells = [[para(v, "CellSmall") for v in row] for row in data]
        t = LongTable(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE8F0")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F7")]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#758DA7")),
                ]
            )
        )
        return t

    def outcomes(data):
        rows = [["Method", "J", "Discovery", "Coverage", "Energy", "Time (s)", "Complete", "Incidents"]]
        for row in data.sort_values("operational_score", ascending=False).itertuples():
            rows.append([row.strategy, *[f"{getattr(row, m):.4f}" for m in metrics]])
        return table(rows, [155, 39, 49, 46, 46, 46, 46, 47])

    story += [
        para("SafeSwarm-CityTwin", "Title"),
        para("Comprehensive publication evidence report", "Title"),
        Spacer(1, 12),
        para(scope, "Heading1"),
        para(narrative),
        para("Protocol and model selection", "Heading1"),
    ]
    if training:
        story += [
            para(
                (
                    "This short synthetic check trains on Vienna and validates on Amsterdam. "
                    if smoke
                    else "Training: Vienna, London and Berlin. Validation: Amsterdam and Prague. "
                )
                + "The eight test cities are excluded from training and checkpoint selection. "
                "Four learned policies are trained; PRISM/EARS configurations are selected "
                "on development data. Fixed heuristics have no trainable weights. "
                "The four information-routing variants share the newly selected EARS-Safe "
                "checkpoint and retain the fixed v9 routing configuration."
            )
        ]
        story += [
            table(
                [
                    ["Stage", "Elapsed seconds"],
                    *[[r["stage"], f"{r['elapsed_seconds']:.1f}"] for r in stage_rows],
                ],
                [360, 164],
            )
        ]
    else:
        story += [
            para(
                "Historical frozen checkpoints are evaluated. Training histories are not "
                "reconstructed by this evaluator; document their original selection provenance."
            )
        ]
    story += [
        Spacer(1, 8),
        para(
            "Evaluation uses identical maps, starts, sensor rules, score, and "
            "step/robot-indexed communication draws for all methods. Mission-stop, fixed-budget "
            "and SWAP panels are reported separately. Means weight cities equally. "
            "The primary contrast is fixed at 8 robots, dropout 0.03, and mission stopping. "
            "The 180-episode smoke test uses a smaller synthetic configuration."
        ),
    ]
    story += [
        para("Score definition", "Heading1"),
        para(
            "J = 0.35 discovery + 0.20 coverage + 0.20 safety + 0.10 energy efficiency "
            "+ 0.10 nonredundancy + 0.05 communication efficiency. Runtime is reported separately "
            "and does not affect J. Definitions and normalization are preserved in the frozen source."
        ),
    ]
    for panel in ["original_setting", "mission_stop", "fixed_budget", "swap", "swap_changed"]:
        data = summary[summary.analysis_panel == panel]
        if data.empty:
            continue
        story += [
            PageBreak(),
            para(panel.replace("_", " ").title(), "Heading1"),
            para(
                f"All 30 implementations; {int(data.cities.iloc[0])} cities; "
                f"{int(data.episodes_per_method.iloc[0]):,} episodes per method. "
                "Between-city standard deviations and adjusted contrasts are provided in CSV."
            ),
            outcomes(data),
        ]
    story += [
        PageBreak(),
        para("All-method comparison", "Heading1"),
        Image(str(out / "all_methods.png"), width=6.7 * inch, height=6.7 * inch),
        PageBreak(),
        para("Primary city contrast", "Heading1"),
        Image(str(out / "city_contrast.png"), width=7.1 * inch, height=3.55 * inch),
    ]
    contrasts = tests[(tests.analysis_panel == "original_setting") & (tests.metric == "operational_score")]
    rows = [["Comparator", "Delta J", "Raw p", "Holm secondary p"]]
    for row in contrasts.itertuples():
        rows.append(
            [
                row.comparator,
                f"{row.delta:+.5f}",
                f"{row.p_two_sided:.5f}",
                "Primary" if row.primary else f"{row.p_holm_secondary:.5f}",
            ]
        )
    story += [
        PageBreak(),
        para("Candidate versus every comparator", "Heading1"),
        para(
            "Positive delta favors EARS-Info-v9. This table uses the original setting. "
            "All metrics and panels, bootstrap intervals, and multiplicity adjustments are in contrasts.csv."
        ),
        table(rows, [235, 85, 85, 119]),
    ]
    for name in plan["cities"]:
        data = city[(city.analysis_panel == "original_setting") & (city.city == name)]
        story += [PageBreak(), para(name + " - all methods", "Heading1"), outcomes(data)]
    story += [PageBreak(), para("Interpretation and publication checks", "Heading1")]
    story += [para(f"{i}. {item}") for i, item in enumerate(limitations, 1)]
    story += [
        para("Remaining manuscript work", "Heading1"),
        para(
            "Integrate these measured results into the original manuscript, retain the original method "
            "and literature coverage, document training budgets and implementation adaptations, and "
            "review claims against all city-level outcomes. Author information, references, formatting, "
            "and journal submission requirements require final editorial review. This report makes no "
            "acceptance prediction or automatic submission-ready designation."
        ),
    ]

    def footer(canvas, document):
        canvas.setFont("ReportSans", 7)
        canvas.setFillColor(colors.HexColor("#516575"))
        canvas.drawString(
            36,
            20,
            "SYNTHETIC CHECK - NOT PAPER RESULTS" if smoke else "SafeSwarm-CityTwin | Exploratory evidence",
        )
        canvas.drawRightString(document.pagesize[0] - 36, 20, str(document.page))

    SimpleDocTemplate(
        str(out / "publication_report.pdf"),
        pagesize=(595.28, 841.89),
        leftMargin=35,
        rightMargin=35,
        topMargin=35,
        bottomMargin=35,
        title="SafeSwarm-CityTwin publication evidence report",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
