"""Read-only evidence audit; descriptive diagnostics are never a publication guarantee.

Kept outside experiments/ so adding this tool does not invalidate historical
v9 source manifests. No historical artifacts are rewritten by this command.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.publication_ears_v9 import (  # noqa: E402
    BASELINE,
    CANDIDATE,
    KEYS,
    digest,
    jobs,
    key,
    read_json,
    read_rows,
    sources,
)
from experiments.report_publication_v9 import analyze, city_test, md_table  # noqa: E402


def read_evidence(root):
    """Verify hashes, complete job matrix, row payloads, diagnostics and CSV parity.

    Source differences are reported separately: arithmetic audit remains possible
    without falsely claiming executable/environment reproduction.
    """
    root = Path(root)
    frozen = root / "frozen"
    spec = read_json(frozen / "study.json")
    manifest = read_json(root / "manifest.json")
    actual_hash = digest(frozen / "study.json")
    if actual_hash != (frozen / "study.sha256").read_text().strip():
        raise ValueError("Frozen study hash mismatch")
    for label, got, expected in [
        ("manifest study", actual_hash, manifest["study_sha256"]),
        ("CSV", digest(root / "episodes.csv"), manifest["episodes_sha256"]),
        ("checkpoint", digest(frozen / "ears_safe.json"), spec["checkpoint_sha256"]),
        ("selection", digest(frozen / "historical_selection.json"), spec["prior_selection_sha256"]),
        *[(m["city"], digest(frozen / m["file"]), m["sha256"]) for m in spec["maps"]],
    ]:
        if got != expected:
            raise ValueError(f"{label} hash mismatch")
    # Immutable mode also prevents SQLite from creating WAL/SHM sidecars. Require
    # a closed, checkpointed snapshot so pending WAL transactions cannot be lost.
    wal = root / "episodes.sqlite-wal"
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Close/checkpoint the source SQLite writer before auditing its snapshot")
    with closing(
        sqlite3.connect((root / "episodes.sqlite").resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    ) as db:
        rows = read_rows(db, spec)
    expected = {key(j) for j in jobs(spec)}
    if len(expected) != spec["expected_rows"] or {key(r) for r in rows} != expected:
        raise ValueError("Incomplete frozen matrix")
    if manifest["status"] != "completed" or any(
        manifest[k] != len(rows) for k in ("expected_rows", "completed_rows")
    ):
        raise ValueError("Manifest completion mismatch")
    frame = pd.DataFrame([{k: v for k, v in r.items() if k != "diagnostics"} for r in rows])
    exported = pd.read_csv(root / "episodes.csv")
    pd.testing.assert_frame_equal(
        frame.sort_values(KEYS).reset_index(drop=True)[exported.columns],
        exported.sort_values(KEYS).reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        atol=1e-12,
        rtol=0,
    )
    diagnostic_rows = [json.loads(line) for line in (root / "diagnostics.jsonl").read_text().splitlines()]
    diagnostics = {key(r): r["diagnostics"] for r in diagnostic_rows}
    if len(diagnostic_rows) != len(rows) or diagnostics != {key(r): r["diagnostics"] for r in rows}:
        raise ValueError("Diagnostic export mismatch")
    failed = frame.completed_all_targets == 0
    if not frame.completed_all_targets.isin([0, 1]).all():
        raise ValueError("Invalid completion indicator")
    if not (frame.loc[failed, "capped_completion_steps"] == spec["max_steps"]).all():
        raise ValueError("Unsuccessful episode not capped at full horizon")
    success = frame.loc[~failed]
    if (
        success.completion_step.isna().any()
        or not (
            (success.completion_step >= 1)
            & (success.completion_step <= success.steps)
            & (success.capped_completion_steps == success.completion_step)
        ).all()
    ):
        raise ValueError("Invalid successful completion time")
    current = sources()
    differences = sorted(k for k, v in spec["source_sha256_lf"].items() if current.get(k) != v)
    additions = sorted(set(current) - set(spec["source_sha256_lf"]))
    return spec, frame, differences, additions


def diagnostic_tables(frame, spec):
    main = frame[frame.panel == "mission_stop"]
    pairkeys = [k for k in KEYS if k != "strategy"]
    pairs = main.pivot(index=pairkeys, columns="strategy", values="operational_score")
    delta = (pairs[CANDIDATE] - pairs[BASELINE]).rename("delta_J").reset_index()
    by_city = delta.groupby("city").delta_J.mean()
    influence = pd.DataFrame(
        [
            {"omitted_city": city, "remaining_cities": len(by_city) - 1, "delta_J": by_city.drop(city).mean()}
            for city in by_city.index
        ]
    )
    strata = delta.groupby(["city", "agents", "dropout"], as_index=False).delta_J.mean()
    failures = strata[strata.delta_J < -0.005].sort_values("delta_J")
    zone_pairs = main.pivot(index=[*pairkeys, "start_zone"], columns="strategy", values="operational_score")
    zone = (zone_pairs[CANDIDATE] - zone_pairs[BASELINE]).rename("delta_J").reset_index()
    zone = zone.groupby(["city", "agents", "dropout", "start_zone"], as_index=False).delta_J.mean()
    maps = pd.DataFrame(
        [
            {
                "city": m["city"],
                "targets": next(iter(m["quality"].values()))["original_targets"],
                "moved_targets": m["metadata"].get("target_projection_audit", {}).get("modified_targets", 0),
                "protocol_amended": bool(m["metadata"].get("target_projection_method")),
                "delta_J": by_city[m["city"]],
            }
            for m in spec["maps"]
        ]
    )
    unchanged = maps.loc[maps.moved_targets == 0, "city"]
    sensitivity = pd.DataFrame(
        [
            {"analysis": "all cities", **city_test(by_city)},
            *(
                [{"analysis": "no targets physically moved (post hoc)", **city_test(by_city.loc[unchanged])}]
                if len(unchanged)
                else []
            ),
        ]
    )
    runtime = (
        main.groupby(["city", "strategy"], as_index=False)[["runtime_seconds", "runtime_overhead", "steps"]]
        .mean()
        .groupby("strategy", as_index=False)
        .mean(numeric_only=True)
    )
    return {
        "city_map_diagnostics": maps,
        "regression_settings": failures,
        "start_zone_deltas": zone,
        "leave_one_city_out": influence,
        "projection_sensitivity": sensitivity,
        "runtime": runtime,
    }


def audit(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("Audit output must be separate from the original evidence directory")
    spec, frame, differences, additions = read_evidence(root)
    _, summary, stats, strata = analyze(frame)
    tables = diagnostic_tables(frame, spec)
    primary = stats[stats.primary].iloc[0]
    count = frame.city.nunique()
    verdict = {
        "scope": "technical evidence review; not an IEEE acceptance prediction",
        "original_evidence_integrity_passed": True,
        "rows": len(frame),
        "cities": count,
        "primary_delta_J": float(primary.delta),
        "city_signflip_p": float(primary.p_two_sided),
        "minimum_two_sided_p": 2 / 2**count,
        "regression_settings": len(tables["regression_settings"]),
        "total_settings": len(strata),
        "worst_setting_delta_J": float(strata.Info_minus_v7.min()),
        "source_mismatches": differences,
        "source_additions": additions,
        "study_sha256": digest(root / "frozen/study.json"),
        "episodes_sha256": digest(root / "episodes.csv"),
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "historical_selected_method": read_json(root / "frozen/historical_selection.json")["selected_method"],
        "independent_confirmation_established": False,
        "submission_recommendation": "Revise evidence and align manuscript before submission",
        "safety_counts": {
            k: int(frame[k].sum()) for k in ["actual_safety_incidents", "battery_failures", "blocked_moves"]
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, table in {**tables, "summary": summary, "paired_tests": stats}.items():
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "audit.json").write_text(json.dumps(verdict, indent=2, allow_nan=False) + "\n")
    sections = [
        "# EARS v9 publication evidence audit",
        "Recommendation: revise evidence and align the manuscript before submission. "
        "Null results can be publishable; a significant superiority result is not "
        "a universal IEEE requirement. "
        "This audit assesses the supplied evidence, not acceptance or originality.",
        f"Verified {len(frame):,} journal rows against the frozen job matrix, payload hashes, raw CSV, "
        "diagnostic export, score formula and artifact hashes. Original evidence was opened read-only.",
        f"Primary delta J = {primary.delta:.8f}; "
        f"exact two-sided city sign-flip p = {primary.p_two_sided:.4f}. "
        f"The {count}-city test cannot attain p < {2 / 2**count:.4f}, regardless of episode count. "
        "The positive percentile bootstrap interval does not overturn this small-city limitation. "
        "All added sensitivity analyses are post hoc and descriptive.",
        "## Geographic concentration and target projection",
        md_table(tables["city_map_diagnostics"]),
        md_table(tables["projection_sensitivity"]),
        "Target projection sensitivity removes entire cities with moved targets; it does NOT isolate "
        "the causal effect of moving targets and is not a new confirmatory test.",
        "## Settings below the original -0.005 robustness threshold",
        md_table(tables["regression_settings"]),
        "## Leave-one-city-out sensitivity",
        md_table(tables["leave_one_city_out"]),
        "## Runtime on the original recorded platform",
        md_table(tables["runtime"]),
        "runtime_overhead is seconds per simulated environment step, not per robot decision. "
        "Episode runtime is affected by stopping time; compare both measures on the same platform.",
        "## Provenance and claim boundaries",
        f"Files differing from frozen hashes: {differences or 'none'}. Additional hashed-source files: "
        f"{additions or 'none'}. Hash consistency supports artifact integrity, not historical independence "
        "or a fresh replay in the original software environment. Archive the evaluated Dockerfile/image "
        "and resolved dependency versions; never edit the original frozen manifest to make checks pass.",
        "The attached manuscript describes v7, while these data evaluate v9 information gain, graph routing, "
        "persistent routes and sensing reservations. Rewrite the algorithm, equations and contributions "
        "before using v9 numbers. Keep the older eight-city comparison separate. The score implementation "
        "uses collisions plus restricted entries in its safety component; battery failures are exported "
        "separately, contrary to the manuscript's broader definition. This did not affect this run's J "
        "because all those counts were zero.",
        "The previous validation selection retained v7. Four-city exploratory findings cannot be "
        "relabeled as independent six-city confirmation. Reused cities may support mechanism diagnostics; "
        "a later confirmatory study needs a prospectively frozen sampling and analysis protocol. "
        "At least six cities are needed merely for p-value resolution, not guaranteed power or significance.",
        "Zero recorded incidents is a simulation observation. Shared-map access during communication "
        "dropout does not validate distributed communication failure robustness or physical robot safety.",
    ]
    (output / "report.md").write_text("\n\n".join(sections) + "\n")
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.study_root, args.output_root), indent=2))
