"""Stress-test SafeSwarm ranking against operational-score weight choices.

The baseline score weights remain unchanged.  This post-selection analysis
samples reasonable ±25% multiplicative perturbations around the published
weights, renormalizes them, and reports how often EARS remains ahead of Ant.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_WEIGHTS = {
    "discovery": 0.35,
    "coverage": 0.20,
    "safety": 0.20,
    "energy": 0.10,
    "coordination": 0.10,
    "communication": 0.05,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results/publication/test/tables/episode_results.csv")
    p.add_argument("--output-root", default="results/publication/sensitivity")
    p.add_argument("--candidate", default="EARS-Safe")
    p.add_argument("--baseline", default="AntSwarmSafe")
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--variation", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=314159)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def _components(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["component_discovery"] = out["weighted_target_discovery"].astype(float)
    out["component_coverage"] = out["coverage_ratio"].astype(float)
    out["component_safety"] = 1.0 - np.minimum(
        1.0,
        out["actual_safety_incidents"].astype(float)
        / np.maximum(1.0, out["steps"].astype(float) * out["agents"].astype(float)),
    )
    out["component_energy"] = 1.0 - np.minimum(
        1.0,
        out["energy_consumption"].astype(float)
        / np.maximum(1.0, 100.0 * out["agents"].astype(float)),
    )
    out["component_coordination"] = 1.0 - np.minimum(1.0, out["redundant_coverage"].astype(float))
    out["component_communication"] = out["communication_efficiency"].astype(float)
    return out


def _score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    total = np.zeros(len(frame), dtype=float)
    for name, weight in weights.items():
        total += float(weight) * frame[f"component_{name}"].to_numpy(float)
    return pd.Series(np.clip(total, 0.0, 1.0), index=frame.index)


def _city_equal_means(frame: pd.DataFrame, scores: pd.Series) -> pd.Series:
    tmp = frame[["city", "strategy"]].copy()
    tmp["score"] = scores.to_numpy(float)
    city = tmp.groupby(["city", "strategy"], as_index=False)["score"].mean()
    return city.groupby("strategy")["score"].mean()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.samples = min(args.samples, 200)
    if not (0.0 <= args.variation < 1.0):
        raise ValueError("--variation must be in [0, 1)")

    frame = _components(pd.read_csv(args.results))
    names = set(frame["strategy"].astype(str))
    if args.candidate not in names or args.baseline not in names:
        raise ValueError("candidate/baseline missing from result table")

    rng = np.random.default_rng(args.seed)
    base = np.array([BASE_WEIGHTS[name] for name in BASE_WEIGHTS], dtype=float)
    scenarios: list[dict[str, float | int | str]] = []

    deterministic = {
        "baseline": BASE_WEIGHTS,
        "discovery_high": {"discovery": 0.40, "coverage": 0.15, "safety": 0.20, "energy": 0.10, "coordination": 0.10, "communication": 0.05},
        "coverage_high": {"discovery": 0.30, "coverage": 0.25, "safety": 0.20, "energy": 0.10, "coordination": 0.10, "communication": 0.05},
        "energy_high": {"discovery": 0.32, "coverage": 0.18, "safety": 0.20, "energy": 0.15, "coordination": 0.10, "communication": 0.05},
        "coordination_high": {"discovery": 0.32, "coverage": 0.18, "safety": 0.20, "energy": 0.10, "coordination": 0.15, "communication": 0.05},
        "efficiency_high": {"discovery": 0.30, "coverage": 0.17, "safety": 0.20, "energy": 0.15, "coordination": 0.13, "communication": 0.05},
    }

    def evaluate(label: str, weights: dict[str, float], sample_id: int) -> None:
        means = _city_equal_means(frame, _score(frame, weights))
        candidate_score = float(means[args.candidate])
        baseline_score = float(means[args.baseline])
        winner = str(means.idxmax())
        scenarios.append(
            {
                "sample": sample_id,
                "label": label,
                **{f"weight_{k}": float(v) for k, v in weights.items()},
                "candidate_score": candidate_score,
                "baseline_score": baseline_score,
                "candidate_minus_baseline": candidate_score - baseline_score,
                "overall_winner": winner,
                "candidate_beats_baseline": int(candidate_score > baseline_score),
            }
        )

    sample_id = 0
    for label, weights in deterministic.items():
        evaluate(label, weights, sample_id)
        sample_id += 1

    names_order = list(BASE_WEIGHTS)
    for index in range(args.samples):
        multipliers = rng.uniform(1.0 - args.variation, 1.0 + args.variation, size=len(base))
        perturbed = base * multipliers
        perturbed /= float(np.sum(perturbed))
        weights = {name: float(perturbed[i]) for i, name in enumerate(names_order)}
        evaluate("random_perturbation", weights, sample_id + index)

    output = Path(args.output_root)
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(scenarios)
    result.to_csv(tables / "weight_scenarios.csv", index=False)

    random_rows = result[result["label"] == "random_perturbation"]
    margin = random_rows["candidate_minus_baseline"].to_numpy(float)
    summary = pd.DataFrame(
        [
            {
                "candidate": args.candidate,
                "baseline": args.baseline,
                "random_scenarios": int(len(random_rows)),
                "variation_fraction": float(args.variation),
                "candidate_beats_baseline_fraction": float(np.mean(margin > 0.0)),
                "candidate_is_overall_winner_fraction": float(np.mean(random_rows["overall_winner"] == args.candidate)),
                "margin_mean": float(np.mean(margin)),
                "margin_std": float(np.std(margin, ddof=1)) if len(margin) > 1 else 0.0,
                "margin_p02_5": float(np.quantile(margin, 0.025)),
                "margin_median": float(np.quantile(margin, 0.5)),
                "margin_p97_5": float(np.quantile(margin, 0.975)),
                "margin_min": float(np.min(margin)),
                "margin_max": float(np.max(margin)),
            }
        ]
    )
    summary.to_csv(tables / "sensitivity_summary.csv", index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_weights": BASE_WEIGHTS,
        "weight_perturbation": f"independent multiplicative ±{100 * args.variation:.1f}% then renormalized",
        "post_selection_only": True,
        "used_for_model_selection": False,
        "summary": summary.iloc[0].to_dict(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
