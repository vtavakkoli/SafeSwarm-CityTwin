"""Publication statistics for frozen EARS vs Ant comparisons.

Reports paired bootstrap confidence intervals, a paired randomization/permutation
test, hierarchical bootstrap intervals with episodes nested inside cities, and
mean ± standard deviation summaries.  No statistic is used for model selection.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

METRICS = {
    "operational_score": "higher",
    "mission_success_rate": "higher",
    "weighted_target_discovery": "higher",
    "coverage_ratio": "higher",
    "energy_consumption": "lower",
    "redundant_coverage": "lower",
    "distance_travelled": "lower",
    "runtime_seconds": "lower",
    "actual_safety_incidents": "lower",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-results", default="results/publication/test/tables/episode_results.csv")
    p.add_argument("--swap-results", default="results/publication/swap-test/tables/episode_results.csv")
    p.add_argument("--output-root", default="results/publication/statistics")
    p.add_argument("--candidate", default="EARS-Safe")
    p.add_argument("--baseline", default="AntSwarmSafe")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--permutations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=90210)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def _ci(samples: np.ndarray) -> tuple[float, float]:
    if samples.size == 0:
        return float("nan"), float("nan")
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _bootstrap_mean(diff: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    if diff.size == 0:
        return np.array([], dtype=float)
    indices = rng.integers(0, diff.size, size=(n_boot, diff.size))
    return diff[indices].mean(axis=1)


def _hierarchical_bootstrap(
    paired: pd.DataFrame,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    cities = list(paired["city"].drop_duplicates())
    if not cities:
        return np.array([], dtype=float)
    groups = {city: paired.loc[paired["city"] == city, "delta"].to_numpy(float) for city in cities}
    values = np.zeros(n_boot, dtype=float)
    for b in range(n_boot):
        selected = rng.choice(cities, size=len(cities), replace=True)
        city_means: list[float] = []
        for city in selected:
            data = groups[str(city)]
            sample = rng.choice(data, size=len(data), replace=True)
            city_means.append(float(np.mean(sample)))
        values[b] = float(np.mean(city_means))
    return values


def _permutation_pvalue(diff: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    if diff.size == 0:
        return float("nan")
    observed = abs(float(np.mean(diff)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, diff.size), replace=True)
    permuted = np.abs((signs * diff).mean(axis=1))
    return float((1.0 + np.sum(permuted >= observed)) / (n_perm + 1.0))


def _paired_table(
    frame: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    swap: bool,
    n_boot: int,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["city", "episode", "start_zone"]
    if swap:
        keys.insert(1, "swap_seed")
    rows: list[dict[str, Any]] = []
    city_rows: list[dict[str, Any]] = []

    available = set(frame["strategy"].astype(str))
    if candidate not in available or baseline not in available:
        raise ValueError(f"Required strategies missing: candidate={candidate!r}, baseline={baseline!r}")

    for metric, preferred in METRICS.items():
        pivot = frame.pivot_table(index=keys, columns="strategy", values=metric, aggfunc="first")
        pivot = pivot[[candidate, baseline]].dropna().reset_index()
        pivot["delta"] = pivot[candidate] - pivot[baseline]
        diff = pivot["delta"].to_numpy(float)
        boot = _bootstrap_mean(diff, n_boot, rng)
        hierarchical = _hierarchical_bootstrap(pivot, n_boot, rng)
        boot_low, boot_high = _ci(boot)
        h_low, h_high = _ci(hierarchical)
        std = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
        effect = float(np.mean(diff) / std) if std > 1e-12 else 0.0
        desirable = diff if preferred == "higher" else -diff
        rows.append(
            {
                "metric": metric,
                "preferred_direction": preferred,
                "pairs": int(diff.size),
                "candidate_mean": float(pivot[candidate].mean()),
                "baseline_mean": float(pivot[baseline].mean()),
                "paired_delta_mean": float(np.mean(diff)),
                "paired_delta_std": std,
                "paired_bootstrap_ci95_low": boot_low,
                "paired_bootstrap_ci95_high": boot_high,
                "hierarchical_bootstrap_ci95_low": h_low,
                "hierarchical_bootstrap_ci95_high": h_high,
                "paired_permutation_pvalue_two_sided": _permutation_pvalue(diff, n_perm, rng),
                "paired_cohens_d": effect,
                "candidate_better_pair_fraction": float(np.mean(desirable > 0.0)),
                "ties_fraction": float(np.mean(np.isclose(diff, 0.0))),
            }
        )
        for city, group in pivot.groupby("city"):
            city_diff = group["delta"].to_numpy(float)
            city_rows.append(
                {
                    "metric": metric,
                    "city": city,
                    "pairs": int(len(group)),
                    "candidate_mean": float(group[candidate].mean()),
                    "baseline_mean": float(group[baseline].mean()),
                    "paired_delta_mean": float(np.mean(city_diff)),
                    "paired_delta_std": float(np.std(city_diff, ddof=1)) if len(city_diff) > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(city_rows)


def _mean_std_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_names = list(METRICS)
    episode = frame.groupby("strategy")[metric_names].agg(["mean", "std"]).reset_index()
    episode.columns = [
        "strategy" if col[0] == "strategy" else f"{col[0]}_{col[1]}"
        for col in episode.columns.to_flat_index()
    ]

    city_means = frame.groupby(["city", "strategy"], as_index=False)[metric_names].mean(numeric_only=True)
    across_city = city_means.groupby("strategy")[metric_names].agg(["mean", "std"]).reset_index()
    across_city.columns = [
        "strategy" if col[0] == "strategy" else f"{col[0]}_citymean_{col[1]}"
        for col in across_city.columns.to_flat_index()
    ]
    return episode, across_city


def main() -> None:
    args = parse_args()
    if args.quick:
        args.bootstrap = min(args.bootstrap, 500)
        args.permutations = min(args.permutations, 1000)
    rng = np.random.default_rng(args.seed)
    test = pd.read_csv(args.test_results)
    swap_path = Path(args.swap_results)
    swap = pd.read_csv(swap_path) if swap_path.exists() else pd.DataFrame()

    output = Path(args.output_root)
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    paired_test, city_test = _paired_table(
        test, candidate=args.candidate, baseline=args.baseline, swap=False,
        n_boot=args.bootstrap, n_perm=args.permutations, rng=rng,
    )
    paired_test.to_csv(tables / "paired_test.csv", index=False)
    city_test.to_csv(tables / "paired_test_by_city.csv", index=False)
    episode_test, citymean_test = _mean_std_tables(test)
    episode_test.to_csv(tables / "test_episode_mean_std.csv", index=False)
    citymean_test.to_csv(tables / "test_citymean_mean_std.csv", index=False)

    if not swap.empty:
        paired_swap, city_swap = _paired_table(
            swap, candidate=args.candidate, baseline=args.baseline, swap=True,
            n_boot=args.bootstrap, n_perm=args.permutations, rng=rng,
        )
        paired_swap.to_csv(tables / "paired_swap.csv", index=False)
        city_swap.to_csv(tables / "paired_swap_by_city.csv", index=False)
        episode_swap, citymean_swap = _mean_std_tables(swap)
        episode_swap.to_csv(tables / "swap_episode_mean_std.csv", index=False)
        citymean_swap.to_csv(tables / "swap_citymean_mean_std.csv", index=False)
    else:
        paired_swap = pd.DataFrame()

    primary = paired_test.loc[paired_test["metric"] == "operational_score"].iloc[0].to_dict()
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": args.candidate,
        "baseline": args.baseline,
        "post_selection_only": True,
        "bootstrap_samples": args.bootstrap,
        "permutation_samples": args.permutations,
        "primary_operational_test": primary,
        "swap_available": bool(not paired_swap.empty),
        "interpretation_rule": (
            "publication support is strongest when the paired and hierarchical 95% CIs for "
            "operational-score delta exclude zero, the paired randomization p-value is small, "
            "and raw discovery/redundancy/energy metrics support the same mechanism"
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
