"""Validation-only checkpoint selection utilities.

A single validation city/start zone can produce a deceptively strong checkpoint
that fails on a new urban topology. v4 therefore selects checkpoints with a
robust score that rewards a high mean while penalizing confidence-interval width
and variation across validation city/start-zone domains.

No test-split metric is accepted by these helpers.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def ci95(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size <= 1:
        return 0.0
    return float(1.96 * np.std(array, ddof=1) / np.sqrt(array.size))


def validation_selection_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Return mean and domain-robust validation statistics.

    ``robust_score`` is intentionally conservative:

    mean - 0.5 * CI95 - 0.20 * domain_std - 0.10 * (mean - worst_domain)

    The final term penalizes a policy that is excellent in one validation
    geography and weak in another. This directly targets the Amsterdam-only
    model-selection failure observed in v3 without ever looking at test cities.
    """

    if not rows:
        return {
            "mean_score": float("nan"),
            "ci95": float("nan"),
            "domain_std": float("nan"),
            "worst_domain_score": float("nan"),
            "robust_score": float("nan"),
            "domains": 0.0,
        }

    frame = pd.DataFrame(rows)
    if "operational_score" not in frame:
        raise ValueError("Validation rows must contain operational_score")
    scores = frame["operational_score"].astype(float)
    mean = float(scores.mean())
    interval = ci95(scores.to_numpy())

    domain_columns = [column for column in ("city", "start_zone") if column in frame]
    if domain_columns:
        domain_means = frame.groupby(domain_columns, dropna=False)["operational_score"].mean()
    else:
        domain_means = pd.Series([mean], dtype=float)
    domain_values = domain_means.to_numpy(dtype=float)
    domain_std = float(np.std(domain_values, ddof=0)) if domain_values.size else 0.0
    worst = float(np.min(domain_values)) if domain_values.size else mean
    robust = float(
        mean
        - 0.50 * interval
        - 0.20 * domain_std
        - 0.10 * max(0.0, mean - worst)
    )
    return {
        "mean_score": mean,
        "ci95": interval,
        "domain_std": domain_std,
        "worst_domain_score": worst,
        "robust_score": robust,
        "domains": float(domain_values.size),
    }
