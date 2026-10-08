"""Fail-closed v8 publication artifact validation (not model selection)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import episode_operational_score


def validate(directory: str | Path, *, require_confirm: bool = False) -> dict:
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    episodes = pd.read_csv(root / "episodes.csv")
    topology = pd.read_csv(root / "topology.csv")
    summary = pd.read_csv(root / "summary.csv")
    paired = pd.read_csv(root / "paired_city_tests.csv")
    keys = ["city", "start_zone", "episode", "agents", "dropout"]
    checks = {}
    checks["nonempty"] = bool(len(episodes) and len(summary) and len(paired))
    checks["finite_scores"] = bool(np.isfinite(episodes["operational_score"]).all())
    checks["paired_complete"] = bool(episodes.groupby(keys)["strategy"].nunique().eq(4).all())
    checks["unique_episode_keys"] = bool(not episodes.duplicated(keys + ["strategy"]).any())
    expected = episodes.apply(episode_operational_score, axis=1)
    checks["recomputed_scores_equal"] = bool(np.allclose(
        expected, episodes["operational_score"], rtol=0, atol=1e-10))
    checks["no_actual_safety_incidents"] = bool(episodes["actual_safety_incidents"].eq(0).all())
    checks["no_battery_failures"] = bool(episodes["battery_failures"].eq(0).all())
    checks["protocol_unchanged"] = bool(
        hashlib.sha256(Path(manifest["protocol"]).read_bytes()).hexdigest()
        == manifest["protocol_sha256"])
    checks["checkpoint_unchanged"] = bool(
        hashlib.sha256(
            (Path(manifest["arguments"]["model_dir"]) / "ears_safe.json").read_bytes()
        ).hexdigest() == manifest["checkpoint_sha256"])
    checks["topology_covered"] = (
        set(topology["city"]) == set(episodes["city"]))
    if require_confirm:
        checks["confirmation_mode"] = manifest["status"] == "confirmatory"
        checks["real_osm_only"] = bool(episodes["data_source"].eq("openstreetmap").all())
        checks["adequate_target_counts"] = bool(topology["original_targets"].ge(6).all())
        checks["reachable_targets"] = bool(
            topology["reachable_mission_fraction"].ge(1 - 1e-12).all())
    result = {"passed": all(checks.values()), "checks": checks,
              "rows": len(episodes), "cities": int(episodes["city"].nunique()),
              "warning": "Passing validation does NOT imply model superiority or physical safety."}
    (root / "quality_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(f"v8 evidence validation failed: {result}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", default="results/v8/development")
    p.add_argument("--confirm", action="store_true")
    a = p.parse_args()
    print(json.dumps(validate(a.output_root, require_confirm=a.confirm), indent=2))
