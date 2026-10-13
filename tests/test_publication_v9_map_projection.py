"""Synthetic-fixture tests for publication v9 OSM target-cell projection."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.prepare_publication_v9_maps import project_targets
from experiments.benchmark_ears_v8 import topology_metrics
from src.training.geography import apply_start_zone


def isolated_poi_fixture():
    return {
        "obstacles": {(4, 3), (4, 5), (3, 4), (5, 4)},
        "restricted_zones": set(),
        "mission_zones": {(4, 4), (6, 6)},
        "base_stations": {(0, 0)},
        "priority_cells": {(4, 4): 0.85, (6, 6): 0.55},
        "metadata": {"source": "openstreetmap", "mission_source": "openstreetmap"},
    }


def test_projection_preserves_targets_priorities_obstacles_and_osm_provenance():
    original = isolated_poi_fixture()
    unchanged_copy = deepcopy(original)
    zones = ["north_east", "south_west"]
    projected, audit = project_targets(original, 9, zones, 3)
    assert original == unchanged_copy
    assert projected["obstacles"] == original["obstacles"]
    assert projected["restricted_zones"] == original["restricted_zones"]
    assert len(projected["mission_zones"]) == len(original["mission_zones"])
    assert (6, 6) in projected["mission_zones"]
    assert (4, 4) not in projected["mission_zones"]
    assert sorted(projected["priority_cells"].values()) == [0.55, 0.85]
    assert audit["modified_targets"] == 1
    assert len(audit["modifications"]) == 1
    assert projected["metadata"]["mission_source"] == "openstreetmap"
    assert projected["metadata"]["target_projection_audit"]["preprocessing_is_protocol_amendment"]
    for zone in zones:
        metrics = topology_metrics(apply_start_zone(projected, 9, zone), 9)
        assert metrics["reachable_mission_fraction"] == 1.0


def test_bounded_projection_rejects_large_target_shift():
    with pytest.raises(ValueError, match="max-shift=1"):
        project_targets(isolated_poi_fixture(), 9, ["north_east", "south_west"], 1)


def test_connected_targets_are_unchanged():
    original = isolated_poi_fixture()
    original["mission_zones"] = {(6, 6)}
    original["priority_cells"] = {(6, 6): 0.55}
    projected, audit = project_targets(original, 9, ["north_east", "south_west"], 0)
    assert projected["mission_zones"] == original["mission_zones"]
    assert projected["priority_cells"] == original["priority_cells"]
    assert audit["modified_targets"] == 0


def test_map_preparation_runs_as_direct_script():
    """Regression: Docker Compose invokes this path directly, not as -m."""
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "prepare_publication_v9_maps.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--max-shift" in result.stdout


def test_audit_only_reports_full_shift_without_modifying_maps(tmp_path):
    """Preflight measures over-limit moves but never commits a derived map."""
    from src.environment.obstacles import _cache_path, save_layer_cache

    root = Path(__file__).resolve().parents[1]
    city = "Fixture, OSM"
    raw = tmp_path / "raw"
    projected = tmp_path / "derived"
    audit_path = tmp_path / "preflight.json"
    cache_file = _cache_path(raw, city, 9, 1400)
    save_layer_cache(cache_file, isolated_poi_fixture())
    original_bytes = cache_file.read_bytes()

    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "cities": [{"name": "Fixture", "place": city, "split": "test", "radius_m": 1400}],
        "start_zones": {"test": ["north_east", "south_west"]},
    }), encoding="utf-8")

    result = subprocess.run([
        sys.executable, "-m", "experiments.prepare_publication_v9_maps",
        "--protocol", str(protocol),
        "--grid-size", "9",
        "--source-cache-dir", str(raw),
        "--output-cache-dir", str(projected),
        "--audit-path", str(audit_path),
        "--max-shift", "1",
        "--audit-only",
    ], cwd=root, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    assert report["status"] == "preflight_only"
    assert not report["all_within_requested_limit"]
    assert report["cities"][0]["city"] == "Fixture"
    assert report["cities"][0]["required_max_shift"] >= 2
    assert report["cities"][0]["over_limit"]
    assert cache_file.read_bytes() == original_bytes
    assert not projected.exists()
