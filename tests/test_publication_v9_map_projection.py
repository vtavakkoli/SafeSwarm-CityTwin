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


def test_preflight_records_missing_city_and_keeps_other_diagnostics(tmp_path):
    """A missing sixth city must not discard the preceding five city checks."""
    from src.environment.obstacles import _cache_path, save_layer_cache

    root = Path(__file__).resolve().parents[1]
    raw = tmp_path / "raw"
    derived = tmp_path / "derived"
    report_path = tmp_path / "complete_preflight.json"
    existing = "Existing, OSM"
    cache_file = _cache_path(raw, existing, 9, 1400)
    save_layer_cache(cache_file, isolated_poi_fixture())
    original = cache_file.read_bytes()

    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "cities": [
            {"name": "Existing", "place": existing, "split": "test"},
            {"name": "Copenhagen", "place": "Copenhagen, Denmark", "split": "test"},
        ],
        "start_zones": {"test": ["north_east", "south_west"]},
    }), encoding="utf-8")
    command = [
        sys.executable, "-m", "experiments.prepare_publication_v9_maps",
        "--protocol", str(protocol),
        "--grid-size", "9",
        "--source-cache-dir", str(raw),
        "--output-cache-dir", str(derived),
        "--audit-path", str(report_path),
        "--max-shift", "3",
        "--audit-only",
    ]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(report["cities"]) == 2
    assert report["cities_expected"] == 2
    assert report["cities_with_valid_osm"] == 1
    assert not report["all_source_cities_available"]
    assert not report["all_within_requested_limit"]
    assert report["cities"][0]["city"] == "Existing"
    assert report["cities"][1]["city"] == "Copenhagen"
    assert report["cities"][1]["status"] == "missing_osm_cache"
    assert "--fetch-missing" in report["cities"][1]["error"]
    assert cache_file.read_bytes() == original
    assert not derived.exists()


def test_missing_osm_can_be_fetched_without_refreshing_existing_cache(tmp_path, monkeypatch):
    """Explicit fetch permits new verified snapshots, but never synthetic ones."""
    from experiments import prepare_publication_v9_maps as prep
    from src.environment.obstacles import _cache_path, save_layer_cache

    city = {"name": "Copenhagen", "place": "Copenhagen, Denmark", "radius_m": 1400}
    expected = _cache_path(tmp_path, city["place"], 9, 1400)
    calls = []

    def fake_loader(place, grid, *, radius_m, cache_dir, allow_network):
        calls.append(allow_network)
        assert place == city["place"]
        if allow_network and not expected.exists():
            save_layer_cache(expected, isolated_poi_fixture())
        return prep.load_layer_cache(expected)

    monkeypatch.setattr(prep, "load_real_city_layers", fake_loader)
    with pytest.raises(FileNotFoundError, match="--fetch-missing"):
        prep.load_verified_source(city, 9, tmp_path, fetch_missing=False)
    layers, path = prep.load_verified_source(city, 9, tmp_path, fetch_missing=True)
    assert path == expected
    assert layers["metadata"]["source"] == "openstreetmap"
    assert calls == [True]
    prep.load_verified_source(city, 9, tmp_path, fetch_missing=False)
    assert calls == [True, False]


def test_fetch_missing_rejects_synthetic_fallback(tmp_path, monkeypatch):
    from experiments import prepare_publication_v9_maps as prep

    city = {"name": "Copenhagen", "place": "Copenhagen, Denmark"}
    def synthetic(*a, **kw):
        return {"metadata": {
            "source": "synthetic", "mission_source": "synthetic_fallback",
            "fallback_reason": "Overpass unavailable",
        }}
    monkeypatch.setattr(prep, "load_real_city_layers", synthetic)
    with pytest.raises(ValueError, match="No synthetic fallback"):
        prep.load_verified_source(city, 9, tmp_path, fetch_missing=True)


def test_overpass_endpoint_override_uses_osmnx_base_url(monkeypatch):
    import osmnx as ox
    from experiments.prepare_publication_v9_maps import configure_overpass_endpoint

    original = ox.settings.overpass_url
    try:
        configure_overpass_endpoint("https://overpass.private.coffee/api")
        assert ox.settings.overpass_url == "https://overpass.private.coffee/api"
        configure_overpass_endpoint(None)
        assert ox.settings.overpass_url == "https://overpass.private.coffee/api"
    finally:
        monkeypatch.setattr(ox.settings, "overpass_url", original)


@pytest.mark.parametrize("url", [
    "http://overpass.private.coffee/api",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.private.coffee/api?token=unsafe",
    "https://user:secret@overpass.private.coffee/api",
    "https://overpass.private.coffee",
])
def test_overpass_endpoint_override_rejects_invalid_urls(url):
    from experiments.prepare_publication_v9_maps import configure_overpass_endpoint

    with pytest.raises(ValueError, match="base URL"):
        configure_overpass_endpoint(url)


def test_cli_lists_overpass_endpoint_option():
    from experiments.prepare_publication_v9_maps import parse_args
    import sys
    from unittest.mock import patch

    with patch.object(sys, "argv", ["prepare_publication_v9_maps", "--overpass-url", "https://overpass.private.coffee/api"]):
        args = parse_args()
    assert args.overpass_url == "https://overpass.private.coffee/api"


def test_preflight_explicit_city_exclusion_never_fetches_omitted_city(tmp_path):
    """A manually omitted city remains recorded as an exploratory exclusion."""
    from src.environment.obstacles import _cache_path, save_layer_cache

    root = Path(__file__).resolve().parents[1]
    raw = tmp_path / "raw"
    derived = tmp_path / "derived"
    audit = tmp_path / "subset-preflight.json"
    places = ["One, OSM", "Two, OSM", "Copenhagen, Denmark"]
    for place in places[:2]:
        save_layer_cache(_cache_path(raw, place, 9, 1400), isolated_poi_fixture())
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "cities": [
            {"name": "One", "place": places[0], "split": "test"},
            {"name": "Two", "place": places[1], "split": "test"},
            {"name": "Copenhagen", "place": places[2], "split": "test"},
        ],
        "start_zones": {"test": ["north_east", "south_west"]},
    }), encoding="utf-8")

    result = subprocess.run([
        sys.executable, "-m", "experiments.prepare_publication_v9_maps",
        "--protocol", str(protocol),
        "--grid-size", "9",
        "--source-cache-dir", str(raw),
        "--output-cache-dir", str(derived),
        "--audit-path", str(audit),
        "--audit-only",
        "--max-shift", "3",
        "--exclude-cities", "Copenhagen",
    ], cwd=root, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(audit.read_text(encoding="utf-8"))
    assert data["excluded_cities"] == ["Copenhagen"]
    assert data["included_cities"] == ["One", "Two"]
    assert data["exploratory_subset"]
    assert data["cities_expected"] == 2
    assert len(data["cities"]) == 2
    assert data["all_source_cities_available"]
    assert not derived.exists()
