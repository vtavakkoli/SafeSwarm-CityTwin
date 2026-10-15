"""Training provenance and stage-resume contracts; Docker CI runs actual training."""

import json

import pytest

from src.environment import obstacles
from tools import eight_city_comparison as comparison
from tools import train_all_eight_city as pipeline


def pipeline_args(root):
    return pipeline.parser().parse_args(["--smoke", "--profile", "core", "--output-root", str(root)])


def test_historical_checkpoint_is_still_required(tmp_path):
    args = comparison.parser().parse_args(["preflight"])
    candidate = tmp_path / "ears_safe.json"
    candidate.write_text("{}")
    with pytest.raises(ValueError, match="ears-v9-train-all"):
        comparison.candidate_provenance(args, tmp_path, candidate, False)


def test_retrained_mode_requires_manifest(tmp_path):
    args = comparison.parser().parse_args(["smoke", "--candidate-mode", "retrained"])
    with pytest.raises(ValueError, match="training-manifest"):
        comparison.candidate_provenance(args, tmp_path, tmp_path / "missing", True)


@pytest.mark.parametrize("stage", pipeline.STAGES)
def test_training_commands_keep_test_stages_out(stage, tmp_path):
    cmd = pipeline.command(stage, tmp_path / stage, tmp_path / "cache", False)
    assert "--offline" in cmd and "--require-real-data" in cmd
    assert "--quick" not in cmd
    assert cmd[cmd.index("--protocol") + 1] == pipeline.PROTOCOL
    assert cmd[cmd.index("--grid-size") + 1] == "40"
    assert cmd[cmd.index("--output-root") + 1] == str(tmp_path / stage)
    assert not any("test_real_city" in part or "test_swap" in part for part in cmd)


def test_completed_stage_skips_execution_and_detects_tampering(tmp_path, monkeypatch):
    calls = []

    def fake_execute(cmd, log):
        calls.append(cmd)
        cp = tmp_path / "learn/checkpoints"
        cp.mkdir()
        for name in comparison.CHECKPOINTS[:4]:
            (cp / name).write_text("{}")
        log.write_text("test fixture")

    monkeypatch.setattr(pipeline, "execute", fake_execute)
    config = {"commands": {"learn": ["fixture"]}, "smoke_only": True}
    first = pipeline.train_stage(tmp_path, "learn", None, config)
    assert pipeline.train_stage(tmp_path, "learn", None, config) == first
    assert len(calls) == 1
    (tmp_path / "learn/checkpoints/grpo_safe.json").write_text("changed")
    with pytest.raises(ValueError, match="Completed stage changed"):
        pipeline.train_stage(tmp_path, "learn", None, config)


def test_interrupted_stage_is_preserved(tmp_path):
    folder = tmp_path / "learn"
    folder.mkdir()
    (folder / "partial-weights.json").write_text("valuable partial work")
    pipeline.start_stage(folder)
    assert not list(folder.iterdir())
    saved = list((tmp_path / "attempts").glob("*/partial-weights.json"))
    assert len(saved) == 1 and saved[0].read_text() == "valuable partial work"


@pytest.fixture
def fresh_record(tmp_path):
    root = tmp_path / "new-study"
    args = pipeline_args(root)
    config = pipeline.initialize(root, args)
    pipeline.prepare(root, args, config["protocol"])
    cp = root / "ears/checkpoints"
    comparison.create_smoke_checkpoints(cp)
    stages = {s: {"command": ["fixture"], "seconds": 1} for s in pipeline.STAGES}
    manifest = pipeline.provenance(root, config, stages)
    evaluation_args = comparison.parser().parse_args(
        [
            "smoke",
            "--candidate-mode",
            "retrained",
            "--training-manifest",
            str(manifest),
            "--cache-dir",
            str(root / "maps/cache"),
            "--model-dir",
            str(cp),
            "--candidate-checkpoint",
            str(cp / "ears_safe.json"),
        ]
    )
    return root, cp, manifest, evaluation_args


def test_new_training_accepts_new_hash_and_rejects_changed_models(fresh_record):
    _, cp, _, args = fresh_record
    candidate = cp / "ears_safe.json"
    assert comparison.digest(candidate) != comparison.EXPECTED_CANDIDATE
    result = comparison.candidate_provenance(args, cp, candidate, True)
    assert result["mode"] == "retrained"
    (cp / "ippo_safe.json").write_text("{}")
    with pytest.raises(ValueError, match="Retrained checkpoints changed"):
        comparison.candidate_provenance(args, cp, candidate, True)


def test_training_scope_and_prepared_maps_cannot_change(fresh_record):
    root, cp, manifest, args = fresh_record
    with pytest.raises(ValueError, match="provenance/scope"):
        comparison.candidate_provenance(args, cp, cp / "ears_safe.json", False)
    data = json.loads(manifest.read_text())
    assert data["training_replicates"] == 1
    map_file = next((root / "maps/cache").glob("*.json"))
    map_file.write_text("{}")
    with pytest.raises(ValueError, match="Map changed"):
        comparison.candidate_provenance(args, cp, cp / "ears_safe.json", True)


def test_resume_configuration_cannot_change(tmp_path):
    root = tmp_path / "study"
    args = pipeline_args(root)
    config = pipeline.initialize(root, args)
    assert pipeline.initialize(root, args) == config
    args.profile = "full"
    with pytest.raises(ValueError, match="configuration"):
        pipeline.initialize(root, args)


@pytest.fixture
def map_setup(tmp_path):
    city = {"name": "San Francisco", "place": "San Francisco, USA", "radius_m": 1500, "split": "test"}
    # Minimal mocked OSM response; these fixtures are not research data.
    valid = dict(
        obstacles={(2, 2)}, restricted_zones=set(), mission_zones={(1, 1)},
        base_stations={(0, 0)}, priority_cells={(1, 1): 1.0},
        metadata={"source": "openstreetmap", "mission_source": "openstreetmap"},
    )
    source = comparison._cache_path(tmp_path / "shared", city["place"], 40, 1500)
    cache = tmp_path / "study/maps/cache"
    cache.mkdir(parents=True)
    return city, valid, source, cache


def test_valid_cache_does_not_download(map_setup, monkeypatch):
    city, valid, source, cache = map_setup
    comparison.save_layer_cache(source, valid)

    def forbidden(*args):
        pytest.fail("Valid cached map must not be refreshed")

    monkeypatch.setattr(obstacles, "_download_osm_layers", forbidden)
    layers, origin, audit = pipeline.real_map(city, 40, 100, cache, source, False)
    assert layers["mission_zones"] == {(1, 1)} and origin == "copied cache"
    assert audit["source_cache_sha256"] == comparison.digest(source)


@pytest.mark.parametrize("invalid", ["synthetic", "missing_target_provenance", "corrupt_json"])
def test_rejected_cache_refreshes_without_touching_source(map_setup, monkeypatch, invalid):
    city, valid, source, cache = map_setup
    bad = comparison.synthetic_city_layers(40, 1)
    if invalid == "missing_target_provenance":
        bad["metadata"]["source"] = "openstreetmap"
    comparison.save_layer_cache(source, bad)
    if invalid == "corrupt_json":
        source.write_text("{broken")
    original = source.read_bytes()
    calls = []

    def download(*args):
        calls.append(args)
        return valid

    monkeypatch.setattr(obstacles, "_download_osm_layers", download)
    layers, origin, audit = pipeline.real_map(city, 40, 100, cache, source, False)
    assert len(calls) == 1 and layers["metadata"]["mission_source"] == "openstreetmap"
    assert origin == "OSM refresh after rejected copied cache"
    assert audit["rejected_cache_reason"] and source.read_bytes() == original
    target = comparison._cache_path(cache, city["place"], 40, 1500)
    comparison.validate_layers(obstacles.load_layer_cache(target), 40, city["name"], False)


def test_invalid_offline_cache_fails_without_network(map_setup, monkeypatch):
    city, _, source, cache = map_setup
    comparison.save_layer_cache(source, comparison.synthetic_city_layers(40, 1))

    def forbidden(*args):
        pytest.fail("Offline mode must never download")

    monkeypatch.setattr(obstacles, "_download_osm_layers", forbidden)
    with pytest.raises(ValueError, match="Offline mode cannot repair"):
        pipeline.real_map(city, 40, 100, cache, source, True)


@pytest.mark.parametrize("cached", [False, True])
def test_download_failure_reports_reason_and_does_not_complete_maps(map_setup, monkeypatch, cached):
    city, _, source, cache = map_setup
    if cached:
        comparison.save_layer_cache(source, comparison.synthetic_city_layers(40, 1))

    def failed(*args):
        raise RuntimeError("Overpass fixture timeout")

    monkeypatch.setattr(obstacles, "_download_osm_layers", failed)
    root = cache.parents[1]
    args = pipeline.parser().parse_args(["--cache-dir", str(source.parent)])
    with pytest.raises(ValueError, match="Overpass fixture timeout"):
        pipeline.prepare(root, args, {"cities": [city]})
    assert not (root / "maps/stage.json").exists()
    assert not (root / "maps/manifest.json").exists()


def test_refresh_with_synthetic_targets_still_rejected(map_setup, monkeypatch):
    city, valid, source, cache = map_setup
    comparison.save_layer_cache(source, comparison.synthetic_city_layers(40, 1))
    valid["metadata"]["mission_source"] = "synthetic_fallback"
    monkeypatch.setattr(obstacles, "_download_osm_layers", lambda *args: valid)
    with pytest.raises(ValueError, match="mission_source=synthetic_fallback"):
        pipeline.real_map(city, 40, 100, cache, source, False)
