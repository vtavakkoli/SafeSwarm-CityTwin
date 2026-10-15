"""Training provenance and stage-resume contracts; Docker CI runs actual training."""

import json

import pytest

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
