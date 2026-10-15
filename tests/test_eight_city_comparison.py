"""Regression tests for the frozen comparison, not evidence of policy quality."""

import copy
import json
import sqlite3
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from tools import eight_city_comparison as study
from tools.report_eight_city_comparison import analyze, report


def arguments(root, *extra):
    return study.parser().parse_args(["smoke", "--profile", "core", "--output-root", str(root), *extra])


@pytest.mark.parametrize("profile,expected", [("core", 24000), ("full", 168000)])
def test_every_method_in_every_paired_setting(profile, expected):
    plan = dict(
        smoke_only=False,
        cities=study.CITIES,
        panels=["mission_stop", "fixed_budget", "swap"],
        agents_grid=[4, 8, 16, 32] if profile == "full" else [8],
        dropout_grid=[0.0, 0.03, 0.15, 0.30] if profile == "full" else [0.03],
        episodes=20,
        zones=["north_east", "south_west"],
        swap_seeds=[2042, 3051, 4060],
        seed=24102026,
    )
    blocks, zones, identities = {}, Counter(), set()
    for job in study.jobs(plan):
        key = tuple(job[k] for k in study.KEYS if k != "strategy")
        blocks.setdefault(key, set()).add(job["strategy"])
        identities.add(study.identity(job))
        zones[job["start_zone"]] += 1
        assert job["seed"] == plan["seed"] + study.CITIES.index(job["city"]) * 10000 + job["episode"]
    assert len(identities) == expected
    assert set(zones.values()) == {expected // 2}
    assert all(methods == set(study.METHODS) for methods in blocks.values())
    assert len(study.ORIGINAL) == 26 and len(study.METHODS) == 30


def test_checkpoint_validation_fails_closed(tmp_path):
    cp = tmp_path / "models"
    with pytest.raises(FileNotFoundError, match="Missing original trained"):
        study.check_checkpoints(cp)
    study.create_smoke_checkpoints(cp)
    study.check_checkpoints(cp, smoke=True)
    with pytest.raises(ValueError, match="Test-only"):
        study.check_checkpoints(cp)
    path = cp / "grpo_safe.json"
    payload = json.loads(path.read_text())
    payload["behavior_weights"] = [[0.0]]
    study.atomic_json(path, payload)
    with pytest.raises(ValueError, match="behavior matrix"):
        study.check_checkpoints(cp, smoke=True)


def test_missing_inputs_do_not_create_a_study(tmp_path):
    root = tmp_path / "output"
    args = study.parser().parse_args(
        ["preflight", "--model-dir", str(tmp_path / "absent"), "--output-root", str(root)]
    )
    with pytest.raises(FileNotFoundError):
        study.freeze(args)
    assert not root.exists()


def environment():
    layers = study.synthetic_city_layers(12, 43, "fixture")
    return study.PairedEnvironment(
        12,
        4,
        37,
        layers=layers,
        max_steps=8,
        communication_dropout_prob=0.30,
        stop_when_all_missions_found=False,
    )


def test_communication_draws_do_not_shift_when_an_agent_finishes():
    a, b = environment(), environment()
    b.agents[0].done = True
    for step in range(1, 9):
        a.steps = b.steps = step
        a.update_communications()
        b.update_communications()
        for aid in [1, 2, 3]:
            assert a.agents[aid].communication_status == b.agents[aid].communication_status
            assert a.agents[aid].comm_loss_steps == b.agents[aid].comm_loss_steps
    assert np.array_equal(a.communication_draws, b.communication_draws)


def test_observation_view_hides_targets_and_keeps_live_observations():
    env = environment()
    view = study.ObservationView(env)
    env.mission_zones.add((11, 11))
    env.priority_map[11, 11] = 1000
    assert view.priority_map[11, 11] != 1000
    assert view.mission_zones == env.discovered_missions
    assert not view.remaining_missions()
    assert view.weighted_target_discovery() == 0
    env.observation_map[1, 1] = 0.75
    assert view.priority_map[1, 1] == 0.75
    with pytest.raises(AttributeError, match="Privileged"):
        view.step({})


def test_maps_cannot_silently_fall_back_or_project_targets():
    layers = study.synthetic_city_layers(12, 43, "fixture")
    with pytest.raises(ValueError, match="genuine OSM"):
        study.validate_layers(layers, 12, "fixture", False)
    cell = next(iter(layers["mission_zones"]))
    layers["obstacles"].add(cell)
    with pytest.raises(ValueError, match="overlaps"):
        study.validate_layers(layers, 12, "fixture", True)


def test_all_policies_preserve_actions_and_memory_under_view(tmp_path):
    root = tmp_path / "study"
    plan = study.freeze(arguments(root))
    audit = study.contract_audit(plan, root / "frozen")
    assert len(audit) == 30
    assert all(row["native_observable_action_parity"] for row in audit)
    policy = study.make("GRPO-Safe-Ablation-NoPropagation", 778, root / "frozen")
    wrapper = study.ObservablePolicy(policy)
    env = environment()
    for _ in range(8):
        env.step(wrapper.act(env))
    assert policy.memory_updates == 8


def test_frozen_inputs_source_and_profile_cannot_change(tmp_path, monkeypatch):
    root = tmp_path / "study"
    study.freeze(arguments(root))
    args = arguments(root)
    args.profile = "full"
    with pytest.raises(ValueError, match="Resume profile"):
        study.freeze(args)
    monkeypatch.setattr(study, "source_hashes", lambda: {})
    with pytest.raises(ValueError, match="Code, dependencies"):
        study.verify(root)
    study.verify(root, execution=False)
    (root / "frozen/candidate.json").write_text("{}")
    with pytest.raises(ValueError, match="Frozen artifact"):
        study.verify(root, execution=False)


def test_output_cannot_overwrite_an_input_directory(tmp_path):
    args = arguments(tmp_path / "models" / "output")
    args.model_dir = str(tmp_path / "models")
    with pytest.raises(ValueError, match="separate"):
        study.freeze(args)


@pytest.fixture(scope="module")
def completed_smoke(tmp_path_factory):
    root = tmp_path_factory.mktemp("eight-city") / "smoke"
    study.run(arguments(root, "--max-jobs", "7"))
    assert json.loads((root / "progress.json").read_text())["rows"] == 7
    with pytest.raises(ValueError, match="Incomplete study"):
        report(root)
    assert not (root / "completion.json").exists()
    assert not (root / "report").exists()
    with sqlite3.connect(root / "episodes.sqlite") as db:
        first_rows = db.execute("SELECT * FROM episodes").fetchall()
    study.run(arguments(root, "--max-jobs", "11"))
    assert json.loads((root / "progress.json").read_text())["rows"] == 18
    study.run(arguments(root))
    with sqlite3.connect(root / "episodes.sqlite") as db:
        assert db.execute("SELECT * FROM episodes LIMIT 7").fetchall() == first_rows
    return root


def test_resume_and_complete_report(completed_smoke):
    root = completed_smoke
    completion = json.loads((root / "completion.json").read_text())
    assessment = json.loads((root / "report/assessment.json").read_text())
    assert completion["rows"] == completion["expected_rows"] == 180
    assert completion["smoke_only"] and assessment["smoke_only"]
    assert not assessment["publication_acceptance_assessed"]
    assert (root.parent / (root.name + ".zip")).is_file()
    frame = pd.read_csv(root / "episodes.csv")
    assert set(frame.strategy) == set(study.METHODS)
    assert not frame.duplicated(study.KEYS).any()
    summary = pd.read_csv(root / "report/summary.csv")
    assert set(summary.strategy) == set(study.METHODS)
    assert "NOT PAPER RESULTS" in (root / "report/index.html").read_text()
    assert "NOT PAPER RESULTS" in (root / "report/all_methods_original_setting.tex").read_text()


def test_analysis_rejects_missing_pairs_and_unpaired_draws(completed_smoke):
    frame = pd.read_csv(completed_smoke / "episodes.csv")
    plan = study.verify(completed_smoke, execution=False)
    with pytest.raises(ValueError, match="Incomplete matched"):
        analyze(frame.iloc[1:], plan)
    frame.loc[0, "communication_draws_sha256"] = "changed"
    with pytest.raises(ValueError, match="Different communication"):
        analyze(frame, plan)


def test_sparse_swap_city_is_retained_but_not_claimed_as_changed(completed_smoke):
    layers = study.synthetic_city_layers(12, 43, "sparse")
    selected = set(sorted(layers["mission_zones"])[:3])
    layers["mission_zones"] = selected
    layers["priority_cells"] = {c: 1.0 for c in selected}
    signatures = set()
    for seed in [2042, 3051, 4060]:
        view = study.seeded_mission_view(layers, seed)
        assert not view["metadata"]["swap_dataset_changed"]
        assert view["mission_zones"] == selected
        signatures.add(view["metadata"]["swap_signature"])
    assert len(signatures) == 1
    plan = copy.deepcopy(study.verify(completed_smoke, execution=False))
    for v in plan["swap_views"]:
        v["city_eligible_changed_swap"] = False
    summary, _, _, _ = analyze(pd.read_csv(completed_smoke / "episodes.csv"), plan)
    assert "swap" in set(summary.analysis_panel)
    assert "swap_changed" not in set(summary.analysis_panel)


def test_journal_tampering_is_detected(completed_smoke, tmp_path):
    plan = study.verify(completed_smoke, execution=False)
    with sqlite3.connect(completed_smoke / "episodes.sqlite") as source:
        with sqlite3.connect(tmp_path / "corrupt.sqlite") as db:
            source.backup(db)
            db.execute("UPDATE episodes SET sha256='tampered' WHERE rowid=1")
            with pytest.raises(ValueError, match="Corrupt or foreign"):
                study.read_ledger(db, plan)
