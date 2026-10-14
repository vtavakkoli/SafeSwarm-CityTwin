"""Synthetic fixtures test evidence integrity and controls; not research results."""

import json
import shutil
import sqlite3
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments import publication_ears_v9 as original
from src.environment.city_twin import CityTwinEnvironment
from tools import mechanism_probe_v9 as probe
from tools import publication_readiness_v9 as audit


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("historical-fixture")
    args = SimpleNamespace(output_root=str(root), checkpoint="unused", cache_dir="unused", offline=True)
    original.freeze(args, smoke=True)
    original.evaluate(args)
    return root


def test_read_only_audit_and_score_parity(evidence):
    before = {p.name: original.digest(p) for p in evidence.iterdir() if p.is_file()}
    spec, frame, mismatches, additions = audit.read_evidence(evidence)
    assert len(frame) == spec["expected_rows"] == 56
    assert not mismatches and not additions
    after = {p.name: original.digest(p) for p in evidence.iterdir() if p.is_file()}
    assert before == after


@pytest.mark.parametrize("target", ["episodes.csv", "diagnostics.jsonl", "frozen/map-0.json"])
def test_tampered_exports_rejected(evidence, tmp_path, target):
    root = tmp_path / "copy"
    shutil.copytree(evidence, root)
    path = root / target
    path.write_text(path.read_text() + ("{}\n" if target.endswith("jsonl") else " "))
    with pytest.raises((ValueError, KeyError)):
        audit.read_evidence(root)


def test_incomplete_ledger_rejected(evidence, tmp_path):
    root = tmp_path / "copy"
    shutil.copytree(evidence, root)
    with sqlite3.connect(root / "episodes.sqlite") as db:
        db.execute("DELETE FROM episodes WHERE rowid=(SELECT MAX(rowid) FROM episodes)")
    db.close()
    with pytest.raises(ValueError, match="Incomplete"):
        audit.read_evidence(root)


def test_live_wal_is_rejected(evidence, tmp_path):
    root = tmp_path / "copy"
    shutil.copytree(evidence, root)
    (root / "episodes.sqlite-wal").write_bytes(b"pending transactions")
    with pytest.raises(ValueError, match="checkpoint"):
        audit.read_evidence(root)


def test_source_difference_reported_not_hidden(evidence, monkeypatch):
    current = original.sources()
    current["docker/Dockerfile"] = "changed"
    monkeypatch.setattr(audit, "sources", lambda: current)
    assert audit.read_evidence(evidence)[2] == ["docker/Dockerfile"]


def test_output_cannot_overwrite_evidence(evidence):
    with pytest.raises(ValueError, match="separate"):
        audit.audit(evidence, evidence / "audit")
    with pytest.raises(ValueError, match="separate"):
        probe.run(evidence, evidence, "pilot")


def test_probe_pairing_and_balanced_start_zones():
    plan = dict(
        panels=["mission_stop", "fixed_budget"],
        cities=["A", "B"],
        agents_grid=[4, 8],
        dropout_grid=[0.03],
        episodes=4,
        zones=["north_east", "south_west"],
        seed=7142026,
    )
    jobs = list(probe.planned_jobs(plan))
    frame = pd.DataFrame(jobs)
    assert len(jobs) == len({probe.identity(j) for j in jobs}) == 128
    assert (frame.groupby(probe.PAIR).strategy.nunique() == 4).all()
    assert (frame.groupby(["panel", "city", "agents", "strategy", "start_zone"]).size() == 2).all()


def test_controls_preserve_parameters_and_return_logic():
    routing = asdict(probe.InformationRoutingConfig())
    full = probe.make_policy(probe.FULL, 5, None, routing)
    for method in probe.METHODS:
        policy = probe.make_policy(method, 5, None, routing)
        config = asdict(policy.routing_config)
        assert config.pop("enable_relocation") == (method != probe.LOCAL)
        expected = {k: v for k, v in routing.items() if k != "enable_relocation"}
        assert config == expected
        assert policy.config == full.config
        assert type(policy)._battery_return_needed is type(full)._battery_return_needed
        assert type(policy)._goal_action is type(full)._goal_action


def test_periodic_schedule_honors_offline_backoff_and_warmup():
    policy = probe.PeriodicInformationPolicy()
    start = policy.config.min_steps_before_reallocation
    state = SimpleNamespace(communication_status=True)
    env = SimpleNamespace(steps=start, agents={0: state})
    assert policy._event_reason(env, 0, {}) == "periodic"
    env.steps += 1
    assert policy._event_reason(env, 0, {}) is None
    env.steps = start + 16
    state.communication_status = False
    assert policy._event_reason(env, 0, {}) is None
    state.communication_status = True
    policy._retry_until[0] = env.steps + 1
    assert policy._event_reason(env, 0, {}) is None
    env.steps = start - 1
    assert policy._event_reason(env, 0, {}) is None


def test_sensing_ablation_leaves_collision_reservations_intact(monkeypatch):
    layers = original.synthetic_city_layers(12, 42, "synthetic test only")
    env = CityTwinEnvironment(grid_size=12, n_agents=2, seed=5, layers=layers, allow_network=False)
    full = probe.EARSInformationPolicy()
    control = probe.UnreservedInformationPolicy()
    reserved = {(x, y) for x in range(12) for y in range(12)}
    assert np.allclose(full._information_gain(env, reserved), 0)
    assert np.array_equal(control._information_gain(env, reserved), full._information_gain(env))
    passed = {}

    def capture(self, env, aid, planned, positions, reserved):
        passed.update(planned=planned, reserved=reserved)
        return "STAY"

    monkeypatch.setattr(probe.EARSInformationPolicy, "_information_action", capture)
    planned = {1: (2, 2)}
    control._information_action(env, 0, planned, {}, {(3, 3)})
    assert passed["planned"] is planned
    assert passed["reserved"] == set()


def test_probe_resume_and_artifact_tamper(evidence, tmp_path, monkeypatch):
    spec = original.read_json(evidence / "frozen/study.json")
    spec["smoke_only"] = False  # Internal synthetic fixture; never exported as research evidence.
    spec["agents_grid"] = [2]
    spec["dropout_grid"] = [0.03]
    spec["episodes"] = 2
    monkeypatch.setattr(probe, "read_evidence", lambda _: (spec, None, [], []))
    output = tmp_path / "probe"
    observe = probe.observe
    calls = 0

    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise KeyboardInterrupt()
        return observe(*args)

    monkeypatch.setattr(probe, "observe", interrupted)
    with pytest.raises(KeyboardInterrupt):
        probe.run(evidence, output, "full")
    with sqlite3.connect(output / "episodes.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 3
        saved = dict(db.execute("SELECT id,sha256 FROM episodes"))
    db.close()
    monkeypatch.setattr(probe, "observe", observe)
    probe.run(evidence, output, "full")
    with sqlite3.connect(output / "episodes.sqlite") as db:
        resumed = dict(db.execute("SELECT id,sha256 FROM episodes"))
    db.close()
    assert all(resumed[k] == value for k, value in saved.items())
    assert json.loads((output / "completion.json").read_text())["rows"] == 16
    original_rows = (output / "episodes.csv").read_bytes()
    monkeypatch.setattr(probe, "observe", lambda *a: pytest.fail("Completed row rerun"))
    probe.run(evidence, output, "full")
    assert (output / "episodes.csv").read_bytes() == original_rows
    with pytest.raises(ValueError, match="same profile"):
        probe.run(evidence, output, "pilot")
    path = output / "frozen/ears_safe.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="artifact modified"):
        probe.run(evidence, output, "full")


def test_primary_geographic_resolution_is_not_episode_count():
    result = audit.city_test([0.01, 0.02, 0.03, 0.04])
    assert result["p_two_sided"] == 0.125
    assert result["cities"] == 4
