"""Publication integrity tests. All generated experiments here are synthetic fixtures."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments import publication_ears_v9 as study
from experiments import report_publication_v9 as reporting


def args(root):
    return SimpleNamespace(
        output_root=str(root),
        checkpoint=str(root / "missing.json"),
        cache_dir=str(root / "cache"),
        offline=True,
    )


def test_full_plan_is_balanced_and_unique():
    spec = study.study_spec()
    jobs = list(study.jobs(spec))
    assert len(jobs) == 16800
    assert len({study.key(j) for j in jobs}) == len(jobs)
    f = pd.DataFrame(jobs)
    assert f.groupby("panel").size().to_dict() == {"fixed_budget": 7680, "mission_stop": 7680, "swap": 1440}
    pairs = [k for k in study.KEYS if k != "strategy"]
    assert (f.groupby(pairs).strategy.nunique() == 4).all()
    assert (f.groupby(pairs).seed.nunique() == 1).all()
    assert set(f.city) == {"Zurich", "Madrid", "Lisbon", "Stockholm", "Helsinki", "Copenhagen"}


def test_holm_and_small_city_exact_resolution():
    assert np.allclose(reporting.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert reporting.city_test([0.1] * 6)["p_two_sided"] == 2 / 64
    assert reporting.city_test([0.0] * 6)["p_two_sided"] == 1


def test_real_data_gate_rejects_legacy_and_synthetic_targets(monkeypatch):
    for metadata in (
        {"source": "synthetic"},
        {"source": "openstreetmap"},
        {"source": "openstreetmap", "mission_source": "synthetic_fallback"},
    ):
        monkeypatch.setattr(
            study, "load_real_city_layers", lambda *a, meta=metadata, **kw: {"metadata": meta}
        )
        with pytest.raises(ValueError, match="Verified OSM"):
            study.real_layers({"name": "fixture", "place": "fixture"}, 16, "none", True)


def test_freeze_refuses_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing checkpoint"):
        study.freeze(args(tmp_path))
    assert not (tmp_path / "frozen/study.json").exists()


@pytest.mark.parametrize("target", ["ears_safe.json", "map-0.json", "study.json"])
def test_frozen_artifact_edits_are_detected(tmp_path, target):
    study.freeze(args(tmp_path), smoke=True)
    p = tmp_path / "frozen" / target
    p.write_text(p.read_text() + " ")
    with pytest.raises(ValueError, match="modified"):
        study.verify(tmp_path)


def test_changed_source_is_rejected(tmp_path, monkeypatch):
    study.freeze(args(tmp_path), smoke=True)
    monkeypatch.setattr(study, "sources", lambda: {"edited.py": "different"})
    with pytest.raises(ValueError, match="Source changed"):
        study.verify(tmp_path)


def test_interruption_resume_exactly_once_and_complete_report(tmp_path, monkeypatch):
    a = args(tmp_path)
    spec = study.freeze(a, smoke=True)
    observe = study.observe
    count = 0

    def interrupt(*a, **kw):
        nonlocal count
        count += 1
        if count == 6:
            raise KeyboardInterrupt()
        return observe(*a, **kw)

    monkeypatch.setattr(study, "observe", interrupt)
    with pytest.raises(KeyboardInterrupt):
        study.evaluate(a)
    assert study.read_json(tmp_path / "manifest.json")["completed_rows"] == 5
    with pytest.raises(ValueError, match="full study"):
        reporting.report(tmp_path)
    monkeypatch.setattr(study, "observe", observe)
    study.evaluate(a)
    f = pd.read_csv(tmp_path / "episodes.csv")
    assert len(f) == spec["expected_rows"] == 56
    assert not f.duplicated(study.KEYS).any()
    # A completed rerun must not execute any episode or change its measurements.
    monkeypatch.setattr(study, "observe", lambda *a, **kw: pytest.fail("Reran completed episode"))
    original = (tmp_path / "episodes.csv").read_bytes()
    study.evaluate(a)
    assert (tmp_path / "episodes.csv").read_bytes() == original
    reporting.report(tmp_path)
    conclusion = study.read_json(tmp_path / "report/conclusion.json")
    assert conclusion["smoke_only"] and not conclusion["primary_average_advantage_supported"]
    assert (tmp_path / "report/report.pdf").read_bytes().startswith(b"%PDF")
    assert "SYNTHETIC" in (tmp_path / "report/report.md").read_text()
    stats = pd.read_csv(tmp_path / "report/paired_tests.csv")
    assert stats.primary.sum() == 1
    assert stats.p_holm.notna().sum() == 89
    failed = f.completed_all_targets == 0
    assert (f.loc[failed, "capped_completion_steps"] == spec["max_steps"]).all()


def test_journal_tampering_is_rejected(tmp_path):
    spec = study.freeze(args(tmp_path), smoke=True)
    job = next(study.jobs(spec))
    db = study.ledger(tmp_path)
    with db:
        db.execute("INSERT INTO episodes VALUES (?,?,?)", (study.key(job), json.dumps(job), "bad-digest"))
    with pytest.raises(ValueError, match="Corrupt"):
        study.read_rows(db, spec)
    db.close()


def test_completion_observation_and_fixed_budget_preserve_metric_definition(tmp_path):
    layers = {
        "obstacles": set(),
        "restricted_zones": set(),
        "mission_zones": {(0, 0)},
        "base_stations": {(0, 0)},
        "priority_cells": {(0, 0): 1.0},
        "metadata": {"source": "synthetic", "test_fixture_only": True},
    }
    checkpoint = tmp_path / "ears.json"
    study.EARSPolicy().save_checkpoint(checkpoint)
    spec = study.study_spec(smoke=True)
    for panel in ("mission_stop", "fixed_budget"):
        job = dict(
            panel=panel,
            city="fixture",
            agents=1,
            dropout=0.0,
            dataset_seed=0,
            episode=0,
            strategy=study.CANDIDATE,
            start_zone="fixture",
            seed=5,
        )
        row = study.observe(layers, spec, job, checkpoint)
        assert row["completed_all_targets"] == 1
        assert row["completion_step"] == 1
        assert row["capped_completion_steps"] == 1
        assert row["steps"] == (1 if panel == "mission_stop" else spec["max_steps"])
        assert row["operational_score"] == study.episode_operational_score(row)


def test_historical_validation_is_not_rewritten_as_promotion():
    selection = study.read_json(study.EVIDENCE / "selected_model.json")
    assert selection["selected_method"] == study.BASELINE
    assert not any(d["eligible"] for d in selection["decisions"])
    text = str(reporting.historical_sections())
    assert "Independent six-city confirmation has not been supplied" in text
    assert "failed" in text


def test_training_uses_only_train_validation_and_matched_seeds(tmp_path, monkeypatch):
    from src.evaluation.metrics import EpisodeMetrics

    visited = []

    def fixture(city, grid, cache, offline):
        visited.append(city["name"])
        return study.synthetic_city_layers(grid, 9, "synthetic training fixture")

    def metrics(env, policy):
        # Explicit synthetic test fixture: no real-data evidence is produced.
        fields = EpisodeMetrics.__dataclass_fields__
        return EpisodeMetrics(**{name: 0 for name in fields})

    monkeypatch.setattr(study, "real_layers", fixture)
    monkeypatch.setattr(study, "quality_check", lambda *a: None)
    monkeypatch.setattr(study, "run_episode", metrics)
    study.training(args(tmp_path))
    assert set(visited) == {"Vienna", "London", "Berlin", "Amsterdam", "Prague"}
    f = pd.read_csv(tmp_path / "training/episodes.csv")
    assert len(f) == 176
    assert (f.groupby(["split", "city", "episode"]).seed.nunique() == 1).all()
    checkpoint = study.read_json(tmp_path / "training/checkpoints/ears_safe.json")
    assert checkpoint["metadata"]["test_cities_used"] == []
    with pytest.raises(FileExistsError):
        study.training(args(tmp_path))


def test_dependency_change_blocks_evaluation(tmp_path, monkeypatch):
    study.freeze(args(tmp_path), smoke=True)
    monkeypatch.setattr(study, "software", lambda: {"python": "changed"})
    with pytest.raises(ValueError, match="Dependencies changed"):
        study.evaluate(args(tmp_path))
