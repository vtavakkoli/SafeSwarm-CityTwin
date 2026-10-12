"""Selection integrity and durable-output tests using explicit synthetic fixtures."""

import json

import pandas as pd
import pytest

from experiments import benchmark_ears_v9 as benchmark


def fixture_frame(delta=0.01):
    rows = []
    for city in ("fixture-A", "fixture-B"):
        for agents in (4, 16):
            for episode in range(4):
                for method in (benchmark.BASELINE, "AntSwarmSafe", *benchmark.CANDIDATES):
                    candidate = method in benchmark.CANDIDATES
                    rows.append(
                        {
                            "city": city,
                            "agents": agents,
                            "dropout": 0.03,
                            "episode": episode,
                            "start_zone": str(episode),
                            "strategy": method,
                            "operational_score": 0.75 + (delta if candidate else 0),
                            "weighted_target_discovery": 0.8,
                            "runtime_seconds": 0.1 if candidate else 1,
                            "actual_safety_incidents": 0,
                            "battery_failures": 0,
                            "blocked_moves": 0,
                        }
                    )
    return pd.DataFrame(rows)


def test_selection_keeps_v7_when_candidates_fail():
    selected, decisions = benchmark.choose_candidate(fixture_frame(delta=-0.01))
    assert selected == benchmark.BASELINE
    assert not any(d["eligible"] for d in decisions)


def test_selection_accepts_a_better_safe_candidate_but_rejects_hidden_stratum_failure():
    frame = fixture_frame()
    assert benchmark.choose_candidate(frame)[0] in benchmark.CANDIDATES
    mask = (frame.city == "fixture-B") & (frame.agents == 16) & frame.strategy.isin(benchmark.CANDIDATES)
    frame.loc[mask, "operational_score"] = 0.73
    # Overall improvement is still positive, but this stratum must block promotion.
    selected, decisions = benchmark.choose_candidate(frame)
    assert selected == benchmark.BASELINE
    assert all("city/setting regression exceeds limit" in d["reasons"] for d in decisions)


def test_selection_blocks_battery_failures_even_if_composite_score_is_high():
    frame = fixture_frame()
    frame.loc[frame.strategy.isin(benchmark.CANDIDATES), "battery_failures"] = 1
    assert benchmark.choose_candidate(frame)[0] == benchmark.BASELINE


def test_selection_requires_all_matched_episodes_and_rejects_duplicates():
    frame = fixture_frame()
    with pytest.raises(ValueError, match="paired"):
        benchmark.choose_candidate(frame.iloc[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        benchmark.choose_candidate(pd.concat([frame, frame.iloc[:1]]))


def test_fallback_evaluation_still_reports_v7_against_ant():
    frame = fixture_frame()
    frame = frame[frame.strategy.isin([benchmark.BASELINE, "AntSwarmSafe"])]
    comparisons = benchmark.paired_comparisons(frame)
    assert len(comparisons) == 2
    assert set(comparisons.method) == {benchmark.BASELINE}
    assert set(comparisons.baseline) == {"AntSwarmSafe"}


def test_smoke_saves_complete_raw_records_and_cannot_create_selected_model(tmp_path):
    output = tmp_path / "smoke"
    args = benchmark.parser().parse_args(
        [
            "--mode",
            "smoke",
            "--offline",
            "--agents-grid",
            "2",
            "--dropout-grid",
            "0",
            "--episodes",
            "1",
            "--grid-size",
            "10",
            "--max-steps",
            "3",
            "--model-dir",
            str(tmp_path / "missing"),
            "--output-root",
            str(output),
        ]
    )
    benchmark.run(args)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["completed_rows"] == manifest["expected_rows"] == 6
    assert len(pd.read_csv(output / "episodes.csv")) == 6
    assert not (output / "selected_model.json").exists()
    before = (output / "episodes.csv").read_bytes()
    with pytest.raises(FileExistsError):
        benchmark.run(args)
    assert (output / "episodes.csv").read_bytes() == before


def test_interruption_leaves_flushed_rows_and_failed_manifest(tmp_path, monkeypatch):
    output = tmp_path / "interrupted"
    original = benchmark.run_episode
    calls = 0

    def interrupt(env, policy):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return original(env, policy)

    monkeypatch.setattr(benchmark, "run_episode", interrupt)
    args = benchmark.parser().parse_args(
        [
            "--mode",
            "smoke",
            "--offline",
            "--agents-grid",
            "2",
            "--dropout-grid",
            "0",
            "--episodes",
            "1",
            "--grid-size",
            "10",
            "--max-steps",
            "3",
            "--model-dir",
            str(tmp_path / "missing"),
            "--output-root",
            str(output),
        ]
    )
    with pytest.raises(KeyboardInterrupt):
        benchmark.run(args)
    assert len(pd.read_csv(output / "episodes.csv")) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "interrupted" and manifest["completed_rows"] == 1


def test_confirmation_requires_frozen_selection_and_selection_requires_all_starts():
    with pytest.raises(ValueError, match="Freeze"):
        benchmark.run(benchmark.parser().parse_args(["--mode", "confirm"]))
    with pytest.raises(ValueError, match="four validation starts"):
        benchmark.run(benchmark.parser().parse_args(["--mode", "select", "--episodes", "2"]))
    with pytest.raises(ValueError, match="balanced"):
        benchmark.run(benchmark.parser().parse_args(["--mode", "select", "--episodes", "5"]))


def test_select_to_frozen_evaluation_and_code_change_rejection(tmp_path, monkeypatch):
    # Explicit mock of the OSM loader to exercise the real-data control flow.
    # These temporary fixture results are NOT scientific evidence or real OSM data.
    from src.agents.ears_v6 import EARSPolicy
    from src.environment.obstacles import synthetic_city_layers

    def fixture_loader(place, grid, seed, **kwargs):
        layers = synthetic_city_layers(grid, seed, place)
        layers["metadata"].update(source="openstreetmap", test_fixture_only=True)
        return layers

    monkeypatch.setattr(benchmark, "load_real_city_layers", fixture_loader)
    models = tmp_path / "models"
    EARSPolicy().save_checkpoint(models / "ears_safe.json", metadata={"test_fixture_only": True})
    common = [
        "--offline",
        "--agents-grid",
        "2",
        "--dropout-grid",
        "0",
        "--grid-size",
        "10",
        "--max-steps",
        "3",
        "--model-dir",
        str(models),
    ]
    select_dir = tmp_path / "select"
    benchmark.run(
        benchmark.parser().parse_args(
            common + ["--mode", "select", "--episodes", "4", "--output-root", str(select_dir)]
        )
    )
    selected = select_dir / "selected_model.json"
    frozen = json.loads(selected.read_text())
    assert frozen["checkpoint_sha256"] == benchmark.digest(models / "ears_safe.json")
    assert frozen["source_sha256"] == benchmark.source_fingerprint()
    evaluation = tmp_path / "eval"
    arguments = common + [
        "--mode",
        "development",
        "--selection",
        str(selected),
        "--episodes",
        "1",
        "--output-root",
        str(evaluation),
    ]
    benchmark.run(benchmark.parser().parse_args(arguments))
    result = json.loads((evaluation / "manifest.json").read_text())
    assert result["status"] == "completed"
    assert result["selected_model_sha256"] == benchmark.digest(selected)
    assert result["completed_rows"] == result["expected_rows"]
    frozen["source_sha256"] = {"changed-code.py": "different"}
    selected.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="changed since selection"):
        benchmark.run(benchmark.parser().parse_args(arguments))
