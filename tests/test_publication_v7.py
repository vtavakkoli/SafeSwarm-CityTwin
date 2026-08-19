from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analyze_publication_statistics_v7 import _paired_table
from experiments.score_sensitivity_v7 import BASE_WEIGHTS, _components, _score
from src.agents.ears_ablations_v7 import EARSAblationPolicy
from src.environment.city_twin import CityTwinEnvironment
from src.training.geography import apply_start_zone


def _layers() -> dict:
    return {
        "obstacles": {(7, 7)},
        "restricted_zones": {(8, 7)},
        "mission_zones": {(12, 12), (3, 12)},
        "base_stations": {(1, 1)},
        "priority_cells": {(12, 12): 1.0, (3, 12): 0.7},
        "metadata": {"source": "synthetic", "feature_count": 2, "place_name": "unit-city"},
    }


def _checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "ears.json"
    payload = {
        "format": "safeswarm-ears-v1",
        "algorithm": "EARS: Event-driven Ant Reallocation Search",
        "ears_config": {},
        "metadata": {"selection": "validation-only"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_publication_protocol_has_eight_frozen_test_cities():
    protocol = json.loads(Path("configs/publication_protocol_v7.json").read_text(encoding="utf-8"))
    test = [city for city in protocol["cities"] if city["split"] == "test"]
    assert len(test) == 8
    assert protocol["publication_contract"]["post_selection_only"] is True
    assert protocol["publication_contract"]["test_results_may_not_change_models"] is True


def test_ears_ablation_modes_act_without_hidden_target_access(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    for mode in ("stagnation_only", "revisit_only", "congestion_only", "no_energy_battery"):
        env = CityTwinEnvironment(
            grid_size=15, n_agents=3, seed=4,
            layers=apply_start_zone(_layers(), 15, "north_west"),
            max_steps=20, allow_network=False, communication_dropout_prob=0.0,
        )
        policy = EARSAblationPolicy(seed=3, model_path=checkpoint, ablation=mode)
        actions = policy.act(env)
        assert set(actions) == set(env.agents)
        assert policy.diagnostics()["ears_ablation"] == mode


def test_paired_statistics_recover_positive_known_delta():
    rows = []
    for city in ("A", "B"):
        for episode in range(4):
            for strategy, score in (("EARS-Safe", 0.80 + 0.01 * episode), ("AntSwarmSafe", 0.75 + 0.01 * episode)):
                rows.append(
                    {
                        "city": city, "episode": episode, "start_zone": "north_east",
                        "strategy": strategy, "operational_score": score,
                        "mission_success_rate": score, "weighted_target_discovery": score,
                        "coverage_ratio": score, "energy_consumption": 400.0,
                        "redundant_coverage": 0.2, "distance_travelled": 700.0,
                        "runtime_seconds": 1.0, "actual_safety_incidents": 0.0,
                    }
                )
    table, _ = _paired_table(
        pd.DataFrame(rows), candidate="EARS-Safe", baseline="AntSwarmSafe", swap=False,
        n_boot=200, n_perm=500, rng=np.random.default_rng(7),
    )
    op = table[table["metric"] == "operational_score"].iloc[0]
    assert np.isclose(op["paired_delta_mean"], 0.05)
    assert op["paired_bootstrap_ci95_low"] > 0.0


def test_baseline_weight_sensitivity_reproduces_component_score():
    frame = pd.DataFrame(
        [
            {
                "city": "A", "strategy": "EARS-Safe", "weighted_target_discovery": 0.9,
                "coverage_ratio": 0.5, "actual_safety_incidents": 0, "steps": 100, "agents": 8,
                "energy_consumption": 400.0, "redundant_coverage": 0.2, "communication_efficiency": 0.97,
            }
        ]
    )
    component = _components(frame)
    score = float(_score(component, BASE_WEIGHTS).iloc[0])
    assert 0.0 <= score <= 1.0
    assert score > 0.7
