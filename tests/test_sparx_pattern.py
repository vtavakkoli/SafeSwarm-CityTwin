from __future__ import annotations

import json

import numpy as np

from src.agents.sparx_pattern import SPARXPolicy
from src.environment.city_twin import CityTwinEnvironment
from src.training.geography import apply_start_zone, load_protocol, validate_protocol
from src.training.validation_selection import validation_selection_stats


def _layers(grid_size: int = 15, mission: tuple[int, int] = (12, 12)) -> dict:
    return {
        "obstacles": {(7, 7), (7, 8)},
        "restricted_zones": {(8, 7)},
        "mission_zones": {mission, (3, 12)},
        "base_stations": {(1, 1)},
        "priority_cells": {mission: 1.0, (3, 12): 0.7},
        "metadata": {"source": "synthetic", "feature_count": 2},
    }


def _env(seed: int = 4, mission: tuple[int, int] = (12, 12), agents: int = 4) -> CityTwinEnvironment:
    return CityTwinEnvironment(
        grid_size=15,
        n_agents=agents,
        seed=seed,
        layers=apply_start_zone(_layers(mission=mission), 15, "north_west"),
        max_steps=20,
        allow_network=False,
        communication_dropout_prob=0.0,
    )


def test_v4_protocol_has_disjoint_multi_domain_validation():
    protocol = load_protocol("configs/real_city_protocol.json")
    checks = validate_protocol(protocol)
    assert checks["city_splits_disjoint"]
    assert checks["validation_start_zones_unseen_during_training"]
    assert checks["validation_start_zones_disjoint_from_test"]
    assert checks["test_start_zones_unseen_during_training"]
    validation_cities = {c["name"] for c in protocol["cities"] if c["split"] == "validation"}
    assert len(validation_cities) >= 2
    assert len(protocol["start_zones"]["validation"]) >= 4


def test_probability_map_is_normalized_and_blocks_unsafe_cells():
    env = _env()
    policy = SPARXPolicy(seed=9, pattern_mode="star")
    policy._update_probability_state(env)
    probability = policy.search_probability_map
    assert probability is not None
    assert np.isclose(float(probability.sum()), 1.0)
    for cell in env.obstacles | env.restricted_zones:
        assert probability[cell] == 0.0
    assert float(probability.max()) > 0.0


def test_pattern_geometry_x_plus_and_star():
    env = _env()
    region = [
        (x, y)
        for x in range(env.grid_size)
        for y in range(env.grid_size)
        if (x, y) not in env.obstacles | env.restricted_zones
    ]
    anchor = (5, 5)
    x_cells = set(SPARXPolicy(seed=1, pattern_mode="x")._pattern_cells(env, anchor, region))
    plus_cells = set(SPARXPolicy(seed=1, pattern_mode="plus")._pattern_cells(env, anchor, region))
    star_cells = set(SPARXPolicy(seed=1, pattern_mode="star")._pattern_cells(env, anchor, region))
    assert (6, 6) in x_cells and (6, 5) not in x_cells
    assert (6, 5) in plus_cells and (6, 6) not in plus_cells
    assert x_cells <= star_cells
    assert plus_cells <= star_cells


def test_spatial_assignment_spreads_agents_across_regions():
    env = _env(agents=4)
    policy = SPARXPolicy(seed=3, pattern_mode="star")
    policy._update_probability_state(env)
    segments = policy._segments(env)
    policy._refresh_assignments(env, segments)
    assigned = [policy._assignments[aid] for aid in sorted(policy._assignments)]
    assert len(assigned) == 4
    assert len(set(assigned)) == 4


def test_unsensed_hidden_target_does_not_change_initial_action():
    env_a = _env(seed=11, mission=(12, 12))
    env_b = _env(seed=11, mission=(11, 11))
    policy_a = SPARXPolicy(seed=17, pattern_mode="star")
    policy_b = SPARXPolicy(seed=17, pattern_mode="star")
    assert np.array_equal(env_a.observation_map, env_b.observation_map)
    assert np.array_equal(env_a.uncertainty_map, env_b.uncertainty_map)
    assert policy_a.act(env_a) == policy_b.act(env_b)


def test_sparx_checkpoint_roundtrip(tmp_path):
    checkpoint = tmp_path / "sparx.json"
    policy = SPARXPolicy(seed=5, pattern_mode="plus", frontier_weight=2.1, memory_weight=1.4)
    policy.save_checkpoint(checkpoint, {"selected_pattern": "plus", "validation_score": 0.61})
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["format"] == "safeswarm-sparx-v2"
    restored = SPARXPolicy(seed=8, model_path=checkpoint, strategy_name="SPARX-Safe")
    assert restored.pattern_mode == "plus"
    assert np.isclose(restored.config.frontier_weight, 2.1)
    assert np.isclose(restored.config.memory_weight, 1.4)
    assert restored.checkpoint_metadata["selected_pattern"] == "plus"


def test_robust_validation_penalizes_single_domain_overfit():
    stable = [
        {"city": "A", "start_zone": "west", "operational_score": 0.60},
        {"city": "A", "start_zone": "east", "operational_score": 0.61},
        {"city": "B", "start_zone": "west", "operational_score": 0.59},
        {"city": "B", "start_zone": "east", "operational_score": 0.60},
    ]
    unstable = [
        {"city": "A", "start_zone": "west", "operational_score": 0.84},
        {"city": "A", "start_zone": "east", "operational_score": 0.82},
        {"city": "B", "start_zone": "west", "operational_score": 0.37},
        {"city": "B", "start_zone": "east", "operational_score": 0.37},
    ]
    stable_stats = validation_selection_stats(stable)
    unstable_stats = validation_selection_stats(unstable)
    assert np.isclose(unstable_stats["mean_score"], stable_stats["mean_score"])
    assert stable_stats["robust_score"] > unstable_stats["robust_score"]
