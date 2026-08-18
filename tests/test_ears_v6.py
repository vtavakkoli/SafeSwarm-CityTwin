from __future__ import annotations

import json

import numpy as np

from src.agents.ears_v6 import (
    EARSConfig,
    EARSNegativePheromonePolicy,
    EARSPolicy,
    HMAPPOEARSPolicy,
)
from src.environment.city_twin import MOVE_ACTIONS, CityTwinEnvironment
from src.training.geography import apply_start_zone


def _layers(mission: tuple[int, int] = (12, 12)) -> dict:
    return {
        "obstacles": {(7, 7), (7, 8)},
        "restricted_zones": {(8, 7)},
        "mission_zones": {mission, (3, 12)},
        "base_stations": {(1, 1)},
        "priority_cells": {mission: 1.0, (3, 12): 0.7},
        "metadata": {"source": "synthetic", "feature_count": 2, "place_name": "unit-city"},
    }


def _env(seed: int = 4, mission: tuple[int, int] = (12, 12), agents: int = 4) -> CityTwinEnvironment:
    return CityTwinEnvironment(
        grid_size=15,
        n_agents=agents,
        seed=seed,
        layers=apply_start_zone(_layers(mission), 15, "north_west"),
        max_steps=30,
        allow_network=False,
        communication_dropout_prob=0.0,
    )


def test_ears_observable_only_initial_action_ignores_hidden_target_location():
    env_a = _env(seed=11, mission=(12, 12))
    env_b = _env(seed=11, mission=(11, 11))
    policy_a = EARSPolicy(seed=17)
    policy_b = EARSPolicy(seed=17)
    assert np.array_equal(env_a.observation_map, env_b.observation_map)
    assert np.array_equal(env_a.uncertainty_map, env_b.uncertainty_map)
    assert policy_a.act(env_a) == policy_b.act(env_b)


def test_stagnation_event_triggers_reallocation_goal():
    env = _env(seed=5, agents=3)
    policy = EARSPolicy(
        seed=2,
        config=EARSConfig(
            history_window=6,
            min_steps_before_reallocation=1,
            stagnation_unique_ratio=0.60,
            local_revisit_trigger=2.0,
            congestion_trigger=99.0,
        ),
    )
    aid = 0
    env.steps = 8
    history = policy._history_for(aid)
    history.extend([env.agents[aid].position] * 6)
    reason = policy._event_reason(env, aid, env.get_positions())
    assert reason == "stagnation"
    policy._select_relocation_goals(env, [aid], env.get_positions())
    assert aid in policy._relocation_goals
    assert policy._relocation_until[aid] > env.steps


def test_negative_pheromone_creates_repulsive_visit_halo():
    env = _env(seed=6)
    env.visit_counts[4, 4] = 12
    policy = EARSNegativePheromonePolicy(seed=3)
    policy._update_negative_field(env)
    field = policy.negative_pheromone
    assert field is not None
    assert float(field[4, 4]) > float(field[10, 10])
    assert float(field[4, 5]) >= 0.0
    assert np.all((field >= 0.0) & (field <= 1.0))


def test_ears_checkpoint_roundtrip(tmp_path):
    checkpoint = tmp_path / "ears.json"
    config = EARSConfig(
        relocation_duration=5,
        global_energy_penalty=0.41,
        negative_pheromone_weight=0.77,
    )
    policy = EARSNegativePheromonePolicy(seed=1, config=config)
    policy.save_checkpoint(checkpoint, {"validation_score": 0.78})
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["format"] == "safeswarm-ears-np-v1"
    restored = EARSNegativePheromonePolicy(seed=2, model_path=checkpoint)
    assert restored.config.relocation_duration == 5
    assert np.isclose(restored.config.global_energy_penalty, 0.41)
    assert np.isclose(restored.config.negative_pheromone_weight, 0.77)
    assert restored.checkpoint_metadata["validation_score"] == 0.78


def test_h_mappo_ears_without_checkpoint_falls_back_to_safe_hierarchy():
    env = _env(seed=7)
    policy = HMAPPOEARSPolicy(seed=4)
    actions = policy.act(env)
    assert set(actions) == set(env.agents)
    assert all(action in MOVE_ACTIONS for action in actions.values())


def test_ears_default_is_ant_dominant_before_events():
    env = _env(seed=8)
    policy = EARSPolicy(seed=9)
    policy.act(env)
    diagnostics = policy.diagnostics()
    assert diagnostics["ears_event_triggers"] == 0
    assert diagnostics["ears_ant_fraction"] > 0.9
    assert diagnostics["ears_relocation_fraction"] == 0.0
