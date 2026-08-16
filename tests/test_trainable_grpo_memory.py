from __future__ import annotations

import json

import numpy as np

from src.agents.trainable_policies import TrainableGRPOMemoryPolicy
from src.environment.city_twin import CityTwinEnvironment
from src.training.geography import apply_start_zone, load_protocol, validate_protocol


def _layers(grid_size: int = 12) -> dict:
    return {
        "obstacles": {(5, 5)},
        "restricted_zones": {(6, 6)},
        "mission_zones": {(9, 9), (2, 9)},
        "base_stations": {(1, 1)},
        "priority_cells": {(9, 9): 1.0, (2, 9): 0.7},
        "metadata": {"source": "synthetic", "feature_count": 2},
    }


def test_protocol_keeps_test_geography_disjoint():
    protocol = load_protocol("configs/real_city_protocol.json")
    checks = validate_protocol(protocol)
    assert checks["city_splits_disjoint"]
    assert checks["test_start_zones_unseen_during_training"]


def test_geographic_start_zone_moves_bases_safely():
    layers = _layers()
    nw = apply_start_zone(layers, 12, "north_west")
    se = apply_start_zone(layers, 12, "south_east")
    assert nw["base_stations"] != se["base_stations"]
    for cell in nw["base_stations"] | se["base_stations"]:
        assert cell not in layers["obstacles"]
        assert cell not in layers["restricted_zones"]


def test_grpo_memory_propagates_and_checkpoint_roundtrips(tmp_path):
    env = CityTwinEnvironment(
        grid_size=12,
        n_agents=3,
        seed=7,
        layers=apply_start_zone(_layers(), 12, "north_west"),
        max_steps=8,
        allow_network=False,
    )
    policy = TrainableGRPOMemoryPolicy(seed=7, propagation_steps=2)
    for _ in range(3):
        env.step(policy.act(env))
    diagnostics = policy.diagnostics()
    assert diagnostics["swarm_memory_updates"] >= 3
    assert diagnostics["swarm_memory_coverage"] > 0.0

    policy.residual_weights[:] = np.linspace(-0.2, 0.2, len(policy.residual_weights))
    checkpoint = tmp_path / "grpo.json"
    policy.save_checkpoint(checkpoint, {"unit_test": True})
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["format"] == "safeswarm-grpo-memory-ppo-v1"

    restored = TrainableGRPOMemoryPolicy(seed=9, model_path=checkpoint)
    assert np.allclose(restored.residual_weights, policy.residual_weights)
    assert restored.propagation_steps == 2


def test_ppo_update_changes_residual_weights():
    policy = TrainableGRPOMemoryPolicy(seed=3)
    features = np.zeros((2, len(policy.residual_weights)), dtype=float)
    features[0, 0] = 1.0
    features[1, 1] = 1.0
    decision = {
        "features": features,
        "base_scores": np.asarray([0.0, 0.0]),
        "action_index": 0,
        "old_probability": 0.5,
    }
    before = policy.residual_weights.copy()
    policy.ppo_update(
        [(decision, 1.0), (dict(decision, action_index=1), -1.0)], learning_rate=0.1
    )
    assert not np.allclose(before, policy.residual_weights)
