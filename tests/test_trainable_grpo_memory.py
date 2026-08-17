from __future__ import annotations

import json

import numpy as np

from src.agents.grpo_v3 import BEHAVIOR_FEATURE_NAMES
from src.agents.safe_ppo_core import VALUE_FEATURE_NAMES
from src.agents.trainable_policies import TrainableGRPOMemoryPolicy
from src.environment.city_twin import CityTwinEnvironment
from src.safety.runtime_monitor import RuntimeSafetyMonitor
from src.training.geography import apply_start_zone, load_protocol, validate_protocol
from src.training.policy_learning import gae_advantages


def _layers(grid_size: int = 12) -> dict:
    return {
        "obstacles": {(5, 5)},
        "restricted_zones": {(6, 6)},
        "mission_zones": {(9, 9), (2, 9)},
        "base_stations": {(1, 1)},
        "priority_cells": {(9, 9): 1.0, (2, 9): 0.7},
        "metadata": {"source": "synthetic", "feature_count": 2},
    }


def _env(seed: int = 7, agents: int = 3) -> CityTwinEnvironment:
    return CityTwinEnvironment(
        grid_size=12,
        n_agents=agents,
        seed=seed,
        layers=apply_start_zone(_layers(), 12, "north_west"),
        max_steps=8,
        allow_network=False,
        communication_dropout_prob=0.0,
    )


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


def test_safe_mask_makes_logged_action_equal_executed_action():
    env = _env()
    policy = TrainableGRPOMemoryPolicy(seed=7)
    actions = policy.act(env)
    decisions = policy.drain_decisions()

    assert decisions
    assert policy.monitor.intervention_count == 0
    for decision in decisions:
        assert actions[decision["agent_id"]] == decision["selected_action"]

    monitor = RuntimeSafetyMonitor()
    planned = {}
    for agent_id in sorted(env.agents):
        ok, violations, candidate = monitor.is_action_safe(
            env, agent_id, actions[agent_id], planned
        )
        assert ok, violations
        planned[agent_id] = candidate


def test_grpo_memory_and_state_behavior_checkpoint_roundtrip(tmp_path):
    env = _env()
    policy = TrainableGRPOMemoryPolicy(seed=7, propagation_steps=2)
    for _ in range(3):
        env.step(policy.act(env))
        policy.drain_decisions()
    diagnostics = policy.diagnostics()
    assert diagnostics["swarm_memory_updates"] >= 3
    assert diagnostics["swarm_memory_coverage"] > 0.0
    assert diagnostics["swarm_memory_peak"] > 0.0

    policy.residual_weights[:] = np.linspace(-0.2, 0.2, len(policy.residual_weights))
    policy.critic_weights[:] = np.linspace(-0.1, 0.1, len(policy.critic_weights))
    policy.behavior_bias[:] = np.linspace(-0.05, 0.05, len(policy.behavior_bias))
    policy.behavior_weights[:] = np.linspace(
        -0.03, 0.03, policy.behavior_weights.size
    ).reshape(policy.behavior_weights.shape)
    checkpoint = tmp_path / "grpo.json"
    policy.save_checkpoint(checkpoint, {"unit_test": True})
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["format"] == "safeswarm-grpo-memory-ppo-v3"
    assert payload["behavior_feature_names"] == list(BEHAVIOR_FEATURE_NAMES)

    restored = TrainableGRPOMemoryPolicy(seed=9, model_path=checkpoint)
    assert np.allclose(restored.residual_weights, policy.residual_weights)
    assert np.allclose(restored.critic_weights, policy.critic_weights)
    assert np.allclose(restored.behavior_bias, policy.behavior_bias)
    assert np.allclose(restored.behavior_weights, policy.behavior_weights)
    assert restored.propagation_steps == 2


def test_ppo_update_changes_action_and_state_conditioned_behavior_weights():
    policy = TrainableGRPOMemoryPolicy(seed=3)
    features = np.zeros((2, len(policy.residual_weights)), dtype=float)
    features[0, 0] = 1.0
    features[1, 1] = 1.0
    behavior_features = np.zeros(len(BEHAVIOR_FEATURE_NAMES), dtype=float)
    behavior_features[0] = 1.0
    behavior_features[1] = 0.75
    base = {
        "features": features,
        "base_scores": np.asarray([0.0, 0.0]),
        "old_probability": 0.5,
        "value_features": np.zeros(len(VALUE_FEATURE_NAMES), dtype=float),
        "old_value": 0.0,
        "behavior_base_scores": np.zeros(policy.group_size, dtype=float),
        "behavior_features": behavior_features,
        "behavior_old_probability": 1.0 / policy.group_size,
    }
    positive = dict(base, action_index=0, behavior_index=0)
    negative = dict(base, action_index=1, behavior_index=1)
    before_actor = policy.residual_weights.copy()
    before_behavior = policy.behavior_weights.copy()
    policy.ppo_update(
        [(positive, 1.0, 1.0), (negative, -1.0, -1.0)],
        learning_rate=0.05,
        critic_learning_rate=0.02,
        entropy_coef=0.01,
        epochs=1,
    )
    assert not np.allclose(before_actor, policy.residual_weights)
    assert not np.allclose(before_behavior, policy.behavior_weights)


def test_imitation_update_moves_behavior_and_action_parameters():
    env = _env(agents=2)
    policy = TrainableGRPOMemoryPolicy(seed=4)
    policy._update_swarm_memory(env)
    positions = env.get_positions()
    aid = 0
    current = env.agents[aid].position
    candidates = env.get_neighbors(current)
    target = candidates[-1]
    example = policy.imitation_example(env, aid, positions, candidates, target)
    assert example is not None
    before_actor = policy.residual_weights.copy()
    before_behavior = policy.behavior_weights.copy()
    stats = policy.imitation_update([example], learning_rate=0.05, epochs=2)
    assert stats["imitation_examples"] == 1.0
    assert not np.allclose(before_actor, policy.residual_weights)
    assert not np.allclose(before_behavior, policy.behavior_weights)


def test_gae_uses_value_baseline():
    advantages, targets = gae_advantages(
        [0.0, 0.0, 1.0],
        [0.2, 0.3, 0.4],
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert len(advantages) == 3
    assert len(targets) == 3
    assert np.isclose(advantages[-1], 0.6)
    assert np.isclose(targets[-1], 1.0)
