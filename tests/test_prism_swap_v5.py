from __future__ import annotations

import json

import numpy as np

from src.agents.ppo_v3 import TrainableIPPOPolicy
from src.agents.prism_ant import PRISMAntPolicy
from src.agents.registry import strategy_factories
from src.environment.city_twin import CityTwinEnvironment
from src.safety.runtime_monitor import RuntimeSafetyMonitor
from src.training.swap_protocol import seeded_mission_view


def _layers() -> dict:
    missions = {
        (2, 10), (3, 11), (4, 12), (5, 10), (6, 11), (7, 12),
        (9, 2), (10, 3), (11, 4), (12, 5), (10, 7), (12, 8),
    }
    return {
        "obstacles": {(7, 7)},
        "restricted_zones": {(8, 8)},
        "mission_zones": missions,
        "base_stations": {(1, 1)},
        "priority_cells": {cell: 0.7 for cell in missions},
        "metadata": {"source": "synthetic", "place_name": "v5-test-city"},
    }


def _env(seed: int = 7, agents: int = 4) -> CityTwinEnvironment:
    return CityTwinEnvironment(
        grid_size=15,
        n_agents=agents,
        seed=seed,
        layers=_layers(),
        max_steps=20,
        allow_network=False,
        communication_dropout_prob=0.0,
    )


def test_v5_registry_uses_prism_not_sparx():
    names = set(strategy_factories(seed=3))
    assert "PRISM-Ant-Safe" in names
    assert {"PRISM-X-Safe", "PRISM-Plus-Safe", "PRISM-Star-Safe"} <= names
    assert not any(name.startswith("SPARX") for name in names)


def test_prism_ant_actions_are_safe():
    env = _env()
    policy = PRISMAntPolicy(seed=9, pattern_mode="star")
    actions = policy.act(env)
    monitor = RuntimeSafetyMonitor()
    planned = {}
    for aid in sorted(env.agents):
        ok, violations, candidate = monitor.is_action_safe(
            env, aid, actions[aid], planned
        )
        assert ok, violations
        planned[aid] = candidate


def test_prism_ant_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "hybrid.json"
    policy = PRISMAntPolicy(seed=1, pattern_mode="star")
    policy.save_checkpoint(path, {"selected_pattern": "star"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "safeswarm-prism-ant-v1"
    restored = PRISMAntPolicy(seed=2, model_path=path)
    assert restored.pattern_mode == "star"
    assert np.isclose(restored.hybrid_config.ant_blend, policy.hybrid_config.ant_blend)


def test_swap_same_seed_reproducible_and_new_seed_changes_targets():
    base = _layers()
    a = seeded_mission_view(base, 2042, target_fraction=0.5, min_targets=3)
    b = seeded_mission_view(base, 2042, target_fraction=0.5, min_targets=3)
    c = seeded_mission_view(base, 3051, target_fraction=0.5, min_targets=3)
    assert a["mission_zones"] == b["mission_zones"]
    assert a["metadata"]["swap_signature"] == b["metadata"]["swap_signature"]
    assert a["metadata"]["swap_signature"] != c["metadata"]["swap_signature"]
    assert a["mission_zones"] != c["mission_zones"]
    assert a["obstacles"] == c["obstacles"] == base["obstacles"]
    assert a["restricted_zones"] == c["restricted_zones"] == base["restricted_zones"]
    assert a["base_stations"] == c["base_stations"] == base["base_stations"]


def test_ippo_deterministic_evaluation_is_seed_invariant():
    env_a = _env(seed=13)
    env_b = _env(seed=13)
    policy_a = TrainableIPPOPolicy(seed=1, deterministic_eval=True)
    policy_b = TrainableIPPOPolicy(seed=999, deterministic_eval=True)
    assert policy_a.act(env_a) == policy_b.act(env_b)


def test_ippo_global_coordination_assigns_multiple_goals():
    env = _env(agents=4)
    policy = TrainableIPPOPolicy(seed=4, deterministic_eval=True)
    policy.act(env)
    goals = list(policy._coordination_goals.values())
    assert len(goals) == 4
    assert len(set(goals)) >= 2


def test_generic_ippo_teacher_imitation_updates_actor():
    env = _env(agents=2)
    policy = TrainableIPPOPolicy(seed=5, deterministic_eval=False)
    aid = 0
    current = env.agents[aid].position
    positions = env.get_positions()
    candidates = env.get_neighbors(current)
    target = max(candidates, key=lambda cell: env.uncertainty_map[cell])
    example = policy.imitation_example(env, aid, positions, candidates, target)
    assert example is not None
    before = policy.residual_weights.copy()
    stats = policy.imitation_update([example], learning_rate=0.05, epochs=2)
    assert stats["imitation_examples"] == 1.0
    assert not np.allclose(before, policy.residual_weights)
