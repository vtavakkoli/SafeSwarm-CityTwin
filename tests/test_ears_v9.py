"""Behavioral contracts for information routing; no network/checkpoints needed."""

from dataclasses import replace

import numpy as np
import pytest

from src.agents.ears_v9 import EARSInformationPolicy, InformationRoutingConfig
from src.environment.city_twin import CityTwinEnvironment


def make_env(*, obstacles=(), targets=((8, 8),), agents=1, sensor_radius=1):
    layers = {
        "obstacles": set(obstacles),
        "restricted_zones": set(),
        "mission_zones": set(targets),
        "priority_cells": {c: 1.0 for c in targets},
        "base_stations": {(1, 1)},
        "metadata": {"source": "synthetic"},
    }
    return CityTwinEnvironment(
        grid_size=10,
        n_agents=agents,
        seed=7,
        layers=layers,
        max_steps=80,
        sensor_radius=sensor_radius,
        allow_network=False,
        communication_dropout_prob=0,
    )


def prepare(env, **kwargs):
    policy = EARSInformationPolicy(**kwargs)
    policy._ensure_topology(env)
    policy._gain = policy._information_gain(env)
    return policy


def test_gain_matches_sensor_footprint_and_reservations():
    env = make_env(obstacles=((4, 5),), sensor_radius=2)
    policy = prepare(env)
    reserved = {(5, 4), (5, 5)}
    gain = policy._information_gain(env, reserved)
    for cell in [(0, 0), (5, 5), (9, 9), (1, 1)]:
        expected = (
            sum(env.uncertainty_map[c] >= 0.99 and c not in reserved for c in policy._footprint(env, cell))
            / 13
        )
        assert gain[cell] == pytest.approx(expected)


def test_route_cache_survives_steps_but_invalidates_on_topology_change():
    env = make_env()
    policy = prepare(env)
    field = policy._distance_field(env, (8, 8))
    env.steps += 1
    assert policy._distance_field(env, (8, 8)) is field
    assert policy.bfs_builds == 1
    env.obstacles.update((5, y) for y in range(10))
    assert (1, 1) not in policy._distance_field(env, (8, 8))
    assert policy.bfs_builds == 2


def test_cache_is_bounded_and_reset_clears_episode_memory():
    env = make_env()
    policy = prepare(env, routing_config=InformationRoutingConfig(cache_fields=2))
    for goal in [(4, 4), (5, 5), (6, 6)]:
        policy._distance_field(env, goal)
    assert len(policy._route_cache) == 2
    policy.act(env)
    policy.reset_episode_state()
    assert not policy._history and not policy._route_cache
    assert not policy._relocation_goals and policy.bfs_builds == 0


def test_route_progress_beats_large_visit_penalty():
    env = make_env()
    env.agents[0].position = (3, 3)
    env.visit_counts[4, 3] = 100
    policy = prepare(env)
    action = policy._goal_action(env, 0, (6, 3), {})
    assert env.next_position((3, 3), action) == (4, 3)


def test_return_uses_graph_distance_and_reachable_base():
    wall = {(3, y) for y in range(9)}
    env = make_env(obstacles=wall)
    env.agents[0].position = (4, 1)
    env.agents[0].battery_level = 25
    policy = prepare(env)
    assert env.nearest_base_distance((4, 1)) == 3
    assert policy._base_distances[(4, 1)] == 19
    assert policy._battery_return_needed(env, 0)


def test_rejected_request_backs_off_and_history_is_every_step():
    env = make_env()
    policy = prepare(env, routing_config=InformationRoutingConfig(min_route_gain=1000))
    env.uncertainty_map.fill(0)
    env.steps = 12
    policy.act(env)
    assert policy.rejected_relocations == 1
    for _ in range(4):
        env.steps += 1
        policy.act(env)
    assert policy.rejected_relocations == 1
    assert len(policy._history_for(0)) == 5


def test_no_hidden_targets_or_city_identity_affect_actions():
    a = make_env(targets=((8, 8),), agents=3)
    b = make_env(targets=((8, 7), (7, 8)), agents=3)
    assert np.array_equal(a.observation_map, b.observation_map)
    assert prepare(a).act(a) == prepare(b).act(b)


def test_expired_reservation_removed_and_no_relocation_ablation():
    env = make_env()
    policy = prepare(env, routing_config=InformationRoutingConfig(enable_relocation=False))
    policy._relocation_goals[0] = (8, 8)
    policy._relocation_until[0] = -1
    policy.act(env)
    assert not policy._relocation_goals
    assert policy.accepted_relocations == 0


def test_active_goal_footprints_are_reserved_between_planning_steps():
    env = make_env(agents=2)
    env.uncertainty_map.fill(0)
    env.uncertainty_map[8, 8] = 1
    policy = prepare(env)
    policy._relocation_goals[0] = (8, 8)
    policy._relocation_until[0] = 50
    policy._select_relocation_goals(env, [1], env.get_positions())
    assert 1 not in policy._relocation_goals


@pytest.mark.parametrize(
    "field,value", [("retry_steps", 0), ("cache_fields", 1.5), ("information_weight", float("nan"))]
)
def test_invalid_configuration_rejected(field, value):
    with pytest.raises(ValueError):
        replace(InformationRoutingConfig(), **{field: value})


def test_multi_agent_episode_has_no_collisions_or_invalid_moves():
    env = make_env(agents=8, obstacles=((5, 4), (5, 5), (5, 6)))
    policy = prepare(env)
    for _ in range(50):
        info = env.step(policy.act(env))
        if info["done"]:
            break
    assert env.actual_collisions == env.actual_restricted_entries == env.blocked_moves == 0
    assert all(a.battery_level > 0 for a in env.agents.values())
