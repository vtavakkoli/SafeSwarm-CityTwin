from __future__ import annotations

from src.agents.bio_swarm_agents import AntSwarmPolicy
from src.environment.city_twin import CityTwinEnvironment
from src.safety.runtime_monitor import RuntimeSafetyMonitor
from src.training.policy_learning import (
    agent_credit_snapshot,
    agent_step_rewards,
    dense_reward,
    reward_state,
)


def _layers(mission=(10, 10)) -> dict:
    return {
        "obstacles": set(),
        "restricted_zones": set(),
        "mission_zones": {mission},
        "base_stations": {(1, 1)},
        "priority_cells": {mission: 1.0},
        "metadata": {"source": "synthetic", "feature_count": 1},
    }


def test_ant_policy_cannot_see_hidden_mission_location():
    env_a = CityTwinEnvironment(
        grid_size=12, n_agents=2, seed=1, layers=_layers((10, 10)),
        max_steps=10, communication_dropout_prob=0.0,
    )
    env_b = CityTwinEnvironment(
        grid_size=12, n_agents=2, seed=1, layers=_layers((9, 10)),
        max_steps=10, communication_dropout_prob=0.0,
    )
    # Both missions are outside the initial sensor footprint, so the observable
    # state is identical. A fair baseline must therefore choose the same actions.
    assert env_a.observation_map.sum() == 0.0
    assert env_b.observation_map.sum() == 0.0
    assert AntSwarmPolicy().act(env_a) == AntSwarmPolicy().act(env_b)


def test_low_battery_agent_can_park_safely_at_base():
    env = CityTwinEnvironment(
        grid_size=12, n_agents=1, seed=2, layers=_layers(),
        max_steps=160, communication_dropout_prob=0.0,
    )
    state = env.agents[0]
    state.position = (1, 1)
    state.battery_level = 10.0
    info = env.step({0: "STAY"})
    assert state.done
    assert state.safe_returned
    assert state.battery_level > 0.0
    assert info["safe_returns"] == 1.0
    assert info["done"] == 1.0


def test_return_guard_restricts_low_battery_motion_toward_base():
    env = CityTwinEnvironment(
        grid_size=12, n_agents=1, seed=3, layers=_layers(),
        max_steps=160, communication_dropout_prob=0.0,
    )
    env.agents[0].position = (5, 1)
    env.agents[0].battery_level = 18.0
    monitor = RuntimeSafetyMonitor()
    actions = monitor.safe_actions(env, 0, {}, count_masked=True)
    assert actions
    current_distance = env.nearest_base_distance(env.agents[0].position)
    assert all(
        env.nearest_base_distance(env.next_position(env.agents[0].position, action))
        < current_distance
        for action in actions
    )
    assert monitor.mask_rule_counts.get("return_guard", 0) > 0


def test_mission_completion_ends_episode_before_energy_exhaustion():
    env = CityTwinEnvironment(
        grid_size=12, n_agents=1, seed=4, layers=_layers((3, 1)),
        max_steps=160, communication_dropout_prob=0.0,
    )
    assert not env.discovered_missions
    info = env.step({0: "RIGHT"})
    assert (3, 1) in env.discovered_missions
    assert info["done"] == 1.0
    assert env.agents[0].battery_level > 90.0


def test_agent_credit_is_not_broadcast_identically():
    env = CityTwinEnvironment(
        grid_size=12, n_agents=2, seed=5, layers=_layers(),
        max_steps=20, communication_dropout_prob=0.0,
    )
    before_credit = agent_credit_snapshot(env)
    before_team = reward_state(env)
    env.step({0: "RIGHT", 1: "STAY"})
    after_team = reward_state(env)
    team = dense_reward(after_team, before_team)
    rewards = agent_step_rewards(env, before_credit, team)
    assert rewards[0] != rewards[1]
