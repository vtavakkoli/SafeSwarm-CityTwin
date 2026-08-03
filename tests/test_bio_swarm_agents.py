import pytest

from src.agents.bio_swarm_agents import (
    AntSwarmPolicy,
    BeeSwarmPolicy,
    PSOSwarmPolicy,
    UncertaintyAwareBeeAntSwarmPolicy,
)
from src.environment.city_twin import CityTwinEnvironment
from src.environment.obstacles import synthetic_city_layers


@pytest.mark.parametrize(
    "policy_cls",
    [AntSwarmPolicy, BeeSwarmPolicy, PSOSwarmPolicy, UncertaintyAwareBeeAntSwarmPolicy],
)
def test_bio_swarm_policy_runs_safely(policy_cls):
    layers = synthetic_city_layers(20, seed=11)
    env = CityTwinEnvironment(
        grid_size=20,
        n_agents=3,
        seed=11,
        layers=layers,
        allow_network=False,
        max_steps=15,
        communication_dropout_prob=0.0,
    )
    policy = policy_cls()
    while True:
        actions = policy.act(env)
        assert set(actions) == set(env.agents)
        assert all(action in env.all_candidate_actions() for action in actions.values())
        info = env.step(actions)
        if info["done"]:
            break
    assert env.actual_restricted_entries == 0
    assert env.actual_collisions == 0
    assert env.weighted_target_discovery() >= 0.0
