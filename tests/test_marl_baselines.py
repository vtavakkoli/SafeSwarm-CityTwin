import pytest

from src.agents.marl_baselines import (
    GRPOPolicy,
    HAPPOPolicy,
    IPPOPolicy,
    MADDPGPolicy,
    MAPPOPolicy,
    MATPolicy,
    QMIXPolicy,
)
from src.agents.registry import strategy_factories
from src.environment.city_twin import CityTwinEnvironment
from src.environment.obstacles import synthetic_city_layers


POLICIES = [
    GRPOPolicy,
    IPPOPolicy,
    MAPPOPolicy,
    QMIXPolicy,
    MADDPGPolicy,
    HAPPOPolicy,
    MATPolicy,
]


@pytest.mark.parametrize("policy_cls", POLICIES)
def test_marl_policy_completes_safe_episode(policy_cls):
    layers = synthetic_city_layers(20, seed=19)
    env = CityTwinEnvironment(
        grid_size=20,
        n_agents=3,
        seed=19,
        layers=layers,
        allow_network=False,
        max_steps=12,
        communication_dropout_prob=0.0,
    )
    policy = policy_cls(seed=19)
    while True:
        actions = policy.act(env)
        assert set(actions) == set(env.agents)
        assert all(action in env.all_candidate_actions() for action in actions.values())
        info = env.step(actions)
        if info["done"]:
            break

    assert env.actual_restricted_entries == 0
    assert env.actual_collisions == 0
    assert policy.diagnostics()["name"] == policy.name


def test_grpo_exposes_group_relative_diagnostics():
    layers = synthetic_city_layers(20, seed=23)
    env = CityTwinEnvironment(
        grid_size=20,
        n_agents=3,
        seed=23,
        layers=layers,
        allow_network=False,
        max_steps=3,
        communication_dropout_prob=0.0,
    )
    policy = GRPOPolicy(seed=23)
    env.step(policy.act(env))
    diagnostics = policy.diagnostics()

    assert diagnostics["group_size"] == 7
    assert diagnostics["last_behavior"] in GRPOPolicy.behavior_names
    assert sum(diagnostics["behavior_distribution"].values()) == pytest.approx(1.0)


def test_registry_contains_grpo_and_well_known_marl_baselines():
    factories = strategy_factories(seed=5)
    expected = {
        "GRPO-Safe",
        "IPPO-Safe",
        "MAPPO-Safe",
        "QMIX-Safe",
        "MADDPG-Safe",
        "HAPPO-Safe",
        "MAT-Safe",
    }
    assert expected <= set(factories)
    assert all(factories[name]().monitor is not None for name in expected)
