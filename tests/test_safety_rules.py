from src.environment.city_twin import CityTwinEnvironment
from src.safety.rules import (
    battery_reserve_for_return,
    inside_operational_boundary,
    no_collision,
    no_restricted_zone,
)


def test_boundary_rule():
    env = CityTwinEnvironment(grid_size=20, n_agents=2, seed=1)
    assert inside_operational_boundary((0, 0), env)
    assert not inside_operational_boundary((-1, 0), env)


def test_restricted_zone_rule():
    env = CityTwinEnvironment(grid_size=20, n_agents=2, seed=1)
    cell = next(iter(env.restricted_zones)) if env.restricted_zones else (1, 1)
    env.restricted_zones.add(cell)
    assert not no_restricted_zone(cell, env)


def test_collision_rule():
    env = CityTwinEnvironment(grid_size=20, n_agents=2, seed=1)
    a0 = env.agents[0].position
    planned = {1: a0}
    assert not no_collision(0, a0, env, planned)


def test_battery_rule():
    env = CityTwinEnvironment(grid_size=20, n_agents=1, seed=1)
    agent = env.agents[0]
    agent.battery_level = 3
    assert not battery_reserve_for_return(agent, (10, 10), env)
