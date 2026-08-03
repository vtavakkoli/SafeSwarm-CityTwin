from src.environment.city_twin import CityTwinEnvironment
from src.environment.obstacles import synthetic_city_layers
from src.safety.rules import battery_reserve_for_return, inside_operational_boundary, no_restricted_zone


def make_env() -> CityTwinEnvironment:
    layers = synthetic_city_layers(15, seed=2)
    return CityTwinEnvironment(grid_size=15, n_agents=1, seed=2, layers=layers, allow_network=False)


def test_boundary_rule():
    env = make_env()
    assert inside_operational_boundary((0, 0), env)
    assert not inside_operational_boundary((-1, 0), env)


def test_restricted_zone_rule():
    env = make_env()
    env.restricted_zones.add((1, 1))
    assert not no_restricted_zone((1, 1), env)


def test_battery_reserve_rule():
    env = make_env()
    agent = env.agents[0]
    agent.battery_level = 3
    assert not battery_reserve_for_return(agent, (10, 10), env)
