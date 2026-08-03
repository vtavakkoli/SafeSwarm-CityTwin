from src.environment.city_twin import CityTwinEnvironment
from src.environment.obstacles import synthetic_city_layers
from src.safety.runtime_monitor import RuntimeSafetyMonitor


def make_env() -> CityTwinEnvironment:
    layers = synthetic_city_layers(15, seed=2)
    return CityTwinEnvironment(grid_size=15, n_agents=1, seed=2, layers=layers, allow_network=False)


def test_monitor_blocks_restricted_and_selects_safe_alternative():
    env = make_env()
    monitor = RuntimeSafetyMonitor()
    current = env.agents[0].position
    restricted = (current[0] + 1, current[1])
    if env.in_bounds(restricted):
        env.restricted_zones.add(restricted)
    filtered = monitor.filter_actions(env, {0: "RIGHT"})
    candidate = env.next_position(current, filtered[0])
    assert candidate not in env.restricted_zones
    assert monitor.intervention_count >= 1


def test_monitor_detects_comm_loss_violation():
    env = make_env()
    monitor = RuntimeSafetyMonitor(max_comm_loss_steps=1)
    env.agents[0].comm_loss_steps = 3
    is_safe, violations, _ = monitor.is_action_safe(env, 0, "STAY", planned_positions={})
    assert not is_safe
    assert "communication_loss" in violations
