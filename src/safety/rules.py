"""Safety rules for city-twin multi-agent systems."""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.environment.city_twin import AgentState, CityTwinEnvironment

Cell = Tuple[int, int]


def no_restricted_zone(candidate: Cell, env: CityTwinEnvironment) -> bool:
    return candidate not in env.restricted_zones


def no_collision(agent_id: int, candidate: Cell, env: CityTwinEnvironment, planned_positions: Dict[int, Cell]) -> bool:
    occupied_positions = {pos for aid, pos in planned_positions.items() if aid != agent_id}
    occupied_positions.update(
        state.position for aid, state in env.agents.items() if aid != agent_id and not state.done
    )
    return candidate not in occupied_positions


def battery_reserve_for_return(
    agent: AgentState,
    candidate: Cell,
    env: CityTwinEnvironment,
    reserve_margin: float = 5.0,
) -> bool:
    """Require enough energy to reach a base, but never reject arrival at base.

    The previous rule could classify ``STAY`` at a base as unsafe once battery
    fell below the reserve margin, creating an impossible fallback loop. A base
    is the terminal safe state: any positive battery is sufficient to enter or
    remain there, after which the environment can safely park the agent.
    """

    if candidate in env.base_stations:
        return agent.battery_level > 0.0
    dist = env.nearest_base_distance(candidate)
    move_cost = float(getattr(env, "move_energy_cost", 1.5))
    required = dist * move_cost + reserve_margin
    return agent.battery_level >= required


def communication_within_limit(agent: AgentState, max_loss_steps: int = 5) -> bool:
    return agent.comm_loss_steps <= max_loss_steps


def inside_operational_boundary(candidate: Cell, env: CityTwinEnvironment) -> bool:
    return env.in_bounds(candidate)


def evaluate_all_rules(
    agent_id: int,
    agent: AgentState,
    candidate: Cell,
    env: CityTwinEnvironment,
    planned_positions: Dict[int, Cell],
    max_comm_loss_steps: int = 5,
) -> List[str]:
    violations: List[str] = []
    if not inside_operational_boundary(candidate, env):
        violations.append("boundary")
    if candidate in env.obstacles:
        violations.append("obstacle")
    if not no_restricted_zone(candidate, env):
        violations.append("restricted_zone")
    if not no_collision(agent_id, candidate, env, planned_positions):
        violations.append("collision")
    if not battery_reserve_for_return(agent, candidate, env):
        violations.append("battery_reserve")
    if not communication_within_limit(agent, max_comm_loss_steps):
        violations.append("communication_loss")
    return violations
