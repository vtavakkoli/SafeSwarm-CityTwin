"""Shared rollout, reward, per-agent credit and GAE utilities.

The v2 trainer broadcast one global timestep advantage to every agent. With
8 agents, useful and harmful actions therefore received identical credit and
their policy gradients largely cancelled. v3 keeps a small cooperative team
reward but adds agent-local discovery, coverage, uncertainty, redundancy,
energy and safe-return credit before computing GAE independently per agent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.environment.city_twin import Cell, CityTwinEnvironment
from src.evaluation.metrics import EpisodeMetrics
from src.safety.runtime_monitor import RuntimeSafetyMonitor


def episode_metrics(env: CityTwinEnvironment, monitor: RuntimeSafetyMonitor) -> EpisodeMetrics:
    actual_incidents = env.actual_collisions + env.actual_restricted_entries
    battery_failures = sum(state.battery_level <= 0 for state in env.agents.values())
    coverage = len(env.visited - env.restricted_zones) / env.traversable_cell_count
    return EpisodeMetrics(
        mission_success_rate=len(env.discovered_missions) / max(1, len(env.mission_zones)),
        weighted_target_discovery=env.weighted_target_discovery(),
        coverage_ratio=float(np.clip(coverage, 0.0, 1.0)),
        number_of_safety_violations=int(monitor.safety_violations),
        safety_interventions=int(monitor.intervention_count),
        actual_safety_incidents=int(actual_incidents),
        collision_count=int(env.actual_collisions),
        restricted_zone_entries=int(env.actual_restricted_entries),
        battery_failures=int(battery_failures),
        energy_consumption=env.energy_consumption(),
        redundant_coverage=env.redundant_coverage(),
        communication_efficiency=env.communication_efficiency(),
        distance_travelled=sum(state.distance_travelled for state in env.agents.values()),
        runtime_seconds=0.0,
        runtime_overhead=0.0,
        blocked_moves=int(env.blocked_moves),
    )


def _reserve_pressure(env: CityTwinEnvironment) -> float:
    values: list[float] = []
    move_cost = float(getattr(env, "move_energy_cost", 0.55))
    for state in env.agents.values():
        if state.safe_returned:
            values.append(0.0)
            continue
        required = move_cost * env.nearest_base_distance(state.position) + 5.0
        values.append(max(0.0, required - state.battery_level) / 100.0)
    return float(np.mean(values)) if values else 0.0


def reward_state(env: CityTwinEnvironment) -> dict[str, float]:
    return {
        "target": env.weighted_target_discovery(),
        "coverage": len(env.visited - env.restricted_zones) / env.traversable_cell_count,
        "uncertainty": float(np.mean(env.uncertainty_map)),
        "incidents": float(env.actual_collisions + env.actual_restricted_entries),
        "energy": env.energy_consumption() / max(1.0, 100.0 * env.n_agents),
        "redundancy": env.redundant_coverage(),
        "reserve_pressure": _reserve_pressure(env),
        "battery_failures": float(sum(state.battery_level <= 0 for state in env.agents.values())),
        "safe_returns": float(env.safe_return_count / max(1, env.n_agents)),
    }


def dense_reward(current: dict[str, float], previous: dict[str, float]) -> float:
    """Dense team reward aligned with the final operational ranking.

    Target discovery is the largest component, followed by coverage and actual
    safety. Energy/redundancy are regularizers rather than dominant terminal
    penalties. Safe return is explicitly rewarded so returning to base does not
    look worse than dying at the end of a fixed horizon.
    """

    return float(
        4.0 * (current["target"] - previous["target"])
        + 1.5 * (current["coverage"] - previous["coverage"])
        + 0.50 * max(0.0, previous["uncertainty"] - current["uncertainty"])
        - 2.0 * max(0.0, current["incidents"] - previous["incidents"])
        - 0.10 * max(0.0, current["energy"] - previous["energy"])
        - 0.30 * max(0.0, current["redundancy"] - previous["redundancy"])
        - 0.40 * max(0.0, current["reserve_pressure"] - previous["reserve_pressure"])
        - 1.5 * max(0.0, current["battery_failures"] - previous["battery_failures"])
        + 0.80 * max(0.0, current["safe_returns"] - previous["safe_returns"])
    )


def _sensor_cells(env: CityTwinEnvironment, position: Cell) -> list[Cell]:
    px, py = position
    cells: list[Cell] = []
    for x in range(max(0, px - env.sensor_radius), min(env.grid_size, px + env.sensor_radius + 1)):
        for y in range(max(0, py - env.sensor_radius), min(env.grid_size, py + env.sensor_radius + 1)):
            if abs(x - px) + abs(y - py) <= env.sensor_radius and (x, y) not in env.obstacles:
                cells.append((x, y))
    return cells


def agent_credit_snapshot(env: CityTwinEnvironment) -> dict[str, Any]:
    return {
        "positions": {aid: state.position for aid, state in env.agents.items()},
        "batteries": {aid: float(state.battery_level) for aid, state in env.agents.items()},
        "base_distance": {aid: env.nearest_base_distance(state.position) for aid, state in env.agents.items()},
        "safe_returned": {aid: bool(state.safe_returned) for aid, state in env.agents.items()},
        "visited": set(env.visited),
        "visit_counts": env.visit_counts.copy(),
        "uncertainty": env.uncertainty_map.copy(),
        "discovered": set(env.discovered_missions),
    }


def agent_step_rewards(
    env: CityTwinEnvironment,
    previous: dict[str, Any],
    team_reward: float,
) -> dict[int, float]:
    """Assign difference-style credit to the agents that produced progress."""

    new_missions = set(env.discovered_missions) - set(previous["discovered"])
    total_priority = sum(env.priority_cells.get(cell, 0.5) for cell in env.mission_zones)
    target_credit = {aid: 0.0 for aid in env.agents}
    for mission in new_missions:
        contributors = [
            aid
            for aid, state in env.agents.items()
            if abs(state.position[0] - mission[0]) + abs(state.position[1] - mission[1]) <= env.sensor_radius
        ]
        if not contributors:
            continue
        weight = env.priority_cells.get(mission, 0.5) / max(total_priority, 1e-12)
        share = float(weight / len(contributors))
        for aid in contributors:
            target_credit[aid] += share

    rewards: dict[int, float] = {}
    positions_now = {aid: state.position for aid, state in env.agents.items()}
    for aid, state in env.agents.items():
        before_pos = previous["positions"][aid]
        after_pos = state.position
        cells = _sensor_cells(env, after_pos)
        uncertainty_gain = 0.0
        if cells:
            uncertainty_gain = float(np.mean([
                max(0.0, float(previous["uncertainty"][cell]) - float(env.uncertainty_map[cell]))
                for cell in cells
            ]))

        first_visit = 1.0 if after_pos not in previous["visited"] else 0.0
        prior_visits = float(previous["visit_counts"][after_pos])
        repeated = float(np.clip(max(0.0, prior_visits - 1.0) / 3.0, 0.0, 1.0))
        energy_used = max(0.0, float(previous["batteries"][aid]) - float(state.battery_level))
        energy_scale = max(float(getattr(env, "move_energy_cost", 0.55)), 1e-6)
        energy_used = float(np.clip(energy_used / energy_scale, 0.0, 2.0))

        other_distances = [
            float(np.hypot(after_pos[0] - pos[0], after_pos[1] - pos[1]))
            for oid, pos in positions_now.items()
            if oid != aid and not env.agents[oid].done
        ]
        spread = 0.0 if not other_distances else float(np.clip(min(other_distances) / max(1.0, env.grid_size / 4.0), 0.0, 1.0))

        return_progress = 0.0
        if float(previous["batteries"][aid]) <= 35.0:
            return_progress = float(np.clip(
                previous["base_distance"][aid] - env.nearest_base_distance(after_pos),
                -1.0,
                1.0,
            ))
        safe_return = 1.0 if state.safe_returned and not previous["safe_returned"][aid] else 0.0
        moving = float(after_pos != before_pos)
        useful_idle_penalty = 0.15 if moving < 0.5 and float(previous["batteries"][aid]) > 35.0 else 0.0

        rewards[aid] = float(
            0.20 * team_reward
            + 5.0 * target_credit[aid]
            + 0.85 * first_visit
            + 0.65 * uncertainty_gain
            + 0.18 * spread
            + 0.45 * return_progress
            + 0.90 * safe_return
            - 0.30 * repeated
            - 0.06 * energy_used
            - useful_idle_penalty
        )
    return rewards


def gae_advantages(
    rewards: list[float],
    values: list[float],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    if len(rewards) != len(values):
        raise ValueError("rewards and values must have equal length")
    advantages = [0.0] * len(rewards)
    targets = [0.0] * len(rewards)
    gae = 0.0
    next_value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + gamma * next_value - values[index]
        gae = delta + gamma * gae_lambda * gae
        advantages[index] = float(gae)
        targets[index] = float(gae + values[index])
        next_value = values[index]
    return advantages, targets


def rollout_episode(
    env: CityTwinEnvironment,
    policy: Any,
    *,
    gamma: float,
    gae_lambda: float,
    training: bool,
) -> tuple[EpisodeMetrics, list[tuple[dict[str, Any], float, float]], dict[str, float]]:
    policy.monitor = RuntimeSafetyMonitor()
    step_decisions: list[dict[int, dict[str, Any]]] = []
    per_agent_rewards: list[dict[int, float]] = []
    team_rewards: list[float] = []
    previous_team = reward_state(env)

    while True:
        credit_before = agent_credit_snapshot(env)
        actions = policy.act(env)
        decisions = policy.drain_decisions() if hasattr(policy, "drain_decisions") else []
        decision_map = {int(decision["agent_id"]): decision for decision in decisions if "agent_id" in decision}
        info = env.step(actions)
        current_team = reward_state(env)
        team_reward = dense_reward(current_team, previous_team)
        previous_team = current_team
        local_rewards = agent_step_rewards(env, credit_before, team_reward)
        team_rewards.append(team_reward)
        per_agent_rewards.append(local_rewards)
        step_decisions.append(decision_map)
        if info["done"] > 0:
            break

    samples: list[tuple[dict[str, Any], float, float]] = []
    if training:
        horizon = len(step_decisions)
        for aid in env.agents:
            rewards = [float(per_agent_rewards[t].get(aid, 0.0)) for t in range(horizon)]
            values = [
                float(step_decisions[t].get(aid, {}).get("old_value", 0.0))
                for t in range(horizon)
            ]
            advantages, targets = gae_advantages(
                rewards,
                values,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
            for t in range(horizon):
                decision = step_decisions[t].get(aid)
                if decision is not None:
                    samples.append((decision, advantages[t], targets[t]))

    diagnostics = policy.diagnostics() if hasattr(policy, "diagnostics") else {}
    flat_local = [value for step in per_agent_rewards for value in step.values()]
    extra = {
        "episode_reward": float(np.sum(team_rewards)),
        "mean_agent_reward": float(np.mean(flat_local)) if flat_local else 0.0,
        "safe_returns": float(env.safe_return_count),
        "safety_mask_rejections": float(diagnostics.get("safety_mask_rejections", 0)),
        "forced_fallbacks": float(diagnostics.get("forced_fallbacks", 0)),
        "return_guard_interventions": float(getattr(policy.monitor, "return_guard_interventions", 0)),
        "swarm_memory_coverage": float(diagnostics.get("swarm_memory_coverage", 0.0)),
        "swarm_memory_peak": float(diagnostics.get("swarm_memory_peak", 0.0)),
    }
    return episode_metrics(env, policy.monitor), samples, extra
