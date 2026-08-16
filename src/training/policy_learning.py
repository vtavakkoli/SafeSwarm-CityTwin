"""Shared rollout, reward and GAE utilities for SafeSwarm policy learning."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.environment.city_twin import CityTwinEnvironment
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
    for state in env.agents.values():
        required = 1.5 * env.nearest_base_distance(state.position) + 8.0
        values.append(max(0.0, required - state.battery_level) / 100.0)
    return float(np.mean(values)) if values else 0.0


def reward_state(env: CityTwinEnvironment) -> dict[str, float]:
    return {
        "target": env.weighted_target_discovery(),
        "coverage": len(env.visited - env.restricted_zones) / env.traversable_cell_count,
        "incidents": float(env.actual_collisions + env.actual_restricted_entries),
        "energy": env.energy_consumption() / max(1.0, 100.0 * env.n_agents),
        "redundancy": env.redundant_coverage(),
        "reserve_pressure": _reserve_pressure(env),
        "battery_failures": float(sum(state.battery_level <= 0 for state in env.agents.values())),
    }


def dense_reward(current: dict[str, float], previous: dict[str, float]) -> float:
    """Reward discovery/coverage while penalizing unsafe and stranded behavior."""

    return float(
        2.8 * (current["target"] - previous["target"])
        + 1.0 * (current["coverage"] - previous["coverage"])
        - 1.8 * (current["incidents"] - previous["incidents"])
        - 0.18 * max(0.0, current["energy"] - previous["energy"])
        - 0.12 * max(0.0, current["redundancy"] - previous["redundancy"])
        - 0.75 * max(0.0, current["reserve_pressure"] - previous["reserve_pressure"])
        - 1.25 * max(0.0, current["battery_failures"] - previous["battery_failures"])
    )


def gae_advantages(
    rewards: list[float],
    values: list[float],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    """Return terminal GAE advantages and critic targets for one episode."""

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
    step_decisions: list[list[dict[str, Any]]] = []
    rewards: list[float] = []
    values: list[float] = []
    previous = reward_state(env)

    while True:
        actions = policy.act(env)
        decisions = policy.drain_decisions() if hasattr(policy, "drain_decisions") else []
        value = float(np.mean([d.get("old_value", 0.0) for d in decisions])) if decisions else 0.0
        info = env.step(actions)
        current = reward_state(env)
        rewards.append(dense_reward(current, previous))
        previous = current
        step_decisions.append(decisions)
        values.append(value)
        if info["done"] > 0:
            break

    samples: list[tuple[dict[str, Any], float, float]] = []
    if training:
        advantages, targets = gae_advantages(
            rewards, values, gamma=gamma, gae_lambda=gae_lambda
        )
        for decisions, advantage, target in zip(step_decisions, advantages, targets):
            samples.extend((decision, advantage, target) for decision in decisions)

    diagnostics = policy.diagnostics() if hasattr(policy, "diagnostics") else {}
    extra = {
        "episode_reward": float(np.sum(rewards)),
        "safety_mask_rejections": float(diagnostics.get("safety_mask_rejections", 0)),
        "forced_fallbacks": float(diagnostics.get("forced_fallbacks", 0)),
        "swarm_memory_coverage": float(diagnostics.get("swarm_memory_coverage", 0.0)),
        "swarm_memory_peak": float(diagnostics.get("swarm_memory_peak", 0.0)),
    }
    return episode_metrics(env, policy.monitor), samples, extra
