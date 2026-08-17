"""Teacher distillation utilities for the GRPO warm start.

BioSwarm's strongest PPO pipeline does not start from random behavior selection:
it uses imitation pretraining from strong swarm teachers. SafeSwarm v3 mirrors
that idea without copying hidden target labels. Teachers and student operate on
the same partially observable environment and runtime-safety contract.
"""

from __future__ import annotations

from typing import Any

from src.environment.city_twin import CityTwinEnvironment
from src.safety.runtime_monitor import RuntimeSafetyMonitor


def collect_grpo_teacher_examples(
    policy: Any,
    teacher: Any,
    env: CityTwinEnvironment,
    *,
    max_examples: int = 5000,
) -> list[dict[str, Any]]:
    """Collect safe action/behavior labels from one teacher rollout."""

    if not hasattr(policy, "imitation_example"):
        return []
    if hasattr(policy, "reset_episode_state"):
        policy.reset_episode_state()
    policy.monitor = RuntimeSafetyMonitor()
    examples: list[dict[str, Any]] = []

    while True:
        if hasattr(policy, "_update_swarm_memory"):
            policy._update_swarm_memory(env)  # execution-available memory only

        teacher_actions = teacher.act(env)
        positions = env.get_positions()
        planned: dict[int, tuple[int, int]] = {}
        for agent_id in sorted(env.agents):
            state = env.agents[agent_id]
            if state.done:
                planned[agent_id] = state.position
                continue
            safe_actions = policy.monitor.safe_actions(
                env, agent_id, planned, count_masked=False
            )
            if not safe_actions:
                planned[agent_id] = state.position
                continue
            target_action = teacher_actions.get(agent_id, "STAY")
            if target_action not in safe_actions:
                target_action = policy.monitor.nearest_safe_action(
                    env, agent_id, planned, target_action
                )
            candidates = [env.next_position(state.position, action) for action in safe_actions]
            target_cell = env.next_position(state.position, target_action)
            example = policy.imitation_example(
                env, agent_id, positions, candidates, target_cell
            )
            if example is not None:
                examples.append(example)
            planned[agent_id] = target_cell
            positions[agent_id] = target_cell
            if len(examples) >= max_examples:
                return examples

        info = env.step(teacher_actions)
        if info["done"] > 0:
            break
    return examples
