"""Observable teacher-distillation utilities for SafeSwarm trainable policies.

v5 generalizes the former GRPO-only warm start.  Any policy implementing
``imitation_example`` can learn from a strong observable teacher without hidden
target labels.  GRPO keeps its richer hierarchical labels; IPPO/MAPPO/HAPPO use
a generic safe-action imitation target.
"""

from __future__ import annotations

from typing import Any

from src.environment.city_twin import CityTwinEnvironment
from src.safety.runtime_monitor import RuntimeSafetyMonitor


def collect_teacher_examples(
    policy: Any,
    teacher: Any,
    env: CityTwinEnvironment,
    *,
    max_examples: int = 5000,
) -> list[dict[str, Any]]:
    """Collect safe observable action labels from one teacher rollout."""

    if not hasattr(policy, "imitation_example"):
        return []
    if hasattr(policy, "reset_episode_state"):
        policy.reset_episode_state()
    policy.monitor = RuntimeSafetyMonitor()
    examples: list[dict[str, Any]] = []

    while True:
        if hasattr(policy, "_update_swarm_memory"):
            policy._update_swarm_memory(env)

        teacher_actions = teacher.act(env)
        positions = env.get_positions()
        planned: dict[int, tuple[int, int]] = {}
        executable: dict[int, str] = {}

        for agent_id in sorted(env.agents):
            state = env.agents[agent_id]
            if state.done:
                executable[agent_id] = "STAY"
                planned[agent_id] = state.position
                continue

            safe_actions = policy.monitor.safe_actions(
                env, agent_id, planned, count_masked=False
            )
            if not safe_actions:
                executable[agent_id] = "STAY"
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
            executable[agent_id] = target_action
            planned[agent_id] = target_cell
            positions[agent_id] = target_cell
            if len(examples) >= max_examples:
                return examples

        # Execute the same safety-adjusted labels that were distilled.  This avoids
        # teacher/student state drift when the teacher proposed an unsafe move.
        info = env.step(executable)
        if info["done"] > 0:
            break
    return examples


def collect_grpo_teacher_examples(
    policy: Any,
    teacher: Any,
    env: CityTwinEnvironment,
    *,
    max_examples: int = 5000,
) -> list[dict[str, Any]]:
    """Backward-compatible alias used by v3/v4 experiment scripts."""

    return collect_teacher_examples(
        policy, teacher, env, max_examples=max_examples
    )
