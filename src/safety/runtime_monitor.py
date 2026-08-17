"""Runtime monitor for safety-constrained action filtering.

Trainable policies sample only from executable actions. A proactive energy guard
also narrows the admissible set toward a base *before* the hard reserve rule is
violated; this avoids the old state where every action eventually became
invalid and the monitor spent the end of an episode repeatedly falling back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.environment.city_twin import CityTwinEnvironment
from src.safety.rules import evaluate_all_rules

Cell = Tuple[int, int]


@dataclass
class RuntimeSafetyMonitor:
    max_comm_loss_steps: int = 5
    return_guard_margin: float = 12.0
    safety_violations: int = 0
    collision_count: int = 0
    restricted_zone_entries: int = 0
    battery_failures: int = 0
    intervention_count: int = 0
    mask_rejections: int = 0
    return_guard_interventions: int = 0
    violation_log: List[Dict[str, object]] = field(default_factory=list)
    rule_counts: Dict[str, int] = field(default_factory=dict)
    mask_rule_counts: Dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.safety_violations = 0
        self.collision_count = 0
        self.restricted_zone_entries = 0
        self.battery_failures = 0
        self.intervention_count = 0
        self.mask_rejections = 0
        self.return_guard_interventions = 0
        self.violation_log.clear()
        self.rule_counts.clear()
        self.mask_rule_counts.clear()

    def is_action_safe(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        action: str,
        planned_positions: Dict[int, Cell],
    ) -> tuple[bool, List[str], Cell]:
        agent = env.agents[agent_id]
        candidate = env.next_position(agent.position, action)
        violations = evaluate_all_rules(
            agent_id=agent_id,
            agent=agent,
            candidate=candidate,
            env=env,
            planned_positions=planned_positions,
            max_comm_loss_steps=self.max_comm_loss_steps,
        )
        return len(violations) == 0, violations, candidate

    def _return_guard_active(self, env: CityTwinEnvironment, agent_id: int) -> bool:
        state = env.agents[agent_id]
        if state.done:
            return False
        if state.position in env.base_stations:
            return state.battery_level <= float(getattr(env, "safe_park_battery", 18.0))
        distance = env.nearest_base_distance(state.position)
        move_cost = float(getattr(env, "move_energy_cost", 1.5))
        hard_reserve = distance * move_cost + 5.0
        return state.battery_level <= hard_reserve + float(self.return_guard_margin)

    def safe_actions(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        planned_positions: Dict[int, Cell],
        *,
        count_masked: bool = False,
    ) -> List[str]:
        """Return the executable action set under hard and proactive safety."""

        if env.agents[agent_id].done:
            return ["STAY"]

        safe: List[str] = []
        for action in env.all_candidate_actions():
            ok, violations, _ = self.is_action_safe(env, agent_id, action, planned_positions)
            if ok:
                safe.append(action)
                continue
            if count_masked:
                self.mask_rejections += 1
                for violation in violations:
                    self.mask_rule_counts[violation] = self.mask_rule_counts.get(violation, 0) + 1

        if not safe or not self._return_guard_active(env, agent_id):
            return safe

        current = env.agents[agent_id].position
        current_distance = env.nearest_base_distance(current)
        progress = [
            action
            for action in safe
            if env.nearest_base_distance(env.next_position(current, action)) < current_distance
        ]
        guarded = progress or [
            action
            for action in safe
            if env.nearest_base_distance(env.next_position(current, action)) <= current_distance
        ]
        if guarded and count_masked:
            rejected = len(safe) - len(guarded)
            self.mask_rejections += rejected
            if rejected:
                self.mask_rule_counts["return_guard"] = self.mask_rule_counts.get("return_guard", 0) + rejected
        return guarded or safe

    def record_intervention(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        action: str,
        violations: List[str],
    ) -> None:
        self.intervention_count += 1
        self.safety_violations += len(violations)
        self.collision_count += int("collision" in violations)
        self.restricted_zone_entries += int("restricted_zone" in violations)
        self.battery_failures += int("battery_reserve" in violations)
        for violation in violations:
            self.rule_counts[violation] = self.rule_counts.get(violation, 0) + 1
        self.violation_log.append(
            {"step": env.steps, "agent_id": agent_id, "action": action, "violations": violations}
        )

    def nearest_safe_action(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        planned_positions: Dict[int, Cell],
        preferred_action: str,
    ) -> str:
        agent = env.agents[agent_id]
        if agent.done:
            return "STAY"
        preferred_cell = env.next_position(agent.position, preferred_action)
        guarded = self.safe_actions(env, agent_id, planned_positions, count_masked=False)
        safe_candidates: List[tuple[float, float, float, str]] = []
        for action in guarded:
            ok, _, candidate = self.is_action_safe(env, agent_id, action, planned_positions)
            if ok:
                base_distance = float(env.nearest_base_distance(candidate))
                preferred_distance = float(abs(candidate[0] - preferred_cell[0]) + abs(candidate[1] - preferred_cell[1]))
                utility = float(env.observation_map[candidate] + env.uncertainty_map[candidate])
                primary = base_distance if self._return_guard_active(env, agent_id) else preferred_distance
                safe_candidates.append((primary, preferred_distance, -utility, action))
        if not safe_candidates:
            return "STAY"
        safe_candidates.sort()
        return safe_candidates[0][3]

    def filter_actions(self, env: CityTwinEnvironment, proposed_actions: Dict[int, str]) -> Dict[int, str]:
        filtered: Dict[int, str] = {}
        planned_positions: Dict[int, Cell] = {}
        for aid in sorted(env.agents):
            if env.agents[aid].done:
                filtered[aid] = "STAY"
                planned_positions[aid] = env.agents[aid].position
                continue

            action = proposed_actions.get(aid, "STAY")
            is_safe, violations, _ = self.is_action_safe(env, aid, action, planned_positions)
            guarded_actions = self.safe_actions(env, aid, planned_positions, count_masked=False)
            guard_blocks = self._return_guard_active(env, aid) and action not in guarded_actions

            if is_safe and not guard_blocks:
                filtered[aid] = action
            elif guard_blocks and is_safe:
                self.return_guard_interventions += 1
                filtered[aid] = self.nearest_safe_action(env, aid, planned_positions, action)
            else:
                self.record_intervention(env, aid, action, violations)
                filtered[aid] = self.nearest_safe_action(env, aid, planned_positions, action)

            planned_positions[aid] = env.next_position(env.agents[aid].position, filtered[aid])
        return filtered
