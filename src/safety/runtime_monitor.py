"""Runtime monitor for safety-constrained action filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.environment.city_twin import CityTwinEnvironment
from src.safety.rules import evaluate_all_rules

Cell = Tuple[int, int]


@dataclass
class RuntimeSafetyMonitor:
    max_comm_loss_steps: int = 5
    safety_violations: int = 0
    collision_count: int = 0
    restricted_zone_entries: int = 0
    battery_failures: int = 0
    intervention_count: int = 0
    violation_log: List[Dict[str, object]] = field(default_factory=list)
    rule_counts: Dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.safety_violations = 0
        self.collision_count = 0
        self.restricted_zone_entries = 0
        self.battery_failures = 0
        self.intervention_count = 0
        self.violation_log.clear()
        self.rule_counts.clear()

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

    def nearest_safe_action(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        planned_positions: Dict[int, Cell],
        preferred_action: str,
    ) -> str:
        agent = env.agents[agent_id]
        preferred_cell = env.next_position(agent.position, preferred_action)
        safe_candidates: List[tuple[float, float, str]] = []
        for action in env.all_candidate_actions():
            ok, _, candidate = self.is_action_safe(env, agent_id, action, planned_positions)
            if ok:
                distance = abs(candidate[0] - preferred_cell[0]) + abs(candidate[1] - preferred_cell[1])
                utility = env.priority_map[candidate] + env.uncertainty_map[candidate]
                safe_candidates.append((float(distance), -float(utility), action))
        if not safe_candidates:
            return "STAY"
        safe_candidates.sort()
        return safe_candidates[0][2]

    def filter_actions(self, env: CityTwinEnvironment, proposed_actions: Dict[int, str]) -> Dict[int, str]:
        filtered: Dict[int, str] = {}
        planned_positions: Dict[int, Cell] = {}
        for aid in sorted(env.agents):
            action = proposed_actions.get(aid, "STAY")
            is_safe, violations, _ = self.is_action_safe(env, aid, action, planned_positions)
            if is_safe:
                filtered[aid] = action
            else:
                self.intervention_count += 1
                self.safety_violations += len(violations)
                self.collision_count += int("collision" in violations)
                self.restricted_zone_entries += int("restricted_zone" in violations)
                self.battery_failures += int("battery_reserve" in violations)
                for violation in violations:
                    self.rule_counts[violation] = self.rule_counts.get(violation, 0) + 1
                self.violation_log.append(
                    {"step": env.steps, "agent_id": aid, "action": action, "violations": violations}
                )
                filtered[aid] = self.nearest_safe_action(env, aid, planned_positions, action)
            planned_positions[aid] = env.next_position(env.agents[aid].position, filtered[aid])
        return filtered
