"""Safety-filtered wrapper around the greedy city monitoring policy."""

from __future__ import annotations

from typing import Dict

from src.agents.greedy_agent import GreedyAgentPolicy
from src.environment.city_twin import CityTwinEnvironment
from src.safety.runtime_monitor import RuntimeSafetyMonitor


class SafetyFilteredAgentPolicy:
    name = "SafetyFilteredGreedy"

    def __init__(self, monitor: RuntimeSafetyMonitor | None = None) -> None:
        self.base = GreedyAgentPolicy()
        self.monitor = monitor or RuntimeSafetyMonitor()

    def act(self, env: CityTwinEnvironment) -> Dict[int, str]:
        return self.monitor.filter_actions(env, self.base.act(env))
