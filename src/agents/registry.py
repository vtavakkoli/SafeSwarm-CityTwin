"""Central strategy registry used by experiments and tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Dict

from src.agents.bio_swarm_agents import (
    AntSwarmPolicy,
    BeeSwarmPolicy,
    PSOSwarmPolicy,
    UncertaintyAwareBeeAntSwarmPolicy,
)
from src.agents.greedy_agent import GreedyAgentPolicy
from src.agents.random_agent import RandomAgentPolicy
from src.agents.safe_swarm_agent import SafeSwarmAgentPolicy
from src.agents.safety_filtered_agent import SafetyFilteredAgentPolicy


def strategy_factories(seed: int = 42) -> Dict[str, Callable[[], object]]:
    """Return fresh policy factories so state never leaks across episodes."""
    return {
        "RandomAgent": lambda: RandomAgentPolicy(seed=seed),
        "GreedyAgent": GreedyAgentPolicy,
        "SafetyFilteredGreedy": SafetyFilteredAgentPolicy,
        "SafeSwarmAgent": SafeSwarmAgentPolicy,
        "AntSwarmSafe": AntSwarmPolicy,
        "BeeSwarmSafe": BeeSwarmPolicy,
        "PSOSwarmSafe": PSOSwarmPolicy,
        "UA-HBAS-Safe": UncertaintyAwareBeeAntSwarmPolicy,
    }
