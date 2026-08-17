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
from src.agents.observable_marl import (
    ObservableGRPOPolicy,
    ObservableHAPPOPolicy,
    ObservableIPPOPolicy,
    ObservableMADDPGPolicy,
    ObservableMAPPOPolicy,
    ObservableMATPolicy,
    ObservableQMIXPolicy,
)
from src.agents.random_agent import RandomAgentPolicy
from src.agents.safe_swarm_agent import SafeSwarmAgentPolicy
from src.agents.safety_filtered_agent import SafetyFilteredAgentPolicy


def strategy_factories(seed: int = 42) -> Dict[str, Callable[[], object]]:
    """Return fresh policy factories so state never leaks across episodes.

    Primary benchmark strategies obey the same partial-observability contract;
    hidden mission coordinates/priority labels are reserved for evaluation.
    """

    return {
        "RandomAgent": lambda: RandomAgentPolicy(seed=seed),
        "GreedyAgent": GreedyAgentPolicy,
        "SafetyFilteredGreedy": SafetyFilteredAgentPolicy,
        "SafeSwarmAgent": SafeSwarmAgentPolicy,
        "AntSwarmSafe": AntSwarmPolicy,
        "BeeSwarmSafe": BeeSwarmPolicy,
        "PSOSwarmSafe": PSOSwarmPolicy,
        "UA-HBAS-Safe": UncertaintyAwareBeeAntSwarmPolicy,
        "GRPO-Safe": lambda: ObservableGRPOPolicy(seed=seed),
        "IPPO-Safe": lambda: ObservableIPPOPolicy(seed=seed),
        "MAPPO-Safe": lambda: ObservableMAPPOPolicy(seed=seed),
        "QMIX-Safe": lambda: ObservableQMIXPolicy(seed=seed),
        "MADDPG-Safe": lambda: ObservableMADDPGPolicy(seed=seed),
        "HAPPO-Safe": lambda: ObservableHAPPOPolicy(seed=seed),
        "MAT-Safe": lambda: ObservableMATPolicy(seed=seed),
    }
