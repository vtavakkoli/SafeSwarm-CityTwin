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
from src.agents.prism_ant import PRISMAntPolicy
from src.agents.prism_pattern import prism_pattern_factories
from src.agents.random_agent import RandomAgentPolicy
from src.agents.safe_swarm_agent import SafeSwarmAgentPolicy
from src.agents.safety_filtered_agent import SafetyFilteredAgentPolicy


def strategy_factories(seed: int = 42) -> Dict[str, Callable[[], object]]:
    """Return fresh observable-only policy factories.

    PRISM X/Plus/Star are untuned mechanism baselines in the generic benchmark.
    The train/test workflow replaces them with validation-selected checkpoints.
    ``SPARX`` is intentionally absent from v5 strategy names; the old module is
    retained only as a source-compatibility alias for v4 experiments.
    """

    factories: Dict[str, Callable[[], object]] = {
        "RandomAgent": lambda: RandomAgentPolicy(seed=seed),
        "GreedyAgent": GreedyAgentPolicy,
        "SafetyFilteredGreedy": SafetyFilteredAgentPolicy,
        "SafeSwarmAgent": SafeSwarmAgentPolicy,
        "AntSwarmSafe": AntSwarmPolicy,
        "BeeSwarmSafe": BeeSwarmPolicy,
        "PSOSwarmSafe": PSOSwarmPolicy,
        "UA-HBAS-Safe": UncertaintyAwareBeeAntSwarmPolicy,
        "PRISM-Ant-Safe": lambda: PRISMAntPolicy(
            seed=seed, pattern_mode="star", strategy_name="PRISM-Ant-Safe"
        ),
        "GRPO-Safe": lambda: ObservableGRPOPolicy(seed=seed),
        "IPPO-Safe": lambda: ObservableIPPOPolicy(seed=seed),
        "MAPPO-Safe": lambda: ObservableMAPPOPolicy(seed=seed),
        "QMIX-Safe": lambda: ObservableQMIXPolicy(seed=seed),
        "MADDPG-Safe": lambda: ObservableMADDPGPolicy(seed=seed),
        "HAPPO-Safe": lambda: ObservableHAPPOPolicy(seed=seed),
        "MAT-Safe": lambda: ObservableMATPolicy(seed=seed),
    }
    factories.update(prism_pattern_factories(seed=seed))
    return factories
