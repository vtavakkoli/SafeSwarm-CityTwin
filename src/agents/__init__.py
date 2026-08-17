"""Agent policies available in SafeSwarm-CityTwin."""

from src.agents.bio_swarm_agents import (
    AntSwarmPolicy,
    BeeSwarmPolicy,
    PSOSwarmPolicy,
    UncertaintyAwareBeeAntSwarmPolicy,
)
from src.agents.marl_baselines import (
    GRPOPolicy,
    HAPPOPolicy,
    IPPOPolicy,
    MADDPGPolicy,
    MAPPOPolicy,
    MATPolicy,
    QMIXPolicy,
)
from src.agents.sparx_pattern import SPARXPolicy
from src.agents.registry import strategy_factories

__all__ = [
    "AntSwarmPolicy",
    "BeeSwarmPolicy",
    "PSOSwarmPolicy",
    "UncertaintyAwareBeeAntSwarmPolicy",
    "GRPOPolicy",
    "IPPOPolicy",
    "MAPPOPolicy",
    "QMIXPolicy",
    "MADDPGPolicy",
    "HAPPOPolicy",
    "MATPolicy",
    "SPARXPolicy",
    "strategy_factories",
]
