"""Deprecated SPARX compatibility layer.

SafeSwarm v5 renamed SPARX to PRISM (Probability-guided Region-Integrated
Search with Memory) to avoid confusion with unrelated swarm-robotics projects.
New code must import :mod:`src.agents.prism_pattern`.

The aliases below intentionally remain for loading/reproducing v4 experiments.
They are not registered as v5 benchmark strategy names.
"""

from src.agents.prism_pattern import (
    PATTERN_LABELS,
    PATTERN_MODES,
    PRISMConfig,
    PRISMPolicy,
    PRISM_PARAMETER_BOUNDS,
    PRISM_TUNABLE_FIELDS,
    prism_pattern_factories,
)

SPARXConfig = PRISMConfig
SPARXPolicy = PRISMPolicy
SPARX_PARAMETER_BOUNDS = PRISM_PARAMETER_BOUNDS
SPARX_TUNABLE_FIELDS = PRISM_TUNABLE_FIELDS
sparx_pattern_factories = prism_pattern_factories

__all__ = [
    "PATTERN_LABELS",
    "PATTERN_MODES",
    "PRISMConfig",
    "PRISMPolicy",
    "PRISM_PARAMETER_BOUNDS",
    "PRISM_TUNABLE_FIELDS",
    "prism_pattern_factories",
    "SPARXConfig",
    "SPARXPolicy",
    "SPARX_PARAMETER_BOUNDS",
    "SPARX_TUNABLE_FIELDS",
    "sparx_pattern_factories",
]
