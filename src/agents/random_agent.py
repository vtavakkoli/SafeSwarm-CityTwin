"""Random baseline agent without a runtime safety filter."""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.environment.city_twin import CityTwinEnvironment


class RandomAgentPolicy:
    name = "RandomAgent"

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, env: CityTwinEnvironment) -> Dict[int, str]:
        actions = env.all_candidate_actions()
        return {aid: str(self.rng.choice(actions)) for aid in env.agents}
