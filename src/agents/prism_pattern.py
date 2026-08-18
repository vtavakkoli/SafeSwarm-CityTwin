"""PRISM: Probability-guided Region-Integrated Search with Memory.

PRISM is the renamed successor to the v4 SPARX controller.  The rename avoids
confusion with unrelated swarm-robotics projects while preserving the scientific
mechanism: an observable-only shared search-utility map, signed swarm memory,
spatial region allocation and X / Plus / Star local search patterns.

Ground-truth mission coordinates and hidden priority labels are evaluation-only
and are never used to choose actions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

from src.agents.observable_utils import frontier_map
from src.environment.city_twin import Cell, CityTwinEnvironment
from src.safety.runtime_monitor import RuntimeSafetyMonitor

PATTERN_MODES = ("x", "plus", "star")
PATTERN_LABELS = {"x": "X", "plus": "Plus", "star": "Star"}


@dataclass
class PRISMConfig:
    """Auditable weights for PRISM probability, memory and allocation."""

    observed_weight: float = 2.20
    uncertainty_weight: float = 0.70
    frontier_weight: float = 1.55
    memory_weight: float = 1.10
    pheromone_weight: float = 0.22
    novelty_weight: float = 0.95
    revisit_penalty: float = 1.10
    congestion_penalty: float = 0.70
    distance_weight: float = 0.10
    segment_mass_weight: float = 1.00
    segment_frontier_weight: float = 0.65
    segment_unseen_weight: float = 0.35
    memory_decay: float = 0.94
    memory_diffusion: float = 0.16
    probability_temperature: float = 0.55
    pattern_radius: float = 6.0
    assignment_interval: float = 6.0


PRISM_TUNABLE_FIELDS = tuple(field.name for field in fields(PRISMConfig))
PRISM_PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "observed_weight": (0.5, 4.0),
    "uncertainty_weight": (0.1, 2.5),
    "frontier_weight": (0.2, 3.5),
    "memory_weight": (0.0, 3.0),
    "pheromone_weight": (0.0, 1.5),
    "novelty_weight": (0.1, 2.5),
    "revisit_penalty": (0.1, 3.0),
    "congestion_penalty": (0.0, 2.5),
    "distance_weight": (0.01, 0.40),
    "segment_mass_weight": (0.2, 2.5),
    "segment_frontier_weight": (0.0, 2.0),
    "segment_unseen_weight": (0.0, 2.0),
    "memory_decay": (0.75, 0.995),
    "memory_diffusion": (0.0, 0.40),
    "probability_temperature": (0.20, 1.50),
    "pattern_radius": (3.0, 10.0),
    "assignment_interval": (2.0, 16.0),
}


def _clip_config(config: PRISMConfig) -> PRISMConfig:
    values = asdict(config)
    for name, (low, high) in PRISM_PARAMETER_BOUNDS.items():
        values[name] = float(np.clip(float(values[name]), low, high))
    return PRISMConfig(**values)


def _pattern_name(mode: str) -> str:
    if mode not in PATTERN_MODES:
        raise ValueError(f"Unknown PRISM pattern {mode!r}; expected one of {PATTERN_MODES}")
    return PATTERN_LABELS[mode]


class PRISMPolicy:
    """Shared-memory probability-map search with region/pattern allocation."""

    name = "PRISM-Safe"

    def __init__(
        self,
        seed: int = 42,
        *,
        pattern_mode: str = "star",
        monitor: RuntimeSafetyMonitor | None = None,
        model_path: str | Path | None = None,
        strategy_name: str | None = None,
        **overrides: Any,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.monitor = monitor or RuntimeSafetyMonitor()
        self.model_path = str(model_path) if model_path else None
        self.checkpoint_metadata: dict[str, Any] = {}
        config_values = asdict(PRISMConfig())
        loaded_mode: str | None = None
        if model_path:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"PRISM checkpoint not found: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            # v5 can consume v4 checkpoints so old experiments remain reproducible.
            supported = {
                "safeswarm-prism-v1",
                "safeswarm-prism-ant-v1",
                "safeswarm-sparx-v1",
                "safeswarm-sparx-v2",
            }
            if payload.get("format") not in supported:
                raise ValueError(f"Unsupported PRISM checkpoint format in {path}")
            config_values.update(dict(payload.get("config", {})))
            loaded_mode = str(payload.get("pattern_mode", "star"))
            self.checkpoint_metadata = dict(payload.get("metadata", {}))

        known = set(config_values)
        for key, value in overrides.items():
            if key in known:
                config_values[key] = float(value)
        self.config = _clip_config(PRISMConfig(**config_values))
        self.pattern_mode = loaded_mode or str(pattern_mode)
        _pattern_name(self.pattern_mode)
        selected = self.checkpoint_metadata.get("selected_pattern")
        self.name = strategy_name or (
            "PRISM-Safe" if model_path and selected else f"PRISM-{_pattern_name(self.pattern_mode)}-Safe"
        )

        self.memory_map: np.ndarray | None = None
        self.search_probability_map: np.ndarray | None = None
        self.search_utility_map: np.ndarray | None = None
        self._last_update_step = -1
        self._last_assignment_step = -10_000
        self._assignments: dict[int, int] = {}
        self._region_anchors: dict[int, Cell] = {}
        self._region_scores: dict[int, float] = {}
        self._goals: dict[int, Cell] = {}
        self._distance_cache: dict[tuple[int, Cell], dict[Cell, int]] = {}
        self.memory_updates = 0
        self.assignment_refreshes = 0
        self.goal_switches = 0
        self.pattern_goal_counts = {mode: 0 for mode in PATTERN_MODES}

    def reset_episode_state(self) -> None:
        self.memory_map = None
        self.search_probability_map = None
        self.search_utility_map = None
        self._last_update_step = -1
        self._last_assignment_step = -10_000
        self._assignments.clear()
        self._region_anchors.clear()
        self._region_scores.clear()
        self._goals.clear()
        self._distance_cache.clear()
        self.memory_updates = 0
        self.assignment_refreshes = 0
        self.goal_switches = 0
        self.pattern_goal_counts = {mode: 0 for mode in PATTERN_MODES}
        self.monitor.reset()

    @staticmethod
    def _blocked_mask(env: CityTwinEnvironment) -> np.ndarray:
        mask = np.zeros((env.grid_size, env.grid_size), dtype=bool)
        for cell in env.obstacles | env.restricted_zones:
            mask[cell] = True
        return mask

    @staticmethod
    def _four_neighbor_mean(values: np.ndarray) -> np.ndarray:
        padded = np.pad(values, 1, mode="constant", constant_values=0.0)
        return (
            padded[:-2, 1:-1] + padded[2:, 1:-1]
            + padded[1:-1, :-2] + padded[1:-1, 2:]
        ) / 4.0

    def _ensure_memory(self, env: CityTwinEnvironment) -> np.ndarray:
        shape = (env.grid_size, env.grid_size)
        if self.memory_map is None or self.memory_map.shape != shape:
            self.memory_map = np.zeros(shape, dtype=float)
        return self.memory_map

    def _update_probability_state(self, env: CityTwinEnvironment) -> None:
        if self._last_update_step == env.steps:
            return
        memory = self._ensure_memory(env)
        frontier = frontier_map(env)
        blocked = self._blocked_mask(env)
        confidence = np.clip(1.0 - env.uncertainty_map, 0.0, 1.0)
        observed = np.clip(env.observation_map * confidence, 0.0, 1.0)
        novelty = 1.0 / (1.0 + env.visit_counts.astype(float))
        uncertain_novel = np.clip(env.uncertainty_map, 0.0, 1.0) * novelty
        pheromone = np.clip(env.pheromone_map / 2.0, 0.0, 1.0) * novelty
        visits = env.visit_counts.astype(float)
        visit_pressure = visits / (1.0 + visits)

        evidence = (
            1.50 * observed + 0.85 * frontier + 0.40 * uncertain_novel
            + 0.12 * pheromone - 0.45 * visit_pressure
        )
        memory *= float(self.config.memory_decay)
        memory += evidence
        diffusion = float(self.config.memory_diffusion)
        if diffusion > 0:
            propagated = self._four_neighbor_mean(np.maximum(memory, 0.0))
            memory[:] = (1.0 - diffusion) * memory + diffusion * propagated
        for cell in env.discovered_missions:
            memory[cell] = min(float(memory[cell]), -0.75)
        memory[blocked] = -2.0
        memory[:] = np.clip(memory, -3.0, 4.0)

        c = self.config
        utility = (
            c.observed_weight * observed
            + c.uncertainty_weight * uncertain_novel
            + c.frontier_weight * frontier
            + c.memory_weight * np.maximum(memory, 0.0)
            + c.pheromone_weight * pheromone
            + c.novelty_weight * novelty
            - c.revisit_penalty * visit_pressure
        )
        for aid, position in env.get_positions().items():
            if not env.agents[aid].done:
                utility[position] -= c.congestion_penalty
        utility[blocked] = -np.inf

        probability = np.zeros_like(utility, dtype=float)
        traversable = np.isfinite(utility)
        if np.any(traversable):
            logits = utility[traversable] / max(0.05, float(c.probability_temperature))
            logits -= float(np.max(logits))
            weights = np.exp(np.clip(logits, -60.0, 20.0))
            total = float(np.sum(weights))
            if not np.isfinite(total) or total <= 1e-12:
                weights = np.ones_like(weights)
                total = float(weights.size)
            probability[traversable] = weights / total
        self.search_utility_map = utility
        self.search_probability_map = probability
        self._last_update_step = int(env.steps)
        self._distance_cache.clear()
        self.memory_updates += 1

    def _segments(self, env: CityTwinEnvironment) -> list[list[Cell]]:
        axis = max(2, int(math.ceil(math.sqrt(max(1, env.n_agents)))))
        edges = np.linspace(0, env.grid_size, axis + 1, dtype=int)
        blocked = env.obstacles | env.restricted_zones
        segments: list[list[Cell]] = []
        for ix in range(axis):
            for iy in range(axis):
                cells = [
                    (x, y)
                    for x in range(int(edges[ix]), int(edges[ix + 1]))
                    for y in range(int(edges[iy]), int(edges[iy + 1]))
                    if (x, y) not in blocked
                ]
                if cells:
                    segments.append(cells)
        return segments

    def _segment_statistics(
        self, env: CityTwinEnvironment, segments: list[list[Cell]]
    ) -> tuple[dict[int, float], dict[int, Cell]]:
        assert self.search_probability_map is not None
        frontier = frontier_map(env)
        scores: dict[int, float] = {}
        anchors: dict[int, Cell] = {}
        c = self.config
        for rid, cells in enumerate(segments):
            mass = float(sum(self.search_probability_map[cell] for cell in cells))
            frontier_mass = float(np.mean([frontier[cell] for cell in cells]))
            unseen = float(np.mean([env.uncertainty_map[cell] > 0.90 for cell in cells]))
            revisit = float(np.mean([env.visit_counts[cell] > 1 for cell in cells]))
            scores[rid] = (
                c.segment_mass_weight * mass
                + c.segment_frontier_weight * frontier_mass
                + c.segment_unseen_weight * unseen
                - 0.35 * revisit
            )
            anchors[rid] = max(
                cells,
                key=lambda cell: (
                    float(self.search_probability_map[cell]),
                    float(frontier[cell]),
                    -float(env.visit_counts[cell]),
                ),
            )
        return scores, anchors

    @staticmethod
    def _manhattan(a: Cell, b: Cell) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _refresh_assignments(self, env: CityTwinEnvironment, segments: list[list[Cell]]) -> None:
        scores, anchors = self._segment_statistics(env, segments)
        self._region_scores = scores
        self._region_anchors = anchors
        available = set(scores)
        assignments: dict[int, int] = {}
        active = [aid for aid in sorted(env.agents) if not env.agents[aid].done]
        active.sort(key=lambda aid: (env.agents[aid].battery_level, aid))
        for aid in active:
            if not available:
                available = set(scores)
            current = env.agents[aid].position
            best = max(
                available,
                key=lambda rid: (
                    scores[rid]
                    - self.config.distance_weight
                    * self._manhattan(current, anchors[rid]) / max(1.0, env.grid_size),
                    scores[rid], -rid,
                ),
            )
            assignments[aid] = best
            available.discard(best)
        if assignments != self._assignments:
            self.goal_switches += sum(
                self._assignments.get(aid) != rid for aid, rid in assignments.items()
            )
        self._assignments = assignments
        self._last_assignment_step = int(env.steps)
        self.assignment_refreshes += 1

    def _pattern_cells(
        self, env: CityTwinEnvironment, anchor: Cell, region_cells: Iterable[Cell]
    ) -> list[Cell]:
        region = set(region_cells)
        radius = max(1, int(round(self.config.pattern_radius)))
        ordered: list[Cell] = [anchor]
        seen = {anchor}
        def add(cell: Cell) -> None:
            if cell in region and cell not in seen and cell not in env.obstacles and cell not in env.restricted_zones:
                ordered.append(cell)
                seen.add(cell)
        for r in range(1, radius + 1):
            if self.pattern_mode in {"plus", "star"}:
                for dx, dy in ((r, 0), (-r, 0), (0, r), (0, -r)):
                    add((anchor[0] + dx, anchor[1] + dy))
            if self.pattern_mode in {"x", "star"}:
                for dx, dy in ((r, r), (r, -r), (-r, r), (-r, -r)):
                    add((anchor[0] + dx, anchor[1] + dy))
        return ordered

    def _select_pattern_goal(
        self, env: CityTwinEnvironment, agent_id: int, region_cells: list[Cell], anchor: Cell
    ) -> Cell:
        assert self.search_probability_map is not None
        assert self.search_utility_map is not None
        pattern = self._pattern_cells(env, anchor, region_cells) or [anchor]
        current = env.agents[agent_id].position
        frontier = frontier_map(env)
        unresolved = [c for c in pattern if env.uncertainty_map[c] > 0.35 or env.visit_counts[c] == 0]
        candidates = unresolved or pattern
        goal = max(
            candidates,
            key=lambda cell: (
                1.25 * float(self.search_probability_map[cell]) * env.traversable_cell_count
                + 0.55 * float(np.nan_to_num(self.search_utility_map[cell], nan=-10.0, neginf=-10.0))
                + 0.90 * float(frontier[cell])
                + 0.65 / (1.0 + float(env.visit_counts[cell]))
                - 0.42 * float(env.visit_counts[cell])
                - 0.035 * self._manhattan(current, cell)
            ),
        )
        self.pattern_goal_counts[self.pattern_mode] += 1
        return goal

    def _distance_field(self, env: CityTwinEnvironment, goal: Cell) -> dict[Cell, int]:
        key = (int(env.steps), goal)
        if key in self._distance_cache:
            return self._distance_cache[key]
        blocked = env.obstacles | env.restricted_zones
        distances: dict[Cell, int] = {}
        if goal in blocked or not env.in_bounds(goal):
            return distances
        queue: deque[Cell] = deque([goal])
        distances[goal] = 0
        while queue:
            x, y = queue.popleft()
            d = distances[(x, y)] + 1
            for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if env.in_bounds(nxt) and nxt not in blocked and nxt not in distances:
                    distances[nxt] = d
                    queue.append(nxt)
        self._distance_cache[key] = distances
        return distances

    def _goal_for_agent(self, env: CityTwinEnvironment, agent_id: int, segments: list[list[Cell]]) -> Cell:
        state = env.agents[agent_id]
        hard_return = env.nearest_base_distance(state.position) * env.move_energy_cost + 17.0
        if state.battery_level <= hard_return:
            return min(env.base_stations, key=lambda c: self._manhattan(state.position, c))
        rid = self._assignments.get(agent_id)
        if rid is None or rid >= len(segments):
            return state.position
        goal = self._select_pattern_goal(env, agent_id, segments[rid], self._region_anchors[rid])
        if self._goals.get(agent_id) != goal:
            self.goal_switches += 1
        self._goals[agent_id] = goal
        return goal

    @staticmethod
    def _congestion(cell: Cell, agent_id: int, positions: Dict[int, Cell]) -> float:
        return float(sum(
            1.0 / (1.0 + float(np.hypot(cell[0] - other[0], cell[1] - other[1])))
            for oid, other in positions.items() if oid != agent_id
        ))

    def _choose_safe_action(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        goal: Cell,
        planned_positions: Dict[int, Cell],
        positions: Dict[int, Cell],
    ) -> str:
        if env.agents[agent_id].done:
            return "STAY"
        safe = self.monitor.safe_actions(env, agent_id, planned_positions, count_masked=True)
        if not safe:
            return "STAY"
        assert self.search_probability_map is not None
        assert self.search_utility_map is not None
        distances = self._distance_field(env, goal)
        current = env.agents[agent_id].position
        current_distance = distances.get(current, env.grid_size * env.grid_size)
        def score(action: str) -> tuple[float, float, str]:
            cell = env.next_position(current, action)
            distance = distances.get(cell, env.grid_size * env.grid_size)
            progress = float(current_distance - distance)
            p = float(self.search_probability_map[cell]) * env.traversable_cell_count
            u = float(np.nan_to_num(self.search_utility_map[cell], nan=-10.0, neginf=-10.0))
            novelty = 1.0 / (1.0 + float(env.visit_counts[cell]))
            value = (
                2.20 * progress + 0.35 * p + 0.20 * u + 0.35 * novelty
                - self.config.congestion_penalty * self._congestion(cell, agent_id, positions)
                - 0.06 * float(action == "STAY")
            )
            return value, -float(distance), action
        return max(safe, key=score)

    def act(self, env: CityTwinEnvironment) -> Dict[int, str]:
        self._update_probability_state(env)
        segments = self._segments(env)
        interval = max(1, int(round(self.config.assignment_interval)))
        if (
            not self._assignments
            or env.steps - self._last_assignment_step >= interval
            or any(aid not in self._assignments for aid in env.agents if not env.agents[aid].done)
        ):
            self._refresh_assignments(env, segments)
        positions = env.get_positions()
        actions: Dict[int, str] = {}
        planned: Dict[int, Cell] = {}
        for aid in sorted(env.agents):
            goal = self._goal_for_agent(env, aid, segments)
            action = self._choose_safe_action(env, aid, goal, planned, positions)
            actions[aid] = action
            planned[aid] = env.next_position(env.agents[aid].position, action)
        return actions

    def config_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self.config).items()}

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "safeswarm-prism-v1",
            "algorithm": "PRISM: Probability-guided Region-Integrated Search with Memory",
            "pattern_mode": self.pattern_mode,
            "config": self.config_dict(),
            "metadata": dict(metadata or {}),
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def diagnostics(self) -> dict[str, Any]:
        entropy = peak = 0.0
        if self.search_probability_map is not None:
            p = self.search_probability_map[self.search_probability_map > 0]
            if p.size:
                entropy = float(-np.sum(p * np.log(p + 1e-12)))
                peak = float(np.max(p))
        memory_coverage = memory_peak = 0.0
        if self.memory_map is not None:
            memory_coverage = float(np.mean(np.abs(self.memory_map) > 0.05))
            memory_peak = float(np.max(self.memory_map))
        return {
            "name": self.name,
            "checkpoint": self.model_path,
            "checkpoint_metadata": self.checkpoint_metadata,
            "prism_pattern": self.pattern_mode,
            "prism_memory_updates": self.memory_updates,
            "prism_memory_coverage": memory_coverage,
            "prism_memory_peak": memory_peak,
            "prism_assignment_refreshes": self.assignment_refreshes,
            "prism_goal_switches": self.goal_switches,
            "prism_probability_entropy": entropy,
            "prism_probability_peak": peak,
            "prism_region_count": len(self._region_scores),
            "prism_config": self.config_dict(),
            "safety_mask_rejections": int(self.monitor.mask_rejections),
            "forced_fallbacks": int(self.monitor.intervention_count),
        }


def prism_pattern_factories(seed: int = 42) -> dict[str, Any]:
    return {
        "PRISM-X-Safe": lambda: PRISMPolicy(seed=seed, pattern_mode="x", strategy_name="PRISM-X-Safe"),
        "PRISM-Plus-Safe": lambda: PRISMPolicy(seed=seed, pattern_mode="plus", strategy_name="PRISM-Plus-Safe"),
        "PRISM-Star-Safe": lambda: PRISMPolicy(seed=seed, pattern_mode="star", strategy_name="PRISM-Star-Safe"),
    }
