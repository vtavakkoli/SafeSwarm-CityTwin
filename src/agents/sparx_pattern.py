"""SPARX: Swarm Probability-map Allocation & Region eXploration.

SPARX is an observable-only, safety-aware multi-agent search controller aimed at
the failure mode seen in PPO-family policies on held-out cities: agents move a
lot, but revisit the same areas and do not explicitly allocate the team across
high-value unexplored regions.

The controller builds a shared search-probability map from sensed evidence,
uncertainty/frontiers, pheromones, visit history and signed swarm memory. It
segments the traversable map, assigns different agents to different promising
regions, and then searches each assigned region with one of three geometric
patterns:

* ``x``    - diagonal X rays;
* ``plus`` - cardinal + rays;
* ``star`` - the union of X and + (eight rays).

Ground-truth mission coordinates and hidden priority labels are never consulted
for action selection. The resulting map is a *search utility probability*, not
a calibrated probability that a hidden target exists at a cell.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

from src.agents.observable_utils import frontier_map, observable_priority
from src.environment.city_twin import Cell, CityTwinEnvironment
from src.safety.runtime_monitor import RuntimeSafetyMonitor

PATTERN_MODES = ("x", "plus", "star")
PATTERN_LABELS = {"x": "X", "plus": "Plus", "star": "Star"}


@dataclass
class SPARXConfig:
    """Auditable weights used by the SPARX probability/assignment controller."""

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


SPARX_TUNABLE_FIELDS = tuple(field.name for field in fields(SPARXConfig))
SPARX_PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
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


def _clip_config(config: SPARXConfig) -> SPARXConfig:
    values = asdict(config)
    for name, (low, high) in SPARX_PARAMETER_BOUNDS.items():
        values[name] = float(np.clip(float(values[name]), low, high))
    return SPARXConfig(**values)


def _pattern_name(mode: str) -> str:
    if mode not in PATTERN_MODES:
        raise ValueError(f"Unknown SPARX pattern {mode!r}; expected one of {PATTERN_MODES}")
    return PATTERN_LABELS[mode]


class SPARXPolicy:
    """Shared-memory probability-map search with region/pattern allocation."""

    name = "SPARX-Safe"

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

        config_values = asdict(SPARXConfig())
        loaded_mode: str | None = None
        if model_path:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"SPARX checkpoint not found: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("format") not in {"safeswarm-sparx-v1", "safeswarm-sparx-v2"}:
                raise ValueError(f"Unsupported SPARX checkpoint format in {path}")
            config_values.update(dict(payload.get("config", {})))
            loaded_mode = str(payload.get("pattern_mode", "star"))
            self.checkpoint_metadata = dict(payload.get("metadata", {}))

        known = set(config_values)
        for key, value in overrides.items():
            if key in known:
                config_values[key] = float(value)
        self.config = _clip_config(SPARXConfig(**config_values))
        self.pattern_mode = loaded_mode or str(pattern_mode)
        _pattern_name(self.pattern_mode)
        self.name = strategy_name or (
            "SPARX-Safe" if model_path and self.checkpoint_metadata.get("selected_pattern")
            else f"SPARX-{_pattern_name(self.pattern_mode)}-Safe"
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
        self.pattern_goal_counts: dict[str, int] = {mode: 0 for mode in PATTERN_MODES}

    # ------------------------------------------------------------------
    # State / probability map
    # ------------------------------------------------------------------
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
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
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
        visit_pressure = env.visit_counts.astype(float) / (1.0 + env.visit_counts.astype(float))

        evidence = (
            1.50 * observed
            + 0.85 * frontier
            + 0.40 * uncertain_novel
            + 0.12 * pheromone
            - 0.45 * visit_pressure
        )
        memory *= float(self.config.memory_decay)
        memory += evidence

        # Positive evidence may spread a short distance through traversable space,
        # but never through known obstacles/restricted cells.
        diffusion = float(self.config.memory_diffusion)
        if diffusion > 0.0:
            positive = np.maximum(memory, 0.0)
            propagated = self._four_neighbor_mean(positive)
            memory[:] = (1.0 - diffusion) * memory + diffusion * propagated

        for cell in env.discovered_missions:
            # Once resolved, the location becomes repulsive so the team leaves it.
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

        # Current swarm occupancy should not become a set of attractive beacons.
        for aid, position in env.get_positions().items():
            if not env.agents[aid].done:
                utility[position] -= c.congestion_penalty
        utility[blocked] = -np.inf

        traversable = np.isfinite(utility)
        probability = np.zeros_like(utility, dtype=float)
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

    # ------------------------------------------------------------------
    # Segmentation / assignment
    # ------------------------------------------------------------------
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
        for region_id, cells in enumerate(segments):
            probs = np.asarray([self.search_probability_map[cell] for cell in cells], dtype=float)
            frontiers = np.asarray([frontier[cell] for cell in cells], dtype=float)
            unseen = np.asarray([env.uncertainty_map[cell] > 0.90 for cell in cells], dtype=float)
            visits = np.asarray([env.visit_counts[cell] for cell in cells], dtype=float)
            mass = float(np.sum(probs))
            frontier_mass = float(np.mean(frontiers)) if frontiers.size else 0.0
            unseen_share = float(np.mean(unseen)) if unseen.size else 0.0
            revisit_share = float(np.mean(visits > 1)) if visits.size else 0.0
            scores[region_id] = float(
                c.segment_mass_weight * mass
                + c.segment_frontier_weight * frontier_mass
                + c.segment_unseen_weight * unseen_share
                - 0.35 * revisit_share
            )
            anchors[region_id] = max(
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
        # Agents with less battery are assigned first so they get nearer regions.
        active.sort(key=lambda aid: (env.agents[aid].battery_level, aid))
        for aid in active:
            if not available:
                available = set(scores)
            current = env.agents[aid].position
            best = max(
                available,
                key=lambda region_id: (
                    scores[region_id]
                    - self.config.distance_weight
                    * self._manhattan(current, anchors[region_id])
                    / max(1.0, env.grid_size),
                    scores[region_id],
                    -region_id,
                ),
            )
            assignments[aid] = best
            available.discard(best)

        if assignments != self._assignments:
            self.goal_switches += sum(
                self._assignments.get(aid) != region for aid, region in assignments.items()
            )
        self._assignments = assignments
        self._last_assignment_step = int(env.steps)
        self.assignment_refreshes += 1

    # ------------------------------------------------------------------
    # Pattern planning
    # ------------------------------------------------------------------
    def _pattern_cells(
        self,
        env: CityTwinEnvironment,
        anchor: Cell,
        region_cells: Iterable[Cell],
    ) -> list[Cell]:
        region = set(region_cells)
        radius = max(1, int(round(self.config.pattern_radius)))
        ordered: list[Cell] = [anchor]
        seen = {anchor}

        def add(cell: Cell) -> None:
            if (
                cell in region
                and cell not in seen
                and env.in_bounds(cell)
                and cell not in env.obstacles
                and cell not in env.restricted_zones
            ):
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
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        region_cells: list[Cell],
        anchor: Cell,
    ) -> Cell:
        assert self.search_probability_map is not None
        assert self.search_utility_map is not None
        pattern = self._pattern_cells(env, anchor, region_cells)
        if not pattern:
            pattern = [anchor]

        current = env.agents[agent_id].position
        phase = (env.steps // 3 + agent_id) % max(1, len(pattern))
        frontier = frontier_map(env)

        def score(item: tuple[int, Cell]) -> tuple[float, float, float]:
            index, cell = item
            visits = float(env.visit_counts[cell])
            novelty = 1.0 / (1.0 + visits)
            probability_scale = float(self.search_probability_map[cell]) * env.traversable_cell_count
            utility = float(np.nan_to_num(self.search_utility_map[cell], nan=-10.0, neginf=-10.0))
            phase_distance = min(
                abs(index - phase),
                len(pattern) - abs(index - phase),
            ) if len(pattern) > 1 else 0
            value = (
                1.20 * probability_scale
                + 0.55 * utility
                + 0.90 * float(frontier[cell])
                + 0.65 * novelty
                - 0.42 * visits
                - 0.035 * self._manhattan(current, cell)
                + 0.18 / (1.0 + phase_distance)
            )
            return value, novelty, -float(self._manhattan(current, cell))

        # Prefer unresolved/unseen pattern cells; once exhausted, reuse the best
        # observable probability cell in the region rather than orbiting forever.
        unresolved = [
            (index, cell)
            for index, cell in enumerate(pattern)
            if env.uncertainty_map[cell] > 0.35 or env.visit_counts[cell] == 0
        ]
        candidates = unresolved or list(enumerate(pattern))
        goal = max(candidates, key=score)[1]
        self.pattern_goal_counts[self.pattern_mode] += 1
        return goal

    def _distance_field(self, env: CityTwinEnvironment, goal: Cell) -> dict[Cell, int]:
        key = (int(env.steps), goal)
        cached = self._distance_cache.get(key)
        if cached is not None:
            return cached
        blocked = env.obstacles | env.restricted_zones
        distances: dict[Cell, int] = {}
        if goal in blocked or not env.in_bounds(goal):
            self._distance_cache[key] = distances
            return distances
        queue: deque[Cell] = deque([goal])
        distances[goal] = 0
        while queue:
            cell = queue.popleft()
            distance = distances[cell] + 1
            x, y = cell
            for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (
                    env.in_bounds(nxt)
                    and nxt not in blocked
                    and nxt not in distances
                ):
                    distances[nxt] = distance
                    queue.append(nxt)
        self._distance_cache[key] = distances
        return distances

    def _goal_for_agent(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        segments: list[list[Cell]],
    ) -> Cell:
        state = env.agents[agent_id]
        # Proactively switch the planning target to a base before the hard safety
        # mask activates. The monitor still owns the actual safety decision.
        hard_return = (
            env.nearest_base_distance(state.position) * env.move_energy_cost + 17.0
        )
        if state.battery_level <= hard_return:
            return min(env.base_stations, key=lambda cell: self._manhattan(state.position, cell))

        region_id = self._assignments.get(agent_id)
        if region_id is None or region_id >= len(segments):
            return state.position
        anchor = self._region_anchors[region_id]
        goal = self._select_pattern_goal(env, agent_id, segments[region_id], anchor)
        previous = self._goals.get(agent_id)
        if previous != goal:
            self.goal_switches += 1
        self._goals[agent_id] = goal
        return goal

    def _congestion(self, cell: Cell, agent_id: int, positions: Dict[int, Cell]) -> float:
        total = 0.0
        for other_id, other in positions.items():
            if other_id == agent_id:
                continue
            distance = float(np.hypot(cell[0] - other[0], cell[1] - other[1]))
            total += 1.0 / (1.0 + distance)
        return total

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
        safe = self.monitor.safe_actions(
            env, agent_id, planned_positions, count_masked=True
        )
        if not safe:
            return "STAY"

        assert self.search_probability_map is not None
        assert self.search_utility_map is not None
        distances = self._distance_field(env, goal)
        current = env.agents[agent_id].position
        current_distance = distances.get(current, env.grid_size * env.grid_size)

        def action_score(action: str) -> tuple[float, float, str]:
            cell = env.next_position(current, action)
            distance = distances.get(cell, env.grid_size * env.grid_size)
            progress = float(current_distance - distance)
            probability_scale = float(self.search_probability_map[cell]) * env.traversable_cell_count
            local_utility = float(np.nan_to_num(self.search_utility_map[cell], nan=-10.0, neginf=-10.0))
            novelty = 1.0 / (1.0 + float(env.visit_counts[cell]))
            congestion = self._congestion(cell, agent_id, positions)
            score = (
                2.20 * progress
                + 0.35 * probability_scale
                + 0.20 * local_utility
                + 0.35 * novelty
                - self.config.congestion_penalty * congestion
                - 0.06 * float(action == "STAY")
            )
            return score, -float(distance), action

        return max(safe, key=action_score)

    def act(self, env: CityTwinEnvironment) -> Dict[int, str]:
        self._update_probability_state(env)
        segments = self._segments(env)
        interval = max(1, int(round(self.config.assignment_interval)))
        refresh = (
            not self._assignments
            or env.steps - self._last_assignment_step >= interval
            or any(aid not in self._assignments for aid in env.agents if not env.agents[aid].done)
        )
        if refresh:
            self._refresh_assignments(env, segments)

        positions = env.get_positions()
        actions: Dict[int, str] = {}
        planned: Dict[int, Cell] = {}
        for agent_id in sorted(env.agents):
            goal = self._goal_for_agent(env, agent_id, segments)
            action = self._choose_safe_action(
                env, agent_id, goal, planned, positions
            )
            actions[agent_id] = action
            planned[agent_id] = env.next_position(env.agents[agent_id].position, action)
        return actions

    # ------------------------------------------------------------------
    # Checkpoint / diagnostics
    # ------------------------------------------------------------------
    def config_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self.config).items()}

    def save_checkpoint(
        self, path: str | Path, metadata: dict[str, Any] | None = None
    ) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "safeswarm-sparx-v2",
            "algorithm": "SPARX: Swarm Probability-map Allocation & Region eXploration",
            "pattern_mode": self.pattern_mode,
            "config": self.config_dict(),
            "metadata": dict(metadata or {}),
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def diagnostics(self) -> dict[str, Any]:
        probability_entropy = 0.0
        peak_probability = 0.0
        if self.search_probability_map is not None:
            p = self.search_probability_map[self.search_probability_map > 0]
            if p.size:
                probability_entropy = float(-np.sum(p * np.log(p + 1e-12)))
                peak_probability = float(np.max(p))
        memory_coverage = 0.0
        memory_peak = 0.0
        if self.memory_map is not None:
            memory_coverage = float(np.mean(np.abs(self.memory_map) > 0.05))
            memory_peak = float(np.max(self.memory_map))
        return {
            "name": self.name,
            "checkpoint": self.model_path,
            "checkpoint_metadata": self.checkpoint_metadata,
            "sparx_pattern": self.pattern_mode,
            "sparx_memory_updates": self.memory_updates,
            "sparx_memory_coverage": memory_coverage,
            "sparx_memory_peak": memory_peak,
            "sparx_assignment_refreshes": self.assignment_refreshes,
            "sparx_goal_switches": self.goal_switches,
            "sparx_probability_entropy": probability_entropy,
            "sparx_probability_peak": peak_probability,
            "sparx_region_count": len(self._region_scores),
            "sparx_config": self.config_dict(),
            "safety_mask_rejections": int(self.monitor.mask_rejections),
            "forced_fallbacks": int(self.monitor.intervention_count),
        }


def sparx_pattern_factories(seed: int = 42) -> dict[str, Any]:
    """Untuned pattern variants used in the generic benchmark."""

    return {
        "SPARX-X-Safe": lambda: SPARXPolicy(seed=seed, pattern_mode="x", strategy_name="SPARX-X-Safe"),
        "SPARX-Plus-Safe": lambda: SPARXPolicy(seed=seed, pattern_mode="plus", strategy_name="SPARX-Plus-Safe"),
        "SPARX-Star-Safe": lambda: SPARXPolicy(seed=seed, pattern_mode="star", strategy_name="SPARX-Star-Safe"),
    }
