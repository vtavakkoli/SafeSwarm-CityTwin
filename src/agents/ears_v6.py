"""SafeSwarm v6 event-driven Ant successors.

EARS (Event-driven Ant Reallocation Search) keeps AntSwarmSafe as the default
local controller and invokes global relocation only when observable evidence
shows that an agent is stagnating, revisiting too heavily, or clustering with
teammates.  EARS-NP adds a signed repulsive/negative pheromone field that
creates exclusion halos around repeatedly searched or congested cells.

H-MAPPO-EARS is deliberately described as a MAPPO-*assisted* hierarchical
controller: the trained MAPPO policy is one local option inside an explicit
high-level option gate.  It is not presented as a new implementation of the
MAPPO training algorithm.

All action selection obeys the same partial-observability contract as the rest
of SafeSwarm.  Ground-truth mission coordinates and hidden priority labels are
never read by these policies.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.agents.observable_utils import frontier_map, observable_priority, observable_target_distance
from src.environment.city_twin import Cell, CityTwinEnvironment
from src.safety.runtime_monitor import RuntimeSafetyMonitor


@dataclass
class EARSConfig:
    # AntSwarmSafe-equivalent local behavior.  Defaults intentionally mirror
    # AntSwarmPolicy so EARS adds intervention rather than silently replacing it.
    explore_weight: float = 1.00
    uncertainty_weight: float = 0.55
    revisit_penalty: float = 0.55
    risk_weight: float = 1.40
    pheromone_weight: float = 0.45
    cluster_penalty: float = 0.85
    target_distance_weight: float = 0.025

    # Event detector.
    history_window: int = 12
    stagnation_unique_ratio: float = 0.46
    local_revisit_trigger: float = 0.54
    congestion_trigger: float = 1.15
    min_steps_before_reallocation: int = 7

    # Event-driven global relocation.  Distance and energy are explicit because
    # v5 showed that PRISM-Ant already had enough discovery but travelled too far.
    relocation_duration: int = 9
    relocation_cooldown: int = 7
    global_frontier_weight: float = 1.35
    global_evidence_weight: float = 1.10
    global_uncertainty_weight: float = 0.45
    global_novelty_weight: float = 0.72
    global_pheromone_weight: float = 0.18
    global_revisit_penalty: float = 0.95
    global_distance_penalty: float = 0.16
    global_energy_penalty: float = 0.22
    global_congestion_penalty: float = 0.70
    goal_spacing: int = 5

    # Negative pheromone / exclusion halo.  EARS-Safe sets the weight to zero;
    # EARS-NP-Safe uses the non-zero default from the factory/checkpoint.
    negative_pheromone_weight: float = 0.0
    negative_decay: float = 0.92
    negative_diffusion: float = 0.12
    negative_visit_deposit: float = 0.24
    negative_congestion_deposit: float = 0.18
    negative_escape_trigger: float = 0.48


@dataclass
class HMAPPOEARSConfig:
    """High-level option-gate parameters for MAPPO-assisted EARS."""

    mappo_uncertainty_min: float = 0.38
    mappo_uncertainty_max: float = 0.88
    mappo_max_local_revisit: float = 0.42
    evidence_priority_trigger: float = 0.18
    negative_escape_margin: float = 0.10


class EARSPolicy:
    """Ant by default; event-driven global relocation only when needed."""

    name = "EARS-Safe"

    def __init__(
        self,
        seed: int = 42,
        *,
        monitor: RuntimeSafetyMonitor | None = None,
        config: EARSConfig | None = None,
        model_path: str | Path | None = None,
        strategy_name: str | None = None,
        **overrides: Any,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.monitor = monitor or RuntimeSafetyMonitor()
        self.model_path = str(model_path) if model_path else None
        self.checkpoint_metadata: dict[str, Any] = {}
        values = asdict(config or EARSConfig())
        if model_path:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"EARS checkpoint not found: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("format") not in {
                "safeswarm-ears-v1",
                "safeswarm-ears-np-v1",
                "safeswarm-h-mappo-ears-v1",
            }:
                raise ValueError(f"Unsupported EARS checkpoint format in {path}")
            values.update(dict(payload.get("ears_config", {})))
            self.checkpoint_metadata = dict(payload.get("metadata", {}))
        for key, value in overrides.items():
            if key in values:
                values[key] = value
        # Keep integer knobs integral after JSON round-trip / tuning.
        for key in (
            "history_window", "min_steps_before_reallocation", "relocation_duration",
            "relocation_cooldown", "goal_spacing",
        ):
            values[key] = int(round(float(values[key])))
        self.config = EARSConfig(**values)
        self.name = strategy_name or self.name

        self.negative_pheromone: np.ndarray | None = None
        self._history: dict[int, deque[Cell]] = {}
        self._relocation_goals: dict[int, Cell] = {}
        self._relocation_until: dict[int, int] = {}
        self._cooldown_until: dict[int, int] = {}
        self._distance_cache: dict[tuple[int, Cell], dict[Cell, int]] = {}
        self._last_field_step = -1

        self.event_triggers = 0
        self.stagnation_triggers = 0
        self.revisit_triggers = 0
        self.congestion_triggers = 0
        self.ant_steps = 0
        self.relocation_steps = 0
        self.negative_escape_steps = 0
        self.return_steps = 0

    def reset_episode_state(self) -> None:
        self.negative_pheromone = None
        self._history.clear()
        self._relocation_goals.clear()
        self._relocation_until.clear()
        self._cooldown_until.clear()
        self._distance_cache.clear()
        self._last_field_step = -1
        self.event_triggers = 0
        self.stagnation_triggers = 0
        self.revisit_triggers = 0
        self.congestion_triggers = 0
        self.ant_steps = 0
        self.relocation_steps = 0
        self.negative_escape_steps = 0
        self.return_steps = 0
        self.monitor.reset()

    @staticmethod
    def _manhattan(a: Cell, b: Cell) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _congestion(cell: Cell, agent_id: int, positions: Dict[int, Cell]) -> float:
        total = 0.0
        for other_id, other_pos in positions.items():
            if other_id == agent_id:
                continue
            distance = float(np.hypot(cell[0] - other_pos[0], cell[1] - other_pos[1]))
            total += 2.0 if distance < 1e-9 else 1.0 / distance
        return total

    @staticmethod
    def _neighbor_mean(values: np.ndarray) -> np.ndarray:
        padded = np.pad(values, 1, mode="constant", constant_values=0.0)
        return (
            padded[:-2, 1:-1] + padded[2:, 1:-1]
            + padded[1:-1, :-2] + padded[1:-1, 2:]
        ) / 4.0

    def _ensure_negative_field(self, env: CityTwinEnvironment) -> np.ndarray:
        shape = (env.grid_size, env.grid_size)
        if self.negative_pheromone is None or self.negative_pheromone.shape != shape:
            self.negative_pheromone = np.zeros(shape, dtype=float)
        return self.negative_pheromone

    def _update_negative_field(self, env: CityTwinEnvironment) -> None:
        if self._last_field_step == env.steps:
            return
        field = self._ensure_negative_field(env)
        c = self.config
        field *= float(np.clip(c.negative_decay, 0.70, 0.999))

        visits = env.visit_counts.astype(float)
        visit_pressure = visits / (1.0 + visits)
        field += float(max(0.0, c.negative_visit_deposit)) * visit_pressure

        occupancy = np.zeros_like(field)
        positions = env.get_positions()
        for aid, pos in positions.items():
            if not env.agents[aid].done:
                occupancy[pos] += 1.0 + min(2.0, self._congestion(pos, aid, positions))
        if float(np.max(occupancy)) > 0:
            occupancy /= float(np.max(occupancy))
        field += float(max(0.0, c.negative_congestion_deposit)) * occupancy

        diffusion = float(np.clip(c.negative_diffusion, 0.0, 0.45))
        if diffusion > 0:
            field[:] = (1.0 - diffusion) * field + diffusion * self._neighbor_mean(field)
        for cell in env.obstacles | env.restricted_zones:
            field[cell] = 1.0
        field[:] = np.clip(field, 0.0, 1.0)
        self._last_field_step = int(env.steps)
        self._distance_cache.clear()

    def _history_for(self, agent_id: int) -> deque[Cell]:
        window = max(4, int(self.config.history_window))
        history = self._history.get(agent_id)
        if history is None or history.maxlen != window:
            history = deque(maxlen=window)
            self._history[agent_id] = history
        return history

    def _local_revisit_ratio(self, env: CityTwinEnvironment, cell: Cell) -> float:
        neighbors = env.get_neighbors(cell)
        if not neighbors:
            return 1.0
        return float(np.mean([float(env.visit_counts[n] > 1) for n in neighbors]))

    def _event_reason(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        positions: Dict[int, Cell],
    ) -> str | None:
        if env.steps < int(self.config.min_steps_before_reallocation):
            return None
        if env.steps < self._cooldown_until.get(agent_id, -1):
            return None
        current = env.agents[agent_id].position
        history = self._history_for(agent_id)
        history.append(current)

        if len(history) >= max(4, int(0.75 * history.maxlen)):
            unique_ratio = len(set(history)) / max(1, len(history))
            if unique_ratio <= float(self.config.stagnation_unique_ratio):
                return "stagnation"
        if self._local_revisit_ratio(env, current) >= float(self.config.local_revisit_trigger):
            return "revisit"
        if self._congestion(current, agent_id, positions) >= float(self.config.congestion_trigger):
            return "congestion"
        return None

    def _global_utility(self, env: CityTwinEnvironment) -> np.ndarray:
        c = self.config
        frontier = frontier_map(env)
        confidence = np.clip(1.0 - env.uncertainty_map, 0.0, 1.0)
        evidence = np.clip(env.observation_map * confidence, 0.0, 1.0)
        novelty = 1.0 / (1.0 + env.visit_counts.astype(float))
        uncertainty = np.clip(env.uncertainty_map, 0.0, 1.0) * novelty
        pheromone = np.clip(env.pheromone_map / 2.0, 0.0, 1.0) * novelty
        visits = env.visit_counts.astype(float)
        revisit = visits / (1.0 + visits)
        negative = self._ensure_negative_field(env)
        utility = (
            c.global_frontier_weight * frontier
            + c.global_evidence_weight * evidence
            + c.global_uncertainty_weight * uncertainty
            + c.global_novelty_weight * novelty
            + c.global_pheromone_weight * pheromone
            - c.global_revisit_penalty * revisit
            - c.negative_pheromone_weight * negative
        )
        for cell in env.obstacles | env.restricted_zones:
            utility[cell] = -np.inf
        return utility

    def _select_relocation_goals(
        self,
        env: CityTwinEnvironment,
        agents: list[int],
        positions: Dict[int, Cell],
    ) -> None:
        if not agents:
            return
        utility = self._global_utility(env)
        finite_cells = [tuple(map(int, cell)) for cell in np.argwhere(np.isfinite(utility))]
        chosen: list[Cell] = []
        for aid in sorted(agents, key=lambda i: (env.agents[i].battery_level, i)):
            current = env.agents[aid].position
            remaining_battery = float(env.agents[aid].battery_level)

            def score(cell: Cell) -> float:
                distance = float(self._manhattan(current, cell))
                congestion = self._congestion(cell, aid, positions)
                spacing_penalty = 0.0
                if chosen:
                    nearest_chosen = min(self._manhattan(cell, goal) for goal in chosen)
                    spacing_penalty = max(0.0, float(self.config.goal_spacing - nearest_chosen))
                estimated_energy = distance * float(env.move_energy_cost)
                reserve = float(env.nearest_base_distance(cell) * env.move_energy_cost + 12.0)
                infeasible = estimated_energy + reserve > remaining_battery
                return float(
                    utility[cell]
                    - self.config.global_distance_penalty * distance / max(1.0, env.grid_size)
                    - self.config.global_energy_penalty * estimated_energy / 100.0
                    - self.config.global_congestion_penalty * congestion
                    - 0.20 * spacing_penalty
                    - 10.0 * float(infeasible)
                )

            goal = max(finite_cells, key=score) if finite_cells else current
            self._relocation_goals[aid] = goal
            self._relocation_until[aid] = int(env.steps + max(1, self.config.relocation_duration))
            self._cooldown_until[aid] = int(
                self._relocation_until[aid] + max(1, self.config.relocation_cooldown)
            )
            chosen.append(goal)

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

    def _battery_return_needed(self, env: CityTwinEnvironment, agent_id: int) -> bool:
        state = env.agents[agent_id]
        reserve = env.nearest_base_distance(state.position) * env.move_energy_cost + 16.0
        return bool(state.battery_level <= reserve)

    def _goal_action(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        goal: Cell,
        planned: Dict[int, Cell],
    ) -> str:
        safe = self.monitor.safe_actions(env, agent_id, planned, count_masked=True)
        if not safe:
            return "STAY"
        current = env.agents[agent_id].position
        distances = self._distance_field(env, goal)
        current_distance = distances.get(current, env.grid_size * env.grid_size)
        negative = self._ensure_negative_field(env)

        def key(action: str) -> tuple[float, float, str]:
            cell = env.next_position(current, action)
            distance = distances.get(cell, env.grid_size * env.grid_size)
            progress = float(current_distance - distance)
            novelty = 1.0 / (1.0 + float(env.visit_counts[cell]))
            value = (
                2.35 * progress
                + 0.35 * novelty
                - 0.48 * float(env.visit_counts[cell])
                - self.config.negative_pheromone_weight * float(negative[cell])
                - 0.04 * float(action == "STAY")
            )
            return value, -float(distance), action
        return max(safe, key=key)

    def _ant_action(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        planned: Dict[int, Cell],
        positions: Dict[int, Cell],
    ) -> str:
        safe = self.monitor.safe_actions(env, agent_id, planned, count_masked=True)
        if not safe:
            return "STAY"
        current = env.agents[agent_id].position
        c = self.config
        negative = self._ensure_negative_field(env)

        def key(action: str) -> tuple[float, float, str]:
            cell = env.next_position(current, action)
            visits = float(env.visit_counts[cell])
            novelty = 1.0 / (1.0 + visits)
            priority = observable_priority(env, cell)
            uncertainty = float(env.uncertainty_map[cell])
            pheromone = float(env.pheromone_map[cell])
            score = (
                c.explore_weight * novelty
                + c.uncertainty_weight * uncertainty * novelty
                - c.revisit_penalty * visits
                + c.risk_weight * priority
                + c.pheromone_weight * pheromone * novelty
                - c.cluster_penalty * self._congestion(cell, agent_id, positions)
                - c.target_distance_weight * observable_target_distance(env, cell)
                - c.negative_pheromone_weight * float(negative[cell])
                - 8.0 * float(cell in env.restricted_zones)
            )
            return score, novelty, action
        return max(safe, key=key)

    def _negative_escape_action(
        self,
        env: CityTwinEnvironment,
        agent_id: int,
        planned: Dict[int, Cell],
        positions: Dict[int, Cell],
    ) -> str | None:
        if self.config.negative_pheromone_weight <= 0:
            return None
        field = self._ensure_negative_field(env)
        current = env.agents[agent_id].position
        if float(field[current]) < float(self.config.negative_escape_trigger):
            return None
        safe = self.monitor.safe_actions(env, agent_id, planned, count_masked=True)
        if not safe:
            return None
        action = min(
            safe,
            key=lambda a: (
                float(field[env.next_position(current, a)]),
                float(env.visit_counts[env.next_position(current, a)]),
                self._congestion(env.next_position(current, a), agent_id, positions),
                a,
            ),
        )
        target = env.next_position(current, action)
        if float(field[current]) - float(field[target]) < 0.06:
            return None
        return action

    def act(self, env: CityTwinEnvironment) -> Dict[int, str]:
        self._update_negative_field(env)
        positions = env.get_positions()
        newly_triggered: list[int] = []
        for aid in sorted(env.agents):
            if env.agents[aid].done or self._battery_return_needed(env, aid):
                continue
            active = env.steps <= self._relocation_until.get(aid, -1)
            goal = self._relocation_goals.get(aid)
            if active and goal is not None and env.agents[aid].position != goal:
                continue
            if active and goal == env.agents[aid].position:
                self._relocation_until[aid] = env.steps - 1
            reason = self._event_reason(env, aid, positions)
            if reason is not None:
                newly_triggered.append(aid)
                self.event_triggers += 1
                if reason == "stagnation":
                    self.stagnation_triggers += 1
                elif reason == "revisit":
                    self.revisit_triggers += 1
                elif reason == "congestion":
                    self.congestion_triggers += 1
        self._select_relocation_goals(env, newly_triggered, positions)

        actions: Dict[int, str] = {}
        planned: Dict[int, Cell] = {}
        for aid in sorted(env.agents):
            state = env.agents[aid]
            if state.done:
                actions[aid] = "STAY"
                planned[aid] = state.position
                continue
            if self._battery_return_needed(env, aid):
                goal = min(env.base_stations, key=lambda cell: self._manhattan(state.position, cell))
                action = self._goal_action(env, aid, goal, planned)
                self.return_steps += 1
            elif env.steps <= self._relocation_until.get(aid, -1) and aid in self._relocation_goals:
                action = self._goal_action(env, aid, self._relocation_goals[aid], planned)
                self.relocation_steps += 1
            else:
                escape = self._negative_escape_action(env, aid, planned, positions)
                if escape is not None:
                    action = escape
                    self.negative_escape_steps += 1
                else:
                    action = self._ant_action(env, aid, planned, positions)
                    self.ant_steps += 1
            actions[aid] = action
            planned[aid] = env.next_position(state.position, action)
        return actions

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fmt = "safeswarm-ears-np-v1" if self.config.negative_pheromone_weight > 0 else "safeswarm-ears-v1"
        target.write_text(
            json.dumps(
                {
                    "format": fmt,
                    "algorithm": "EARS: Event-driven Ant Reallocation Search",
                    "ears_config": self.config_dict(),
                    "metadata": dict(metadata or {}),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def diagnostics(self) -> dict[str, Any]:
        negative_mean = negative_peak = 0.0
        if self.negative_pheromone is not None:
            negative_mean = float(np.mean(self.negative_pheromone))
            negative_peak = float(np.max(self.negative_pheromone))
        total_option_steps = max(
            1,
            self.ant_steps + self.relocation_steps + self.negative_escape_steps + self.return_steps,
        )
        return {
            "name": self.name,
            "checkpoint": self.model_path,
            "checkpoint_metadata": self.checkpoint_metadata,
            "ears_event_triggers": self.event_triggers,
            "ears_stagnation_triggers": self.stagnation_triggers,
            "ears_revisit_triggers": self.revisit_triggers,
            "ears_congestion_triggers": self.congestion_triggers,
            "ears_ant_fraction": self.ant_steps / total_option_steps,
            "ears_relocation_fraction": self.relocation_steps / total_option_steps,
            "ears_negative_escape_fraction": self.negative_escape_steps / total_option_steps,
            "ears_return_fraction": self.return_steps / total_option_steps,
            "ears_negative_pheromone_mean": negative_mean,
            "ears_negative_pheromone_peak": negative_peak,
            "ears_active_goals": len(self._relocation_goals),
            "ears_config": self.config_dict(),
            "safety_mask_rejections": int(self.monitor.mask_rejections),
            "forced_fallbacks": int(self.monitor.intervention_count),
        }


class EARSNegativePheromonePolicy(EARSPolicy):
    """EARS with a repulsive exclusion-halo pheromone field."""

    name = "EARS-NP-Safe"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        config = kwargs.pop("config", None)
        if config is None and not kwargs.get("model_path"):
            config = EARSConfig(
                negative_pheromone_weight=0.82,
                negative_decay=0.91,
                negative_diffusion=0.14,
                negative_visit_deposit=0.27,
                negative_congestion_deposit=0.20,
            )
        kwargs.setdefault("strategy_name", "EARS-NP-Safe")
        super().__init__(*args, config=config, **kwargs)


class HMAPPOEARSPolicy(EARSNegativePheromonePolicy):
    """MAPPO-assisted hierarchical option controller over EARS primitives."""

    name = "H-MAPPO-EARS-Safe"

    def __init__(
        self,
        *args: Any,
        mappo_model_path: str | Path | None = None,
        hierarchical_config: HMAPPOEARSConfig | None = None,
        **kwargs: Any,
    ) -> None:
        model_path = kwargs.get("model_path")
        loaded_hconfig: dict[str, Any] = {}
        loaded_mappo: str | None = None
        if model_path:
            path = Path(model_path)
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                loaded_hconfig = dict(payload.get("hierarchical_config", {}))
                loaded_mappo = payload.get("mappo_checkpoint")
        super().__init__(*args, **kwargs)
        defaults = asdict(hierarchical_config or HMAPPOEARSConfig())
        defaults.update(loaded_hconfig)
        self.hierarchical_config = HMAPPOEARSConfig(**defaults)
        self.name = "H-MAPPO-EARS-Safe"
        self.mappo_model_path = str(mappo_model_path or loaded_mappo or "") or None
        self._mappo = None
        if self.mappo_model_path:
            from src.agents.ppo_v3 import TrainableMAPPOPolicy

            self._mappo = TrainableMAPPOPolicy(
                seed=int(self.rng.integers(0, 2**31 - 1)),
                model_path=self.mappo_model_path,
                deterministic_eval=True,
            )
        self.mappo_option_steps = 0
        self.evidence_option_steps = 0

    def reset_episode_state(self) -> None:
        super().reset_episode_state()
        self.mappo_option_steps = 0
        self.evidence_option_steps = 0
        if self._mappo is not None and hasattr(self._mappo, "reset_episode_state"):
            self._mappo.reset_episode_state()

    def _use_mappo_option(self, env: CityTwinEnvironment, aid: int) -> bool:
        if self._mappo is None:
            return False
        current = env.agents[aid].position
        local_priority = max(observable_priority(env, cell) for cell in env.get_neighbors(current))
        if local_priority >= self.hierarchical_config.evidence_priority_trigger:
            return False
        uncertainty = float(env.uncertainty_map[current])
        if not (
            self.hierarchical_config.mappo_uncertainty_min
            <= uncertainty
            <= self.hierarchical_config.mappo_uncertainty_max
        ):
            return False
        if self._local_revisit_ratio(env, current) > self.hierarchical_config.mappo_max_local_revisit:
            return False
        return True

    def act(self, env: CityTwinEnvironment) -> Dict[int, str]:
        # Reuse EARS event detection and negative-field maintenance, but preserve
        # explicit high-level choices so diagnostics can identify what helped.
        self._update_negative_field(env)
        positions = env.get_positions()
        newly_triggered: list[int] = []
        for aid in sorted(env.agents):
            if env.agents[aid].done or self._battery_return_needed(env, aid):
                continue
            active = env.steps <= self._relocation_until.get(aid, -1)
            if active and aid in self._relocation_goals and env.agents[aid].position != self._relocation_goals[aid]:
                continue
            reason = self._event_reason(env, aid, positions)
            if reason is not None:
                newly_triggered.append(aid)
                self.event_triggers += 1
                if reason == "stagnation":
                    self.stagnation_triggers += 1
                elif reason == "revisit":
                    self.revisit_triggers += 1
                elif reason == "congestion":
                    self.congestion_triggers += 1
        self._select_relocation_goals(env, newly_triggered, positions)

        mappo_actions = self._mappo.act(env) if self._mappo is not None else {}
        actions: Dict[int, str] = {}
        planned: Dict[int, Cell] = {}
        for aid in sorted(env.agents):
            state = env.agents[aid]
            if state.done:
                action = "STAY"
            elif self._battery_return_needed(env, aid):
                goal = min(env.base_stations, key=lambda cell: self._manhattan(state.position, cell))
                action = self._goal_action(env, aid, goal, planned)
                self.return_steps += 1
            elif env.steps <= self._relocation_until.get(aid, -1) and aid in self._relocation_goals:
                action = self._goal_action(env, aid, self._relocation_goals[aid], planned)
                self.relocation_steps += 1
            else:
                escape = self._negative_escape_action(env, aid, planned, positions)
                if escape is not None:
                    action = escape
                    self.negative_escape_steps += 1
                else:
                    local_priority = max(
                        observable_priority(env, cell) for cell in env.get_neighbors(state.position)
                    )
                    if local_priority >= self.hierarchical_config.evidence_priority_trigger:
                        action = self._ant_action(env, aid, planned, positions)
                        self.evidence_option_steps += 1
                    elif self._use_mappo_option(env, aid) and mappo_actions.get(aid) is not None:
                        candidate = mappo_actions[aid]
                        safe = self.monitor.safe_actions(env, aid, planned, count_masked=True)
                        action = candidate if candidate in safe else self._ant_action(env, aid, planned, positions)
                        self.mappo_option_steps += int(action == candidate)
                    else:
                        action = self._ant_action(env, aid, planned, positions)
                        self.ant_steps += 1
            actions[aid] = action
            planned[aid] = env.next_position(state.position, action)
        return actions

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "format": "safeswarm-h-mappo-ears-v1",
                    "algorithm": "H-MAPPO-EARS: MAPPO-assisted hierarchical EARS option controller",
                    "ears_config": self.config_dict(),
                    "hierarchical_config": asdict(self.hierarchical_config),
                    "mappo_checkpoint": self.mappo_model_path,
                    "metadata": dict(metadata or {}),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def diagnostics(self) -> dict[str, Any]:
        data = super().diagnostics()
        total = max(
            1,
            self.ant_steps + self.relocation_steps + self.negative_escape_steps
            + self.return_steps + self.mappo_option_steps + self.evidence_option_steps,
        )
        data.update(
            {
                "name": self.name,
                "h_mappo_ears_mappo_fraction": self.mappo_option_steps / total,
                "h_mappo_ears_evidence_fraction": self.evidence_option_steps / total,
                "h_mappo_ears_config": asdict(self.hierarchical_config),
                "mappo_checkpoint": self.mappo_model_path,
            }
        )
        return data
