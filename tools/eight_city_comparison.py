"""Frozen comparison of all 26 original implementations and four EARS-Info variants.

New, exploratory evaluation on reused geography; never silently train, replace a
checkpoint, project a target, omit a method, or substitute a synthetic real map.
Core controllers and the simulator are not modified. See README for commands.
"""

from __future__ import annotations

# ruff: noqa: E402 -- allow direct CLI execution with the repository on sys.path.
import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import shutil
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from experiments.benchmark_ears_v8 import topology_metrics
from experiments.publication_ears_v9 import ObservedEnvironment, canonical_hash, digest
from experiments.run_city_benchmark import run_episode
from src.agents.ears_v6 import EARSNegativePheromonePolicy, EARSPolicy, HMAPPOEARSPolicy
from src.agents.grpo_v3 import BEHAVIOR_FEATURE_NAMES
from src.agents.prism_ant import PRISMAntPolicy
from src.agents.prism_pattern import PRISMPolicy
from src.agents.safe_ppo_core import FEATURE_NAMES, VALUE_FEATURE_NAMES
from src.agents.trainable_policies import TRAINABLE_POLICY_CLASSES, evaluation_factories
from src.environment.obstacles import (
    _cache_path,
    load_layer_cache,
    load_real_city_layers,
    save_layer_cache,
    synthetic_city_layers,
)
from src.evaluation.metrics import episode_operational_score
from src.training.geography import apply_start_zone, load_protocol, select_cities, validate_protocol
from src.training.swap_protocol import seeded_mission_view
from tools.mechanism_probe_v9 import METHODS as INFO_METHODS
from tools.mechanism_probe_v9 import InformationRoutingConfig, make_policy

ORIGINAL = [
    "RandomAgent",
    "GreedyAgent",
    "SafetyFilteredGreedy",
    "SafeSwarmAgent",
    "AntSwarmSafe",
    "BeeSwarmSafe",
    "PSOSwarmSafe",
    "UA-HBAS-Safe",
    "PRISM-Ant-Safe",
    "GRPO-Safe",
    "IPPO-Safe",
    "MAPPO-Safe",
    "QMIX-Safe",
    "MADDPG-Safe",
    "HAPPO-Safe",
    "MAT-Safe",
    "PRISM-X-Safe",
    "PRISM-Plus-Safe",
    "PRISM-Star-Safe",
    "GRPO-Safe-Ablation-NoMemory",
    "GRPO-Safe-Ablation-NoPropagation",
    "GRPO-Safe-Ablation-NoLearnedBehavior",
    "PRISM-Safe",
    "EARS-Safe",
    "EARS-NP-Safe",
    "H-MAPPO-EARS-Safe",
]
METHODS = ORIGINAL + INFO_METHODS
CITIES = ["San Francisco", "Paris", "Barcelona", "Rome", "New York", "Chicago", "Tokyo", "Melbourne"]
CHECKPOINTS = [
    "grpo_safe.json",
    "ippo_safe.json",
    "mappo_safe.json",
    "happo_safe.json",
    "prism_safe.json",
    "prism_x_safe.json",
    "prism_plus_safe.json",
    "prism_star_safe.json",
    "prism_ant_safe.json",
    "ears_safe.json",
    "ears_np_safe.json",
    "h_mappo_ears_safe.json",
]
KEYS = ["panel", "city", "agents", "dropout", "dataset_seed", "episode", "start_zone", "seed", "strategy"]
EXPECTED_CANDIDATE = "3023a2cb283206b9dfd0c0679bdfbb29aaad9e633b599c2470cd4b196bf326bf"


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def method_manifest():
    result = []
    for name in METHODS:
        group = "repository heuristic/hybrid"
        if name in ["RandomAgent", "GreedyAgent"]:
            group = "unfiltered reference control"
        elif name in ["QMIX-Safe", "MADDPG-Safe", "MAT-Safe"]:
            group = "lightweight policy-style adaptation; NOT full published algorithm"
        elif name.split("-Ablation")[0] in TRAINABLE_POLICY_CLASSES:
            group = "repository-trained linear/residual policy; NOT external SOTA reproduction"
        elif name in INFO_METHODS:
            group = "frozen information-routing candidate/control"
        result.append(
            dict(strategy=name, implementation=group, filtered=name not in ["RandomAgent", "GreedyAgent"])
        )
    return result


class ObservationView:
    """Hide unsensed target fields, including legacy reads overwritten downstream.

    Shared observed maps and geometry remain available. This is an explicit
    benchmark observation contract, not a sandbox or a decentralized network.
    """

    def __init__(self, env):
        self._env = env
        self.data_metadata = {"source": env.data_source}

    @property
    def priority_map(self):
        return self._env.observation_map

    @property
    def mission_zones(self):
        return set(self._env.discovered_missions)

    @property
    def priority_cells(self):
        return {c: float(self._env.observation_map[c]) for c in self.mission_zones}

    def remaining_missions(self):
        return set()

    def weighted_target_discovery(self):
        # Training-only critic diagnostics are also computed during inference.
        # Withhold the privileged target denominator; actions must pass the
        # native-versus-restricted parity audit before real evaluation.
        return 0.0

    def __getattr__(self, name):
        if name in {"weighted_target_discovery", "step", "reset"} or name.startswith("_"):
            raise AttributeError(f"Privileged environment access from policy: {name}")
        return getattr(self._env, name)


class PairedEnvironment(ObservedEnvironment):
    """Pre-sampled communication draws indexed by step/robot, independent of deaths."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 824726]))
        self.communication_draws = rng.random((self.max_steps, self.n_agents))
        # Preserve true initial completion (base sensing may already find all targets).
        if self.mission_zones and self.discovered_missions >= self.mission_zones:
            self.completion_step = 0

    def update_communications(self):
        for aid, agent in self.agents.items():
            if agent.done:
                continue
            offline = self.communication_draws[self.steps - 1, aid] < self.communication_dropout_prob
            agent.communication_status = not offline
            agent.comm_loss_steps = agent.comm_loss_steps + 1 if offline else 0
            self.communication_total_samples += 1
            self.communication_online_samples += int(not offline)


class ObservablePolicy:
    def __init__(self, policy):
        self.policy = policy
        self.monitor = getattr(policy, "monitor", None)
        self.policy_seconds = 0.0
        self.view = None

    def act(self, env):
        start = time.perf_counter()
        # Several controllers use environment identity to retain episode memory.
        if self.view is None or self.view._env is not env:
            self.view = ObservationView(env)
        actions = self.policy.act(self.view)
        self.policy_seconds += time.perf_counter() - start
        if set(actions) != set(env.agents) or any(
            a not in env.all_candidate_actions() for a in actions.values()
        ):
            raise ValueError("Invalid/incomplete action dictionary")
        return actions


def make(method, seed, frozen):
    if method in INFO_METHODS:
        return make_policy(method, seed, frozen / "candidate.json", asdict(InformationRoutingConfig()))
    factories = evaluation_factories(seed, frozen / "checkpoints")
    if set(factories) != set(ORIGINAL):
        raise ValueError(f"Original registry mismatch: {set(factories) ^ set(ORIGINAL)}")
    policy = factories[method]()
    # The inherited GRPO ablation factory did not explicitly request deterministic
    # inference. Fix evaluation mode here without editing its controller or weights.
    if hasattr(policy, "deterministic_eval"):
        policy.deterministic_eval = True
    return policy


def _finite_tree(value):
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite_tree(v) for v in value)
    return not isinstance(value, float) or np.isfinite(value)


def check_checkpoints(model_dir, smoke=False):
    missing = [str(model_dir / n) for n in CHECKPOINTS if not (model_dir / n).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing original trained checkpoints (no defaults or retraining):\n" + "\n".join(missing)
        )
    for name in CHECKPOINTS:
        payload = json.loads((model_dir / name).read_text())
        if not _finite_tree(payload):
            raise ValueError(f"Nonfinite checkpoint: {name}")
        if not smoke and any(
            word in json.dumps(payload.get("metadata", {})).lower()
            for word in ["smoke", "synthetic", "unit_test", "ci-only"]
        ):
            raise ValueError(f"Test-only checkpoint cannot be used for real evaluation: {name}")
        if name in ["grpo_safe.json", "ippo_safe.json", "mappo_safe.json", "happo_safe.json"]:
            if payload.get("feature_names") != list(FEATURE_NAMES) or payload.get(
                "value_feature_names"
            ) != list(VALUE_FEATURE_NAMES):
                raise ValueError(f"Checkpoint feature schema mismatch: {name}")
            for field, size in [
                ("ppo_residual_weights", len(FEATURE_NAMES)),
                ("critic_weights", len(VALUE_FEATURE_NAMES)),
            ]:
                if np.asarray(payload.get(field, [])).shape != (size,):
                    raise ValueError(f"Malformed {field}: {name}")
            if not payload.get("params"):
                raise ValueError(f"Missing selected policy parameters: {name}")
            if name == "grpo_safe.json":
                if payload.get("behavior_feature_names") != list(BEHAVIOR_FEATURE_NAMES):
                    raise ValueError("GRPO behavior feature schema mismatch")
                if np.asarray(payload.get("behavior_weights", [])).shape != (len(BEHAVIOR_FEATURE_NAMES), 7):
                    raise ValueError("GRPO behavior matrix mismatch")
                if np.asarray(payload.get("behavior_bias", [])).shape != (7,) or not payload.get("memory"):
                    raise ValueError("Missing GRPO behavior bias or memory configuration")
        elif name.startswith("prism"):
            if not payload.get("config") or payload.get("pattern_mode") not in ["x", "plus", "star"]:
                raise ValueError(f"Missing PRISM configuration: {name}")
            if name == "prism_ant_safe.json" and not payload.get("hybrid_config"):
                raise ValueError("Missing PRISM-Ant hybrid configuration")
        elif not payload.get("ears_config"):
            raise ValueError(f"Missing EARS configuration: {name}")
        if name == "h_mappo_ears_safe.json" and not payload.get("hierarchical_config"):
            raise ValueError("Missing hierarchical configuration")
    # Validate actual checkpoint deserialization, not merely JSON presence.
    factories = evaluation_factories(42, model_dir)
    if set(factories) != set(ORIGINAL):
        raise ValueError("All 26 original methods are mandatory")
    for name, factory in factories.items():
        policy = factory()
        if name in TRAINABLE_POLICY_CLASSES:
            payload = json.loads((model_dir / (name.lower().replace("-", "_") + ".json")).read_text())
            if not np.array_equal(policy.residual_weights, payload["ppo_residual_weights"]):
                raise ValueError(f"Checkpoint silently failed to load: {name}")
            if not np.array_equal(policy.critic_weights, payload["critic_weights"]):
                raise ValueError(f"Critic silently failed to load: {name}")
            if name == "GRPO-Safe" and not np.array_equal(
                policy.behavior_weights, payload["behavior_weights"]
            ):
                raise ValueError("GRPO behavior weights silently failed to load")


def source_hashes():
    paths = list((ROOT / "src").rglob("*.py")) + list((ROOT / "experiments").glob("*.py"))
    paths += list((ROOT / "tools").glob("*.py"))
    paths += [
        ROOT / n
        for n in [
            "docker/Dockerfile",
            "docker-compose.yaml",
            "requirements.txt",
            "requirements-dev.txt",
            "configs/publication_protocol_v7.json",
        ]
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for p in sorted(paths)
    }


def runtime():
    cpu = platform.processor()
    path = Path("/proc/cpuinfo")
    if path.exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in path.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        cpu=cpu,
        logical_cpus=os.cpu_count(),
        packages={
            d.metadata["Name"].lower(): d.version
            for d in importlib.metadata.distributions()
            if d.metadata.get("Name")
        },
        threads={
            k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
        },
        image_digest=os.environ.get("EIGHT_CITY_IMAGE_DIGEST", "not supplied"),
    )


def jobs(plan):
    for panel in plan["panels"]:
        settings = (
            [(8, 0.03)]
            if panel == "swap"
            else list(itertools.product(plan["agents_grid"], plan["dropout_grid"]))
        )
        if plan["smoke_only"]:
            settings = [(2, 0.03)]
        for ci, city in enumerate(plan["cities"]):
            for agents, dropout in settings:
                for dataset_seed in plan["swap_seeds"] if panel == "swap" else [0]:
                    for episode in range(plan["episodes"]):
                        offset = (episode + ci + plan["panels"].index(panel)) % len(METHODS)
                        for method in METHODS[offset:] + METHODS[:offset]:
                            yield dict(
                                panel=panel,
                                city=city,
                                agents=agents,
                                dropout=dropout,
                                dataset_seed=dataset_seed,
                                episode=episode,
                                start_zone=plan["zones"][episode % 2],
                                seed=plan["seed"] + ci * 10000 + episode,
                                strategy=method,
                            )


def identity(job):
    return canonical_hash({k: job[k] for k in KEYS})


def validate_layers(layers, grid, city, smoke):
    meta = layers["metadata"]
    if not smoke and (meta.get("source") != "openstreetmap" or meta.get("mission_source") != "openstreetmap"):
        raise ValueError(f"{city}: genuine OSM geometry AND targets required; no fallback accepted")
    if not layers["mission_zones"]:
        raise ValueError(f"{city}: empty mission set")
    for field in ["obstacles", "restricted_zones", "mission_zones", "base_stations"]:
        if any(not (0 <= x < grid and 0 <= y < grid) for x, y in layers[field]):
            raise ValueError(f"{city}: out-of-grid {field}")
    if layers["mission_zones"] & (layers["obstacles"] | layers["restricted_zones"]):
        raise ValueError(f"{city}: target overlaps blocked cell; original maps are not automatically altered")
    if set(layers["priority_cells"]) != layers["mission_zones"] or any(
        not np.isfinite(v) or v <= 0 for v in layers["priority_cells"].values()
    ):
        raise ValueError(f"{city}: missing or invalid target priorities")


def create_smoke_checkpoints(root):
    root.mkdir(parents=True, exist_ok=True)
    meta = {"purpose": "synthetic smoke only; untrained; not publication evidence"}
    for name, cls in TRAINABLE_POLICY_CLASSES.items():
        cls(seed=5).save_checkpoint(root / (name.lower().replace("-", "_") + ".json"), meta)
    for pattern in [None, "x", "plus", "star"]:
        PRISMPolicy(pattern_mode=pattern or "star").save_checkpoint(
            root / f"prism{('_' + pattern) if pattern else ''}_safe.json", meta
        )
    PRISMAntPolicy().save_checkpoint(root / "prism_ant_safe.json", meta)
    EARSPolicy().save_checkpoint(root / "ears_safe.json", meta)
    EARSNegativePheromonePolicy().save_checkpoint(root / "ears_np_safe.json", meta)
    HMAPPOEARSPolicy(mappo_model_path=root / "mappo_safe.json").save_checkpoint(
        root / "h_mappo_ears_safe.json", meta
    )


def candidate_provenance(args, model_dir, candidate, smoke):
    """Keep historical replication strict; new training requires recorded provenance."""
    if args.candidate_mode == "historical":
        if not smoke and (not candidate.is_file() or digest(candidate) != EXPECTED_CANDIDATE):
            actual = digest(candidate) if candidate.is_file() else "missing"
            raise ValueError(
                f"Historical candidate {candidate}: {actual}; expected {EXPECTED_CANDIDATE}. "
                "Restore the original archive, or start a NEW training study with "
                "docker compose run --rm ears-v9-train-all. See README."
            )
        return {"mode": "historical", "expected_sha256": EXPECTED_CANDIDATE}
    if not args.training_manifest:
        raise ValueError("Retrained evaluation requires --training-manifest")
    path = Path(args.training_manifest).resolve()
    record = json.loads(path.read_text())
    protocol = load_protocol(ROOT / "configs/publication_protocol_v7.json")
    if (
        record.get("format") != "eight-city-training-v1"
        or record.get("status") != "complete"
        or record.get("smoke_only") != smoke
        or record.get("protocol") != protocol
        or record.get("source_hashes") != source_hashes()
        or record.get("runtime") != runtime()
    ):
        raise ValueError("Training provenance/scope/environment does not match this evaluation")
    expected_stages = {"learn", "coordinate", "prism", "ears"}
    if set(record.get("stages", {})) != expected_stages:
        raise ValueError("Training provenance lacks completed stages")
    for city in protocol["cities"]:
        cache = _cache_path(args.cache_dir, city["place"], 12 if smoke else 40, city["radius_m"])
        expected = record.get("map_stage", {}).get("files", {}).get("cache/" + cache.name)
        if not cache.is_file() or digest(cache) != expected:
            raise ValueError(f"Map changed or absent since training preparation: {city['name']}")
    hashes = {name: digest(model_dir / name) for name in CHECKPOINTS}
    if hashes != record.get("checkpoint_hashes"):
        raise ValueError("Retrained checkpoints changed since selection")
    if digest(candidate) != hashes["ears_safe.json"] or digest(candidate) != record.get("candidate_sha256"):
        raise ValueError("Retrained candidate must share the selected EARS-Safe checkpoint")
    return {"mode": "retrained", "manifest_sha256": digest(path), "training": record}


def freeze(args):
    out = Path(args.output_root).resolve()
    frozen = out / "frozen"
    if (frozen / "plan.json").exists():
        plan = verify(out)
        if plan["profile"] != args.profile or plan["smoke_only"] != (args.command == "smoke"):
            raise ValueError("Resume profile differs; use the original command/root")
        if plan.get("candidate_provenance", {}).get("mode", "historical") != args.candidate_mode:
            raise ValueError("Resume candidate mode differs; use the original command/root")
        return plan
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty and has no frozen plan; use a new root")
    smoke = args.command == "smoke"
    model_dir = Path(args.model_dir).resolve()
    candidate = Path(args.candidate_checkpoint).resolve()
    for source in [model_dir, candidate.parent, Path(args.cache_dir).resolve()]:
        if out == source or out in source.parents or source in out.parents:
            raise ValueError("Output root must be separate from checkpoints, candidate, and cache")
    if not smoke or args.candidate_mode == "retrained":
        check_checkpoints(model_dir, smoke=smoke)
    provenance = candidate_provenance(args, model_dir, candidate, smoke)
    if not smoke:
        if any(v != "1" for v in runtime()["threads"].values()):
            raise ValueError(
                "Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS to 1 (Compose does this)"
            )
    protocol = load_protocol(ROOT / "configs/publication_protocol_v7.json")
    if not all(validate_protocol(protocol).values()):
        raise ValueError("Invalid original train/validation/test protocol")
    cities = select_cities(protocol, "test")
    if [c["name"] for c in cities] != CITIES:
        raise ValueError("Original eight cities changed")
    if smoke:
        cities = cities[:1]
    grid, horizon, episodes = (12, 8, 2) if smoke else (40, 160, 20)
    # Acquire and validate every map before creating an immutable study.
    maps = []
    for ci, city in enumerate(cities):
        if smoke and args.candidate_mode == "historical":
            layers = synthetic_city_layers(grid, 100 + ci, city["place"])
        else:
            cache = _cache_path(args.cache_dir, city["place"], grid, city["radius_m"])
            if not cache.exists() and not args.allow_download:
                raise FileNotFoundError(
                    f"Missing original cached map: {cache}. Restore it, or explicitly use "
                    "--allow-download for a NEW snapshot study."
                )
            layers = load_real_city_layers(
                city["place"],
                grid,
                100 + ci,
                radius_m=city["radius_m"],
                cache_dir=args.cache_dir,
                allow_network=args.allow_download,
            )
        validate_layers(layers, grid, city["name"], smoke)
        maps.append(layers)
    out.mkdir(parents=True, exist_ok=True)
    staging = out / "preparing"
    staging.mkdir()
    try:
        cp = staging / "checkpoints"
        if smoke and args.candidate_mode == "historical":
            create_smoke_checkpoints(cp)
            shutil.copyfile(cp / "ears_safe.json", staging / "candidate.json")
        else:
            cp.mkdir()
            for name in CHECKPOINTS:
                shutil.copyfile(model_dir / name, cp / name)
            shutil.copyfile(candidate, staging / "candidate.json")
        atomic_json(staging / "candidate_provenance.json", provenance)
        check_checkpoints(cp, smoke)
        audit, views = [], []
        for ci, (city, layers) in enumerate(zip(cities, maps, strict=True)):
            save_layer_cache(staging / f"map-{ci}.json", layers)
            for zone in protocol["start_zones"]["test"]:
                zoned = apply_start_zone(layers, grid, zone)
                audit.append(
                    dict(
                        city=city["name"],
                        zone=zone,
                        targets=len(layers["mission_zones"]),
                        obstacles=len(layers["obstacles"]),
                        **topology_metrics(zoned, grid),
                    )
                )
            signatures = []
            for seed in [2042] if smoke else [2042, 3051, 4060]:
                view = seeded_mission_view(layers, seed)
                meta = view["metadata"]
                signatures.append(meta["swap_signature"])
                views.append(
                    dict(
                        city=city["name"],
                        dataset_seed=seed,
                        signature=meta["swap_signature"],
                        changed=meta["swap_dataset_changed"],
                        targets=len(view["mission_zones"]),
                    )
                )
            eligible = len(set(signatures)) == len(signatures) and all(
                v["changed"] for v in views if v["city"] == city["name"]
            )
            for view in views:
                if view["city"] == city["name"]:
                    view["city_eligible_changed_swap"] = eligible
        plan = dict(
            format="eight-city-all-methods-v1",
            profile=args.profile,
            smoke_only=smoke,
            cities=[c["name"] for c in cities],
            methods=METHODS,
            method_manifest=method_manifest(),
            grid_size=grid,
            max_steps=horizon,
            episodes=episodes,
            agents_grid=[4, 8, 16, 32] if args.profile == "full" else [8],
            dropout_grid=[0.0, 0.03, 0.15, 0.30] if args.profile == "full" else [0.03],
            panels=["mission_stop", "fixed_budget", "swap"],
            swap_seeds=[2042] if smoke else [2042, 3051, 4060],
            zones=protocol["start_zones"]["test"],
            seed=24102026,
            routing_config=asdict(InformationRoutingConfig()),
            map_audit=audit,
            swap_views=views,
            source_hashes=source_hashes(),
            runtime=runtime(),
            primary=(
                "EARS-Info-v9 minus EARS-Safe J: mission_stop, 8 robots, dropout 0.03; "
                "exact two-sided city test"
            ),
            evidence_status="EXPLORATORY REUSED GEOGRAPHY; no claim of untouched confirmation",
            fairness=(
                "same maps/starts/step-indexed communication draws; "
                "observed-only policy view; native filters/return logic"
            ),
            snapshot_policy="preserved supplied cache; explicit downloads may be new geography snapshots",
            allow_download=args.allow_download,
            execution="serial; rotate method order; no fitting or policy selection",
            source_model_dir=str(model_dir),
            source_candidate=str(candidate),
            candidate_provenance=provenance,
        )
        plan["expected_rows"] = sum(1 for _ in jobs(plan))
        # Archive the exact source bytes, Dockerfile, requirements, and loaded environment.
        for name in plan["source_hashes"]:
            dst = staging / "source" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dst)
        plan["artifact_hashes"] = {
            str(p.relative_to(staging)): digest(p) for p in sorted(staging.rglob("*")) if p.is_file()
        }
        atomic_json(staging / "plan.json", plan)
        (staging / "plan.sha256").write_text(digest(staging / "plan.json") + "\n")
        staging.rename(frozen)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if not any(out.iterdir()):
            out.rmdir()
        raise
    return plan


def verify(root, execution=True):
    frozen = root / "frozen"
    plan = json.loads((frozen / "plan.json").read_text())
    if digest(frozen / "plan.json") != (frozen / "plan.sha256").read_text().strip():
        raise ValueError("Frozen plan changed")
    for name, expected in plan["artifact_hashes"].items():
        if digest(frozen / name) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    if execution and (plan["source_hashes"] != source_hashes() or plan["runtime"] != runtime()):
        raise ValueError(
            "Code, dependencies, image provenance, CPU/platform, or thread settings changed; "
            "restore the original environment"
        )
    return plan


def scenario(plan, frozen, job):
    layers = load_layer_cache(frozen / f"map-{plan['cities'].index(job['city'])}.json")
    if job["panel"] == "swap":
        layers = seeded_mission_view(layers, job["dataset_seed"])
    layers = apply_start_zone(layers, plan["grid_size"], job["start_zone"])
    return PairedEnvironment(
        grid_size=plan["grid_size"],
        n_agents=job["agents"],
        seed=job["seed"],
        layers=layers,
        max_steps=plan["max_steps"],
        allow_network=False,
        communication_dropout_prob=job["dropout"],
        stop_when_all_missions_found=job["panel"] != "fixed_budget",
    )


def contract_audit(plan, frozen):
    """Metamorphic audit on a fixture; never tune from eight-city outcomes."""
    layers = synthetic_city_layers(12, 20261014, "contract fixture")
    evidence = []
    for method in METHODS:
        # Check native and restricted observations produce identical actions along
        # the same short trajectory, including partial sensing and dropouts.
        a = make(method, 778, frozen)
        b = make(method, 778, frozen)
        ea = PairedEnvironment(
            12,
            4,
            778,
            layers=copy.deepcopy(layers),
            max_steps=8,
            communication_dropout_prob=0.30,
            stop_when_all_missions_found=False,
        )
        eb = PairedEnvironment(
            12,
            4,
            778,
            layers=copy.deepcopy(layers),
            max_steps=8,
            communication_dropout_prob=0.30,
            stop_when_all_missions_found=False,
        )
        view = ObservationView(eb)
        for _ in range(8):
            aa = a.act(ea)
            bb = b.act(view)
            if aa != bb:
                raise ValueError(f"{method}: action depends on masked information; inspect before evaluation")
            ea.step(aa)
            eb.step(bb)
        evidence.append(
            dict(
                strategy=method,
                fixture_steps=8,
                native_observable_action_parity=True,
                deterministic_eval=getattr(b, "deterministic_eval", None),
                runtime_monitor=type(getattr(b, "monitor", None)).__name__,
            )
        )
    return evidence


def observe(plan, frozen, job):
    env = scenario(plan, frozen, job)
    policy = make(job["strategy"], job["seed"], frozen)
    wrapper = ObservablePolicy(policy)
    row = run_episode(env, wrapper).to_dict(**job, steps=env.steps, data_source=env.data_source)
    row.update(
        operational_score=episode_operational_score(row),
        completed_all_targets=int(env.completion_step is not None),
        capped_completion_steps=plan["max_steps"] if env.completion_step is None else env.completion_step,
        policy_seconds=wrapper.policy_seconds,
        safe_returns=env.safe_return_count,
        planning_seconds=float(getattr(policy, "planning_seconds", 0)),
        accepted_relocations=int(getattr(policy, "accepted_relocations", 0)),
        relocation_steps=int(getattr(policy, "relocation_steps", 0)),
        event_requests=int(getattr(policy, "event_triggers", 0)),
        communication_draws_sha256=hashlib.sha256(env.communication_draws.tobytes()).hexdigest(),
        diagnostics=policy.diagnostics() if hasattr(policy, "diagnostics") else {},
    )
    if not _finite_tree(row):
        raise ValueError("Nonfinite episode output")
    return row


def read_ledger(db, plan):
    expected = {identity(job): job for job in jobs(plan)}
    rows, seen = [], set()
    if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise ValueError("SQLite integrity failure")
    for ident, payload, sha in db.execute("SELECT id,payload,sha256 FROM episodes ORDER BY rowid"):
        row = json.loads(payload)
        if (
            ident not in expected
            or identity(row) != ident
            or canonical_hash(row) != sha
            or any(row[k] != v for k, v in expected[ident].items())
            or not _finite_tree(row)
            or abs(episode_operational_score(row) - row["operational_score"]) > 1e-12
        ):
            raise ValueError("Corrupt or foreign episode journal")
        rows.append(row)
        seen.add(ident)
    return rows, seen


def run(args):
    out = Path(args.output_root).resolve()
    # A sibling lock also covers first freeze/preflight and avoids two creators.
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / (out.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = freeze(args)
        frozen = out / "frozen"
        atomic_json(out / "contract_audit.json", contract_audit(plan, frozen))
        pd.DataFrame(plan["map_audit"]).to_csv(out / "map_audit.csv", index=False)
        pd.DataFrame(plan["method_manifest"]).to_csv(out / "method_manifest.csv", index=False)
        print(
            f"{'SMOKE ONLY' if plan['smoke_only'] else 'EXPLORATORY'}: "
            f"{len(plan['cities'])} cities, {len(METHODS)} methods, {plan['expected_rows']} episodes",
            flush=True,
        )
        if args.command == "preflight":
            print(
                "Preflight passed; no research episodes run. Use the same profile/root for run.", flush=True
            )
            return
        with sqlite3.connect(out / "episodes.sqlite") as db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("CREATE TABLE IF NOT EXISTS episodes (id TEXT PRIMARY KEY, payload TEXT, sha256 TEXT)")
            rows, done = read_ledger(db, plan)
            started, count = time.perf_counter(), 0
            atomic_json(
                out / "progress.json",
                dict(status="running", rows=len(rows), expected_rows=plan["expected_rows"]),
            )
            for job in jobs(plan):
                ident = identity(job)
                if ident in done:
                    continue
                if args.max_jobs is not None and count >= args.max_jobs:
                    break
                try:
                    row = observe(plan, frozen, job)
                except BaseException as exc:
                    atomic_json(
                        out / "failure.json",
                        dict(job=job, error=f"{type(exc).__name__}: {exc}", rows=len(rows)),
                    )
                    raise
                with db:
                    db.execute(
                        "INSERT INTO episodes VALUES (?,?,?)",
                        (ident, json.dumps(row, allow_nan=False), canonical_hash(row)),
                    )
                rows.append(row)
                count += 1
                if count % len(METHODS) == 0:
                    seconds = time.perf_counter() - started
                    remaining_hours = (plan["expected_rows"] - len(rows)) * seconds / count / 3600
                    atomic_json(
                        out / "progress.json",
                        dict(
                            status="running",
                            rows=len(rows),
                            expected_rows=plan["expected_rows"],
                            estimated_remaining_hours=remaining_hours,
                        ),
                    )
                    print(
                        f"{len(rows)}/{plan['expected_rows']} {job['city']} {job['panel']} "
                        f"N={job['agents']} p={job['dropout']} estimated_remaining={remaining_hours:.1f}h",
                        flush=True,
                    )
        if len(rows) != plan["expected_rows"]:
            atomic_json(
                out / "progress.json",
                dict(status="paused_incomplete", rows=len(rows), expected_rows=plan["expected_rows"]),
            )
            print("Incomplete: journal saved; no final rankings generated.", flush=True)
            return
        verify(out)
        from tools.report_eight_city_comparison import report

        report(out, locked=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["preflight", "run", "report", "smoke"])
    p.add_argument(
        "--profile",
        choices=["core", "full"],
        default="full",
        help="core=24,000 episodes; full=168,000, including all methods at every stress setting",
    )
    p.add_argument("--output-root", default="results/eight-city-v9-full")
    p.add_argument("--model-dir", default="results/train/checkpoints")
    p.add_argument("--candidate-mode", choices=["historical", "retrained"], default="historical")
    p.add_argument("--training-manifest", help="Completed training provenance for a NEW retrained study")
    p.add_argument(
        "--candidate-checkpoint", default="results/publication-v9-sparse-filter-v2/frozen/ears_safe.json"
    )
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument(
        "--allow-download",
        action="store_true",
        help=(
            "Explicitly permit new OSM snapshots when original cached maps are missing; "
            "never synthetic fallback"
        ),
    )
    p.add_argument(
        "--max-jobs", type=int, help="Pause after this many new episodes; never certify partial results"
    )
    return p


def main():
    args = parser().parse_args()
    if args.max_jobs is not None and args.max_jobs < 1:
        raise ValueError("--max-jobs must be positive")
    if args.command == "report":
        from tools.report_eight_city_comparison import report

        report(Path(args.output_root).resolve())
    else:
        run(args)


if __name__ == "__main__":
    main()
