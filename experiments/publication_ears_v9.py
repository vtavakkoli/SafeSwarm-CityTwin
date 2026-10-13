"""Frozen, resumable EARS research study. Training is controller calibration, not SGD."""

from __future__ import annotations

import argparse
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
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_ears_v8 import topology_metrics  # noqa: E402
from experiments.benchmark_ears_v9 import BASELINE, atomic_json, digest, layer_digest  # noqa: E402
from experiments.run_city_benchmark import run_episode  # noqa: E402
from experiments.train_ears_v6 import _ears_candidates  # noqa: E402
from src.agents.bio_swarm_agents import AntSwarmPolicy  # noqa: E402
from src.agents.ears_v6 import EARSPolicy  # noqa: E402
from src.agents.ears_v9 import EARSInformationPolicy, InformationRoutingConfig  # noqa: E402
from src.environment.city_twin import CityTwinEnvironment  # noqa: E402
from src.environment.obstacles import (  # noqa: E402
    load_layer_cache,
    load_real_city_layers,
    save_layer_cache,
    synthetic_city_layers,
)
from src.evaluation.metrics import episode_operational_score  # noqa: E402
from src.training.geography import apply_start_zone, load_protocol, select_cities, validate_protocol  # noqa: E402
from src.training.swap_protocol import seeded_mission_view  # noqa: E402

CANDIDATE = "EARS-Info-v9"
ABLATION = "EARS-Info-Local-v9"
METHODS = ["AntSwarmSafe", BASELINE, CANDIDATE, ABLATION]
EVIDENCE = ROOT / "evidence/validation-20261012"
PROTOCOL_PATH = ROOT / "configs/confirmation_protocol_v9.json"
KEYS = ["panel", "city", "agents", "dropout", "dataset_seed", "episode", "strategy"]


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sources():
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "experiments").glob("*.py"))
    paths += [ROOT / p for p in ("requirements.txt", "requirements-dev.txt", "docker/Dockerfile")]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for p in paths
    }


def software():
    names = ("numpy", "pandas", "matplotlib", "osmnx", "geopandas", "shapely", "reportlab")
    installed = {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()}
    return {"python": platform.python_version(), **{name: installed.get(name) for name in names}}


def key(row):
    return canonical_hash({k: row[k] for k in KEYS})


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def real_layers(city, grid, cache, offline):
    layers = load_real_city_layers(
        city["place"], grid, radius_m=city.get("radius_m", 1400), cache_dir=cache, allow_network=not offline
    )
    # A legacy OSM source label does not establish where its mission targets came from.
    if layers["metadata"].get("mission_source") is None and not offline:
        layers = load_real_city_layers(
            city["place"],
            grid,
            radius_m=city.get("radius_m", 1400),
            cache_dir=cache,
            force_refresh=True,
            allow_network=True,
        )
    meta = layers["metadata"]
    if meta.get("source") != "openstreetmap" or meta.get("mission_source") != "openstreetmap":
        raise ValueError(
            f"Verified OSM geography AND targets required for {city['name']}: {meta}. "
            "Refresh legacy caches online; synthetic fallback is never publication evidence."
        )
    return layers


def quality_check(layers, grid, zone):
    q = topology_metrics(apply_start_zone(layers, grid, zone), grid)
    if q["original_targets"] < 6 or q["reachable_mission_fraction"] < 1:
        raise ValueError(f"Map/start quality failure for {zone}: {q}; do not silently drop this city.")
    return q


def make_policy(method, seed, checkpoint, routing):
    if method == "AntSwarmSafe":
        return AntSwarmPolicy()
    kw = dict(seed=seed, model_path=checkpoint, strategy_name=method)
    if method == BASELINE:
        return EARSPolicy(**kw)
    config = dict(routing)
    config["enable_relocation"] = method != ABLATION
    return EARSInformationPolicy(**kw, routing_config=InformationRoutingConfig(**config))


class ObservedEnvironment(CityTwinEnvironment):
    """Evaluator-only observation; policies get no extra target information."""

    completion_step = None

    def step(self, actions):
        info = super().step(actions)
        if self.completion_step is None and self.mission_zones <= self.discovered_missions:
            self.completion_step = self.steps
        return info


def observe(layers, spec, job, checkpoint):
    env = ObservedEnvironment(
        grid_size=spec["grid_size"],
        n_agents=job["agents"],
        seed=job["seed"],
        layers=layers,
        max_steps=spec["max_steps"],
        communication_dropout_prob=job["dropout"],
        allow_network=False,
        stop_when_all_missions_found=job["panel"] != "fixed_budget",
    )
    policy = make_policy(job["strategy"], job["seed"], checkpoint, spec["routing_config"])
    metrics = run_episode(env, policy)
    row = metrics.to_dict(**job, steps=env.steps, data_source=env.data_source)
    row.update(
        operational_score=episode_operational_score(row),
        completed_all_targets=int(env.completion_step is not None),
        completion_step=env.completion_step,
        # Failures receive the full budget, including early all-agent exhaustion.
        capped_completion_steps=env.completion_step if env.completion_step is not None else spec["max_steps"],
        ended_before_budget=int(env.steps < spec["max_steps"]),
        diagnostics=policy.diagnostics() if hasattr(policy, "diagnostics") else {},
    )
    return row


def training(args):
    out = Path(args.output_root) / "training"
    if out.exists():
        raise FileExistsError(f"Training directory already exists: {out}. Use a new study root.")
    out.mkdir(parents=True)
    protocol = load_protocol(PROTOCOL_PATH)
    rows, configs = [], dict(_ears_candidates())
    finalists = list(configs)
    try:
        for split, repeats in (("train", 4), ("validation", 2)):
            cities, zones = select_cities(protocol, split), protocol["start_zones"][split]
            prepared = {}
            for city in cities:
                layers = real_layers(city, 40, args.cache_dir, args.offline)
                for zone in zones:
                    quality_check(layers, 40, zone)
                prepared[city["name"]] = layers
                save_layer_cache(out / f"{city['name']}.json", layers)
            for ci, city in enumerate(cities):
                for episode in range(repeats * len(zones)):
                    seed = (191042 if split == "train" else 291042) + ci * 10000 + episode
                    zone = zones[episode % len(zones)]
                    for name in finalists:
                        env = CityTwinEnvironment(
                            grid_size=40,
                            n_agents=8,
                            seed=seed,
                            max_steps=160,
                            layers=apply_start_zone(prepared[city["name"]], 40, zone),
                            communication_dropout_prob=0.03,
                            allow_network=False,
                        )
                        policy = EARSPolicy(seed=seed, config=configs[name])
                        row = run_episode(env, policy).to_dict(
                            strategy=name,
                            episode=episode,
                            city=city["name"],
                            start_zone=zone,
                            seed=seed,
                            split=split,
                            agents=8,
                            steps=env.steps,
                        )
                        row["operational_score"] = episode_operational_score(row)
                        rows.append(row)
                        pd.DataFrame(rows).to_csv(out / "episodes.csv", index=False)
                        print(f"Training {split}: {city['name']} {episode} {name}", flush=True)
            frame = pd.DataFrame([r for r in rows if r["split"] == split])
            # Equal city weight and matched scenarios across every candidate; safety first.
            means = frame.groupby(["strategy", "city"])["operational_score"].mean().groupby("strategy").mean()
            failures = (
                frame.groupby("strategy")[["actual_safety_incidents", "battery_failures", "blocked_moves"]]
                .sum()
                .sum(axis=1)
            )
            eligible = [n for n in finalists if failures[n] == 0]
            if not eligible:
                raise ValueError(f"No safety-eligible training candidate on {split}")
            finalists = sorted(eligible, key=lambda n: (-means[n], n))[: 2 if split == "train" else 1]
        chosen = finalists[0]
        checkpoint = out / "checkpoints/ears_safe.json"
        EARSPolicy(config=configs[chosen]).save_checkpoint(
            checkpoint,
            metadata={
                "training": "matched discrete controller calibration; no neural optimizer",
                "chosen": chosen,
                "training_cities": [c["name"] for c in select_cities(protocol, "train")],
                "validation_cities": [c["name"] for c in select_cities(protocol, "validation")],
                "test_cities_used": [],
                "episodes_sha256": digest(out / "episodes.csv"),
                "protocol_sha256": digest(PROTOCOL_PATH),
                "source_sha256_lf": sources(),
            },
        )
        atomic_json(
            out / "manifest.json",
            dict(
                status="completed", selected=chosen, checkpoint_sha256=digest(checkpoint), episodes=len(rows)
            ),
        )
        print(f"Trained checkpoint: {checkpoint}", flush=True)
    except BaseException as exc:
        atomic_json(out / "manifest.json", dict(status="failed", error=str(exc), episodes=len(rows)))
        raise


def study_spec(smoke=False, exclude_cities=""):
    geography = load_protocol(PROTOCOL_PATH)
    if not all(validate_protocol(geography).values()):
        raise ValueError("Training, validation and test geographies/start zones must be disjoint")
    reserved = select_cities(geography, "test")
    if len(reserved) != 6:
        raise ValueError("The original prespecified publication study requires six test cities")
    excluded = {name.strip() for name in exclude_cities.split(",") if name.strip()}
    unknown = excluded - {c["name"] for c in reserved}
    if unknown:
        raise ValueError(f"Unknown excluded test cities: {sorted(unknown)}")
    if smoke and excluded:
        raise ValueError("The synthetic smoke run uses its fixed single-city fixture")
    selected = [c for c in reserved if c["name"] not in excluded]
    if len(selected) < 2:
        raise ValueError("Reduced-city analysis requires at least two distinct test cities")
    exploratory_subset = bool(excluded)
    return dict(
        format="ears-publication-study-v1",
        smoke_only=smoke,
        exploratory_subset=exploratory_subset,
        original_six_city_confirmation=not exploratory_subset and not smoke,
        excluded_cities=sorted(excluded),
        candidate=CANDIDATE,
        candidate_status="research candidate; historical promotion failed",
        routing_config=asdict(InformationRoutingConfig()),
        methods=METHODS,
        cities=selected[:1] if smoke else selected,
        zones=geography["start_zones"]["test"],
        agents_grid=[4, 8] if smoke else [4, 8, 16, 32],
        dropout_grid=[0.03] if smoke else [0.0, 0.03, 0.15, 0.30],
        episodes=2 if smoke else 20,
        grid_size=16 if smoke else 40,
        max_steps=16 if smoke else 160,
        seed=391042,
        panels=["mission_stop", "fixed_budget", "swap"],
        swap_seeds=[2042, 3042, 4042],
        primary="Two-sided city sign-flip test of Info minus v7 mean J in mission_stop; alpha=0.05",
        secondary="Holm correction across all non-primary panel/comparator/metric tests",
        fixed_budget=(
            "Disable mission-completion stopping only; all-agent termination and safety remain active"
        ),
        completion=(
            "First full-target discovery step, capped at horizon for failures; "
            "success rate reported separately"
        ),
        independent_geography=(
            "EXPLORATORY, reduced geographic panel: not the original six-city confirmation; "
            "test cities may already have been inspected and cannot be claimed as untouched"
            if exploratory_subset else
            "Only untouched if these six cities have not been inspected for prior tuning"
        ),
        map_quality="OSM geography and targets; >=6 targets; all targets reachable from actual start bases",
        prior_selection_sha256=digest(EVIDENCE / "selected_model.json"),
    )


def jobs(spec):
    for panel in spec["panels"]:
        settings = (
            [(8, 0.03)] if panel == "swap" else itertools.product(spec["agents_grid"], spec["dropout_grid"])
        )
        settings = list(settings)
        for ci, city in enumerate(spec["cities"]):
            for agents, dropout in settings:
                for dataset_seed in spec["swap_seeds"] if panel == "swap" else [0]:
                    for episode in range(spec["episodes"]):
                        offset = episode % len(spec["methods"])
                        for method in spec["methods"][offset:] + spec["methods"][:offset]:
                            yield dict(
                                panel=panel,
                                city=city["name"],
                                agents=agents,
                                dropout=dropout,
                                dataset_seed=dataset_seed,
                                episode=episode,
                                strategy=method,
                                start_zone=spec["zones"][episode % len(spec["zones"])],
                                seed=spec["seed"] + ci * 10000 + episode,
                            )


def freeze(args, smoke=False):
    out = Path(args.output_root) / "frozen"
    requested_exclusions = {
        c.strip() for c in getattr(args, "exclude_cities", "").split(",") if c.strip()
    }
    if (out / "study.json").exists():
        spec = verify(args.output_root)
        if spec["smoke_only"] != smoke:
            raise ValueError("Cannot mix smoke and publication studies")
        if set(spec.get("excluded_cities", [])) != requested_exclusions:
            raise ValueError(
                "Frozen city's exclusion list differs from this command. "
                "Use the original --exclude-cities or a NEW --output-root."
            )
        return spec
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    if smoke:
        EARSPolicy().save_checkpoint(out / "ears_safe.json", metadata={"smoke_only": True})
    else:
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint {checkpoint}. Restore the original or run train, "
                "then freeze with --checkpoint results/publication-v9/training/checkpoints/ears_safe.json"
            )
        EARSPolicy(model_path=checkpoint)  # validate before any evaluation
        shutil.copyfile(checkpoint, out / "ears_safe.json")
    spec = study_spec(smoke, exclude_cities=getattr(args, "exclude_cities", ""))
    if spec["exploratory_subset"]:
        print(
            f"EXPLORATORY, NOT SIX-CITY CONFIRMATION: "
            f"{[c['name'] for c in spec['cities']]}; "
            f"excluded={spec['excluded_cities']}",
            flush=True,
        )
    spec["checkpoint_sha256"] = digest(out / "ears_safe.json")
    prior = read_json(EVIDENCE / "selected_model.json")
    spec["same_checkpoint_as_historical_validation"] = spec["checkpoint_sha256"] == prior["checkpoint_sha256"]
    spec["source_sha256_lf"] = sources()
    spec["software"] = software()
    spec["maps"] = []
    for ci, city in enumerate(spec["cities"]):
        layers = (
            synthetic_city_layers(spec["grid_size"], 42 + ci, city["place"])
            if smoke
            else real_layers(city, spec["grid_size"], args.cache_dir, args.offline)
        )
        quality = {
            z: topology_metrics(apply_start_zone(layers, spec["grid_size"], z), spec["grid_size"])
            if smoke
            else quality_check(layers, spec["grid_size"], z)
            for z in spec["zones"]
        }
        signatures = [layer_digest(seeded_mission_view(layers, s)) for s in spec["swap_seeds"]]
        if not smoke and len(set(signatures)) != len(signatures):
            raise ValueError(f"SWAP views are not distinct for {city['name']}; inspect data before freezing")
        path = out / f"map-{ci}.json"
        save_layer_cache(path, layers)
        spec["maps"].append(
            dict(
                city=city["name"],
                file=path.name,
                sha256=digest(path),
                layers_sha256=layer_digest(layers),
                metadata=layers["metadata"],
                quality=quality,
            )
        )
    spec["expected_rows"] = sum(1 for _ in jobs(spec))
    spec["frozen_utc"] = datetime.now(UTC).isoformat()
    atomic_json(out / "historical_selection.json", prior)
    atomic_json(out / "study.json", spec)
    (out / "study.sha256").write_text(digest(out / "study.json") + "\n")
    print(f"Frozen {CANDIDATE}: {spec['expected_rows']} planned episodes. {out}", flush=True)
    return spec


def verify(root, check_software=False):
    out = Path(root) / "frozen"
    spec = read_json(out / "study.json")
    if digest(out / "study.json") != (out / "study.sha256").read_text().strip():
        raise ValueError("Frozen study modified")
    if spec["source_sha256_lf"] != sources():
        raise ValueError("Source changed after freeze; restore the frozen source or start a new study")
    if check_software and spec["software"] != software():
        raise ValueError("Dependencies changed after freeze; use the original container image to resume")
    if digest(out / "ears_safe.json") != spec["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint modified")
    for m in spec["maps"]:
        if digest(out / m["file"]) != m["sha256"]:
            raise ValueError(f"Frozen map modified: {m['city']}")
    return spec


def ledger(root):
    db = sqlite3.connect(Path(root) / "episodes.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS episodes (id TEXT PRIMARY KEY, payload TEXT, sha256 TEXT)")
    return db


def read_rows(db, spec):
    expected = {key(j): j for j in jobs(spec)}
    rows = []
    for identity, payload, sha in db.execute("SELECT id,payload,sha256 FROM episodes ORDER BY rowid"):
        row = json.loads(payload)
        if canonical_hash(row) != sha or identity != key(row) or identity not in expected:
            raise ValueError("Corrupt or unexpected journal row")
        if any(row[k] != v for k, v in expected[identity].items()):
            raise ValueError("Mismatched journal pairing")
        if abs(episode_operational_score(row) - row["operational_score"]) > 1e-12:
            raise ValueError("Stored score does not reproduce")
        rows.append(row)
    return rows


def export(root, rows):
    path = Path(root)
    frame = pd.DataFrame([{k: v for k, v in r.items() if k != "diagnostics"} for r in rows])
    frame.to_csv(path / "episodes.csv", index=False)
    with (path / "diagnostics.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({**{k: r[k] for k in KEYS}, "diagnostics": r["diagnostics"]}) + "\n")
    return frame


def evaluate(args):
    root = Path(args.output_root)
    spec = verify(root, check_software=True)
    db = ledger(root)
    rows = read_rows(db, spec)
    done = {key(r) for r in rows}
    maps = {m["city"]: load_layer_cache(root / "frozen" / m["file"]) for m in spec["maps"]}
    manifest = dict(
        status="running",
        completed_rows=len(rows),
        expected_rows=spec["expected_rows"],
        study_sha256=digest(root / "frozen/study.json"),
        smoke_only=spec["smoke_only"],
        python=platform.python_version(),
        platform=platform.platform(),
        cpu=platform.processor(),
        numpy=np.__version__,
        pandas=pd.__version__,
        resumed=bool(rows),
        thread_settings={
            k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    )
    atomic_json(root / "manifest.json", manifest)
    try:
        for job in jobs(spec):
            identity = key(job)
            if identity in done:
                continue
            layers = maps[job["city"]]
            if job["panel"] == "swap":
                layers = seeded_mission_view(layers, job["dataset_seed"])
            layers = apply_start_zone(layers, spec["grid_size"], job["start_zone"])
            row = observe(layers, spec, job, root / "frozen/ears_safe.json")
            with db:
                db.execute(
                    "INSERT INTO episodes VALUES (?,?,?)",
                    (identity, json.dumps(row, allow_nan=False), canonical_hash(row)),
                )
            rows.append(row)
            manifest["completed_rows"] = len(rows)
            if len(rows) % len(spec["methods"]) == 0:
                atomic_json(root / "manifest.json", manifest)
            print(
                f"{len(rows)}/{spec['expected_rows']} {job['panel']} {job['city']} "
                f"n={job['agents']} drop={job['dropout']} ep={job['episode']} "
                f"{job['strategy']} J={row['operational_score']:.5f}",
                flush=True,
            )
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc)
        )
        raise
    finally:
        export(root, rows)
        manifest.update(
            episodes_sha256=digest(root / "episodes.csv"),
            completed_rows=len(rows),
            updated_utc=datetime.now(UTC).isoformat(),
        )
        atomic_json(root / "manifest.json", manifest)
        db.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["train", "freeze", "test", "report", "all", "smoke"])
    p.add_argument("--output-root", default="results/publication-v9")
    p.add_argument("--checkpoint", default="results/train/checkpoints/ears_safe.json")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--offline", action="store_true")
    p.add_argument(
        "--exclude-cities", default="",
        help="Explicit comma-separated cities omitted from a NEW exploratory subset (e.g. Copenhagen)",
    )
    return p


def main(args):
    if getattr(args, "exclude_cities", "") and args.stage not in ("freeze", "all"):
        raise ValueError("--exclude-cities is supported only for freeze/all of a NEW exploratory study")
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process owns this study. Do not run simultaneous writers.") from exc
        if args.stage == "train":
            training(args)
        if args.stage in ("freeze", "all", "smoke"):
            freeze(args, smoke=args.stage == "smoke")
        if args.stage in ("test", "all", "smoke"):
            evaluate(args)
        if args.stage in ("report", "test", "all", "smoke"):
            from experiments.report_publication_v9 import report

            report(root)


if __name__ == "__main__":
    main(parser().parse_args())
