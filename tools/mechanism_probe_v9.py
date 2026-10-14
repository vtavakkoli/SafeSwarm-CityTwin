"""Frozen, resumable, exploratory controls for the v9 routing mechanism.

Uses only supplied verified maps. Reusing these maps is NOT independent geographic
confirmation. Never changes v9, its original ledger, checkpoint or promotion gate.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import shutil
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_ears_v9 import atomic_json  # noqa: E402
from experiments.publication_ears_v9 import (  # noqa: E402
    ObservedEnvironment,
    canonical_hash,
    digest,
    read_json,
    software,
    sources,
)
from experiments.report_publication_v9 import city_test, holm, md_table  # noqa: E402
from experiments.run_city_benchmark import run_episode  # noqa: E402
from src.agents.ears_v9 import EARSInformationPolicy, InformationRoutingConfig  # noqa: E402
from src.environment.obstacles import load_layer_cache  # noqa: E402
from src.evaluation.metrics import episode_operational_score  # noqa: E402
from src.training.geography import apply_start_zone  # noqa: E402
from tools.publication_readiness_v9 import read_evidence  # noqa: E402

FULL = "EARS-Info-v9"
LOCAL = "EARS-Info-Local-v9"
PERIODIC = "EARS-Info-Periodic16"
UNRESERVED = "EARS-Info-NoSensingReservations"
METHODS = [FULL, LOCAL, PERIODIC, UNRESERVED]
PAIR = ["panel", "city", "agents", "dropout", "episode", "start_zone", "seed"]
METRICS = [
    "operational_score",
    "weighted_target_discovery",
    "coverage_ratio",
    "energy_consumption",
    "runtime_seconds",
    "runtime_overhead",
    "completed_all_targets",
    "capped_completion_steps",
    "actual_safety_incidents",
    "battery_failures",
    "planning_seconds",
    "accepted_relocations",
    "relocation_steps",
    "event_requests",
]


class PeriodicInformationPolicy(EARSInformationPolicy):
    """Change only event requests; retain routing, feasibility, return and safety.

    Period 16 is a fixed diagnostic control, not tuned on the results. The
    accepted intervention budget is measured, NOT assumed to match event-driven.
    """

    def _event_reason(self, env, agent_id, positions):
        start = self.config.min_steps_before_reallocation
        if not self.routing_config.enable_relocation or env.steps < start:
            return None
        if env.steps < max(self._retry_until.get(agent_id, -1), self._cooldown_until.get(agent_id, -1)):
            return None
        if not env.agents[agent_id].communication_status:
            return None
        return "periodic" if (env.steps - start) % 16 == 0 else None


class UnreservedInformationPolicy(EARSInformationPolicy):
    """Remove sensing-footprint reservations, keeping collision reservations intact."""

    def _information_gain(self, env, reserved=None):
        return super()._information_gain(env, reserved=None)

    def _information_action(self, env, aid, planned, positions, reserved):
        return super()._information_action(env, aid, planned, positions, reserved=set())


def make_policy(method, seed, checkpoint, routing):
    config = InformationRoutingConfig(**routing)
    cls = EARSInformationPolicy
    if method == LOCAL:
        config = replace(config, enable_relocation=False)
    elif method == PERIODIC:
        cls = PeriodicInformationPolicy
    elif method == UNRESERVED:
        cls = UnreservedInformationPolicy
    elif method != FULL:
        raise ValueError(f"Unknown control: {method}")
    return cls(seed=seed, model_path=checkpoint, routing_config=config, strategy_name=method)


def probe_sources():
    return {
        **sources(),
        **{
            str(p.relative_to(ROOT)): digest(p)
            for p in [Path(__file__).resolve(), ROOT / "tools/publication_readiness_v9.py"]
        },
    }


def planned_jobs(plan):
    for panel in plan["panels"]:
        for ci, city in enumerate(plan["cities"]):
            for agents in plan["agents_grid"]:
                for dropout in plan["dropout_grid"]:
                    for episode in range(plan["episodes"]):
                        offset = episode % len(METHODS)
                        for method in METHODS[offset:] + METHODS[:offset]:
                            yield dict(
                                panel=panel,
                                city=city,
                                agents=agents,
                                dropout=dropout,
                                episode=episode,
                                start_zone=plan["zones"][episode % len(plan["zones"])],
                                seed=plan["seed"] + ci * 10000 + episode,
                                strategy=method,
                            )


def identity(job):
    return canonical_hash({k: job[k] for k in [*PAIR, "strategy"]})


def freeze(source, out, profile):
    frozen = out / "frozen"
    if (frozen / "plan.json").exists():
        plan = read_json(frozen / "plan.json")
        if digest(frozen / "plan.json") != (frozen / "plan.sha256").read_text().strip():
            raise ValueError("Probe plan modified")
        if plan["profile"] != profile or plan["source_study_sha256"] != digest(source / "frozen/study.json"):
            raise ValueError("Resume requires the same profile and source study")
        if plan["source_hashes"] != probe_sources() or plan["software"] != software():
            raise ValueError("Probe code/dependencies changed; use a new output root")
        if plan["thread_settings"] != thread_settings():
            raise ValueError("Thread settings changed; restore them before resuming")
        for name, expected in plan["artifact_hashes"].items():
            if digest(frozen / name) != expected:
                raise ValueError(f"Probe artifact modified: {name}")
        return plan
    spec, _, mismatches, additions = read_evidence(source)
    if any(p.endswith(".py") for p in mismatches + additions):
        raise ValueError("Historical Python source differs: restore it before probing its mechanism")
    if spec["smoke_only"]:
        raise ValueError("Use a complete real-map study for this exploratory probe")
    frozen.mkdir(parents=True, exist_ok=True)
    artifacts = ["ears_safe.json", *[m["file"] for m in spec["maps"]]]
    for name in artifacts:
        shutil.copyfile(source / "frozen" / name, frozen / name)
    plan = dict(
        format="ears-v9-mechanism-probe-v1",
        profile=profile,
        evidence_status="EXPLORATORY REUSED GEOGRAPHY; NOT CONFIRMATION OR MODEL SELECTION",
        source_study_sha256=digest(source / "frozen/study.json"),
        source_episodes_sha256=digest(source / "episodes.csv"),
        original_source_mismatches=mismatches,
        source_hashes=probe_sources(),
        software=software(),
        platform=platform.platform(),
        thread_settings=thread_settings(),
        artifact_hashes={name: digest(frozen / name) for name in artifacts},
        maps={m["city"]: m["file"] for m in spec["maps"]},
        map_metadata={m["city"]: m["metadata"] for m in spec["maps"]},
        cities=[m["city"] for m in spec["maps"]],
        zones=spec["zones"],
        grid_size=spec["grid_size"],
        max_steps=spec["max_steps"],
        routing_config=spec["routing_config"],
        agents_grid=[4, 8] if profile == "pilot" else spec["agents_grid"],
        dropout_grid=[0.03] if profile == "pilot" else spec["dropout_grid"],
        episodes=4 if profile == "pilot" else spec["episodes"],
        panels=["mission_stop", "fixed_budget"],
        seed=7142026,
        methods=METHODS,
        periodic_steps=16,
        uncertainty="descriptive exact city tests; Holm across all comparisons",
        intervention_budget="not matched; report requests, accepted routes, route steps and planning time",
    )
    plan["expected_rows"] = sum(1 for _ in planned_jobs(plan))
    atomic_json(frozen / "plan.json", plan)
    (frozen / "plan.sha256").write_text(digest(frozen / "plan.json") + "\n")
    return plan


def thread_settings():
    return {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]}


def observe(layers, plan, job, checkpoint):
    env = ObservedEnvironment(
        grid_size=plan["grid_size"],
        n_agents=job["agents"],
        seed=job["seed"],
        layers=layers,
        max_steps=plan["max_steps"],
        communication_dropout_prob=job["dropout"],
        allow_network=False,
        stop_when_all_missions_found=job["panel"] != "fixed_budget",
    )
    policy = make_policy(job["strategy"], job["seed"], checkpoint, plan["routing_config"])
    row = run_episode(env, policy).to_dict(**job, steps=env.steps, data_source=env.data_source)
    row.update(
        operational_score=episode_operational_score(row),
        completed_all_targets=int(env.completion_step is not None),
        capped_completion_steps=env.completion_step if env.completion_step is not None else plan["max_steps"],
        planning_seconds=policy.planning_seconds,
        accepted_relocations=policy.accepted_relocations,
        relocation_steps=policy.relocation_steps,
        event_requests=policy.event_triggers,
        diagnostics=policy.diagnostics(),
    )
    return row


def summarize(rows, out):
    frame = pd.DataFrame([{k: v for k, v in r.items() if k != "diagnostics"} for r in rows])
    if frame.duplicated([*PAIR, "strategy"]).any():
        raise ValueError("Duplicate probe rows")
    city = frame.groupby(["panel", "city", "strategy"], as_index=False)[METRICS].mean()
    summary = city.groupby(["panel", "strategy"], as_index=False)[METRICS].mean()
    tests = []
    for panel, group in frame.groupby("panel"):
        for metric in METRICS:
            pivot = group.pivot(index=PAIR, columns="strategy", values=metric)
            if pivot.isna().any().any() or set(pivot.columns) != set(METHODS):
                raise ValueError("Incomplete paired probe")
            for control in METHODS[1:]:
                deltas = (pivot[FULL] - pivot[control]).groupby(level="city").mean()
                tests.append(dict(panel=panel, metric=metric, comparator=control, **city_test(deltas)))
    tests = pd.DataFrame(tests)
    tests["p_holm_descriptive"] = holm(tests.p_two_sided)
    for name, f in [("episodes", frame), ("city_means", city), ("summary", summary), ("contrasts", tests)]:
        f.to_csv(out / f"{name}.csv", index=False)
    text = [
        "# v9 exploratory mechanism probe",
        "Reused geography; no promotion or confirmation claim.",
        "All variants retain the same safety monitor, return logic, routing score, checkpoint and paired "
        "scenario seeds. Local disables relocation. Periodic16 replaces the event request schedule. "
        "NoSensingReservations removes local and relocation sensing-footprint overlap suppression; "
        "collision avoidance reservations stay active. "
        "Safety, score, routing and stopping rules remain unchanged.",
        md_table(
            summary[
                [
                    "panel",
                    "strategy",
                    "operational_score",
                    "weighted_target_discovery",
                    "runtime_overhead",
                    "accepted_relocations",
                    "relocation_steps",
                ]
            ]
        ),
        "Periodic interventions are not budget-matched to the event-driven method. Differences test the "
        "whole request schedule, not a causal advantage at equal intervention budgets. Future work should "
        "calibrate matched budgets on development maps and freeze them before untouched-map evaluation.",
        "Fixed-budget disables mission completion stopping only; all-agent termination remains possible. "
        "All tests and intervals are descriptive, with cities as units and Holm adjustment across all "
        "reported panel/metric/comparator contrasts. "
        "Small samples and convenience maps limit generalization.",
    ]
    (out / "report.md").write_text("\n\n".join(text) + "\n")


def run(source, output, profile):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("Probe output must be separate from original evidence")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = freeze(source, output, profile)
        expected = {identity(j): j for j in planned_jobs(plan)}
        with sqlite3.connect(output / "episodes.sqlite") as db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("CREATE TABLE IF NOT EXISTS episodes (id TEXT PRIMARY KEY, payload TEXT, sha256 TEXT)")
            rows, done = [], set()
            for ident, payload, sha in db.execute("SELECT id,payload,sha256 FROM episodes ORDER BY rowid"):
                row = json.loads(payload)
                if (
                    canonical_hash(row) != sha
                    or ident != identity(row)
                    or ident not in expected
                    or any(row[k] != v for k, v in expected[ident].items())
                    or abs(episode_operational_score(row) - row["operational_score"]) > 1e-12
                ):
                    raise ValueError("Corrupt probe journal")
                rows.append(row)
                done.add(ident)
            maps = {city: load_layer_cache(output / "frozen" / name) for city, name in plan["maps"].items()}
            for ident, job in expected.items():
                if ident in done:
                    continue
                layers = apply_start_zone(maps[job["city"]], plan["grid_size"], job["start_zone"])
                row = observe(layers, plan, job, output / "frozen/ears_safe.json")
                with db:
                    db.execute(
                        "INSERT INTO episodes VALUES (?,?,?)",
                        (ident, json.dumps(row, allow_nan=False), canonical_hash(row)),
                    )
                rows.append(row)
                if len(rows) % len(METHODS) == 0:
                    print(
                        f"{len(rows)}/{len(expected)} {job['panel']} {job['city']} n={job['agents']}",
                        flush=True,
                    )
            summarize(rows, output)
        atomic_json(
            output / "completion.json",
            dict(
                status="completed",
                rows=len(rows),
                plan_sha256=digest(output / "frozen/plan.json"),
                episodes_sha256=digest(output / "episodes.csv"),
                independent_confirmation=False,
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    args = parser.parse_args()
    run(args.study_root, args.output_root, args.profile)
