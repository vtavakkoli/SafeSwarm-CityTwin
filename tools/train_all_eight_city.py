"""Train/select on disjoint development cities, freeze, compare all methods, report.

Completed stages are immutable and resumable. An interrupted training stage is
preserved under attempts/ and restarted; evaluation resumes episode by episode.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ruff: noqa: E402 -- direct CLI entry point.
from tools import eight_city_comparison as comparison

STAGES = ("learn", "coordinate", "prism", "ears")
PROTOCOL = "configs/publication_protocol_v7.json"


def file_hashes(directory):
    return {
        str(p.relative_to(directory)): comparison.digest(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "stage.json"
    }


def verify_stage(directory):
    record = json.loads((directory / "stage.json").read_text())
    if record["files"] != file_hashes(directory):
        raise ValueError(f"Completed stage changed: {directory}")
    return record


def start_stage(directory):
    if directory.exists():
        attempts = directory.parent / "attempts"
        attempts.mkdir(exist_ok=True)
        directory.rename(attempts / f"{directory.name}-{time.time_ns()}")
    directory.mkdir(parents=True)


def finish_stage(directory, started, **metadata):
    comparison.atomic_json(
        directory / "stage.json",
        dict(
            status="complete",
            seconds=time.perf_counter() - started,
            files=file_hashes(directory),
            **metadata,
        ),
    )
    return verify_stage(directory)


def command(stage, directory, cache, smoke):
    scripts = dict(
        learn="train_real_city_policies.py",
        coordinate="upgrade_ppo_coordination_v5.py",
        prism="train_prism_patterns.py",
        ears="train_ears_v6.py",
    )
    cmd = [
        sys.executable,
        "experiments/" + scripts[stage],
        "--protocol",
        PROTOCOL,
        "--output-root",
        str(directory),
        "--cache-dir",
        str(cache),
        "--offline",
        "--agents",
        "2" if smoke else "8",
        "--grid-size",
        "12" if smoke else "40",
        "--max-steps",
        "8" if smoke else "160",
    ]
    if stage in {"coordinate", "ears"}:
        cmd += ["--model-dir", str(directory / "checkpoints")]
    budgets = {
        "learn": [
            "--episodes",
            "108",
            "--validation-episodes",
            "12",
            "--validation-repeats",
            "2",
            "--teacher-bootstrap-scenarios",
            "9",
        ],
        "coordinate": [
            "--rounds",
            "2",
            "--teacher-scenarios",
            "9",
            "--validation-episodes",
            "12",
            "--validation-repeats",
            "2",
        ],
        "prism": [
            "--rounds",
            "8",
            "--train-scenarios-per-candidate",
            "4",
            "--validation-episodes",
            "8",
            "--validation-repeats",
            "2",
        ],
        "ears": ["--train-episodes", "6", "--validation-episodes", "8", "--validation-repeats", "2"],
    }
    cmd += budgets[stage]
    cmd += ["--quick"] if smoke else ["--require-real-data"]
    return cmd


def execute(cmd, log):
    print("+ " + " ".join(cmd), flush=True)
    with log.open("w") as stream:
        process = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, cmd)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def map_error(layers, error):
    meta = layers.get("metadata", {}) if isinstance(layers, dict) else {}
    return (
        f"{error}; source={meta.get('source', 'missing')}, "
        f"mission_source={meta.get('mission_source', 'missing')}, "
        f"fallback_reason={meta.get('fallback_reason', 'none reported')}"
    )


def real_map(city, grid, seed, cache, original, offline):
    """Reuse valid study maps, retry failed downloads, and preserve source caches."""
    target = comparison._cache_path(cache, city["place"], grid, city["radius_m"])
    existing = target.is_file()
    copied = not existing and original.is_file()
    if copied:
        shutil.copyfile(original, target)
    audit = {"source_cache_sha256": comparison.digest(original) if copied else None}

    def load(refresh=False):
        return comparison.load_real_city_layers(
            city["place"], grid, seed, radius_m=city["radius_m"], cache_dir=cache,
            allow_network=refresh and not offline, force_refresh=refresh,
        )

    layers = None
    try:
        layers = load()
        comparison.validate_layers(layers, grid, city["name"], False)
        return layers, "copied cache" if copied else "resumed study cache", audit
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        reason = map_error(layers, exc)
        if copied or existing:
            audit["rejected_cache_reason"] = reason
            print(f"[maps] {city['name']}: rejected cache ({reason})", flush=True)

    if not offline:
        audit["download_failures"] = []
        for attempt in range(1, 4):
            print(f"[maps] {city['name']}: OSM download attempt {attempt}/3", flush=True)
            layers = None
            try:
                layers = load(refresh=True)
                comparison.validate_layers(layers, grid, city["name"], False)
                origin = "OSM refresh after rejected copied cache" if copied else "new OSM snapshot"
                return layers, origin, audit
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                reason = map_error(layers, exc)
                audit["download_failures"].append(reason)
            # Retry download failures, never repeatedly replace genuine but invalid data.
            meta = layers.get("metadata", {}) if isinstance(layers, dict) else {}
            if not meta.get("fallback_reason") or attempt == 3:
                break
            delay = 10 if attempt == 1 else 30
            print(f"[maps] {city['name']}: {reason}; retrying in {delay}s", flush=True)
            time.sleep(delay)

    advice = (
        "Offline mode cannot repair this cache. Supply a valid OSM snapshot or start a new online study."
        if offline else
        "Check the download/geocoder/Overpass error above and network access, then rerun the same command."
    )
    raise ValueError(
        f"{city['name']}: no valid OSM snapshot available. {reason}. "
        f"Source cache: {original}. Study cache: {target}. {advice} "
        "Training has not started; no synthetic data accepted and the source cache is unchanged."
    )


def prepare(root, args, protocol):
    directory = root / "maps"
    if (directory / "stage.json").exists():
        return verify_stage(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cache = directory / "cache"
    cache.mkdir(exist_ok=True)
    progress_path = directory / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    grid = 12 if args.smoke else 40
    manifest = []
    for i, city in enumerate(protocol["cities"]):
        target = comparison._cache_path(cache, city["place"], grid, city["radius_m"])
        original = comparison._cache_path(args.cache_dir, city["place"], grid, city["radius_m"])
        if city["name"] in progress:
            record = progress[city["name"]]
            if (
                record["cache_file"] != target.name or not target.is_file()
                or comparison.digest(target) != record["sha256"]
            ):
                raise ValueError(f"Previously validated map changed: {city['name']}")
            comparison.validate_layers(comparison.load_layer_cache(target), grid, city["name"], args.smoke)
            manifest.append(record)
            print(f"[maps] {city['name']} ({city['split']}): verified completed map", flush=True)
            continue
        if args.smoke:
            layers = comparison.synthetic_city_layers(grid, 100 + i, city["place"])
            comparison.save_layer_cache(target, layers)
            origin = "synthetic software check"
            cache_validation = {}
        else:
            layers, origin, cache_validation = real_map(
                city, grid, 100 + i, cache, original, args.offline,
            )
        comparison.validate_layers(layers, grid, city["name"], args.smoke)
        if not target.is_file():
            raise ValueError(f"No genuine cached map saved for {city['name']}")
        manifest.append(
            dict(
                city=city["name"],
                split=city["split"],
                origin=origin,
                cache_file=target.name,
                sha256=comparison.digest(target),
                metadata=layers["metadata"],
                cache_validation=cache_validation,
            )
        )
        progress[city["name"]] = manifest[-1]
        comparison.atomic_json(progress_path, progress)
        print(f"[maps] {city['name']} ({city['split']}): {origin}", flush=True)
    comparison.atomic_json(directory / "manifest.json", manifest)
    return finish_stage(directory, started, smoke_only=args.smoke)


def changed_fields(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        fields = []
        for key in sorted(before.keys() | after.keys()):
            name = f"{prefix}.{key}" if prefix else key
            if key not in before or key not in after:
                fields.append(name)
            else:
                fields.extend(changed_fields(before[key], after[key], name))
        return fields
    return [prefix] if before != after else []


def initialize(root, args):
    protocol = comparison.load_protocol(ROOT / PROTOCOL)
    if not all(comparison.validate_protocol(protocol).values()):
        raise ValueError("Invalid geographic protocol")
    config = dict(
        format="train-all-eight-city-v1",
        smoke_only=args.smoke,
        profile=args.profile,
        protocol=protocol,
        source_hashes=comparison.source_hashes(),
        runtime=comparison.runtime(),
        source_cache=str(Path(args.cache_dir).resolve()),
        offline=args.offline,
        commands={s: command(s, root / s, root / "maps/cache", args.smoke) for s in STAGES},
    )
    path = root / "pipeline.json"
    if path.is_file():
        if comparison.digest(path) != (root / "pipeline.sha256").read_text().strip():
            raise ValueError("Pipeline specification changed")
        changes = changed_fields(json.loads(path.read_text()), config)
        if changes:
            raise ValueError(
                "Pipeline code, configuration, runtime or hardware changed; retain the original image. "
                "Changed fields: " + ", ".join(changes[:12]) +
                (f" (and {len(changes) - 12} more)" if len(changes) > 12 else "")
            )
        for name, expected in config["source_hashes"].items():
            data = (root / "source" / name).read_bytes().replace(b"\r\n", b"\n")
            if comparison.hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Archived source changed: {name}")
        return config
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new empty output root for a new training study")
    source_cache = Path(args.cache_dir).resolve()
    if root == source_cache or root in source_cache.parents or source_cache in root.parents:
        raise ValueError("Pipeline output must be separate from the source cache")
    if not args.smoke and any(v != "1" for v in config["runtime"]["threads"].values()):
        raise ValueError("Use Compose or set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS to 1")
    root.mkdir(parents=True, exist_ok=True)
    for name in config["source_hashes"]:
        dst = root / "source" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dst)
    comparison.atomic_json(path, config)
    (root / "pipeline.sha256").write_text(comparison.digest(path) + "\n")
    return config


def train_stage(root, stage, previous, config):
    directory = root / stage
    if (directory / "stage.json").exists():
        return verify_stage(directory)
    start_stage(directory)
    started = time.perf_counter()
    if previous is not None:
        shutil.copytree(root / previous / "checkpoints", directory / "checkpoints")
    execute(config["commands"][stage], directory / "console.log")
    # Required outputs are checked before declaring a stage reusable.
    expected = list(comparison.CHECKPOINTS[:4])
    if stage in {"prism", "ears"}:
        expected += comparison.CHECKPOINTS[4:9]
    if stage == "ears":
        expected = comparison.CHECKPOINTS
        comparison.check_checkpoints(directory / "checkpoints", smoke=config["smoke_only"])
    if any(not (directory / "checkpoints" / name).is_file() for name in expected):
        raise ValueError(f"Stage {stage} did not produce its required checkpoints")
    return finish_stage(directory, started, command=config["commands"][stage])


def provenance(root, config, stages):
    model_dir = root / "ears/checkpoints"
    record = dict(
        format="eight-city-training-v1",
        status="complete",
        smoke_only=config["smoke_only"],
        protocol=config["protocol"],
        source_hashes=config["source_hashes"],
        runtime=config["runtime"],
        stages=stages,
        map_stage=verify_stage(root / "maps"),
        checkpoint_hashes={n: comparison.digest(model_dir / n) for n in comparison.CHECKPOINTS},
        candidate_sha256=comparison.digest(model_dir / "ears_safe.json"),
        selection="Train and validation only; all test metrics excluded from selection",
        training_replicates=1,
        routing="Fixed v9 routing configuration; no test-city tuning",
    )
    path = root / "training_manifest.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("Completed training manifest changed")
    if not path.exists():
        comparison.atomic_json(path, record)
    return path


def run(args):
    root = Path(args.output_root).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / (root.name + ".pipeline.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = initialize(root, args)
        prepare(root, args, config["protocol"])
        if args.stop_after == "prepare":
            return
        stages, previous = {}, None
        for stage in STAGES:
            verify_stage(root / "maps")
            stages[stage] = train_stage(root, stage, previous, config)
            verify_stage(root / "maps")
            previous = stage
            if args.stop_after == stage:
                print(f"Paused after {stage}; rerun without --stop-after to continue.", flush=True)
                return
        manifest = provenance(root, config, stages)
        cmd = [
            "smoke" if args.smoke else "run",
            "--profile",
            args.profile,
            "--output-root",
            str(root / "evaluation"),
            "--model-dir",
            str(root / "ears/checkpoints"),
            "--candidate-checkpoint",
            str(root / "ears/checkpoints/ears_safe.json"),
            "--candidate-mode",
            "retrained",
            "--training-manifest",
            str(manifest),
            "--cache-dir",
            str(root / "maps/cache"),
        ]
        if args.max_evaluation_jobs:
            cmd += ["--max-jobs", str(args.max_evaluation_jobs)]
        comparison.run(comparison.parser().parse_args(cmd))
        if not (root / "evaluation/completion.json").exists():
            print("Evaluation paused. No final publication report certified.", flush=True)
            return
        for stage in STAGES:
            verify_stage(root / stage)
        verify_stage(root / "maps")
        completion = json.loads((root / "evaluation/completion.json").read_text())
        if completion["rows"] != completion["expected_rows"]:
            raise ValueError("Incomplete evaluation")
        comparison.atomic_json(
            root / "completion.json",
            dict(
                status="complete",
                smoke_only=args.smoke,
                evaluation=completion,
                training_manifest_sha256=comparison.digest(manifest),
                publication_report="evaluation/report/publication_report.pdf",
                submission_ready=False,
            ),
        )
        archive = root.parent / (root.name + ".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(root.rglob("*")):
                rel = path.relative_to(root)
                if path.is_file() and "attempts" not in rel.parts and path.suffix not in {".zip", ".lock"}:
                    z.write(path, root.name + "/" + str(rel))
        print(f"Complete publication evidence package: {archive}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", default="results/eight-city-publication")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--profile", choices=["core", "full"], default="full")
    p.add_argument("--smoke", action="store_true", help="Short synthetic training plus 180 test episodes")
    p.add_argument("--offline", action="store_true", help="Require cached maps; do not download missing maps")
    p.add_argument("--stop-after", choices=["prepare", *STAGES])
    p.add_argument("--max-evaluation-jobs", type=int, help="Pause after this many new test episodes")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.max_evaluation_jobs is not None and args.max_evaluation_jobs < 1:
        raise ValueError("--max-evaluation-jobs must be positive")
    run(args)
