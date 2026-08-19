"""Run the frozen SafeSwarm v7 publication-validation suite end to end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--require-real-data", action="store_true")
    p.add_argument("--fresh-train", action="store_true", help="rerun original v6 train/validation selection before publication evaluation")
    return p.parse_args()


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    protocol = "configs/publication_protocol_v7.json"
    common: list[str] = []
    if args.offline:
        common.append("--offline")
    if args.require_real_data:
        common.append("--require-real-data")

    checkpoints = Path("results/train/checkpoints")
    required = [
        checkpoints / "ears_safe.json",
        checkpoints / "ears_np_safe.json",
        checkpoints / "h_mappo_ears_safe.json",
        checkpoints / "mappo_safe.json",
    ]
    if args.fresh_train or not all(path.exists() for path in required):
        train_command = [sys.executable, "experiments/run_train_test_pipeline.py", *common]
        if args.quick:
            train_command.append("--quick")
        _run(train_command)

    prepare = [
        sys.executable, "experiments/prepare_real_city_data.py",
        "--protocol", protocol,
        "--output-root", "results/publication/prepare-data",
        *common,
    ]
    test = [
        sys.executable, "experiments/test_real_city_policies.py",
        "--protocol", protocol,
        "--model-dir", "results/train/checkpoints",
        "--output-root", "results/publication/test",
        *common,
    ]
    swap = [
        sys.executable, "experiments/test_swap_protocol.py",
        "--protocol", protocol,
        "--model-dir", "results/train/checkpoints",
        "--output-root", "results/publication/swap-test",
        *common,
    ]
    ablation = [
        sys.executable, "experiments/test_ears_ablations_v7.py",
        "--protocol", protocol,
        "--model-dir", "results/train/checkpoints",
        "--output-root", "results/publication/ablations",
        *common,
    ]
    stats = [sys.executable, "experiments/analyze_publication_statistics_v7.py"]
    sensitivity = [sys.executable, "experiments/score_sensitivity_v7.py"]
    visualization = [
        sys.executable, "experiments/visualize_winner_episode_v7.py",
        "--protocol", protocol,
        "--model-dir", "results/train/checkpoints",
        *common,
    ]

    if args.quick:
        for command in (test, swap, ablation, stats, sensitivity, visualization):
            command.append("--quick")

    _run(prepare)
    _run(test)
    _run(swap)
    _run(ablation)
    _run(stats)
    _run(sensitivity)
    _run(visualization)
    _run([sys.executable, "experiments/build_publication_report_v7.py"])


if __name__ == "__main__":
    main()
