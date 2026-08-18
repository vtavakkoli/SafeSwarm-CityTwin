"""Run SafeSwarm v5 prepare → learn → coordinate → PRISM → test → SWAP."""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--require-real-data", action="store_true")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    common: list[str] = []
    if args.offline:
        common.append("--offline")
    if args.require_real_data:
        common.append("--require-real-data")

    prepare = [sys.executable, "experiments/prepare_real_city_data.py", *common]
    train = [sys.executable, "experiments/train_real_city_policies.py", *common]
    coordinate = [sys.executable, "experiments/upgrade_ppo_coordination_v5.py", *common]
    prism = [sys.executable, "experiments/train_prism_patterns.py", *common]
    test = [sys.executable, "experiments/test_real_city_policies.py", *common]
    swap = [sys.executable, "experiments/test_swap_protocol.py", *common]

    if args.quick:
        for command in (train, coordinate, prism, test, swap):
            command.append("--quick")

    _run(prepare)
    _run(train)
    _run(coordinate)
    _run(prism)
    _run(test)
    _run(swap)
    _run([sys.executable, "experiments/build_train_test_report.py"])


if __name__ == "__main__":
    main()
