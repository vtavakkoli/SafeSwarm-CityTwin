"""Run prepare-data, train, held-out test, and combined report as one reproducible pipeline."""

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
    _run([sys.executable, "experiments/prepare_real_city_data.py", *common])
    train = [sys.executable, "experiments/train_real_city_policies.py", *common]
    test = [sys.executable, "experiments/test_real_city_policies.py", *common]
    if args.quick:
        train.append("--quick")
        test.append("--quick")
    _run(train)
    _run(test)
    _run([sys.executable, "experiments/build_train_test_report.py"])


if __name__ == "__main__":
    main()
