"""Compatibility entry point for the SafeSwarm real-city trainer.

The implementation lives in :mod:`experiments.train_real_city_policies_v3`.
Keeping this file preserves all existing Docker and CI commands.
"""

from experiments.train_real_city_policies_v3 import main


if __name__ == "__main__":
    main()
