"""Compatibility entry point for the SafeSwarm real-city trainer.

The implementation lives beside this file in ``train_real_city_policies_v3``.
Keeping this entry point preserves all existing Docker and CI commands.
"""

from train_real_city_policies_v3 import main


if __name__ == "__main__":
    main()
