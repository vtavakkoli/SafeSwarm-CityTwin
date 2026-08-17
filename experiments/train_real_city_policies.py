"""Compatibility entry point for the SafeSwarm real-city trainer.

The current implementation lives in ``train_real_city_policies_v4``. Keeping
this stable entry point preserves Docker, CI and external reproduction commands.
"""

from train_real_city_policies_v4 import main


if __name__ == "__main__":
    main()
