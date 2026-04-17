"""O-PEaR training entry point.

Extends verl with O-PEaR regularization, then runs the standard training
pipeline. Extensions are applied before Ray forks workers, so they
propagate automatically.

Usage:
    python -m verl071.main_opear [hydra overrides...]
"""
# Step 1: Apply O-PEaR extensions BEFORE verl.trainer.main_ppo is loaded.
from verl071.opear.extensions import apply
apply()

# Step 2: Run the standard verl training entry point.
from verl.trainer.main_ppo import main

if __name__ == "__main__":
    main()
