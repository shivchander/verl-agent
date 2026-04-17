"""O-PEaR entry point: patches verl then runs the standard training loop.

This is a thin wrapper around verl.trainer.main_ppo that applies O-PEaR
monkey-patches before Ray forks workers. The patches propagate to worker
processes via fork semantics.

Usage:
    python -m verl071.main_opear [hydra overrides...]
"""
from verl071.opear_hooks import apply_patches

# Apply patches BEFORE any verl imports that trigger Ray
apply_patches()

# Now run the standard verl training entry point
from verl.trainer.main_ppo import main

if __name__ == "__main__":
    main()
