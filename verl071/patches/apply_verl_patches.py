"""Apply required patches to the installed verl 0.7.1 package.

These patches inject trajectory_info (global_step, rollout_n) into the
interaction kwargs, enabling deterministic game selection for correct
GRPO grouping in multi-turn environments like ALFWorld.

Usage:
    python verl071/patches/apply_verl_patches.py

Run this after installing verl==0.7.1 in your venv.
"""

import importlib.util
import os
import sys


def find_verl_path():
    """Find the installed verl package path."""
    spec = importlib.util.find_spec("verl")
    if spec is None or spec.origin is None:
        print("ERROR: verl package not found. Install it first: pip install verl==0.7.1")
        sys.exit(1)
    return os.path.dirname(spec.origin)


def patch_file(filepath, old, new, description):
    """Apply a single patch to a file."""
    with open(filepath) as f:
        content = f.read()

    if new in content:
        print(f"  SKIP (already patched): {description}")
        return True

    if old not in content:
        print(f"  WARN: Could not find target string for: {description}")
        print(f"         File: {filepath}")
        print(f"         Expected: {old[:80]}...")
        return False

    content = content.replace(old, new, 1)
    with open(filepath, "w") as f:
        f.write(content)
    print(f"  OK: {description}")
    return True


def main():
    verl_path = find_verl_path()
    print(f"verl package found at: {verl_path}")
    print()

    agent_loop_path = os.path.join(
        verl_path, "experimental", "agent_loop", "agent_loop.py"
    )
    tool_agent_loop_path = os.path.join(
        verl_path, "experimental", "agent_loop", "tool_agent_loop.py"
    )

    success = True
    print("Patch 1: Inject _trajectory_info into agent loop kwargs")
    ok = patch_file(
        agent_loop_path,
        '            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}',
        '            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}\n'
        '            kwargs["_trajectory_info"] = trajectory_info[i]',
        "agent_loop.py: add _trajectory_info to kwargs",
    )
    success = success and ok

    print("Patch 2: Pass global_step and rollout_n to interaction_kwargs")
    # This patch injects trajectory info between reading interaction_kwargs
    # and the "name" check. The target string is the two consecutive lines.
    ok = patch_file(
        tool_agent_loop_path,
        'interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]\n'
        '            if "name" not in interaction_kwargs:',
        'interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]\n'
        '            # Inject trajectory info (global_step, rollout_n) for deterministic env selection\n'
        '            traj = kwargs.get("_trajectory_info", {})\n'
        '            if traj:\n'
        '                interaction_kwargs = {**interaction_kwargs, "global_step": traj.get("step", -1), "rollout_n": traj.get("rollout_n", 0)}\n'
        '            if "name" not in interaction_kwargs:',
        "tool_agent_loop.py: inject global_step into interaction_kwargs",
    )
    success = success and ok

    print()
    if success:
        print("All patches applied successfully.")
    else:
        print("Some patches failed — check warnings above.")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
