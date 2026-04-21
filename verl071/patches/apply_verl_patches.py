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

    dp_actor_path = os.path.join(
        verl_path, "workers", "actor", "dp_actor.py"
    )

    print("Patch 3a: Add O-PEaR import to dp_actor.py")
    ok = patch_file(
        dp_actor_path,
        'from verl.utils.torch_functional import logprobs_from_logits',
        'from verl.utils.torch_functional import logprobs_from_logits\n'
        '\n'
        'try:\n'
        '    from verl071.opear.actor_hook import opear_accumulate_gradients\n'
        'except ImportError:\n'
        '    opear_accumulate_gradients = None',
        "dp_actor.py: add O-PEaR import",
    )
    success = success and ok

    print("Patch 3b: Call O-PEaR gradient accumulation before optimizer step")
    ok = patch_file(
        dp_actor_path,
        '                grad_norm = self._optimizer_step()',
        '                # O-PEaR: accumulate contrastive loss gradients with GRPO gradients\n'
        '                if opear_accumulate_gradients is not None:\n'
        '                    opear_accumulate_gradients(self, data, metrics)\n'
        '                grad_norm = self._optimizer_step()',
        "dp_actor.py: call O-PEaR before optimizer step",
    )
    success = success and ok

    print("Patch 4: Propagate interaction extra_info to agent_data.extra_fields")
    ok = patch_file(
        tool_agent_loop_path,
        'if reward is not None:\n'
        '            agent_data.turn_scores.append(reward)\n'
        '\n'
        '        # Update prompt with user responses (similar to _handle_processing_tools_state)',
        'if reward is not None:\n'
        '            agent_data.turn_scores.append(reward)\n'
        '\n'
        '        # Propagate interaction extra_info (e.g. facts_str) to extra_fields\n'
        '        if isinstance(metrics, dict):\n'
        '            for _k in ("facts_str",):\n'
        '                if _k in metrics:\n'
        '                    agent_data.extra_fields[_k] = metrics[_k]\n'
        '\n'
        '        # Update prompt with user responses (similar to _handle_processing_tools_state)',
        "tool_agent_loop.py: propagate facts_str from interaction to extra_fields",
    )
    success = success and ok

    vllm_server_path = os.path.join(
        verl_path, "workers", "rollout", "vllm_rollout", "vllm_async_server.py"
    )

    print("Patch 5: Fix max_tokens=0 crash in multi-turn rollouts")
    ok = patch_file(
        vllm_server_path,
        '            # Default to a calculation that considers configured lengths\n'
        '            # Cap max_tokens by response_length to ensure tensor alignment,\n'
        '            # and by remaining budget to prevent OOM in multi-turn rollouts.\n'
        '            max_tokens = min(\n'
        '                self.config.response_length, self.config.prompt_length + self.config.response_length - len(prompt_ids)\n'
        '            )\n'
        '\n'
        '        # Clamp max_tokens to the valid range [0, max_possible_tokens]\n'
        '        max_tokens = max(0, min(max_tokens, max_possible_tokens))',
        '            # Default: cap by response_length and remaining context space.\n'
        '            # In multi-turn, prompt_ids grows beyond prompt_length (user/env\n'
        '            # messages accumulate), so use max_model_len as the true budget.\n'
        '            max_tokens = min(self.config.response_length, max_possible_tokens)\n'
        '\n'
        '        # Clamp max_tokens to the valid range [1, max_possible_tokens].\n'
        '        # Must be >= 1 to avoid vLLM VLLMValidationError.\n'
        '        max_tokens = max(1, min(max_tokens, max_possible_tokens))',
        "vllm_async_server.py: fix max_tokens=0 crash in multi-turn",
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
