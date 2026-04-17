"""O-PEaR data packing: tokenize contrastive responses into model-ready tensors.

This module bridges the guide model output and the actor model's forward pass.
The guide produces text; this module turns it into token tensors that the student
model can compute log-probabilities on. The prompt tokens stay the same as the
original rollout; only the response tokens change.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def reconstruct_trajectories(batch, tokenizer) -> list[dict]:
    """Reconstruct per-trajectory data from the flat per-turn training batch.

    Groups turns by traj_uid and extracts the text content needed for guide
    prompts.

    Args:
        batch: DataProto-like object with:
            - batch.batch["prompts"]: (N, prompt_len) tensor
            - batch.batch["responses"]: (N, response_len) tensor
            - batch.batch["attention_mask"]: (N, total_len) tensor
            - batch.non_tensor_batch["traj_uid"]: (N,) array of trajectory IDs
            - batch.non_tensor_batch["active_masks"]: (N,) boolean array
            - batch.non_tensor_batch["facts_str"]: (N,) array of fact strings
              (optional)
        tokenizer: HuggingFace tokenizer for decoding

    Returns:
        List of trajectory dicts:
            - traj_uid: str
            - task_description: str (extracted from first observation)
            - facts: str (semicolon-separated PDDL facts from first active turn)
            - turns: list of {"role": "observation"|"assistant", "content": "..."}
              dicts
            - turn_indices: list of batch indices for this trajectory's active
              turns
    """
    traj_uids = batch.non_tensor_batch.get("traj_uid", np.array([]))
    if len(traj_uids) == 0:
        return []

    # Group batch indices by trajectory
    uid_to_indices: dict[str, list[int]] = {}
    for i, uid in enumerate(traj_uids):
        uid_to_indices.setdefault(str(uid), []).append(i)

    trajectories: list[dict] = []
    for uid, indices in uid_to_indices.items():
        # Only include active turns
        active_indices: list[int] = []
        for idx in indices:
            active = batch.non_tensor_batch.get(
                "active_masks", np.ones(len(traj_uids))
            )[idx]
            if active:
                active_indices.append(idx)

        if not active_indices:
            continue

        turns: list[dict] = []
        task_description = ""
        first_facts = ""

        for turn_num, idx in enumerate(active_indices):
            # Decode observation (prompt)
            prompt_ids = batch.batch["prompts"][idx]
            attn = batch.batch["attention_mask"][idx]
            prompt_len = prompt_ids.shape[0]
            valid_len = attn[:prompt_len].sum().item()
            valid_ids = prompt_ids[-int(valid_len):]
            observation = tokenizer.decode(valid_ids, skip_special_tokens=True)

            # Decode student response
            response_ids = batch.batch["responses"][idx]
            resp_len = attn[prompt_len:].sum().item()
            valid_resp = response_ids[: int(resp_len)]
            response = tokenizer.decode(valid_resp, skip_special_tokens=True)

            # Get facts (stored during rollout)
            facts = ""
            if "facts_str" in batch.non_tensor_batch:
                facts = str(batch.non_tensor_batch["facts_str"][idx])

            # Capture facts from first active turn
            if turn_num == 0 and facts:
                first_facts = facts

            # Build alternating observation/assistant turns
            turns.append({"role": "observation", "content": observation})
            turns.append({"role": "assistant", "content": response})

            # Extract task description from first turn's observation
            if not task_description:
                marker = "Your task is to: "
                pos = observation.find(marker)
                if pos != -1:
                    end = observation.find("\n", pos)
                    task_description = observation[
                        pos + len(marker) : end if end != -1 else None
                    ].strip()

        trajectories.append(
            {
                "traj_uid": uid,
                "task_description": task_description or "Complete the task.",
                "facts": first_facts,
                "turns": turns,
                "turn_indices": active_indices,
            }
        )

    return trajectories


def _response_text_from_turn(turn) -> str:
    """Extract response text from a turn entry.

    Accepts either a parsed dict (from parse_guide_response, with "think" and
    "action" keys) or a plain string.  Parsed dicts are reconstructed into the
    canonical ``<think>...</think><action>...</action>`` format.
    """
    if isinstance(turn, dict):
        think = turn.get("think", "")
        action = turn.get("action", "")
        return f"<think>{think}</think><action>{action}</action>"
    # Already a string (e.g. pre-formatted guide output)
    return str(turn)


def tokenize_contrastive_responses(
    trajectories: list[dict],
    contrastive_pairs: list[Optional[dict]],
    batch,
    tokenizer,
    max_response_length: int,
) -> Optional[dict]:
    """Tokenize compliant and violating responses into model-ready tensors.

    For each turn in each selected trajectory, creates new response token
    sequences from the guide model's rewritten responses. The prompt tokens
    stay the same; only the response tokens change.

    Args:
        trajectories: Output of reconstruct_trajectories.
        contrastive_pairs: Parallel list, each is None or dict with:
            - "compliant": list of parsed turn dicts ({"turn": N, "raw": "...",
              "think": "...", "action": "..."}) or plain strings
            - "violating": same format
        batch: Original training batch (for prompt token IDs).
        tokenizer: For encoding guide responses.
        max_response_length: For padding.

    Returns:
        Dict with tensors:
            compliant_input_ids: (M, prompt_len + max_response_length)
            compliant_attention_mask: (M, prompt_len + max_response_length)
            compliant_response_mask: (M, max_response_length)
            violating_input_ids: same shapes
            violating_attention_mask: same shapes
            violating_response_mask: same shapes
        Or None if no valid pairs.
    """
    compliant_entries: list[dict] = []
    violating_entries: list[dict] = []

    for traj, pair in zip(trajectories, contrastive_pairs):
        if pair is None:
            continue

        turn_indices = traj["turn_indices"]
        compliant_turns = pair["compliant"]
        violating_turns = pair["violating"]

        # Match turns: we may have fewer guide turns than batch turns
        num_turns = min(len(turn_indices), len(compliant_turns), len(violating_turns))

        for t in range(num_turns):
            idx = turn_indices[t]
            prompt_ids = batch.batch["prompts"][idx]  # (prompt_len,)
            prompt_attn = batch.batch["attention_mask"][idx]
            prompt_len = prompt_ids.shape[0]

            for turn_data, entries in [
                (compliant_turns[t], compliant_entries),
                (violating_turns[t], violating_entries),
            ]:
                # Reconstruct the text from parsed dict or use directly
                text = _response_text_from_turn(turn_data)

                # Tokenize the guide's response
                resp_ids = tokenizer.encode(text, add_special_tokens=False)
                resp_ids = resp_ids[:max_response_length]  # truncate
                resp_len = len(resp_ids)

                # Pad response to max_response_length
                pad_len = max_response_length - resp_len

                pad_token_id = getattr(tokenizer, "pad_token_id", None)
                if pad_token_id is None:
                    pad_token_id = 0
                padded_resp = resp_ids + [pad_token_id] * pad_len

                # Build full sequence: prompt + response
                full_ids = torch.cat(
                    [
                        prompt_ids,
                        torch.tensor(padded_resp, dtype=prompt_ids.dtype),
                    ]
                )
                full_attn = torch.cat(
                    [
                        prompt_attn[:prompt_len],
                        torch.ones(resp_len, dtype=prompt_attn.dtype),
                        torch.zeros(pad_len, dtype=prompt_attn.dtype),
                    ]
                )
                resp_mask = torch.cat(
                    [
                        torch.ones(resp_len, dtype=torch.float32),
                        torch.zeros(pad_len, dtype=torch.float32),
                    ]
                )

                entries.append(
                    {
                        "input_ids": full_ids,
                        "attention_mask": full_attn,
                        "response_mask": resp_mask,
                    }
                )

    if not compliant_entries:
        return None

    def stack(entries: list[dict], key: str) -> torch.Tensor:
        return torch.stack([e[key] for e in entries])

    return {
        "compliant_input_ids": stack(compliant_entries, "input_ids"),
        "compliant_attention_mask": stack(compliant_entries, "attention_mask"),
        "compliant_response_mask": stack(compliant_entries, "response_mask"),
        "violating_input_ids": stack(violating_entries, "input_ids"),
        "violating_attention_mask": stack(violating_entries, "attention_mask"),
        "violating_response_mask": stack(violating_entries, "response_mask"),
    }
