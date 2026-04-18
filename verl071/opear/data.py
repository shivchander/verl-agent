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


def select_batch_positions(batch, group_size: int, selection_ratio: float) -> list[int]:
    """Select batch positions for contrastive pair generation.

    Selects floor(selection_ratio * group_size) rollouts per task group.
    Only does index math — no decoding, no tokenization.

    Args:
        batch: DataProto with non_tensor_batch["uid"].
        group_size: Number of rollouts per group (rollout.n).
        selection_ratio: Fraction of each group to select.

    Returns:
        List of selected batch position indices.
    """
    import math
    import random
    from collections import defaultdict

    uids = batch.non_tensor_batch.get("uid", None)
    if uids is None:
        logger.warning("No 'uid' field in batch.non_tensor_batch")
        return []

    groups: dict[str, list[int]] = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[str(uid)].append(i)

    k_per_group = max(1, math.floor(selection_ratio * group_size))
    selected: list[int] = []
    for positions in groups.values():
        k = min(k_per_group, len(positions))
        selected.extend(random.sample(positions, k))

    return selected


def _find_assistant_segments(response_mask: torch.Tensor) -> list[tuple[int, int]]:
    """Find contiguous segments where response_mask=1 (assistant turns).

    Args:
        response_mask: 1D tensor, 1 for model-generated tokens, 0 for user/observation.

    Returns:
        List of (start, end) index pairs within the response portion.
    """
    mask = response_mask.cpu().numpy().astype(int)
    segments = []
    in_segment = False
    start = 0
    for i, v in enumerate(mask):
        if v == 1 and not in_segment:
            start = i
            in_segment = True
        elif v == 0 and in_segment:
            segments.append((start, i))
            in_segment = False
    if in_segment:
        segments.append((start, len(mask)))
    return segments


def reconstruct_trajectories(batch, tokenizer, batch_positions: list[int]) -> list[dict]:
    """Reconstruct per-rollout multi-turn trajectory data for selected positions.

    Uses response_mask to identify individual assistant turns within the
    concatenated response. Each assistant turn is decoded separately so the
    guide model can rewrite them individually.

    Args:
        batch: DataProto with batch["prompts"], batch["responses"],
            batch["response_mask"], batch["attention_mask"],
            and non_tensor_batch["uid"].
        tokenizer: HuggingFace tokenizer for decoding.
        batch_positions: Which rows in the batch to reconstruct.

    Returns:
        List of trajectory dicts with keys: traj_uid, group_id,
        task_description, facts, turns, turn_indices, assistant_segments.
    """
    uids = batch.non_tensor_batch.get("uid", None)
    if uids is None:
        return []

    trajectories: list[dict] = []
    for batch_pos in batch_positions:
        group_id = str(uids[batch_pos])
        traj_uid = f"{group_id}_{batch_pos}"

        prompt_ids = batch.batch["prompts"][batch_pos]
        attn = batch.batch["attention_mask"][batch_pos]
        prompt_len = prompt_ids.shape[0]
        valid_len = int(attn[:prompt_len].sum().item())
        valid_ids = prompt_ids[-valid_len:]
        observation = tokenizer.decode(valid_ids, skip_special_tokens=True)

        response_ids = batch.batch["responses"][batch_pos]
        resp_mask = batch.batch["response_mask"][batch_pos]

        # Find assistant turn boundaries using response_mask
        segments = _find_assistant_segments(resp_mask)

        # Decode each turn separately for the guide
        turns: list[dict] = [{"role": "observation", "content": observation}]
        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            # Decode assistant segment
            asst_ids = response_ids[seg_start:seg_end]
            asst_text = tokenizer.decode(asst_ids, skip_special_tokens=True)
            turns.append({"role": "assistant", "content": asst_text})

            # Decode observation between this assistant turn and the next
            if seg_idx + 1 < len(segments):
                next_start = segments[seg_idx + 1][0]
                obs_ids = response_ids[seg_end:next_start]
                if len(obs_ids) > 0:
                    obs_text = tokenizer.decode(obs_ids, skip_special_tokens=True)
                    turns.append({"role": "observation", "content": obs_text})

        facts = ""
        if "facts_str" in batch.non_tensor_batch:
            facts = str(batch.non_tensor_batch["facts_str"][batch_pos])

        task_description = ""
        marker = "Your task is to: "
        pos = observation.find(marker)
        if pos != -1:
            end = observation.find("\n", pos)
            task_description = observation[
                pos + len(marker) : end if end != -1 else None
            ].strip()

        trajectories.append({
            "traj_uid": traj_uid,
            "group_id": group_id,
            "task_description": task_description or "Complete the task.",
            "facts": facts,
            "turns": turns,
            "turn_indices": [batch_pos],
            "assistant_segments": segments,
        })

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


def _swap_assistant_segments(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_len: int,
    segments: list[tuple[int, int]],
    rewrite_turns: list,
    tokenizer,
    pad_token_id: int,
) -> Optional[dict]:
    """Create a contrastive sequence by swapping assistant tokens in-place.

    Keeps the full original sequence (prompt + interleaved observations)
    identical. Only replaces the tokens in assistant segments with the
    guide's rewrite tokens, truncating or padding each segment to match
    the original segment length.

    Args:
        input_ids: Original full sequence (prompt_len + response_len,).
        attention_mask: Original attention mask, same shape.
        response_mask: Original response mask (response_len,).
            1 for model-generated, 0 for user/observation.
        prompt_len: Length of the prompt portion.
        segments: List of (start, end) pairs within the response portion
            marking assistant turns.
        rewrite_turns: Guide's rewritten turns (list of dicts or strings),
            one per assistant segment.
        tokenizer: For encoding rewrite text.
        pad_token_id: Token ID used for padding within segments.

    Returns:
        Dict with input_ids, attention_mask, response_mask tensors,
        or None if no segments could be matched.
    """
    num_turns = min(len(segments), len(rewrite_turns))
    if num_turns == 0:
        return None

    new_input_ids = input_ids.clone()
    new_attn_mask = attention_mask.clone()
    new_resp_mask = response_mask.clone()

    for t in range(num_turns):
        seg_start, seg_end = segments[t]
        seg_len = seg_end - seg_start

        # Tokenize the guide's rewrite for this turn
        text = _response_text_from_turn(rewrite_turns[t])
        rewrite_ids = tokenizer.encode(text, add_special_tokens=False)

        # Truncate or pad to match the original segment length
        actual_len = min(len(rewrite_ids), seg_len)
        pad_len = seg_len - actual_len

        # Write rewrite tokens into the response portion of input_ids
        offset = prompt_len + seg_start
        new_input_ids[offset:offset + actual_len] = torch.tensor(
            rewrite_ids[:actual_len], dtype=input_ids.dtype
        )
        # Pad remainder of segment with pad_token_id
        if pad_len > 0:
            new_input_ids[offset + actual_len:offset + seg_len] = pad_token_id

        # Keep attention_mask=1 for the ENTIRE segment (including padding).
        # Zeroing attention within a sequence creates holes that break causal
        # attention and unpad_input in flash attention. The pad tokens produce
        # garbage logits but response_mask=0 ensures they don't affect the loss.
        new_attn_mask[offset:offset + seg_len] = 1

        # Update response_mask: only compute loss on actual rewrite tokens
        new_resp_mask[seg_start:seg_start + actual_len] = 1
        new_resp_mask[seg_start + actual_len:seg_end] = 0

    return {
        "input_ids": new_input_ids,
        "attention_mask": new_attn_mask,
        "response_mask": new_resp_mask,
    }


def tokenize_contrastive_responses(
    trajectories: list[dict],
    contrastive_pairs: list[Optional[dict]],
    batch,
    tokenizer,
    max_response_length: int,
) -> Optional[dict]:
    """Create contrastive sequences by swapping assistant content in-place.

    For each selected rollout, takes the ORIGINAL full sequence (prompt +
    multi-turn response with interleaved observations) and replaces only
    the assistant-generated segments with the guide's rewrites. The
    observation tokens, prompt, and overall structure stay identical.

    This ensures the contrastive log-probs are computed in the same context
    as the original rollout — the model sees the full conversation history
    with only the assistant actions changed.

    Args:
        trajectories: Output of reconstruct_trajectories, each with
            'turn_indices' and 'assistant_segments'.
        contrastive_pairs: Parallel list, each None or dict with
            'compliant' and 'violating' turn lists.
        batch: Original training batch.
        tokenizer: For encoding guide responses.
        max_response_length: For tensor sizing (unused for structure,
            but kept for API compat).

    Returns:
        Dict with stacked tensors for compliant and violating sequences,
        or None if no valid pairs.
    """
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0

    compliant_entries: list[dict] = []
    violating_entries: list[dict] = []

    for traj, pair in zip(trajectories, contrastive_pairs):
        if pair is None:
            continue

        batch_pos = traj["turn_indices"][0]
        segments = traj["assistant_segments"]

        # Get the original full sequence
        orig_input_ids = batch.batch["input_ids"][batch_pos]
        orig_attn_mask = batch.batch["attention_mask"][batch_pos]
        orig_resp_mask = batch.batch["response_mask"][batch_pos]
        prompt_len = batch.batch["prompts"][batch_pos].shape[0]

        compliant_turns = pair["compliant"]
        violating_turns = pair["violating"]

        c_entry = _swap_assistant_segments(
            orig_input_ids, orig_attn_mask, orig_resp_mask,
            prompt_len, segments, compliant_turns,
            tokenizer, pad_token_id,
        )
        v_entry = _swap_assistant_segments(
            orig_input_ids, orig_attn_mask, orig_resp_mask,
            prompt_len, segments, violating_turns,
            tokenizer, pad_token_id,
        )

        if c_entry is not None and v_entry is not None:
            compliant_entries.append(c_entry)
            violating_entries.append(v_entry)

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
