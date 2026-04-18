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


def _build_contrastive_sequence(
    prompt_ids: torch.Tensor,
    orig_response_ids: torch.Tensor,
    orig_response_mask: torch.Tensor,
    segments: list[tuple[int, int]],
    rewrite_turns: list,
    tokenizer,
    pad_token_id: int,
    max_seq_len: int,
) -> Optional[dict]:
    """Build a contrastive sequence by reassembling prompt + observations + rewrites.

    Keeps prompt and observation tokens from the original. Replaces each
    assistant segment with the guide's full rewrite (no truncation).
    Pads the result to max_seq_len.

    Args:
        prompt_ids: Original prompt tokens (prompt_len,).
        orig_response_ids: Original response tokens (response_len,).
        orig_response_mask: Original response mask (response_len,).
        segments: List of (start, end) pairs within the response marking
            assistant turns.
        rewrite_turns: Guide's rewritten turns, one per segment.
        tokenizer: For encoding rewrite text.
        pad_token_id: Token ID for padding.
        max_seq_len: Pad all sequences to this length.

    Returns:
        Dict with input_ids, attention_mask, response_mask, or None.
    """
    num_turns = min(len(segments), len(rewrite_turns))
    if num_turns == 0:
        return None

    prompt_len = prompt_ids.shape[0]

    # Collect pieces: prompt, then alternating [obs, rewrite] blocks
    token_pieces: list[torch.Tensor] = [prompt_ids]
    mask_pieces: list[torch.Tensor] = []  # 1 for rewrite, 0 for obs

    prev_end = 0  # tracks position in response portion
    for t in range(num_turns):
        seg_start, seg_end = segments[t]

        # Observation tokens before this assistant segment
        if seg_start > prev_end:
            obs_ids = orig_response_ids[prev_end:seg_start]
            token_pieces.append(obs_ids)
            mask_pieces.append(torch.zeros(len(obs_ids), dtype=torch.float32))

        # Guide's rewrite for this turn (full, no truncation)
        text = _response_text_from_turn(rewrite_turns[t])
        rewrite_ids = tokenizer.encode(text, add_special_tokens=False)
        rewrite_tensor = torch.tensor(rewrite_ids, dtype=prompt_ids.dtype)
        token_pieces.append(rewrite_tensor)
        mask_pieces.append(torch.ones(len(rewrite_ids), dtype=torch.float32))

        prev_end = seg_end

    # Trailing observation tokens after the last segment
    if prev_end < len(orig_response_ids):
        # Only include non-padding trailing tokens
        trailing = orig_response_ids[prev_end:]
        # Find where actual content ends (before original padding)
        orig_attn_len = int(orig_response_mask[prev_end:].sum().item())
        # Include observation tokens (mask=0) that aren't padding
        content_end = prev_end
        for i in range(prev_end, len(orig_response_ids)):
            if orig_response_ids[i] == pad_token_id and orig_response_mask[i] == 0:
                break
            content_end = i + 1
        if content_end > prev_end:
            trailing_ids = orig_response_ids[prev_end:content_end]
            token_pieces.append(trailing_ids)
            mask_pieces.append(torch.zeros(len(trailing_ids), dtype=torch.float32))

    # Concatenate all pieces
    full_ids = torch.cat(token_pieces)
    response_mask = torch.cat(mask_pieces) if mask_pieces else torch.zeros(0, dtype=torch.float32)

    real_len = len(full_ids)
    resp_len = len(response_mask)

    # Pad or truncate to max_seq_len
    if real_len < max_seq_len:
        pad_len = max_seq_len - real_len
        full_ids = torch.cat([full_ids, torch.full((pad_len,), pad_token_id, dtype=full_ids.dtype)])
        response_mask = torch.cat([response_mask, torch.zeros(pad_len, dtype=torch.float32)])
    elif real_len > max_seq_len:
        full_ids = full_ids[:max_seq_len]
        resp_len_new = max_seq_len - prompt_len
        response_mask = response_mask[:resp_len_new]

    # Build attention mask: 1 for real tokens, 0 for padding
    attn_mask = torch.zeros(len(full_ids), dtype=torch.long)
    attn_mask[:min(real_len, max_seq_len)] = 1

    return {
        "input_ids": full_ids,
        "attention_mask": attn_mask,
        "response_mask": response_mask,
    }


def tokenize_contrastive_responses(
    trajectories: list[dict],
    contrastive_pairs: list[Optional[dict]],
    batch,
    tokenizer,
    max_response_length: int,
) -> Optional[dict]:
    """Build contrastive sequences from prompt + observations + guide rewrites.

    For each selected rollout, reassembles a new sequence using:
    - The original prompt tokens (unchanged)
    - The original observation tokens between turns (unchanged)
    - The guide's full rewrite for each assistant turn (no truncation)

    All sequences are padded to the same length for batching.

    Args:
        trajectories: Output of reconstruct_trajectories, each with
            'turn_indices' and 'assistant_segments'.
        contrastive_pairs: Parallel list, each None or dict with
            'compliant' and 'violating' turn lists.
        batch: Original training batch.
        tokenizer: For encoding guide responses.
        max_response_length: Used for max sequence length calculation.

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
        prompt_ids = batch.batch["prompts"][batch_pos]
        orig_resp_ids = batch.batch["responses"][batch_pos]
        orig_resp_mask = batch.batch["response_mask"][batch_pos]
        prompt_len = prompt_ids.shape[0]

        # Max sequence length = prompt + max_response_length (same as original batch)
        max_seq_len = prompt_len + max_response_length

        c_entry = _build_contrastive_sequence(
            prompt_ids, orig_resp_ids, orig_resp_mask,
            segments, pair["compliant"],
            tokenizer, pad_token_id, max_seq_len,
        )
        v_entry = _build_contrastive_sequence(
            prompt_ids, orig_resp_ids, orig_resp_mask,
            segments, pair["violating"],
            tokenizer, pad_token_id, max_seq_len,
        )

        if c_entry is not None and v_entry is not None:
            compliant_entries.append(c_entry)
            violating_entries.append(v_entry)

    if not compliant_entries:
        return None

    def pad_and_stack(entries: list[dict], key: str) -> torch.Tensor:
        """Pad all entries to the same length, then stack."""
        max_len = max(e[key].shape[0] for e in entries)
        padded = []
        for e in entries:
            t = e[key]
            if t.shape[0] < max_len:
                pad = torch.zeros(max_len - t.shape[0], dtype=t.dtype)
                t = torch.cat([t, pad])
            padded.append(t)
        return torch.stack(padded)

    return {
        "compliant_input_ids": pad_and_stack(compliant_entries, "input_ids"),
        "compliant_attention_mask": pad_and_stack(compliant_entries, "attention_mask"),
        "compliant_response_mask": pad_and_stack(compliant_entries, "response_mask"),
        "violating_input_ids": pad_and_stack(violating_entries, "input_ids"),
        "violating_attention_mask": pad_and_stack(violating_entries, "attention_mask"),
        "violating_response_mask": pad_and_stack(violating_entries, "response_mask"),
    }
