"""Integration test for the OPEAR data pipeline.

Creates a realistic multi-turn ALFWorld-like batch, runs it through
select_batch_positions → reconstruct_trajectories → (mock guide) →
tokenize_contrastive_responses, and verifies the output structure.
"""
import torch
import numpy as np
from types import SimpleNamespace
from transformers import AutoTokenizer

from verl071.opear.data import (
    select_batch_positions,
    reconstruct_trajectories,
    tokenize_contrastive_responses,
    _find_assistant_segments,
)


def build_mock_batch(tokenizer, num_prompts=2, group_size=2):
    """Build a realistic multi-turn batch like verl's agent loop produces.

    Each rollout has:
    - prompt: system message + initial observation
    - response: [asst_turn_1][obs_1][asst_turn_2][obs_2][asst_turn_3][padding]
    - response_mask: 1 for assistant tokens, 0 for observation + padding
    """
    prompt_text = (
        "You are an expert agent operating in the ALFRED Embodied Environment. "
        "Your task is to: put some pencil on sidetable. "
        "Your current observation is: You are in the middle of a room."
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt_len = len(prompt_ids)

    # Simulate 3 assistant turns with observations between them
    asst_turns = [
        "<think>I need to find a pencil. Let me look around.</think><action>go to desk 1</action>",
        "<think>I see a pencil on the desk. I should take it.</think><action>take pencil 1 from desk 1</action>",
        "<think>Now I need to go to sidetable and put it down.</think><action>go to sidetable 1</action>",
    ]
    obs_turns = [
        "On the desk 1, you see a pencil 1, a pen 2, and a book 3.",
        "You pick up the pencil 1 from the desk 1.",
    ]

    # Tokenize each segment
    asst_ids_list = [tokenizer.encode(t, add_special_tokens=False) for t in asst_turns]
    obs_ids_list = [tokenizer.encode(o, add_special_tokens=False) for o in obs_turns]

    # Build interleaved response: [asst1][obs1][asst2][obs2][asst3]
    response_ids = []
    response_mask = []
    segments_expected = []

    for i, asst_ids in enumerate(asst_ids_list):
        seg_start = len(response_ids)
        response_ids.extend(asst_ids)
        response_mask.extend([1] * len(asst_ids))
        seg_end = len(response_ids)
        segments_expected.append((seg_start, seg_end))

        if i < len(obs_ids_list):
            response_ids.extend(obs_ids_list[i])
            response_mask.extend([0] * len(obs_ids_list[i]))

    resp_len = len(response_ids)

    # Pad to a fixed response length
    max_response_length = resp_len + 500  # generous budget for guide rewrites
    pad_len = max_response_length - resp_len
    pad_id = tokenizer.pad_token_id or 0
    response_ids_padded = response_ids + [pad_id] * pad_len
    response_mask_padded = response_mask + [0] * pad_len

    # Build full input_ids = prompt + response
    full_ids = prompt_ids + response_ids_padded
    full_len = len(full_ids)
    attn_mask = [1] * (prompt_len + resp_len) + [0] * pad_len

    # Create batch with num_prompts * group_size rollouts
    total = num_prompts * group_size
    batch_input_ids = torch.tensor([full_ids] * total, dtype=torch.long)
    batch_prompts = torch.tensor([prompt_ids] * total, dtype=torch.long)
    batch_responses = torch.tensor([response_ids_padded] * total, dtype=torch.long)
    batch_response_mask = torch.tensor([response_mask_padded] * total, dtype=torch.float32)
    batch_attn_mask = torch.tensor([attn_mask] * total, dtype=torch.long)

    uids = []
    for p in range(num_prompts):
        uids.extend([str(p)] * group_size)

    from tensordict import TensorDict
    batch_td = TensorDict({
        "input_ids": batch_input_ids,
        "prompts": batch_prompts,
        "responses": batch_responses,
        "response_mask": batch_response_mask,
        "attention_mask": batch_attn_mask,
    }, batch_size=total)

    batch = SimpleNamespace(
        batch=batch_td,
        non_tensor_batch={
            "uid": np.array(uids, dtype=object),
            "facts_str": np.array(["pencil_1 is on desk_1. sidetable_1 is in room_1."] * total, dtype=object),
        },
    )

    return batch, prompt_len, max_response_length, segments_expected, asst_turns


def build_mock_guide_pairs(num_trajectories, num_assistant_turns=3):
    """Build mock guide responses (compliant + violating) for each trajectory."""
    pairs = []
    for i in range(num_trajectories):
        compliant_turns = [
            {"turn": t + 1, "raw": "", "think": f"Compliant reasoning for turn {t+1}", "action": f"compliant_action_{t+1}"}
            for t in range(num_assistant_turns)
        ]
        violating_turns = [
            {"turn": t + 1, "raw": "", "think": f"Violating reasoning for turn {t+1}", "action": f"violating_action_{t+1}"}
            for t in range(num_assistant_turns)
        ]
        pairs.append({"compliant": compliant_turns, "violating": violating_turns})
    return pairs


def test_find_assistant_segments():
    """Test that _find_assistant_segments correctly identifies turn boundaries."""
    # [asst1: 5 tokens][obs1: 3 tokens][asst2: 4 tokens][padding: 2 tokens]
    mask = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0], dtype=torch.float32)
    segments = _find_assistant_segments(mask)
    assert segments == [(0, 5), (8, 12)], f"Expected [(0,5),(8,12)], got {segments}"
    print("PASS: _find_assistant_segments")


def test_full_pipeline():
    """End-to-end test of the contrastive data pipeline."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

    num_prompts = 2
    group_size = 4
    beta = 0.5

    batch, prompt_len, max_resp_len, expected_segments, orig_asst_turns = build_mock_batch(
        tokenizer, num_prompts=num_prompts, group_size=group_size
    )
    total_rollouts = num_prompts * group_size

    print(f"\n=== Batch created ===")
    print(f"  Total rollouts: {total_rollouts}")
    print(f"  Prompt length: {prompt_len}")
    print(f"  Response length: {max_resp_len}")
    print(f"  Assistant segments: {expected_segments}")
    print(f"  Num assistant turns: {len(expected_segments)}")

    # Step 1: Select positions
    positions = select_batch_positions(batch, group_size=group_size, selection_ratio=beta)
    k_per_group = int(beta * group_size)
    expected_selected = num_prompts * k_per_group
    assert len(positions) == expected_selected, f"Expected {expected_selected} positions, got {len(positions)}"
    print(f"\n=== Selection ===")
    print(f"  k_per_group: {k_per_group}")
    print(f"  Selected positions: {positions} ({len(positions)} total)")

    # Step 2: Reconstruct trajectories
    trajectories = reconstruct_trajectories(batch, tokenizer, positions)
    assert len(trajectories) == len(positions), f"Expected {len(positions)} trajectories, got {len(trajectories)}"

    traj = trajectories[0]
    assistant_turns_in_traj = [t for t in traj["turns"] if t["role"] == "assistant"]
    print(f"\n=== Reconstruction ===")
    print(f"  Trajectories: {len(trajectories)}")
    print(f"  First traj turns: {len(traj['turns'])} ({len(assistant_turns_in_traj)} assistant)")
    print(f"  Assistant segments: {traj['assistant_segments']}")
    print(f"  Facts: {traj['facts'][:50]}...")

    assert len(assistant_turns_in_traj) == len(expected_segments), \
        f"Expected {len(expected_segments)} assistant turns, got {len(assistant_turns_in_traj)}"
    assert traj["assistant_segments"] == expected_segments, \
        f"Segments mismatch: {traj['assistant_segments']} != {expected_segments}"

    # Print decoded assistant turns
    for i, t in enumerate(assistant_turns_in_traj):
        print(f"  Turn {i+1}: {t['content'][:80]}...")

    # Step 3: Mock guide pairs
    pairs = build_mock_guide_pairs(len(trajectories), num_assistant_turns=len(expected_segments))
    print(f"\n=== Guide pairs ===")
    print(f"  Generated {len(pairs)} pairs, each with {len(expected_segments)} turns")

    # Step 4: Tokenize contrastive responses
    opear_data = tokenize_contrastive_responses(
        trajectories, pairs, batch, tokenizer, max_resp_len
    )
    assert opear_data is not None, "tokenize_contrastive_responses returned None"

    c_ids = opear_data["compliant_input_ids"]
    v_ids = opear_data["violating_input_ids"]
    c_mask = opear_data["compliant_response_mask"]
    v_mask = opear_data["violating_response_mask"]
    c_attn = opear_data["compliant_attention_mask"]

    print(f"\n=== Contrastive tensors ===")
    print(f"  compliant_input_ids:     {c_ids.shape}")
    print(f"  violating_input_ids:     {v_ids.shape}")
    print(f"  compliant_response_mask: {c_mask.shape}")
    print(f"  violating_response_mask: {v_mask.shape}")
    print(f"  compliant_attention_mask:{c_attn.shape}")

    num_pairs = c_ids.shape[0]
    assert num_pairs == len(positions), f"Expected {len(positions)} pairs, got {num_pairs}"

    # Verify: prompt portion is identical across original, compliant, and violating
    orig_prompt = batch.batch["prompts"][positions[0]]
    for i in range(num_pairs):
        assert torch.equal(c_ids[i, :prompt_len], orig_prompt), \
            f"Compliant pair {i}: prompt tokens differ!"
        assert torch.equal(v_ids[i, :prompt_len], orig_prompt), \
            f"Violating pair {i}: prompt tokens differ!"
    print("  PASS: Prompt tokens identical across all variants")

    # Verify: observation tokens are present in contrastive sequences
    # (positions may differ from original since rewrites change length)
    orig_input = batch.batch["input_ids"][positions[0]]
    orig_resp_mask = batch.batch["response_mask"][positions[0]]

    # Extract original observation token sequences
    orig_obs_tokens = []
    prev_end = 0
    for s, e in expected_segments:
        if s > prev_end:
            obs = batch.batch["responses"][positions[0]][prev_end:s]
            orig_obs_tokens.append(obs)
        prev_end = e

    # Check that each observation sequence appears in the contrastive sequences
    for i in range(num_pairs):
        c_full = c_ids[i]
        v_full = v_ids[i]
        for j, obs in enumerate(orig_obs_tokens):
            obs_text = tokenizer.decode(obs, skip_special_tokens=True).strip()
            c_text = tokenizer.decode(c_full, skip_special_tokens=True)
            v_text = tokenizer.decode(v_full, skip_special_tokens=True)
            assert obs_text in c_text, \
                f"Pair {i}: observation {j} not found in compliant sequence"
            assert obs_text in v_text, \
                f"Pair {i}: observation {j} not found in violating sequence"
    print("  PASS: Observation tokens present in all variants")

    # Verify: compliant and violating differ
    for i in range(num_pairs):
        assert not torch.equal(c_ids[i], v_ids[i]), \
            f"Pair {i}: compliant and violating sequences are identical!"
    print("  PASS: Compliant and violating sequences differ")

    # Verify: response_mask marks rewrite tokens
    for i in range(num_pairs):
        c_rm = c_mask[i]
        assert c_rm.sum() > 0, f"Pair {i}: compliant response_mask is all zeros!"
    print("  PASS: Response masks correctly mark rewrite tokens")

    # Verify: no truncation — decode response_mask=1 regions and check for complete tags
    for i in range(num_pairs):
        c_rm = c_mask[i]
        c_resp = c_ids[i, prompt_len:]
        # Get rewrite tokens
        rewrite_positions = (c_rm == 1).nonzero(as_tuple=True)[0]
        if len(rewrite_positions) > 0:
            rewrite_text = tokenizer.decode(c_resp[rewrite_positions], skip_special_tokens=True)
            # Should contain complete action tags (not truncated)
            assert "<action>" in rewrite_text or len(rewrite_text) < 10, \
                f"Pair {i}: rewrite appears truncated — no <action> tag found: {rewrite_text[:80]}..."
    print("  PASS: Rewrites contain complete tags (no truncation)")

    # Decode and show a sample — extract rewrite turns from response_mask
    print(f"\n=== Sample decoded (pair 0) ===")
    print(f"  Original assistant turns:")
    for j, (s, e) in enumerate(expected_segments):
        orig_text = tokenizer.decode(orig_input[prompt_len + s:prompt_len + e], skip_special_tokens=True)
        print(f"    Turn {j+1}: {orig_text[:80]}...")

    # For contrastive: find rewrite segments from response_mask
    for label, ids, mask in [("Compliant", c_ids[0], c_mask[0]), ("Violating", v_ids[0], v_mask[0])]:
        resp_part = ids[prompt_len:]
        segs = _find_assistant_segments(mask)
        print(f"  {label} rewrites:")
        for j, (s, e) in enumerate(segs):
            text = tokenizer.decode(resp_part[s:e], skip_special_tokens=True)
            print(f"    Turn {j+1}: {text[:80]}...")

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    test_find_assistant_segments()
    test_full_pipeline()
