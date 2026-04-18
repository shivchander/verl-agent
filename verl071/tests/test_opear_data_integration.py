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
    max_response_length = resp_len + 50  # some padding
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

    # Verify: observation tokens (response_mask=0 regions) are identical
    orig_input = batch.batch["input_ids"][positions[0]]
    orig_resp_mask = batch.batch["response_mask"][positions[0]]
    for i in range(num_pairs):
        orig_resp = orig_input[prompt_len:]
        c_resp = c_ids[i, prompt_len:]
        v_resp = v_ids[i, prompt_len:]
        obs_positions = (orig_resp_mask == 0).nonzero(as_tuple=True)[0]

        # Only check non-padding observation tokens
        orig_attn = batch.batch["attention_mask"][positions[0]]
        for pos in obs_positions:
            abs_pos = prompt_len + pos
            if orig_attn[abs_pos] == 1:  # non-padding
                assert c_resp[pos] == orig_resp[pos], \
                    f"Pair {i}: compliant obs token at pos {pos} differs! {c_resp[pos]} != {orig_resp[pos]}"
                assert v_resp[pos] == orig_resp[pos], \
                    f"Pair {i}: violating obs token at pos {pos} differs! {v_resp[pos]} != {orig_resp[pos]}"
    print("  PASS: Observation tokens identical across all variants")

    # Verify: assistant tokens (response_mask=1 regions) are DIFFERENT
    for i in range(num_pairs):
        c_resp = c_ids[i, prompt_len:]
        v_resp = v_ids[i, prompt_len:]
        orig_resp = orig_input[prompt_len:]

        for seg_start, seg_end in expected_segments:
            c_seg = c_resp[seg_start:seg_end]
            v_seg = v_resp[seg_start:seg_end]
            o_seg = orig_resp[seg_start:seg_end]

            # Compliant and violating should differ from each other
            assert not torch.equal(c_seg, v_seg), \
                f"Pair {i}: compliant and violating are identical at segment ({seg_start},{seg_end})!"
    print("  PASS: Assistant tokens differ between compliant and violating")

    # Verify: response_mask marks only actual rewrite tokens (not padding within segments)
    for i in range(num_pairs):
        c_rm = c_mask[i]
        # response_mask should have 1s only where there are actual rewrite tokens
        # (may be fewer than original segment if rewrite is shorter)
        assert c_rm.sum() > 0, f"Pair {i}: compliant response_mask is all zeros!"
        assert c_rm.sum() <= orig_resp_mask.sum(), \
            f"Pair {i}: compliant mask has more 1s than original!"
    print("  PASS: Response masks correctly mark rewrite tokens")

    # Decode and show a sample
    print(f"\n=== Sample decoded (pair 0) ===")
    print(f"  Original assistant turns:")
    for j, (s, e) in enumerate(expected_segments):
        orig_text = tokenizer.decode(orig_input[prompt_len + s:prompt_len + e], skip_special_tokens=True)
        print(f"    Turn {j+1}: {orig_text[:80]}...")

    print(f"  Compliant rewrites:")
    for j, (s, e) in enumerate(expected_segments):
        c_text = tokenizer.decode(c_ids[0, prompt_len + s:prompt_len + e], skip_special_tokens=True)
        print(f"    Turn {j+1}: {c_text[:80]}...")

    print(f"  Violating rewrites:")
    for j, (s, e) in enumerate(expected_segments):
        v_text = tokenizer.decode(v_ids[0, prompt_len + s:prompt_len + e], skip_special_tokens=True)
        print(f"    Turn {j+1}: {v_text[:80]}...")

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    test_find_assistant_segments()
    test_full_pipeline()
