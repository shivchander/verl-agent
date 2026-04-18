"""End-to-end OPEAR test: rollout → guide → tokenize → forward → loss.

Loads Qwen3-4B on a single GPU, builds a realistic multi-turn batch,
calls the real guide model, computes log-probs and contrastive loss.
Logs everything to a file for review.

Usage:
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python verl071/tests/test_opear_e2e.py
"""
import os
import sys
import json
import torch
import numpy as np
from types import SimpleNamespace
from dotenv import load_dotenv

load_dotenv()

OUTPUT_FILE = "opear_e2e_results.txt"
out = []


def log(s=""):
    out.append(str(s))
    print(s)


def build_realistic_batch(tokenizer):
    """Build a multi-turn ALFWorld-like batch with 2 prompts × 2 group."""
    from tensordict import TensorDict

    prompt_text = (
        "You are an expert agent operating in the ALFRED Embodied Environment. "
        "Your task is to: put some pencil on sidetable. "
        "Your current observation is: You are in the middle of a room. "
        "Your admissible actions are: [go to desk 1, go to shelf 1, go to sidetable 1, look]."
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt_len = len(prompt_ids)

    # 3 assistant turns with observations between them
    asst_turns_text = [
        "<think>I need to find a pencil. Let me check the desk first since pencils are commonly found there.</think><action>go to desk 1</action>",
        "<think>I can see a pencil on the desk. I should take it.</think><action>take pencil 1 from desk 1</action>",
        "<think>Now I have the pencil. I need to go to the sidetable and put it down.</think><action>go to sidetable 1</action>",
    ]
    obs_turns_text = [
        "On the desk 1, you see a pencil 1, a pen 2, a book 3, and a laptop 1. Your admissible actions are: [take pencil 1 from desk 1, take pen 2 from desk 1, take book 3 from desk 1, go to shelf 1, go to sidetable 1, look].",
        "You pick up the pencil 1 from the desk 1. Your admissible actions are: [go to desk 1, go to shelf 1, go to sidetable 1, put pencil 1 in/on desk 1, examine pencil 1, look].",
    ]

    asst_ids_list = [tokenizer.encode(t, add_special_tokens=False) for t in asst_turns_text]
    obs_ids_list = [tokenizer.encode(o, add_special_tokens=False) for o in obs_turns_text]

    response_ids = []
    response_mask = []
    segments = []

    for i, asst_ids in enumerate(asst_ids_list):
        seg_start = len(response_ids)
        response_ids.extend(asst_ids)
        response_mask.extend([1] * len(asst_ids))
        seg_end = len(response_ids)
        segments.append((seg_start, seg_end))
        if i < len(obs_ids_list):
            response_ids.extend(obs_ids_list[i])
            response_mask.extend([0] * len(obs_ids_list[i]))

    resp_len = len(response_ids)
    max_response_length = resp_len + 500  # generous budget for guide rewrites
    pad_len = max_response_length - resp_len
    pad_id = tokenizer.pad_token_id or 0

    response_ids_padded = response_ids + [pad_id] * pad_len
    response_mask_padded = response_mask + [0] * pad_len

    full_ids = prompt_ids + response_ids_padded
    attn_mask = [1] * (prompt_len + resp_len) + [0] * pad_len

    # 2 prompts × 2 group = 4 rollouts
    num_prompts, group_size = 2, 2
    total = num_prompts * group_size

    uids = []
    for p in range(num_prompts):
        uids.extend([str(p)] * group_size)

    batch_td = TensorDict({
        "input_ids": torch.tensor([full_ids] * total, dtype=torch.long),
        "prompts": torch.tensor([prompt_ids] * total, dtype=torch.long),
        "responses": torch.tensor([response_ids_padded] * total, dtype=torch.long),
        "response_mask": torch.tensor([response_mask_padded] * total, dtype=torch.float32),
        "attention_mask": torch.tensor([attn_mask] * total, dtype=torch.long),
    }, batch_size=total)

    # Simulate rewards: 2 rollouts won (reward=10), 2 lost (reward=0)
    rewards = torch.tensor([10.0, 0.0, 10.0, 0.0])

    batch = SimpleNamespace(
        batch=batch_td,
        non_tensor_batch={
            "uid": np.array(uids, dtype=object),
            "facts_str": np.array([
                "pencil_1 is on desk_1. sidetable_1 is in room_1. pen_2 is on desk_1."
            ] * total, dtype=object),
        },
    )

    return batch, prompt_len, max_response_length, segments, rewards, group_size


def compute_grpo_advantages(rewards, group_size):
    """Compute GRPO advantages (group-relative, no critic)."""
    num_groups = len(rewards) // group_size
    advantages = torch.zeros_like(rewards)
    for g in range(num_groups):
        group_rewards = rewards[g * group_size:(g + 1) * group_size]
        mean = group_rewards.mean()
        std = group_rewards.std()
        if std > 0:
            advantages[g * group_size:(g + 1) * group_size] = (group_rewards - mean) / std
        else:
            advantages[g * group_size:(g + 1) * group_size] = 0.0
    return advantages


def forward_logprobs(model, tokenizer, input_ids, attention_mask, response_len, device):
    """Compute per-token log-probs for the response portion."""
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits  # (B, seq_len, vocab)

    # Log-probs for response tokens
    # Shift: logits[t] predicts token[t+1]
    prompt_len = input_ids.shape[1] - response_len
    resp_logits = logits[:, prompt_len - 1:-1, :]  # (B, response_len, vocab)
    resp_tokens = input_ids[:, prompt_len:]  # (B, response_len)

    log_probs = torch.log_softmax(resp_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, resp_tokens.unsqueeze(-1)).squeeze(-1)  # (B, response_len)

    return token_log_probs


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from verl071.opear.data import (
        select_batch_positions, reconstruct_trajectories,
        tokenize_contrastive_responses, _find_assistant_segments,
    )
    from verl071.opear.guide import OPEaRGuide
    from verl071.opear.loss import compute_opear_loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # ── Step 1: Load model and tokenizer ──
    log("\n" + "=" * 70)
    log("STEP 1: Load Qwen3-4B")
    log("=" * 70)
    model_name = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    log(f"Model loaded: {model_name} ({sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params)")

    # ── Step 2: Build batch ──
    log("\n" + "=" * 70)
    log("STEP 2: Build realistic multi-turn batch")
    log("=" * 70)
    batch, prompt_len, max_resp_len, segments, rewards, group_size = build_realistic_batch(tokenizer)
    total_rollouts = len(rewards)
    resp_len = max_resp_len

    log(f"Rollouts: {total_rollouts} ({total_rollouts // group_size} prompts × {group_size} group)")
    log(f"Prompt length: {prompt_len} tokens")
    log(f"Response length: {resp_len} tokens")
    log(f"Assistant segments: {segments}")
    log(f"Rewards: {rewards.tolist()}")

    # ── Step 3: GRPO advantages ──
    log("\n" + "=" * 70)
    log("STEP 3: Compute GRPO advantages (group-relative)")
    log("=" * 70)
    advantages = compute_grpo_advantages(rewards, group_size)
    log(f"Advantages: {advantages.tolist()}")
    for g in range(total_rollouts // group_size):
        g_rewards = rewards[g * group_size:(g + 1) * group_size]
        g_advs = advantages[g * group_size:(g + 1) * group_size]
        log(f"  Group {g}: rewards={g_rewards.tolist()} → advantages={g_advs.tolist()}")

    # ── Step 4: Original rollout log-probs ──
    log("\n" + "=" * 70)
    log("STEP 4: Compute original rollout log-probs under policy")
    log("=" * 70)
    orig_lp = forward_logprobs(
        model, tokenizer,
        batch.batch["input_ids"], batch.batch["attention_mask"],
        resp_len, device,
    )
    orig_resp_mask = batch.batch["response_mask"]
    orig_lp_cpu = orig_lp.cpu()
    orig_masked_lp = (orig_lp_cpu * orig_resp_mask).sum(dim=-1) / orig_resp_mask.sum(dim=-1).clamp(min=1)
    log(f"Per-rollout mean log-prob (assistant tokens only):")
    for i in range(total_rollouts):
        active = int(orig_resp_mask[i].sum())
        log(f"  Rollout {i}: mean_lp={orig_masked_lp[i].item():.4f} (over {active} tokens)")

    # ── Step 5: Select and reconstruct ──
    log("\n" + "=" * 70)
    log("STEP 5: Select rollouts for contrastive pairs")
    log("=" * 70)
    positions = select_batch_positions(batch, group_size=group_size, selection_ratio=0.5)
    log(f"Selection ratio: 0.5, group_size: {group_size}")
    log(f"Selected positions: {positions} ({len(positions)} rollouts)")

    trajectories = reconstruct_trajectories(batch, tokenizer, positions)
    log(f"Reconstructed {len(trajectories)} trajectories")
    traj = trajectories[0]
    asst_turns = [t for t in traj["turns"] if t["role"] == "assistant"]
    log(f"First trajectory: {len(traj['turns'])} turns ({len(asst_turns)} assistant)")
    log(f"  Segments: {traj['assistant_segments']}")
    log(f"  Task: {traj['task_description'][:60]}...")
    for i, t in enumerate(asst_turns):
        log(f"  Assistant turn {i+1}: {t['content'][:70]}...")

    # ── Step 6: Call guide model ──
    log("\n" + "=" * 70)
    log("STEP 6: Call GPT-5.4-nano for contrastive pairs")
    log("=" * 70)
    guide = OPEaRGuide(model="gpt-5.4-nano")
    pairs = guide.generate_contrastive_batch(trajectories)
    n_valid = sum(1 for p in pairs if p is not None)
    log(f"Guide returned {n_valid}/{len(trajectories)} valid pairs")

    for i, pair in enumerate(pairs):
        if pair is None:
            log(f"  Pair {i}: FAILED")
            continue
        log(f"  Pair {i}:")
        for j, (c, v) in enumerate(zip(pair["compliant"], pair["violating"])):
            c_text = f"<think>{c['think'][:40]}...</think><action>{c['action']}</action>" if isinstance(c, dict) else str(c)[:80]
            v_text = f"<think>{v['think'][:40]}...</think><action>{v['action']}</action>" if isinstance(v, dict) else str(v)[:80]
            log(f"    Turn {j+1} compliant: {c_text}")
            log(f"    Turn {j+1} violating: {v_text}")

    # ── Step 7: Tokenize contrastive sequences ──
    log("\n" + "=" * 70)
    log("STEP 7: Tokenize contrastive sequences (in-place swap)")
    log("=" * 70)
    opear_data = tokenize_contrastive_responses(trajectories, pairs, batch, tokenizer, max_resp_len)

    if opear_data is None:
        log("ERROR: No valid contrastive pairs after tokenization")
        _save()
        return

    c_ids = opear_data["compliant_input_ids"]
    v_ids = opear_data["violating_input_ids"]
    c_mask = opear_data["compliant_response_mask"]
    v_mask = opear_data["violating_response_mask"]
    num_pairs = c_ids.shape[0]

    log(f"Contrastive pairs: {num_pairs}")
    log(f"Shapes: input_ids={c_ids.shape}, response_mask={c_mask.shape}")

    # Verify structure preservation
    orig_ids = batch.batch["input_ids"][positions[0]]
    prompt_match = torch.equal(c_ids[0, :prompt_len], orig_ids[:prompt_len])
    log(f"Prompt preserved: {prompt_match}")

    # Check observations are present (not at same positions — rebuild shifts them)
    orig_rm = batch.batch["response_mask"][positions[0]]
    orig_resp = batch.batch["responses"][positions[0]]
    # Extract original observation text between segments
    obs_texts = []
    prev_end = 0
    for s, e in segments:
        if s > prev_end:
            obs_text = tokenizer.decode(orig_resp[prev_end:s], skip_special_tokens=True).strip()
            if obs_text:
                obs_texts.append(obs_text)
        prev_end = e
    c_full_text = tokenizer.decode(c_ids[0], skip_special_tokens=True)
    obs_ok = all(obs in c_full_text for obs in obs_texts)
    log(f"Observations present in contrastive: {obs_ok}")

    # Show token counts using rebuilt segment boundaries
    from verl071.opear.data import _find_assistant_segments
    orig_segs = _find_assistant_segments(orig_rm)
    c_segs = _find_assistant_segments(c_mask[0])
    v_segs = _find_assistant_segments(v_mask[0])
    log(f"  Original segments: {[(s, e, e-s) for s, e in orig_segs]}")
    log(f"  Compliant segments: {[(s, e, e-s) for s, e in c_segs]}")
    log(f"  Violating segments: {[(s, e, e-s) for s, e in v_segs]}")

    # ── Step 8: Forward pass for contrastive log-probs ──
    log("\n" + "=" * 70)
    log("STEP 8: Compute contrastive log-probs under policy")
    log("=" * 70)
    c_attn = opear_data["compliant_attention_mask"]
    v_attn = opear_data["violating_attention_mask"]

    c_lp = forward_logprobs(model, tokenizer, c_ids, c_attn, resp_len, device).cpu()
    v_lp = forward_logprobs(model, tokenizer, v_ids, v_attn, resp_len, device).cpu()

    # Masked mean log-probs
    c_lengths = c_mask.sum(dim=-1).clamp(min=1)
    v_lengths = v_mask.sum(dim=-1).clamp(min=1)
    c_mean_lp = (c_lp * c_mask).sum(dim=-1) / c_lengths
    v_mean_lp = (v_lp * v_mask).sum(dim=-1) / v_lengths

    log(f"Per-pair log-probs:")
    for i in range(num_pairs):
        gap = c_mean_lp[i].item() - v_mean_lp[i].item()
        log(f"  Pair {i}: compliant={c_mean_lp[i].item():.4f} ({int(c_lengths[i])} tok), "
            f"violating={v_mean_lp[i].item():.4f} ({int(v_lengths[i])} tok), gap={gap:.4f}")

    # ── Step 9: Compute OPEAR losses ──
    log("\n" + "=" * 70)
    log("STEP 9: Compute OPEAR losses (both variants)")
    log("=" * 70)

    # Unbounded
    loss_ub, metrics_ub = compute_opear_loss(c_lp, c_mask, v_lp, v_mask, alpha=0.5, loss_type="unbounded")
    log(f"UNBOUNDED (alpha=0.5):")
    for k, v in sorted(metrics_ub.items()):
        log(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # Logsigmoid beta=1.0
    loss_ls1, metrics_ls1 = compute_opear_loss(c_lp, c_mask, v_lp, v_mask, loss_type="logsigmoid", beta=1.0)
    log(f"\nLOGSIGMOID (beta=1.0):")
    for k, v in sorted(metrics_ls1.items()):
        log(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # Logsigmoid beta=0.1
    loss_ls01, metrics_ls01 = compute_opear_loss(c_lp, c_mask, v_lp, v_mask, loss_type="logsigmoid", beta=0.1)
    log(f"\nLOGSIGMOID (beta=0.1):")
    for k, v in sorted(metrics_ls01.items()):
        log(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # ── Step 10: Gradient scale comparison ──
    log("\n" + "=" * 70)
    log("STEP 10: Gradient scale comparison (GRPO vs OPEAR)")
    log("=" * 70)

    # Simulate GRPO scaling
    loss_scale_factor = 15360  # from config (= max_response_length)
    gradient_accumulation = 16  # ppo_mini_batch_size / micro_batch_size
    grpo_raw_loss = 0.1  # typical pg_loss value

    # GRPO: each micro-batch does (raw / loss_scale_factor) * (1/grad_accum)
    # After 16 micro-batches accumulate: raw / loss_scale_factor
    grpo_per_microbatch = grpo_raw_loss / loss_scale_factor / gradient_accumulation
    grpo_total = grpo_raw_loss / loss_scale_factor  # after accumulation

    # OPEAR scaling comparison
    lam = 0.5
    opear_raw = loss_ls1.item()
    opear_no_fix = lam * opear_raw
    opear_div_accum = lam * opear_raw / gradient_accumulation
    opear_div_sf = lam * opear_raw / loss_scale_factor  # CORRECT fix

    log(f"GRPO:")
    log(f"  raw pg_loss:         {grpo_raw_loss:.4f}")
    log(f"  / loss_sf (15360):   {grpo_raw_loss / loss_scale_factor:.8f}")
    log(f"  per micro-batch:     {grpo_per_microbatch:.8f}")
    log(f"  total (16 accumulated): {grpo_total:.8f}")
    log(f"\nOPEAR (lambda={lam}, logsigmoid):")
    log(f"  raw loss:            {opear_raw:.4f}")
    log(f"  NO FIX:              {opear_no_fix:.8f} ({opear_no_fix / grpo_total:.0f}x GRPO)")
    log(f"  / grad_accum (16):   {opear_div_accum:.8f} ({opear_div_accum / grpo_total:.0f}x GRPO)")
    log(f"  / loss_sf (15360):   {opear_div_sf:.8f} ({opear_div_sf / grpo_total:.1f}x GRPO) ← CORRECT")

    # ── Step 11: Summary ──
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"Batch: {total_rollouts} rollouts, {prompt_len}+{resp_len} tokens each")
    log(f"Segments: {len(segments)} assistant turns per rollout")
    log(f"Guide: {n_valid}/{len(trajectories)} valid pairs from GPT-5.4-nano")
    log(f"Contrastive pairs: {num_pairs}")
    log(f"Mean logprob gap: {metrics_ls1['opear/logprob_gap']:.4f}")
    log(f"Unbounded loss: {loss_ub.item():.4f}")
    log(f"Logsigmoid loss (beta=1.0): {loss_ls1.item():.4f}")
    log(f"Logsigmoid loss (beta=0.1): {loss_ls01.item():.4f}")
    log(f"Gradient ratio (no fix):     OPEAR was {opear_no_fix / grpo_total:.0f}x GRPO")
    log(f"Gradient ratio (/ loss_sf):  OPEAR is {opear_div_sf / grpo_total:.1f}x GRPO")

    _save()


def _save():
    path = os.path.join(os.getcwd(), OUTPUT_FILE)
    with open(path, "w") as f:
        f.write("\n".join(out))
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
