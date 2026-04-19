"""Quick gradient scale verification: GRPO vs O-PEaR.

Loads Qwen3-4B, creates two different sequences (compliant vs violating),
computes actual gradients through the model, and compares gradient norms.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python verl071/tests/test_gradient_scale.py
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.float().norm().item() ** 2
    return total ** 0.5


def forward_logprobs(model, input_ids, prompt_len):
    """Get per-token log-probs for the response portion."""
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, prompt_len - 1:-1, :]
    resp_ids = input_ids[:, prompt_len:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(-1, resp_ids.unsqueeze(-1)).squeeze(-1)
    return token_lp


def main():
    device = torch.device("cuda")
    model_name = "Qwen/Qwen3-4B"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.train()

    # Two different responses (compliant vs violating)
    prompt = "You are an agent. Your task is to put a pencil on the sidetable."
    compliant = prompt + " <think>I see a pencil on the desk. Let me pick it up.</think><action>take pencil 1 from desk 1</action>"
    violating = prompt + " <think>Maybe I should look around more first.</think><action>go to shelf 1</action>"

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    c_ids = torch.tensor([tokenizer.encode(compliant, add_special_tokens=True)], device=device)
    v_ids = torch.tensor([tokenizer.encode(violating, add_special_tokens=True)], device=device)
    prompt_len = len(prompt_ids)

    # Pad to same length
    max_len = max(c_ids.shape[1], v_ids.shape[1])
    pad_id = tokenizer.pad_token_id or 0
    if c_ids.shape[1] < max_len:
        c_ids = F.pad(c_ids, (0, max_len - c_ids.shape[1]), value=pad_id)
    if v_ids.shape[1] < max_len:
        v_ids = F.pad(v_ids, (0, max_len - v_ids.shape[1]), value=pad_id)

    c_resp_len = max_len - prompt_len
    v_resp_len = max_len - prompt_len

    print(f"Prompt: {prompt_len} tokens")
    print(f"Compliant: {c_ids.shape[1]} tokens (resp={c_resp_len})")
    print(f"Violating: {v_ids.shape[1]} tokens (resp={v_resp_len})")

    # Create masks (1 for response tokens, 0 for prompt/padding)
    c_mask = torch.ones(1, c_resp_len, device=device)
    v_mask = torch.ones(1, v_resp_len, device=device)

    loss_sf = 15360
    grad_accum = 16
    num_pairs = 64

    print("=" * 60)

    # ── GRPO-style (token-sum / loss_sf / grad_accum) ──
    model.zero_grad()
    c_lp = forward_logprobs(model, c_ids, prompt_len)
    advantage = 1.0
    grpo_loss = -(advantage * c_lp * c_mask).sum() / loss_sf / grad_accum
    grpo_loss.backward()
    grpo_gn = grad_norm(model)
    print(f"\nGRPO-style (token-sum / {loss_sf} / {grad_accum}):")
    print(f"  raw token-sum: {-(advantage * c_lp * c_mask).sum().item():.4f}")
    print(f"  scaled loss:   {grpo_loss.item():.8f}")
    print(f"  grad norm:     {grpo_gn:.6f}")

    # ── O-PEaR OLD (per-token mean, beta=0.1, alpha=0.5, / loss_sf) ──
    model.zero_grad()
    c_lp = forward_logprobs(model, c_ids, prompt_len)
    v_lp = forward_logprobs(model, v_ids, prompt_len)
    c_mean = (c_lp * c_mask).sum(-1) / c_mask.sum(-1)
    v_mean = (v_lp * v_mask).sum(-1) / v_mask.sum(-1)
    gap = 0.5 * c_mean - 0.5 * v_mean  # alpha=0.5
    old_loss = -F.logsigmoid(0.1 * gap).mean()  # beta=0.1
    old_scaled = 0.5 * old_loss / (loss_sf * num_pairs)  # lambda=0.5
    old_scaled.backward()
    old_gn = grad_norm(model)
    print(f"\nO-PEaR OLD (mean, alpha=0.5, beta=0.1, /loss_sf):")
    print(f"  gap (per-token mean): {(c_mean - v_mean).item():.4f}")
    print(f"  loss:        {old_loss.item():.6f}")
    print(f"  scaled loss: {old_scaled.item():.10f}")
    print(f"  grad norm:   {old_gn:.8f}")
    print(f"  ratio vs GRPO: {old_gn / grpo_gn if grpo_gn > 0 else 0:.6f}x")

    # ── O-PEaR NEW (per-token mean, beta=1.0, NO loss_sf, lambda=1.0) ──
    model.zero_grad()
    c_lp = forward_logprobs(model, c_ids, prompt_len)
    v_lp = forward_logprobs(model, v_ids, prompt_len)
    c_mean = (c_lp * c_mask).sum(-1) / c_mask.sum(-1)
    v_mean = (v_lp * v_mask).sum(-1) / v_mask.sum(-1)
    gap = c_mean - v_mean  # no alpha
    new_loss = -F.logsigmoid(1.0 * gap).mean()  # beta=1.0
    new_scaled = 1.0 * new_loss / num_pairs  # lambda=1.0, NO loss_sf
    new_scaled.backward()
    new_gn = grad_norm(model)
    print(f"\nO-PEaR NEW (mean, beta=1.0, NO loss_sf, lambda=1.0):")
    print(f"  gap (per-token mean): {gap.item():.4f}")
    print(f"  loss:        {new_loss.item():.6f}")
    print(f"  scaled loss: {new_scaled.item():.10f}")
    print(f"  grad norm:   {new_gn:.8f}")
    print(f"  ratio vs GRPO: {new_gn / grpo_gn if grpo_gn > 0 else 0:.6f}x")

    # ── O-PEaR NEW lambda=0.1 ──
    model.zero_grad()
    c_lp = forward_logprobs(model, c_ids, prompt_len)
    v_lp = forward_logprobs(model, v_ids, prompt_len)
    c_mean = (c_lp * c_mask).sum(-1) / c_mask.sum(-1)
    v_mean = (v_lp * v_mask).sum(-1) / v_mask.sum(-1)
    gap = c_mean - v_mean
    new_loss2 = -F.logsigmoid(1.0 * gap).mean()
    new_scaled2 = 0.1 * new_loss2 / num_pairs
    new_scaled2.backward()
    new_gn2 = grad_norm(model)
    print(f"\nO-PEaR NEW (mean, beta=1.0, NO loss_sf, lambda=0.1):")
    print(f"  gap:         {gap.item():.4f}")
    print(f"  loss:        {new_loss2.item():.6f}")
    print(f"  scaled loss: {new_scaled2.item():.10f}")
    print(f"  grad norm:   {new_gn2:.8f}")
    print(f"  ratio vs GRPO: {new_gn2 / grpo_gn if grpo_gn > 0 else 0:.6f}x")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"GRPO grad norm:              {grpo_gn:.6f}")
    print(f"O-PEaR OLD grad norm:        {old_gn:.8f}  ({old_gn/grpo_gn:.6f}x GRPO)")
    print(f"O-PEaR NEW λ=1.0 grad norm:  {new_gn:.8f}  ({new_gn/grpo_gn:.6f}x GRPO)")
    print(f"O-PEaR NEW λ=0.1 grad norm:  {new_gn2:.8f}  ({new_gn2/grpo_gn:.6f}x GRPO)")
    print(f"\nImprovement: OLD → NEW λ=1.0 = {new_gn/old_gn if old_gn > 0 else float('inf'):.1f}x")


if __name__ == "__main__":
    main()
