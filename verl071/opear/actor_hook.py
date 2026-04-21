"""O-PEaR gradient accumulation hook for verl's DataParallelPPOActor.

Called from inside update_policy (via verl patch) between the GRPO backward
pass and optimizer.step(). Computes O-PEaR contrastive loss and calls
backward(), accumulating gradients with the existing GRPO gradients.

Uses micro-batching (one pair at a time) to avoid OOM — GRPO processes
sequences with micro_batch_size=1, so forwarding all N contrastive pairs
at once would exceed GPU memory.

No loss_scale_factor division: O-PEaR uses per-token-mean log-probs which
naturally normalizes by sequence length (~800), matching GRPO's token-sum /
loss_sf (~15360) convention at the per-token gradient level.
"""

import torch
from verl071.opear.loss import compute_opear_loss


def opear_accumulate_gradients(actor, data, metrics):
    """Compute O-PEaR loss and accumulate gradients (no optimizer step).

    Processes one contrastive pair at a time to match GRPO's memory footprint.
    Each per-pair loss is scaled by 1/N before backward so that the accumulated
    gradients equal the gradient of the mean loss over all pairs.

    Args:
        actor: DataParallelPPOActor instance (has _forward_micro_batch, actor_module)
        data: DataProto with meta_info containing opear_data, opear_lambda
        metrics: dict to update with O-PEaR metrics
    """
    if not hasattr(data, "meta_info"):
        return
    opear_data = data.meta_info.get("opear_data")
    if opear_data is None:
        return

    from verl.utils.model import compute_position_id_with_mask

    lam = data.meta_info.get("opear_lambda", 0.5)
    beta = data.meta_info.get("opear_beta", 1.0)
    margin = data.meta_info.get("opear_margin", 0.0)
    temperature = data.meta_info.get("temperature", 1.0)
    device = next(actor.actor_module.parameters()).device

    c_ids = opear_data["compliant_input_ids"].to(device)
    c_attn = opear_data["compliant_attention_mask"].to(device)
    c_mask = opear_data["compliant_response_mask"].to(device)
    v_ids = opear_data["violating_input_ids"].to(device)
    v_attn = opear_data["violating_attention_mask"].to(device)
    v_mask = opear_data["violating_response_mask"].to(device)

    # Extract per-token-mean policy logprob for each contrastive pair
    batch_positions = data.meta_info.get("opear_batch_positions", [])
    policy_mean_lp = None
    if batch_positions and "old_log_probs" in data.batch:
        old_lp = data.batch["old_log_probs"]  # (batch_size, response_len)
        resp_mask = data.batch["response_mask"]  # (batch_size, response_len)
        p_lps = []
        for bp in batch_positions:
            lp_row = old_lp[bp]
            mask_row = resp_mask[bp]
            length = mask_row.sum().clamp(min=1.0)
            p_lps.append((lp_row * mask_row).sum() / length)
        policy_mean_lp = torch.stack(p_lps).to(device)

    num_pairs = c_ids.shape[0]
    resp_len = c_mask.shape[-1]

    # Snapshot per-parameter grads BEFORE any O-PEaR backward (captures GRPO-only grads)
    grpo_grad_norm = _grad_norm(actor.actor_module)
    grpo_grads = {
        n: p.grad.data.clone() for n, p in actor.actor_module.named_parameters()
        if p.grad is not None
    }

    # Accumulators for metrics (averaged over pairs at the end)
    total_loss = 0.0
    total_scaled_loss = 0.0
    agg_metrics = {}
    gap_values = []

    # --- Micro-batched loop: one pair at a time ---
    for i in range(num_pairs):
        ci_ids = c_ids[i : i + 1]
        ci_attn = c_attn[i : i + 1]
        ci_mask = c_mask[i : i + 1]
        vi_ids = v_ids[i : i + 1]
        vi_attn = v_attn[i : i + 1]
        vi_mask = v_mask[i : i + 1]

        # Forward compliant sequence
        c_out = actor._forward_micro_batch(
            micro_batch={
                "input_ids": ci_ids,
                "attention_mask": ci_attn,
                "position_ids": compute_position_id_with_mask(ci_attn),
                "responses": ci_ids[:, -resp_len:],
            },
            temperature=temperature,
            calculate_entropy=False,
        )
        c_lp = c_out["log_probs"][:, :resp_len]

        # Forward violating sequence
        v_out = actor._forward_micro_batch(
            micro_batch={
                "input_ids": vi_ids,
                "attention_mask": vi_attn,
                "position_ids": compute_position_id_with_mask(vi_attn),
                "responses": vi_ids[:, -resp_len:],
            },
            temperature=temperature,
            calculate_entropy=False,
        )
        v_lp = v_out["log_probs"][:, :resp_len]

        ci_rm = ci_mask[:, : c_lp.shape[-1]]
        vi_rm = vi_mask[:, : v_lp.shape[-1]]

        pair_policy_lp = policy_mean_lp[i:i+1] if policy_mean_lp is not None else None
        pair_loss, pair_metrics = compute_opear_loss(
            c_lp, ci_rm, v_lp, vi_rm, beta=beta, margin=margin,
            policy_mean_lp=pair_policy_lp,
        )

        # No loss_sf: per-token-mean already normalizes by sequence length,
        # matching GRPO's per-token gradient scale. Lambda controls the
        # relative weight directly.
        scaled = lam * pair_loss / num_pairs
        scaled.backward()

        total_loss += pair_loss.detach().item()
        total_scaled_loss += scaled.detach().item()

        for k, v in pair_metrics.items():
            agg_metrics[k] = agg_metrics.get(k, 0.0) + v

        gap_values.append(pair_metrics.get("opear/logprob_gap", 0.0))

    # Compute O-PEaR grad norm by subtracting snapshotted GRPO grads
    combined_grad_norm = _grad_norm(actor.actor_module)
    opear_grad_sq = 0.0
    for n, p in actor.actor_module.named_parameters():
        if p.grad is not None and n in grpo_grads:
            opear_grad_sq += (p.grad.data.float() - grpo_grads[n].float()).norm().item() ** 2
        elif p.grad is not None:
            opear_grad_sq += p.grad.data.float().norm().item() ** 2
    opear_grad_norm = opear_grad_sq ** 0.5
    del grpo_grads

    # Build final metrics: average per-pair values
    opear_metrics = {}
    for k, v in agg_metrics.items():
        if k == "opear/num_pairs":
            opear_metrics[k] = num_pairs
        elif k in ("opear/gap_std", "opear/gap_min", "opear/gap_max"):
            continue
        else:
            opear_metrics[k] = v / num_pairs

    if gap_values:
        gap_t = torch.tensor(gap_values)
        opear_metrics["opear/gap_min"] = gap_t.min().item()
        opear_metrics["opear/gap_max"] = gap_t.max().item()
        opear_metrics["opear/gap_std"] = gap_t.std().item() if num_pairs > 1 else 0.0

    opear_metrics["opear/loss"] = total_loss / num_pairs
    # Propagate L_cp metrics from aggregated pair metrics
    if "opear/cp_loss" in agg_metrics:
        opear_metrics["opear/cp_loss"] = agg_metrics["opear/cp_loss"] / num_pairs
    if "opear/policy_logprob" in agg_metrics:
        opear_metrics["opear/policy_logprob"] = agg_metrics["opear/policy_logprob"] / num_pairs
    if "opear/cp_gap" in agg_metrics:
        opear_metrics["opear/cp_gap"] = agg_metrics["opear/cp_gap"] / num_pairs
    if "opear/cv_loss" in agg_metrics:
        opear_metrics["opear/cv_loss"] = agg_metrics["opear/cv_loss"] / num_pairs
    opear_metrics["opear/scaled_loss"] = total_scaled_loss
    opear_metrics["opear/grad_norm"] = opear_grad_norm
    opear_metrics["opear/grpo_grad_norm"] = grpo_grad_norm
    opear_metrics["opear/combined_grad_norm"] = combined_grad_norm
    opear_metrics["opear/guide_time_s"] = data.meta_info.get("opear_guide_time_s", 0.0)
    opear_metrics["opear/num_segments"] = data.meta_info.get("opear_num_segments", 0.0)

    for k, v in opear_metrics.items():
        metrics[k] = metrics.get(k, 0.0) + v

    metrics["opear/lambda"] = lam
    metrics["opear/beta"] = beta
    metrics["opear/margin"] = margin
    metrics["opear/selection_ratio"] = data.meta_info.get("opear_selection_ratio", 0.5)


def _grad_norm(module) -> float:
    """Compute total L2 gradient norm across all parameters."""
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.data.float().norm().item() ** 2
    return total ** 0.5
