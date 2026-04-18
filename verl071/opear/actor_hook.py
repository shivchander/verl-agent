"""O-PEaR gradient accumulation hook for verl's DataParallelPPOActor.

Called from inside update_policy (via verl patch) between the GRPO backward
pass and optimizer.step(). Computes O-PEaR contrastive loss and calls
backward(), accumulating gradients with the existing GRPO gradients.

Uses micro-batching (one pair at a time) to avoid OOM — GRPO processes
sequences with micro_batch_size=1, so forwarding all N contrastive pairs
at once would exceed GPU memory.
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
        data: DataProto with meta_info containing opear_data, opear_alpha, opear_lambda
        metrics: dict to update with O-PEaR metrics
    """
    if not hasattr(data, "meta_info"):
        return
    opear_data = data.meta_info.get("opear_data")
    if opear_data is None:
        return

    from verl.utils.model import compute_position_id_with_mask

    alpha = data.meta_info.get("opear_alpha", 0.5)
    lam = data.meta_info.get("opear_lambda", 0.5)
    loss_type = data.meta_info.get("opear_loss_type", "unbounded")
    beta = data.meta_info.get("opear_loss_beta", 1.0)
    temperature = data.meta_info.get("temperature", 1.0)
    device = next(actor.actor_module.parameters()).device

    c_ids = opear_data["compliant_input_ids"].to(device)
    c_attn = opear_data["compliant_attention_mask"].to(device)
    c_mask = opear_data["compliant_response_mask"].to(device)
    v_ids = opear_data["violating_input_ids"].to(device)
    v_attn = opear_data["violating_attention_mask"].to(device)
    v_mask = opear_data["violating_response_mask"].to(device)

    num_pairs = c_ids.shape[0]
    resp_len = c_mask.shape[-1]

    # Scale to match GRPO gradient magnitude. GRPO's agg_loss divides by
    # loss_scale_factor (a constant normalizer, typically = max_response_length).
    # OPEAR runs once per mini-batch (not per micro-batch), so we only need
    # the loss_scale_factor -- NOT gradient_accumulation -- to put both losses
    # on the same footing. This makes lambda=1 mean "equal gradient magnitude."
    loss_sf = getattr(actor.config, "loss_scale_factor", None)
    if loss_sf is None or loss_sf <= 0:
        import logging
        logging.getLogger(__name__).warning(
            "loss_scale_factor not found on actor.config -- OPEAR gradient scaling "
            "may be wrong. Falling back to gradient_accumulation."
        )
        loss_sf = getattr(actor, "gradient_accumulation", 1)

    # Snapshot grad norm BEFORE any O-PEaR backward (captures GRPO-only grads)
    grpo_grad_norm = _grad_norm(actor.actor_module)

    # Accumulators for metrics (averaged over pairs at the end)
    total_loss = 0.0
    total_scaled_loss = 0.0
    agg_metrics = {}
    gap_values = []

    # --- Micro-batched loop: one pair at a time ---
    for i in range(num_pairs):
        # Slice out the i-th pair (keep batch dim via [i:i+1])
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

        # compute_opear_loss with N=1 returns the single-pair loss (mean is a
        # no-op for a single element). We scale by 1/num_pairs so that the
        # sum of backward passes yields grad(mean_loss).
        pair_loss, pair_metrics = compute_opear_loss(
            c_lp, ci_rm, v_lp, vi_rm, alpha=alpha, loss_type=loss_type, beta=beta
        )

        scaled = lam * pair_loss / (loss_sf * num_pairs)
        scaled.backward()

        total_loss += pair_loss.detach().item()
        total_scaled_loss += scaled.detach().item()

        # Accumulate per-pair metrics for later averaging
        for k, v in pair_metrics.items():
            agg_metrics[k] = agg_metrics.get(k, 0.0) + v

        # Track individual gap values for proper min/max/std
        gap_values.append(pair_metrics.get("opear/logprob_gap", 0.0))

    # Grad norm AFTER all O-PEaR backwards (captures GRPO + OPEAR combined)
    combined_grad_norm = _grad_norm(actor.actor_module)
    # OPEAR contribution ~ combined - grpo (approximate, not exact due to direction)
    opear_grad_norm = abs(combined_grad_norm - grpo_grad_norm)

    # Build final metrics: average per-pair values, keep num_pairs as total.
    # gap_min/gap_max/gap_std are recomputed from collected per-pair gaps
    # since N=1 calls can't compute meaningful min/max/std individually.
    opear_metrics = {}
    for k, v in agg_metrics.items():
        if k == "opear/num_pairs":
            opear_metrics[k] = num_pairs
        elif k in ("opear/gap_std", "opear/gap_min", "opear/gap_max"):
            continue  # handled below from gap_values
        else:
            opear_metrics[k] = v / num_pairs

    if gap_values:
        gap_t = torch.tensor(gap_values)
        opear_metrics["opear/gap_min"] = gap_t.min().item()
        opear_metrics["opear/gap_max"] = gap_t.max().item()
        opear_metrics["opear/gap_std"] = gap_t.std().item() if num_pairs > 1 else 0.0
    else:
        opear_metrics["opear/gap_min"] = 0.0
        opear_metrics["opear/gap_max"] = 0.0
        opear_metrics["opear/gap_std"] = 0.0

    # Override loss with the proper mean (not the per-pair average of
    # compute_opear_loss which is the same thing for N=1, but be explicit)
    opear_metrics["opear/loss"] = total_loss / num_pairs
    opear_metrics["opear/scaled_loss"] = total_scaled_loss
    opear_metrics["opear/loss_scale_factor"] = float(loss_sf)
    opear_metrics["opear/grad_norm"] = opear_grad_norm
    opear_metrics["opear/grpo_grad_norm"] = grpo_grad_norm
    opear_metrics["opear/combined_grad_norm"] = combined_grad_norm
    opear_metrics["opear/guide_time_s"] = data.meta_info.get("opear_guide_time_s", 0.0)
    opear_metrics["opear/num_segments"] = data.meta_info.get("opear_num_segments", 0.0)

    # Accumulate per-step metrics (loss, grad norms, etc.)
    for k, v in opear_metrics.items():
        metrics[k] = metrics.get(k, 0.0) + v

    # Set config constants directly (not accumulated)
    metrics["opear/lambda"] = lam
    metrics["opear/alpha"] = alpha
    metrics["opear/loss_type"] = 1.0 if loss_type == "logsigmoid" else 0.0
    metrics["opear/loss_beta"] = beta
    metrics["opear/selection_ratio"] = data.meta_info.get("opear_selection_ratio", 0.5)


def _grad_norm(module) -> float:
    """Compute total L2 gradient norm across all parameters."""
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.data.float().norm().item() ** 2
    return total ** 0.5
