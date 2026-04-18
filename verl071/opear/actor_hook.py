"""O-PEaR gradient accumulation hook for verl's DataParallelPPOActor.

Called from inside update_policy (via verl patch) between the GRPO backward
pass and optimizer.step(). Computes O-PEaR contrastive loss and calls
backward(), accumulating gradients with the existing GRPO gradients.
"""

import torch
from verl071.opear.loss import compute_opear_loss


def opear_accumulate_gradients(actor, data, metrics):
    """Compute O-PEaR loss and accumulate gradients (no optimizer step).

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

    resp_len = c_mask.shape[-1]

    c_out = actor._forward_micro_batch(
        micro_batch={"input_ids": c_ids, "attention_mask": c_attn,
                     "position_ids": compute_position_id_with_mask(c_attn),
                     "responses": c_ids[:, -resp_len:]},
        temperature=temperature, calculate_entropy=False)
    c_lp = c_out["log_probs"]

    v_out = actor._forward_micro_batch(
        micro_batch={"input_ids": v_ids, "attention_mask": v_attn,
                     "position_ids": compute_position_id_with_mask(v_attn),
                     "responses": v_ids[:, -resp_len:]},
        temperature=temperature, calculate_entropy=False)
    v_lp = v_out["log_probs"]

    c_lp = c_lp[:, :resp_len]
    v_lp = v_lp[:, :resp_len]
    c_rm = c_mask[:, :c_lp.shape[-1]]
    v_rm = v_mask[:, :v_lp.shape[-1]]

    loss, opear_metrics = compute_opear_loss(
        c_lp, c_rm, v_lp, v_rm, alpha=alpha, loss_type=loss_type, beta=beta)

    # Scale to match GRPO gradient magnitude. GRPO's agg_loss divides by
    # loss_scale_factor (a constant normalizer, typically = max_response_length).
    # OPEAR runs once per mini-batch (not per micro-batch), so we only need
    # the loss_scale_factor — NOT gradient_accumulation — to put both losses
    # on the same footing. This makes lambda=1 mean "equal gradient magnitude."
    loss_sf = getattr(actor.config, "loss_scale_factor", None)
    if loss_sf is None or loss_sf <= 0:
        grad_accum = getattr(actor, "gradient_accumulation", 1)
        loss_sf = grad_accum
    # Snapshot grad norm BEFORE opear backward (captures GRPO-only grads)
    grpo_grad_norm = _grad_norm(actor.actor_module)

    scaled = lam * loss / loss_sf
    scaled.backward()

    # Grad norm AFTER opear backward (captures GRPO + OPEAR combined)
    combined_grad_norm = _grad_norm(actor.actor_module)
    # OPEAR contribution ≈ combined - grpo (approximate, not exact due to direction)
    opear_grad_norm = abs(combined_grad_norm - grpo_grad_norm)

    opear_metrics["opear/scaled_loss"] = scaled.detach().item()
    opear_metrics["opear/loss_scale_factor"] = float(loss_sf)
    opear_metrics["opear/grad_norm"] = opear_grad_norm
    opear_metrics["opear/grpo_grad_norm"] = grpo_grad_norm
    opear_metrics["opear/combined_grad_norm"] = combined_grad_norm
    opear_metrics["opear/lambda"] = lam
    opear_metrics["opear/alpha"] = alpha
    opear_metrics["opear/loss_type"] = 1.0 if loss_type == "logsigmoid" else 0.0
    opear_metrics["opear/loss_beta"] = beta
    opear_metrics["opear/selection_ratio"] = data.meta_info.get("opear_selection_ratio", 0.5)
    opear_metrics["opear/guide_time_s"] = data.meta_info.get("opear_guide_time_s", 0.0)
    opear_metrics["opear/num_segments"] = data.meta_info.get("opear_num_segments", 0.0)

    for k, v in opear_metrics.items():
        metrics[k] = metrics.get(k, 0.0) + v


def _grad_norm(module) -> float:
    """Compute total L2 gradient norm across all parameters."""
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.data.float().norm().item() ** 2
    return total ** 0.5
