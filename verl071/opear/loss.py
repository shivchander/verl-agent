"""O-PEaR (Off-Policy Environment-aware Regularization) loss computation.

Three-way ranking loss:
  L_cv = -mean(logsigmoid(beta * (c - v - margin)))   # compliant > violating
  L_cp = -mean(logsigmoid(beta * (c - p)))             # compliant > policy (no margin)
  L_opear = L_cv + L_cp

The margin is the target gap for L_cv: gradients are strong when gap < margin,
half-strength at gap = margin, and decay when gap > margin. Beta controls the
sharpness of this transition.

L_cp uses no margin — it anchors compliant logprobs above the policy's own
rollout logprobs to prevent off-policy logprob collapse. The policy logprob p
is detached (no gradient flows through it).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def compute_opear_loss(
    compliant_log_probs: torch.Tensor,
    compliant_mask: torch.Tensor,
    violating_log_probs: torch.Tensor,
    violating_mask: torch.Tensor,
    beta: float = 1.0,
    margin: float = 0.0,
    policy_mean_lp: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """Compute the O-PEaR three-way ranking loss.

    Args:
        compliant_log_probs: Per-token log-probabilities for compliant responses. Shape (N, response_len).
        compliant_mask: Binary mask for compliant responses. Shape (N, response_len).
        violating_log_probs: Per-token log-probabilities for violating responses. Shape (N, response_len).
        violating_mask: Binary mask for violating responses. Shape (N, response_len).
        beta: Sharpness of transition around margin. Default 1.0.
        margin: Target gap for L_cv (compliant vs violating). Default 0.0.
        policy_mean_lp: Per-token-mean logprob of the policy's own rollout, detached. Shape (N,).
            When None, L_cp is skipped (backward compat).

    Returns:
        loss: Scalar tensor (L_cv + L_cp, or L_cv only if policy_mean_lp is None).
        metrics: Dict with diagnostic metrics.
    """
    num_pairs = compliant_log_probs.shape[0]

    compliant_lengths = compliant_mask.sum(dim=-1).clamp(min=1.0)
    violating_lengths = violating_mask.sum(dim=-1).clamp(min=1.0)

    compliant_mean_lp = (compliant_log_probs * compliant_mask).sum(dim=-1) / compliant_lengths
    violating_mean_lp = (violating_log_probs * violating_mask).sum(dim=-1) / violating_lengths

    cv_gap = compliant_mean_lp - violating_mean_lp
    cv_loss = -F.logsigmoid(beta * (cv_gap - margin)).mean()

    if policy_mean_lp is not None:
        policy_mean_lp = policy_mean_lp.detach()
        cp_gap = compliant_mean_lp - policy_mean_lp
        cp_loss = -F.logsigmoid(beta * cp_gap).mean()
        loss = cv_loss + cp_loss
    else:
        cp_gap = None
        cp_loss = None
        loss = cv_loss

    metrics = {
        "opear/loss": loss.detach().item(),
        "opear/cv_loss": cv_loss.detach().item(),
        "opear/compliant_logprob": compliant_mean_lp.mean().detach().item(),
        "opear/violating_logprob": violating_mean_lp.mean().detach().item(),
        "opear/logprob_gap": cv_gap.mean().detach().item(),
        "opear/gap_std": cv_gap.std().detach().item() if num_pairs > 1 else 0.0,
        "opear/gap_min": cv_gap.min().detach().item(),
        "opear/gap_max": cv_gap.max().detach().item(),
        "opear/num_pairs": num_pairs,
        "opear/compliant_length": compliant_lengths.mean().detach().item(),
        "opear/violating_length": violating_lengths.mean().detach().item(),
        "opear/margin": margin,
    }

    if policy_mean_lp is not None:
        metrics["opear/cp_loss"] = cp_loss.detach().item()
        metrics["opear/policy_logprob"] = policy_mean_lp.mean().item()
        metrics["opear/cp_gap"] = cp_gap.mean().detach().item()

    return loss, metrics
