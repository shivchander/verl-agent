"""O-PEaR (Off-Policy Environment-aware Regularization) loss computation.

The O-PEaR regularizer encourages the policy to assign higher probability to
compliant responses (consistent with environment facts) and lower probability
to violating responses (contradicting facts).

Two loss variants:
  - 'unbounded': L = -mean(R)  — original, R grows without limit
  - 'logsigmoid': L = -mean(log_sigmoid(beta * diff))  — bounded, gradients
    vanish naturally as the model learns to distinguish pairs (DPO-style)

Minimizing either loss maximizes compliant log-prob and minimizes violating log-prob.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_opear_loss(
    compliant_log_probs: torch.Tensor,  # (N, response_len)
    compliant_mask: torch.Tensor,       # (N, response_len)
    violating_log_probs: torch.Tensor,  # (N, response_len)
    violating_mask: torch.Tensor,       # (N, response_len)
    alpha: float = 0.5,
    loss_type: str = "unbounded",
    beta: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Compute the O-PEaR regularization loss.

    Per-sequence normalized log-prob is computed as:
        sum(log_prob * mask) / sum(mask)
    with the denominator clamped to avoid division by zero.

    Args:
        compliant_log_probs: Per-token log-probabilities for compliant
            (fact-consistent) responses. Shape (N, response_len).
        compliant_mask: Binary mask indicating valid tokens in compliant
            responses. Shape (N, response_len).
        violating_log_probs: Per-token log-probabilities for violating
            (fact-contradicting) responses. Shape (N, response_len).
        violating_mask: Binary mask indicating valid tokens in violating
            responses. Shape (N, response_len).
        alpha: Balance parameter between compliant and violating terms.
            Used in both loss variants. Default 0.5.
        loss_type: 'unbounded' (original) or 'logsigmoid' (bounded).
        beta: Temperature for logsigmoid. Higher = saturates faster.
            Only used for 'logsigmoid' loss_type. Default 1.0.

    Returns:
        loss: Scalar tensor.
        metrics: Dict with diagnostic metrics.
    """
    num_pairs = compliant_log_probs.shape[0]

    # Per-sequence normalized log-prob: sum(lp * mask) / sum(mask)
    compliant_lengths = compliant_mask.sum(dim=-1).clamp(min=1.0)
    violating_lengths = violating_mask.sum(dim=-1).clamp(min=1.0)

    compliant_norm_lp = (compliant_log_probs * compliant_mask).sum(dim=-1) / compliant_lengths
    violating_norm_lp = (violating_log_probs * violating_mask).sum(dim=-1) / violating_lengths

    # Per-pair logprob gap (length-invariant, always raw)
    gap = compliant_norm_lp - violating_norm_lp  # (N,)

    if loss_type == "logsigmoid":
        weighted_gap = alpha * compliant_norm_lp - (1.0 - alpha) * violating_norm_lp
        loss = -F.logsigmoid(beta * weighted_gap).mean()
    elif loss_type == "unbounded":
        R = alpha * compliant_norm_lp - (1.0 - alpha) * violating_norm_lp
        loss = -R.mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}. Expected 'unbounded' or 'logsigmoid'.")

    metrics = {
        "opear/loss": loss.detach().item(),
        "opear/compliant_logprob": compliant_norm_lp.mean().detach().item(),
        "opear/violating_logprob": violating_norm_lp.mean().detach().item(),
        "opear/logprob_gap": gap.mean().detach().item(),
        "opear/gap_std": gap.std().detach().item() if num_pairs > 1 else 0.0,
        "opear/gap_min": gap.min().detach().item(),
        "opear/gap_max": gap.max().detach().item(),
        "opear/num_pairs": num_pairs,
        "opear/compliant_length": compliant_lengths.mean().detach().item(),
        "opear/violating_length": violating_lengths.mean().detach().item(),
    }

    return loss, metrics
