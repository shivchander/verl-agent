"""O-PEaR (Off-Policy Environment-aware Regularization) loss computation.

The O-PEaR regularizer encourages the policy to assign higher probability to
compliant responses (consistent with environment facts) and lower probability
to violating responses (contradicting facts).

Loss: L = -mean(log_sigmoid(beta * mean_gap))

Uses per-token-mean log-probs to keep the logsigmoid in its active range.
Gradient magnitude matching with GRPO is handled by the scaling in
actor_hook.py (no loss_scale_factor needed).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_opear_loss(
    compliant_log_probs: torch.Tensor,  # (N, response_len)
    compliant_mask: torch.Tensor,       # (N, response_len)
    violating_log_probs: torch.Tensor,  # (N, response_len)
    violating_mask: torch.Tensor,       # (N, response_len)
    beta: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Compute the O-PEaR regularization loss.

    Args:
        compliant_log_probs: Per-token log-probabilities for compliant
            (fact-consistent) responses. Shape (N, response_len).
        compliant_mask: Binary mask indicating valid tokens in compliant
            responses. Shape (N, response_len).
        violating_log_probs: Per-token log-probabilities for violating
            (fact-contradicting) responses. Shape (N, response_len).
        violating_mask: Binary mask indicating valid tokens in violating
            responses. Shape (N, response_len).
        beta: Temperature controlling saturation speed. Default 1.0.

    Returns:
        loss: Scalar tensor.
        metrics: Dict with diagnostic metrics.
    """
    num_pairs = compliant_log_probs.shape[0]

    # Per-sequence length-normalized log-prob
    compliant_lengths = compliant_mask.sum(dim=-1).clamp(min=1.0)
    violating_lengths = violating_mask.sum(dim=-1).clamp(min=1.0)

    compliant_mean_lp = (compliant_log_probs * compliant_mask).sum(dim=-1) / compliant_lengths
    violating_mean_lp = (violating_log_probs * violating_mask).sum(dim=-1) / violating_lengths

    # Per-token-mean gap (keeps logsigmoid in active range ~[-10, 10])
    gap = compliant_mean_lp - violating_mean_lp  # (N,)

    loss = -F.logsigmoid(beta * gap).mean()

    metrics = {
        "opear/loss": loss.detach().item(),
        "opear/compliant_logprob": compliant_mean_lp.mean().detach().item(),
        "opear/violating_logprob": violating_mean_lp.mean().detach().item(),
        "opear/logprob_gap": gap.mean().detach().item(),
        "opear/gap_std": gap.std().detach().item() if num_pairs > 1 else 0.0,
        "opear/gap_min": gap.min().detach().item(),
        "opear/gap_max": gap.max().detach().item(),
        "opear/num_pairs": num_pairs,
        "opear/compliant_length": compliant_lengths.mean().detach().item(),
        "opear/violating_length": violating_lengths.mean().detach().item(),
    }

    return loss, metrics
