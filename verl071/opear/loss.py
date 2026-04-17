"""O-PEaR (Off-Policy Environment-aware Regularization) loss computation.

The O-PEaR regularizer encourages the policy to assign higher probability to
compliant responses (consistent with environment facts) and lower probability
to violating responses (contradicting facts).

Math:
    R(theta) = alpha * E[log p(z_c | c)] - (1 - alpha) * E[log p(z_v | c)]
    L_OPEaR  = -mean(R)

Minimizing L_OPEaR maximizes compliant log-prob and minimizes violating log-prob.
"""

from __future__ import annotations

import torch


def compute_opear_loss(
    compliant_log_probs: torch.Tensor,  # (N, response_len)
    compliant_mask: torch.Tensor,       # (N, response_len)
    violating_log_probs: torch.Tensor,  # (N, response_len)
    violating_mask: torch.Tensor,       # (N, response_len)
    alpha: float = 0.5,
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
            Default 0.5 (equal weight).

    Returns:
        loss: Scalar tensor. Minimizing this maximizes compliant probability
            and minimizes violating probability.
        metrics: Dict with keys:
            - opear/loss: the scalar loss value (detached)
            - opear/compliant_logprob: mean normalized compliant log-prob
            - opear/violating_logprob: mean normalized violating log-prob
            - opear/R_mean: mean of the per-sample regularizer R
            - opear/num_pairs: number of contrastive pairs (N)
    """
    # Number of contrastive pairs
    num_pairs = compliant_log_probs.shape[0]

    # Per-sequence normalized log-prob: sum(lp * mask) / sum(mask)
    # Clamp denominator to 1.0 minimum to avoid division by zero
    compliant_lengths = compliant_mask.sum(dim=-1).clamp(min=1.0)
    violating_lengths = violating_mask.sum(dim=-1).clamp(min=1.0)

    compliant_norm_lp = (compliant_log_probs * compliant_mask).sum(dim=-1) / compliant_lengths
    violating_norm_lp = (violating_log_probs * violating_mask).sum(dim=-1) / violating_lengths

    # R(theta) = alpha * E[log p(z_c | c)] - (1 - alpha) * E[log p(z_v | c)]
    R = alpha * compliant_norm_lp - (1.0 - alpha) * violating_norm_lp

    # L_OPEaR = -mean(R)
    R_mean = R.mean()
    loss = -R_mean

    metrics = {
        "opear/loss": loss.detach().item(),
        "opear/compliant_logprob": compliant_norm_lp.mean().detach().item(),
        "opear/violating_logprob": violating_norm_lp.mean().detach().item(),
        "opear/R_mean": R_mean.detach().item(),
        "opear/num_pairs": num_pairs,
    }

    return loss, metrics
