# Three-Way Ranking Loss for O-PEaR

## Problem

The current O-PEaR contrastive loss `L = -logsigmoid(beta * (c - v - margin))` only enforces a relative ranking between compliant and violating logprobs. Both logprobs collapse together over training (compliant: -8 → -19, violating: -9 → -25) because the loss has no anchor to absolute values. The model satisfies the objective by making all off-policy sequences less likely, with violating falling faster. Val reward degrades despite the gap widening.

## Solution

Add a second contrastive term that anchors compliant logprobs above the policy's own rollout logprobs. Since we only select failed rollouts, the compliant rewrite (fact-consistent) should be more probable than the model's failed attempt. GRPO handles pushing the policy up; O-PEaR provides a gentle contrastive nudge.

## Loss

```
L_cv = -logsigmoid(beta * (c - v - margin_cv))    # compliant > violating
L_cp = -logsigmoid(beta * (c - p))                 # compliant > policy (no margin)
L_opear = L_cv + L_cp
```

Where:
- `c` = per-token-mean logprob of compliant sequence (guide's fact-consistent rewrite, forward pass through current model)
- `v` = per-token-mean logprob of violating sequence (guide's fact-contradicting rewrite, forward pass through current model)
- `p` = per-token-mean logprob of policy's own rollout (from `data.batch["old_log_probs"]`, detached)
- `beta` = 1.0 (sharpness of transition around margin)
- `margin_cv` = 2 (target gap for compliant vs violating)

### Why margin_cv = 2, margin_cp = 0

**margin_cv = 2** (reduced from 3): Based on observed training dynamics, the natural starting gap is ~0.8-1.0. The productive learning window was gap 0.8-2.0 (steps 1-15). With margin=3, gradients stayed strong past the productive region and the gap grew to 6+ before backing off. margin=2 keeps gradients active during the productive phase and self-regulates before the destructive phase (3+).

**margin_cp = 0**: L_cp is an anchor, not a ranking target. Its job is to prevent compliant logprobs from collapsing, not to push them above the policy. The policy's own logprob p is naturally higher than c (off-policy content is less likely). margin=0 says "compliant should be at least as likely as the policy rollout" — the right anti-collapse semantic. GRPO independently handles improving the policy's own rollouts.

### Gradient dynamics at margin_cv = 2

```
gap=0.8 (start):  arg = 1*(0.8-2) = -1.2  -> gradient ~ 0.77 (strong)
gap=1.5 (mid):    arg = 1*(1.5-2) = -0.5  -> gradient ~ 0.62 (moderate)
gap=2.0 (target): arg = 0                  -> gradient = 0.50 (half)
gap=3.0:          arg = 1.0                -> gradient ~ 0.27 (backing off)
```

## Gradient analysis

Since `p` is detached (no gradient), only `c` and `v` receive gradients:

- **L_cv gradient on c**: pushes c UP (away from v)
- **L_cv gradient on v**: pushes v DOWN (away from c)
- **L_cp gradient on c**: pushes c UP (toward p)
- **L_cp gradient on p**: none (detached)

Both terms push compliant logprob UP. This prevents the collapse observed in previous runs because:
- If c drifts to -20 but p is at -8, L_cp produces a massive gradient to pull c back up
- The policy logprob p acts as a natural anchor — it stays reasonable because it comes from the model's own generation distribution

## Data flow

1. `c` and `v`: computed via micro-batched forward pass in `actor_hook.py` (existing)
2. `p`: extracted from `data.batch["old_log_probs"]` and `data.batch["response_mask"]` (already in the batch)
3. Mapping: `batch_positions` list from `data.py` maps each contrastive pair back to its source rollout index, enabling extraction of the correct `p` for each pair

## Files changed

### `loss.py`
- Add `policy_mean_lp` parameter (1-D tensor, one value per pair, detached)
- Compute `L_cp = -logsigmoid(beta * (c_mean_lp - policy_mean_lp)).mean()` (no margin)
- Return combined loss `L_cv + L_cp`
- Add metrics: `opear/policy_logprob`, `opear/cp_gap`, `opear/cp_loss`

### `actor_hook.py`
- Read `batch_positions` from `data.meta_info["opear_batch_positions"]`
- Extract `old_log_probs` and `response_mask` for matched positions
- Compute per-token-mean policy logprob for each pair (detached)
- Pass to `compute_opear_loss()` as `policy_mean_lp`

### `extensions.py`
- Store `batch_positions` in `batch.meta_info["opear_batch_positions"]` so actor_hook can map pairs back to rollout positions

## Config

- `opear_margin` renamed to `opear_margin_cv` (default 2) — controls L_cv target gap
- No new hyperparameters for L_cp (margin is hardcoded to 0)

## Metrics

| Metric | Description |
|--------|-------------|
| `opear/policy_logprob` | Mean per-token logprob of policy's own rollout (anchor) |
| `opear/cp_gap` | Mean gap between compliant and policy (c - p) |
| `opear/cp_loss` | L_cp term value |
| `opear/cv_loss` | L_cv term value |
| `opear/loss` | Total L_cv + L_cp |

## Expected behavior

- Compliant logprob should stabilize or trend upward (anchored by policy logprob)
- Gap c-v should grow toward margin_cv=2, then self-regulate
- Gap c-p should start negative (off-policy content is less likely) and gradually approach 0 as the model learns to assign higher probability to compliant rewrites
- Val reward should improve as the model actually learns fact-consistent behavior rather than just suppressing violating sequences
