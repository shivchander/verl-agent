# Three-Way Ranking Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a policy-anchor term (L_cp) to O-PEaR's contrastive loss to prevent logprob collapse during training.

**Architecture:** Modify the loss function to accept detached policy logprobs and compute a second logsigmoid term. Propagate batch position mapping through `data.py` → `extensions.py` → `actor_hook.py` so each contrastive pair can be matched to its source rollout's logprobs.

**Tech Stack:** PyTorch, verl 0.7.1, existing O-PEaR module

**Spec:** `docs/superpowers/specs/2026-04-21-three-way-ranking-loss-design.md`

---

### Task 1: Update `loss.py` — add policy anchor term

**Files:**
- Modify: `verl071/opear/loss.py`
- Test: `verl071/tests/test_opear_loss.py`

- [ ] **Step 1: Write failing tests for the new loss**

Add to `verl071/tests/test_opear_loss.py`:

```python
class TestThreeWayRankingLoss:
    """Test the three-way ranking loss: L_cv + L_cp."""

    def test_policy_anchor_pushes_compliant_up(self):
        """L_cp gradient on compliant should be negative (pushes c up)."""
        N, L = 1, 1
        c_lp = torch.tensor([[-2.0]], requires_grad=True)
        v_lp = torch.tensor([[-4.0]])
        c_mask = torch.ones(N, L)
        v_mask = torch.ones(N, L)
        p_lp = torch.tensor([-1.0])  # policy logprob (detached)

        loss, metrics = compute_opear_loss(
            c_lp, c_mask, v_lp, v_mask,
            beta=1.0, margin=2.0, policy_mean_lp=p_lp,
        )
        loss.backward()

        # Gradient on compliant should be negative (minimize loss = increase c)
        assert c_lp.grad.item() < 0
        # Should have both cv and cp metrics
        assert "opear/cv_loss" in metrics
        assert "opear/cp_loss" in metrics
        assert "opear/policy_logprob" in metrics
        assert "opear/cp_gap" in metrics

    def test_no_policy_logprob_falls_back_to_cv_only(self):
        """Without policy_mean_lp, loss should equal L_cv only."""
        N, L = 1, 1
        c_lp = torch.tensor([[-1.0]])
        v_lp = torch.tensor([[-3.0]])
        mask = torch.ones(N, L)

        loss_with, _ = compute_opear_loss(
            c_lp, mask, v_lp, mask,
            beta=1.0, margin=2.0, policy_mean_lp=None,
        )
        loss_without, _ = compute_opear_loss(
            c_lp, mask, v_lp, mask,
            beta=1.0, margin=2.0,
        )

        assert loss_with.item() == pytest.approx(loss_without.item(), abs=1e-6)

    def test_cp_loss_no_margin(self):
        """L_cp should use no margin (margin_cp=0)."""
        N, L = 1, 1
        c_lp = torch.tensor([[-3.0]])
        v_lp = torch.tensor([[-5.0]])
        mask = torch.ones(N, L)
        p_lp = torch.tensor([-1.0])

        _, metrics = compute_opear_loss(
            c_lp, mask, v_lp, mask,
            beta=1.0, margin=2.0, policy_mean_lp=p_lp,
        )

        # cp_gap = c - p = -3 - (-1) = -2 (no margin subtracted)
        assert metrics["opear/cp_gap"] == pytest.approx(-2.0, abs=1e-5)

    def test_total_loss_is_sum_of_cv_and_cp(self):
        """Total loss should be L_cv + L_cp."""
        import torch.nn.functional as F
        N, L = 1, 1
        c_lp = torch.tensor([[-2.0]])
        v_lp = torch.tensor([[-4.0]])
        mask = torch.ones(N, L)
        p_lp = torch.tensor([-1.0])

        _, metrics = compute_opear_loss(
            c_lp, mask, v_lp, mask,
            beta=1.0, margin=2.0, policy_mean_lp=p_lp,
        )

        expected_cv = -F.logsigmoid(torch.tensor(1.0 * (2.0 - 2.0))).item()  # gap=2, margin=2
        expected_cp = -F.logsigmoid(torch.tensor(1.0 * (-2.0 - (-1.0)))).item()  # c-p = -1, no margin
        assert metrics["opear/cv_loss"] == pytest.approx(expected_cv, abs=1e-5)
        assert metrics["opear/cp_loss"] == pytest.approx(expected_cp, abs=1e-5)
        assert metrics["opear/loss"] == pytest.approx(expected_cv + expected_cp, abs=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspace/home/lab/shiv/verl-agent && python -m pytest verl071/tests/test_opear_loss.py::TestThreeWayRankingLoss -v`
Expected: FAIL — `compute_opear_loss() got an unexpected keyword argument 'policy_mean_lp'`

- [ ] **Step 3: Implement the three-way loss in `loss.py`**

Replace the entire content of `verl071/opear/loss.py` with:

```python
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
    compliant_log_probs: torch.Tensor,  # (N, response_len)
    compliant_mask: torch.Tensor,       # (N, response_len)
    violating_log_probs: torch.Tensor,  # (N, response_len)
    violating_mask: torch.Tensor,       # (N, response_len)
    beta: float = 1.0,
    margin: float = 0.0,
    policy_mean_lp: Optional[torch.Tensor] = None,  # (N,) detached
) -> tuple[torch.Tensor, dict]:
    """Compute the O-PEaR three-way ranking loss.

    Args:
        compliant_log_probs: Per-token log-probabilities for compliant
            (fact-consistent) responses. Shape (N, response_len).
        compliant_mask: Binary mask for compliant responses. Shape (N, response_len).
        violating_log_probs: Per-token log-probabilities for violating
            (fact-contradicting) responses. Shape (N, response_len).
        violating_mask: Binary mask for violating responses. Shape (N, response_len).
        beta: Sharpness of transition around margin. Default 1.0.
        margin: Target gap for L_cv (compliant vs violating). Default 0.0.
        policy_mean_lp: Per-token-mean logprob of the policy's own rollout,
            one value per pair. Detached (no gradient). Shape (N,).
            When None, L_cp is skipped (backward compat).

    Returns:
        loss: Scalar tensor (L_cv + L_cp, or L_cv only if policy_mean_lp is None).
        metrics: Dict with diagnostic metrics.
    """
    num_pairs = compliant_log_probs.shape[0]

    # Per-sequence length-normalized log-prob
    compliant_lengths = compliant_mask.sum(dim=-1).clamp(min=1.0)
    violating_lengths = violating_mask.sum(dim=-1).clamp(min=1.0)

    compliant_mean_lp = (compliant_log_probs * compliant_mask).sum(dim=-1) / compliant_lengths
    violating_mean_lp = (violating_log_probs * violating_mask).sum(dim=-1) / violating_lengths

    # L_cv: compliant > violating by margin
    cv_gap = compliant_mean_lp - violating_mean_lp  # (N,)
    cv_loss = -F.logsigmoid(beta * (cv_gap - margin)).mean()

    # L_cp: compliant > policy (no margin, anchor only)
    if policy_mean_lp is not None:
        policy_mean_lp = policy_mean_lp.detach()
        cp_gap = compliant_mean_lp - policy_mean_lp  # (N,)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /workspace/home/lab/shiv/verl-agent && python -m pytest verl071/tests/test_opear_loss.py::TestThreeWayRankingLoss -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add verl071/opear/loss.py verl071/tests/test_opear_loss.py
git commit -m "feat(opear): add three-way ranking loss with policy anchor (L_cp)"
```

---

### Task 2: Update `data.py` — return batch positions from tokenization

**Files:**
- Modify: `verl071/opear/data.py:339-390`

- [ ] **Step 1: Add batch position tracking to `tokenize_contrastive_responses`**

In `verl071/opear/data.py`, modify the `tokenize_contrastive_responses` function to collect and return `batch_positions`:

Find this block (line 336-366):
```python
    compliant_entries: list[dict] = []
    violating_entries: list[dict] = []

    for traj, pair in zip(trajectories, contrastive_pairs):
        if pair is None:
            continue

        batch_pos = traj["turn_indices"][0]
```

Replace with:
```python
    compliant_entries: list[dict] = []
    violating_entries: list[dict] = []
    batch_positions: list[int] = []

    for traj, pair in zip(trajectories, contrastive_pairs):
        if pair is None:
            continue

        batch_pos = traj["turn_indices"][0]
```

Then find the block where entries are appended (line 364-366):
```python
        if c_entry is not None and v_entry is not None:
            compliant_entries.append(c_entry)
            violating_entries.append(v_entry)
```

Replace with:
```python
        if c_entry is not None and v_entry is not None:
            compliant_entries.append(c_entry)
            violating_entries.append(v_entry)
            batch_positions.append(batch_pos)
```

Then find the return dict (line 383-390):
```python
    return {
        "compliant_input_ids": pad_and_stack(compliant_entries, "input_ids"),
        "compliant_attention_mask": pad_and_stack(compliant_entries, "attention_mask"),
        "compliant_response_mask": pad_and_stack(compliant_entries, "response_mask"),
        "violating_input_ids": pad_and_stack(violating_entries, "input_ids"),
        "violating_attention_mask": pad_and_stack(violating_entries, "attention_mask"),
        "violating_response_mask": pad_and_stack(violating_entries, "response_mask"),
    }
```

Replace with:
```python
    return {
        "compliant_input_ids": pad_and_stack(compliant_entries, "input_ids"),
        "compliant_attention_mask": pad_and_stack(compliant_entries, "attention_mask"),
        "compliant_response_mask": pad_and_stack(compliant_entries, "response_mask"),
        "violating_input_ids": pad_and_stack(violating_entries, "input_ids"),
        "violating_attention_mask": pad_and_stack(violating_entries, "attention_mask"),
        "violating_response_mask": pad_and_stack(violating_entries, "response_mask"),
        "batch_positions": batch_positions,
    }
```

- [ ] **Step 2: Commit**

```bash
git add verl071/opear/data.py
git commit -m "feat(opear): return batch_positions from tokenize_contrastive_responses"
```

---

### Task 3: Update `extensions.py` — propagate batch positions to meta_info

**Files:**
- Modify: `verl071/opear/extensions.py:105-110`

- [ ] **Step 1: Store batch_positions in meta_info**

In `verl071/opear/extensions.py`, find this block (line 105-106):
```python
    if opear_data is not None:
        batch.meta_info["opear_data"] = opear_data
```

Add one line after:
```python
    if opear_data is not None:
        batch.meta_info["opear_data"] = opear_data
        batch.meta_info["opear_batch_positions"] = opear_data.pop("batch_positions", [])
```

This pops `batch_positions` out of `opear_data` (it's a list, not a tensor — shouldn't be mixed with the tensor dict) and stores it separately in `meta_info`.

- [ ] **Step 2: Commit**

```bash
git add verl071/opear/extensions.py
git commit -m "feat(opear): propagate batch_positions through meta_info"
```

---

### Task 4: Update `actor_hook.py` — extract policy logprobs and pass to loss

**Files:**
- Modify: `verl071/opear/actor_hook.py`

- [ ] **Step 1: Add policy logprob extraction and pass to loss**

In `verl071/opear/actor_hook.py`, add after line 51 (after `v_mask = ...`):

```python
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
```

Then modify the `compute_opear_loss` call inside the micro-batch loop (line 107-109). Find:
```python
        pair_loss, pair_metrics = compute_opear_loss(
            c_lp, ci_rm, v_lp, vi_rm, beta=beta, margin=margin
        )
```

Replace with:
```python
        pair_policy_lp = policy_mean_lp[i:i+1] if policy_mean_lp is not None else None
        pair_loss, pair_metrics = compute_opear_loss(
            c_lp, ci_rm, v_lp, vi_rm, beta=beta, margin=margin,
            policy_mean_lp=pair_policy_lp,
        )
```

Then add L_cp metrics to the final metrics block. Find (line 152):
```python
    opear_metrics["opear/loss"] = total_loss / num_pairs
```

Add after:
```python
    # Propagate L_cp metrics from aggregated pair metrics
    if "opear/cp_loss" in agg_metrics:
        opear_metrics["opear/cp_loss"] = agg_metrics["opear/cp_loss"] / num_pairs
    if "opear/policy_logprob" in agg_metrics:
        opear_metrics["opear/policy_logprob"] = agg_metrics["opear/policy_logprob"] / num_pairs
    if "opear/cp_gap" in agg_metrics:
        opear_metrics["opear/cp_gap"] = agg_metrics["opear/cp_gap"] / num_pairs
    if "opear/cv_loss" in agg_metrics:
        opear_metrics["opear/cv_loss"] = agg_metrics["opear/cv_loss"] / num_pairs
```

- [ ] **Step 2: Commit**

```bash
git add verl071/opear/actor_hook.py
git commit -m "feat(opear): extract policy logprobs and pass to three-way loss"
```

---

### Task 5: Update configs and launch

**Files:**
- Modify: `verl071/configs/opear_b1m3_sr100_lambda01.yaml`
- Modify: `verl071/configs/opear_b1m3_sr050_lambda01.yaml`

- [ ] **Step 1: Update configs with margin=2**

In both `opear_b1m3_sr100_lambda01.yaml` and `opear_b1m3_sr050_lambda01.yaml`, change:
```yaml
opear_margin: 3
```
to:
```yaml
opear_margin: 2
```

- [ ] **Step 2: Run the loss tests to make sure nothing is broken**

Run: `cd /workspace/home/lab/shiv/verl-agent && python -m pytest verl071/tests/test_opear_loss.py::TestThreeWayRankingLoss -v`
Expected: All 4 tests PASS

- [ ] **Step 3: Commit and push all changes**

```bash
git add -A verl071/
git commit -m "feat(opear): three-way ranking loss with policy anchor — full implementation"
git push origin opear
```

- [ ] **Step 4: Clean GPUs and launch both runs**

```bash
# Kill any lingering processes
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null
sleep 5

# Launch Exp 7b (sr=1.0)
cd /workspace/home/lab/shiv/verl-agent && source .venv/bin/activate
nohup python verl071/run_alfworld_opear.py verl071/configs/opear_b1m3_sr100_lambda01.yaml > /tmp/opear-3way-sr100.log 2>&1 &

# Launch Exp 8b (sr=0.5)
nohup python verl071/run_alfworld_opear.py verl071/configs/opear_b1m3_sr050_lambda01.yaml > /tmp/opear-3way-sr050.log 2>&1 &
```

- [ ] **Step 5: Set up monitoring**

Set up a cron job to check training status every 30 minutes, tracking the new metrics:
- `opear/cv_loss` and `opear/cp_loss` — both terms separately
- `opear/policy_logprob` — should stay stable (anchor)
- `opear/cp_gap` — should start negative, approach 0
- `opear/compliant_logprob` — should NOT collapse to -20
