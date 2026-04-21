# O-PEaR Research Log

Chronological record of experiments, hypotheses, findings, and bugs.

---

## 2026-04-17 — Exp 1 & 2: Initial O-PEaR runs (logsigmoid, no margin)

### Setup
- Model: Qwen3-4B, ALFWorld, Dr. GRPO + O-PEaR
- Loss: `L = -logsigmoid(beta * gap)` where `gap = c - v`
- beta=0.1, no margin, random selection (not failed-only)
- Guide model: GPT-5.4-nano

### Exp 1: lambda=0.1
- Gap saturated to ~58 by step 30 (gradient multiplier 0.003 — effectively zero)
- O-PEaR became a no-op; training was pure GRPO after step 30
- Val acc 6.49 during training, but eval only 14.2% (vs 68.7% baseline)
- **Hypothesis**: Early gradient dominance damaged representations before saturation

### Exp 2: lambda=1.0
- O-PEaR gradients dominated GRPO by 2-5x throughout training
- Val acc peaked at 4.03 then declined; killed at step 120
- **Finding**: High lambda causes representation damage from contrastive signal

### Key insight
The logsigmoid loss saturates when `beta * gap` is large. With beta=0.1 and no margin, the gap quickly grew to 50-60, making gradients vanishingly small.

---

## 2026-04-19 — Exp 3 & 4: Margin + failed-only selection

### Changes
- Added margin=3 to prevent saturation: `L = -logsigmoid(beta * gap - margin)`
- Failed-only selection: only rollouts with reward < 10 get contrastive pairs
- Lambda sweep: 0.1 (Exp 3) and 0.3 (Exp 4)
- beta=0.1

### Results (killed at step ~35)
- Margin prevented gap saturation (gap 1→12 vs 50-60 before)
- Failed-only selection created natural curriculum decay
- Val reward oscillated without clear upward trend (Exp 3: 2.98-3.43, Exp 4: 2.83-3.73)
- Gradient ratio grew to 8-14x (O-PEaR dominating GRPO) by step 34

### Expert review findings
- Observation-action coherence problem in violating rewrites
- Growing gradient ratio (O-PEaR dominating GRPO)
- No reference model anchoring → unbounded gap drift

---

## 2026-04-19 — MAJOR BUG: Empty privileged facts

### Discovery
The guide model was receiving **empty privileged facts** for ALL O-PEaR experiments (Exp 1-4). `alfworld_interaction.py` never requested `facts=True` from TextWorld's EnvInfos.

### Root cause
- `textworld.EnvInfos` was created with `won=True, admissible_commands=True` but NOT `facts=True`
- The `facts_str` field was never in `batch.non_tensor_batch`
- `data.py:161` defaulted to `facts=""` when `facts_str` wasn't found
- Guide model received `"Privileged facts:\n\n"` (empty) for every trajectory
- All contrastive pairs were meaningless noise

### Impact
- **Exp 1-4 results are all invalid** — guide was operating blind
- Explains: unbounded gap growth (model learning spurious rewrite-style differences), val oscillation (noise gradients), underperformance vs baseline
- Baseline Dr. GRPO was unaffected (doesn't use O-PEaR)

### Fix
- Added `facts=True` to EnvInfos in `alfworld_interaction.py`
- Added `serialize_facts()` to filter to actionable predicates
- Added facts extraction on reset and step
- Added `facts_str` in `generate_response()` return value
- Added Patch 4 to propagate `facts_str` from interaction to `extra_fields`
- Added empty facts warning in `guide.py` as regression guard

### Why tests didn't catch it
Tests manually injected `facts_str` into mock batches, bypassing the actual verl pipeline.

---

## 2026-04-20 — Exp 5 & 6: Selection ratio sweep (with facts fix)

### Setup
- First runs with actual PDDL facts flowing to guide model
- Margin formula: `beta * gap - margin` (still had the formula bug, see below)
- beta=0.1, margin=3, lambda=0.1
- Exp 5: selection_ratio=1.0 (all failed), Exp 6: selection_ratio=0.5

### Results (killed at steps 12/15)
- Guide model generating 100% valid pairs — facts fix confirmed working
- Gradient ratio balanced (~1:1) — much healthier than Exp 3-4
- **Problem**: Both compliant AND violating logprobs drifting down together
  - Exp 5: compliant -7.98 → -9.87, violating -8.98 → -11.02
  - Exp 6: compliant -7.93 → -10.07, violating -8.88 → -11.86
- Gap widening mostly via violating suppression, not compliant improvement
- Val reward: Exp 5 peaked at 3.43, Exp 6 at 3.50

### Hypothesis
The model found the "cheap path": suppress violating logprobs (push already-unlikely sequences lower) rather than boost compliant (requires learning better actions). The loss only cares about the gap, not absolute values.

---

## 2026-04-20 — BUG: Margin formula error

### Discovery
The margin formula `beta * gap - margin` places the saturation point at `gap = margin/beta`.

With beta=0.1, margin=3: saturation at gap=30 (not gap=3 as intended).

```python
# Bug: saturation at gap = margin/beta = 30
loss = -F.logsigmoid(beta * gap - margin)

# Fix: saturation at gap = margin = 3
loss = -F.logsigmoid(beta * (gap - margin))
```

At the observed gap of ~1.8: old formula gave gradient 0.94 (near-maximum), new formula gives gradient 0.77 (moderate). The old formula kept pushing hard even when the gap was already healthy.

### Fix
Changed `beta * gap - margin` to `beta * (gap - margin)` in `loss.py`.

---

## 2026-04-20 — Exp 7 & 8: Fixed margin formula

### Setup
- Fixed formula: `beta * (gap - margin)`
- Increased beta from 0.1 to 1.0 (needed for meaningful gradient dynamics with corrected formula)
- margin=3, lambda=0.1
- Exp 7: sr=1.0 (GPUs 4-7), Exp 8: sr=0.5 (GPUs 0-3)

### Results (crashed at step 30 — disk full)
- **Early phase (steps 1-8)**: Gap stable around 0.8-1.2, compliant logprobs improving
  - Exp 7 compliant: -7.98 → -8.06 (stable/improving)
  - This was the first time compliant trended UP — the fixed formula was working
- **Mid phase (steps 9-15)**: Gap grew to 1.6-2.3, both logprobs drifting negative
- **Late phase (steps 16-29)**: Gap past margin, O-PEaR self-regulating (gradient weak)
  - Compliant collapsed: -12 → -19
  - Violating collapsed: -14 → -25
  - Gap: 4-6 (past margin=3, O-PEaR effectively off)
- **Val reward**: Peaked at 3.51 (Exp 7, step 20), best O-PEaR result to date
- **Crashed at step 30**: `/workspace` disk full (2TB, 100%)

### Key finding
The fixed margin formula correctly self-regulated the gap (didn't explode to 50+), but the underlying entropy collapse (both logprobs going negative) persisted. The loss has no anchor to absolute logprob values.

---

## 2026-04-20 — BUG: Disk full

### Issue
`/workspace` (2TB) was 100% full from accumulated checkpoints across all experiments (~961GB).

### Fix
- Moved old checkpoints to `/mnt/nvme3n1/opear_checkpoints/archive/` (6.6TB free)
- Changed `run_alfworld_opear.py` to save checkpoints to `/mnt/nvme3n1/opear_checkpoints/`
- Increased save frequency from every 30 steps to every 10 steps

---

## 2026-04-21 — Three-way ranking loss design

### Problem analysis
The contrastive loss `L_cv = -logsigmoid(beta * (c - v - margin))` only depends on the gap (c - v), not the absolute values. GRPO sharpens the policy, making all off-policy sequences (both compliant and violating) increasingly unlikely. The gap self-regulation works (doesn't explode past margin), but the absolute logprob drift is the deeper problem.

### Solution: Policy anchor term
Add L_cp to anchor compliant logprobs above the policy's own rollout logprobs:

```
L_cv = -logsigmoid(beta * (c - v - margin_cv))    # compliant > violating (margin=2)
L_cp = -logsigmoid(beta * (c - p))                 # compliant > policy (no margin)
L_opear = L_cv + L_cp
```

- `p` = per-token-mean logprob of policy's own rollout (detached, no gradient)
- Both terms push compliant UP → prevents collapse
- Policy logprob acts as natural anchor (stays reasonable since it's the model's own output)

### Design decisions
- **margin_cv = 2** (reduced from 3): Based on training data, productive learning happened at gap 0.8-2.0. margin=3 kept gradients too strong past the productive region.
- **margin_cp = 0**: L_cp is an anchor, not a ranking target. GRPO handles pushing the policy up independently.
- **Same beta=1.0** for both terms: No new hyperparameters.

### Expert review (RL + ML)
Both experts validated the approach and independently recommended:
1. margin_cp = 0 (not 3) — demanding c > p + 3 is unreasonably aggressive for off-policy content
2. Consider IPO (quadratic) for L_cv as future improvement
3. Set entropy_coeff > 0 to counteract mode collapse
4. InfoNCE as principled alternative

### Colleague feedback
"The policy should be pushed up by the GRPO loss in any case. Remove the margin [from L_cp] and make the other one [L_cv margin] smaller."

---

## 2026-04-21 — Implementation and launch

### Implementation
- `loss.py`: Added `policy_mean_lp` parameter, compute L_cp, return combined loss + new metrics
- `data.py`: Return `batch_positions` from `tokenize_contrastive_responses`
- `extensions.py`: Pre-compute policy logprobs on full batch (actor only sees sharded subset)
- `actor_hook.py`: Read pre-computed policy logprobs from `meta_info`, pass to loss

### BUG: Batch sharding
Initial implementation tried to index `data.batch["old_log_probs"]` with global batch positions inside `actor_hook.py`, but the actor only sees a sharded subset (128/4 = 32 rows). Index 38 → IndexError.

**Fix**: Pre-compute policy logprobs in `extensions.py` (full batch access) and pass via `meta_info`.

### Exp 7b & 8b: Three-way loss runs (IN PROGRESS)
- beta=1.0, margin_cv=2, margin_cp=0, lambda=0.1
- Exp 7b: sr=1.0 (GPUs 4-7), Exp 8b: sr=0.5 (GPUs 0-3)
- Checkpoints: `/mnt/nvme3n1/opear_checkpoints/`, save every 10 steps
- Monitoring: cron every 30 min
- **Key metrics to watch**: `opear/policy_logprob` (should stay stable), `opear/cp_gap` (should start negative, approach 0), `opear/compliant_logprob` (should NOT collapse to -20)

---

## Bug Tracker

| Bug | Impact | Fix | Commit |
|-----|--------|-----|--------|
| Empty PDDL facts in interaction | Exp 1-4 all invalid | Added `facts=True` to EnvInfos | `c267d29` |
| Margin formula `beta*gap-margin` | Saturation at gap=30 not gap=3 | Changed to `beta*(gap-margin)` | `347072f` |
| `/workspace` disk full | Both runs crashed at step 30 | Moved checkpoints to `/mnt/nvme3n1` | `953b8a0` |
| Batch sharding in actor_hook | IndexError on global batch positions | Pre-compute in extensions.py | `5a7a94f` |

---

## Hyperparameter Evolution

| Param | Exp 1-2 | Exp 3-4 | Exp 5-6 | Exp 7-8 | Exp 7b-8b |
|-------|---------|---------|---------|---------|-----------|
| beta | 0.1 | 0.1 | 0.1 | 1.0 | 1.0 |
| margin | 0 | 3 | 3 | 3 | 2 (cv), 0 (cp) |
| lambda | 0.1/1.0 | 0.1/0.3 | 0.1 | 0.1 | 0.1 |
| selection | random | failed-only | failed-only | failed-only | failed-only |
| formula | `beta*gap` | `beta*gap-m` | `beta*gap-m` | `beta*(gap-m)` | `beta*(gap-m)` + L_cp |
| facts | BROKEN | BROKEN | fixed | fixed | fixed |
| loss terms | L_cv | L_cv | L_cv | L_cv | L_cv + L_cp |
