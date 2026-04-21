# O-PEaR Violating Quality Improvements

## Problem

The O-PEaR contrastive loss saturates early in training (logprob gap reaches 50-60 by step ~30), making the regularization term effectively a no-op for the remainder of training. Diagnosis reveals two root causes:

1. **Selection is reward-blind**: Random selection includes successful trajectories where the agent already took correct actions. The guide can't produce meaningful violations for these — compliant and violating rewrites are near-identical (6/7 turns had identical actions in our diagnostic).

2. **The violating prompt is too vague**: "Sound plausible but contradict the facts" gives the guide no concrete strategies. It defaults to mild paraphrases with the same actions.

## Changes

### 1. Failed-Only Selection

**Current**: `select_batch_positions()` picks `floor(selection_ratio * group_size)` random rollouts per group, regardless of reward.

**New**: Select all rollouts with reward < 10 (non-winning trajectories). `selection_ratio` becomes an optional cap:
- `selection_ratio: 1` (default) — use all failures, no cap
- `selection_ratio: 0.5` — cap at `floor(0.5 * group_size)` per group, randomly sampled from failures
- If fewer failures exist than the cap, use all of them

**Rationale**:
- Failed trajectories contain real decision points where the agent diverged from the correct path. The guide can produce genuinely different compliant (correct path) vs violating (different wrong path) rewrites.
- Natural curriculum decay: as training improves, fewer rollouts fail, so less O-PEaR is applied. This provides automatic gap-adaptive behavior without hyperparameter tuning.
- Successful trajectories are redundant — compliant rewrites are paraphrases of what the model already did.

**File**: `data.py::select_batch_positions()` — add reward tensor parameter, filter by reward < 10 before selection.
**File**: `extensions.py::_generate_contrastive_data()` — pass reward data from batch to selection function.

### 2. Violating Prompt Rewrite

**Current prompt issues**:
- "Should lead away from the correct solution" — too vague
- No requirement to pick different actions
- No concrete violation strategies

**New `VIOLATING_SYSTEM` prompt** — three hard constraints:

**a) Mandate different actions**: Every rewritten turn MUST pick a different action from the correct (fact-consistent) one. The prompt explicitly states: "You MUST NOT choose the same action as the correct one."

**b) Concrete violation strategies**: Provide a menu of violation types:
- Wrong object (take potato instead of lettuce)
- Wrong receptacle (go to cabinet instead of fridge)
- Wrong state assumption (reason as if an object is dirty when facts say it's clean)
- Wrong location belief (reason as if an object is in a drawer when facts say it's in the fridge)
- Premature action (try to use/place before acquiring the needed object)

**c) Reasoning must justify the wrong action**: The `<think>` tag must construct a specific false belief that logically motivates the incorrect action. Not just "I'll check here" but "The lettuce is probably in the cabinet since fresh vegetables are usually stored there" (when facts say fridge).

**File**: `prompts.py::VIOLATING_SYSTEM` — full rewrite.

### 3. Compliant Prompt — Minor Tightening

Add one instruction: "If the original action is already consistent with the privileged facts, you may keep the same action but ensure your reasoning explicitly references observable evidence that supports it."

No structural changes. Failed trajectories will naturally produce more divergent compliant rewrites since the originals were wrong.

**File**: `prompts.py::COMPLIANT_SYSTEM` — one line addition.

### 4. Config Changes

Set `selection_ratio: 1` in all config YAMLs (use all failures).

**Files**: `configs/opear_logsigmoid_lambda1.yaml`, `configs/opear_logsigmoid_lambda01.yaml`, `configs/opear_test.yaml`

## Files Changed

| File | Change | Scope |
|---|---|---|
| `opear/data.py` | `select_batch_positions()` — reward-based selection with optional cap | Medium |
| `opear/prompts.py` | Rewrite `VIOLATING_SYSTEM`, minor tweak to `COMPLIANT_SYSTEM` | Medium |
| `opear/extensions.py` | Pass reward data to selection function | Small |
| `configs/*.yaml` | Set `selection_ratio: 1` | Trivial |

No changes to: `guide.py`, `actor_hook.py`, `loss.py`, verl patches, `main_opear.py`, `run_alfworld_opear.py`.

## Expected Impact

- Contrastive pairs will have genuinely different actions and reasoning between compliant and violating versions
- Gap saturation should be slower and plateau at a lower value, keeping O-PEaR gradients meaningful for longer
- Natural curriculum: O-PEaR signal intensity decreases as the agent improves, avoiding the representation damage seen in the lambda=1.0 run
- Reduced API cost in later training (fewer failures = fewer guide calls)

## Validation

1. Re-run the diagnostic script (`scripts/diagnose_violating.py`) with the new prompt to verify action divergence
2. Short training run (10 steps) to verify gap grows more slowly than before
3. Monitor `opear/logprob_gap` — should stay below 10 for the first 30+ steps (vs shooting to 50+ previously)
