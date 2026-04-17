# OPEAR Branch Review

Reviewed `origin/opear` against our `verl-071-drgrpo` branch. The OPEAR code is cleanly isolated in `verl071/opear/` and won't break our base Dr. GRPO training. But there are a few issues that are likely causing your runtime problems.

## Rebasing on `verl-071-drgrpo`

The `verl-071-drgrpo` branch has important fixes you'll want:

- Correct GRPO grouping (all rollouts in a group play the same ALFWorld game)
- Game cycling across epochs via `global_step`
- Proper train/val split routing (eval_out_of_distribution for val)
- Dr. GRPO config (no ref model, token-level normalization)
- verl patch script for reproducibility

```bash
git fetch origin
git rebase origin/verl-071-drgrpo

# After rebase, apply the verl patches to your venv:
python verl071/patches/apply_verl_patches.py
```

The patches inject `global_step` and `rollout_n` into `interaction_kwargs` — needed for deterministic game selection. Without them, GRPO groups play different games and the advantage signal is noise.

## Critical Issues

### 1. `asyncio.run()` inside Ray worker will crash

`guide.py:generate_contrastive_batch` calls `asyncio.run()`, but Ray workers already have a running event loop. This raises:

```
RuntimeError: This event loop is already running
```

**Fix**: Use `nest_asyncio` or switch to synchronous HTTP calls:

```python
# Option A: nest_asyncio
import nest_asyncio
nest_asyncio.apply()

# Option B: detect existing loop
try:
    loop = asyncio.get_running_loop()
    result = loop.run_until_complete(self._generate_batch(...))
except RuntimeError:
    result = asyncio.run(self._generate_batch(...))

# Option C: just use sync requests (simplest)
import requests
response = requests.post(...)
```

### 2. `traj_uid` is never populated — OPEAR is silently a no-op

`data.py:reconstruct_trajectories` reads `batch.non_tensor_batch.get("traj_uid")` to group rollout tokens into per-episode trajectories. But `traj_uid` is never set in the rollout pipeline.

The rollout_loop changes only add `facts_str` to the batch, not `traj_uid`. So `reconstruct_trajectories` returns empty results and OPEAR never actually runs.

**Fix**: Either populate `traj_uid` during rollout (add it to `non_tensor_batch` in the agent loop), or use the existing `index` field to identify trajectories. With `rollout.n=8`, rollouts are interleaved as `[idx0, idx0, ..., idx0, idx1, idx1, ..., idx1, ...]`, so `index` + position is sufficient to reconstruct per-trajectory data.

## Significant Issues

### 3. Double optimizer step — LR advances 2x

In `extensions.py:_opear_step`, after the GRPO update calls `optimizer.step()`, OPEAR computes its own loss and calls `actor._optimizer_step()` again. This means:

- Two `optimizer.step()` calls per training step
- LR schedule advances twice as fast as configured
- GRPO and OPEAR gradients are never combined — they're applied sequentially

If you want alternating optimization, this is fine but the effective LR is 2x what you set. If you want a combined update, accumulate both losses before stepping:

```python
# Combined approach:
grpo_loss = original_update(...)  # compute but don't step
opear_loss = compute_opear_loss(...)
total_loss = grpo_loss + opear_loss
total_loss.backward()
optimizer.step()  # single step
```

### 4. `meta_info` may not transfer tensors across Ray

OPEAR data (tokenized compliant/violating sequences) is placed in `batch.meta_info["opear_data"]` in the trainer process, then accessed in the actor Ray worker. `meta_info` is designed for lightweight metadata, not large tensors. Depending on how verl serializes `DataProto` across Ray, the tensors may not arrive in the actor.

**Fix**: Use `batch.non_tensor_batch` for numpy arrays, or add the tensors to `batch.batch` (TensorDict) which is guaranteed to transfer.

### 5. Strict guide response parsing

`prompts.py:parse_guide_response` raises `ValueError` if the number of `[TURN N]` markers doesn't exactly match `expected_turns`. LLMs frequently produce slightly off formatting. The 3 retries in `guide.py` catch this, but you'll waste API calls.

**Suggestion**: Accept responses with fewer turns than expected (partial rewrites are still useful) and log warnings instead of raising.

### 6. Fragile internal API usage

`_forward_micro_batch` and `_optimizer_step` are internal verl methods. They work for 0.7.1 but may change in future versions. Consider adding version checks or documenting the dependency.

## What's Working

- The loss math in `loss.py` is correct
- `main_opear.py` TaskRunner subclass is well-designed (patches applied inside Ray worker)
- The env_manager changes (adding `facts_str`) are safe and additive
- The separation into `verl071/opear/` is clean — no interference with base training
