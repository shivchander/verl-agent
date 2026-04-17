# O-PEaR Reimplementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fragile runtime monkey-patching of verl's actor with a static verl patch + trainer subclass, giving combined GRPO + O-PEaR gradients in a single optimizer step.

**Architecture:** The trainer side (contrastive data generation) stays as monkey-patching inside the TaskRunner process — this works reliably. The actor side (O-PEaR loss computation) moves from runtime monkey-patching to a static verl patch that injects a single function call into `dp_actor.py:update_policy`. The O-PEaR logic lives in our own `verl071/opear/actor_hook.py`, keeping the patch minimal and the logic maintainable.

**Tech Stack:** verl 0.7.1, PyTorch, OpenAI async client, nest_asyncio

---

### Task 1: Create `actor_hook.py` — O-PEaR gradient accumulation

**Files:**
- Create: `verl071/opear/actor_hook.py`

- [ ] **Step 1: Create actor_hook.py**

This function is called from inside `dp_actor.py:update_policy` (via the verl patch in Task 2). It runs after GRPO micro-batch backward passes but before `optimizer.step()`, so O-PEaR gradients accumulate with GRPO gradients.

```python
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
    temperature = data.meta_info.get("temperature", 1.0)
    device = next(actor.actor_module.parameters()).device

    c_ids = opear_data["compliant_input_ids"].to(device)
    c_attn = opear_data["compliant_attention_mask"].to(device)
    c_mask = opear_data["compliant_response_mask"].to(device)
    v_ids = opear_data["violating_input_ids"].to(device)
    v_attn = opear_data["violating_attention_mask"].to(device)
    v_mask = opear_data["violating_response_mask"].to(device)

    resp_len = c_mask.shape[-1]

    _, c_lp = actor._forward_micro_batch(
        micro_batch={"input_ids": c_ids, "attention_mask": c_attn,
                     "position_ids": compute_position_id_with_mask(c_attn),
                     "responses": c_ids[:, -resp_len:]},
        temperature=temperature, calculate_entropy=False)

    _, v_lp = actor._forward_micro_batch(
        micro_batch={"input_ids": v_ids, "attention_mask": v_attn,
                     "position_ids": compute_position_id_with_mask(v_attn),
                     "responses": v_ids[:, -resp_len:]},
        temperature=temperature, calculate_entropy=False)

    c_lp = c_lp[:, :resp_len]
    v_lp = v_lp[:, :resp_len]
    c_rm = c_mask[:, :c_lp.shape[-1]]
    v_rm = v_mask[:, :v_lp.shape[-1]]

    loss, opear_metrics = compute_opear_loss(c_lp, c_rm, v_lp, v_rm, alpha=alpha)
    scaled = lam * loss
    scaled.backward()

    opear_metrics["opear/scaled_loss"] = scaled.detach().item()
    opear_metrics["opear/lambda"] = lam
    opear_metrics["opear/alpha"] = alpha

    for k, v in opear_metrics.items():
        metrics[k] = metrics.get(k, 0.0) + v
```

- [ ] **Step 2: Commit**

```bash
git add verl071/opear/actor_hook.py
git commit -m "feat(opear): add actor_hook for combined gradient accumulation"
```

---

### Task 2: Add O-PEaR patch to `apply_verl_patches.py`

**Files:**
- Modify: `verl071/patches/apply_verl_patches.py`

The patch adds two things to `dp_actor.py`:
1. A try/except import of `opear_accumulate_gradients` at the top of the file
2. A function call between the GRPO micro-batch loop and `_optimizer_step()`

- [ ] **Step 1: Add patch 3 to apply_verl_patches.py**

Add the following after the existing `success = success and ok` block (before the final print):

```python
    dp_actor_path = os.path.join(
        verl_path, "workers", "actor", "dp_actor.py"
    )

    print("Patch 3a: Add O-PEaR import to dp_actor.py")
    ok = patch_file(
        dp_actor_path,
        'from verl.utils.torch_functional import logprobs_from_logits',
        'from verl.utils.torch_functional import logprobs_from_logits\n'
        '\n'
        'try:\n'
        '    from verl071.opear.actor_hook import opear_accumulate_gradients\n'
        'except ImportError:\n'
        '    opear_accumulate_gradients = None',
        "dp_actor.py: add O-PEaR import",
    )
    success = success and ok

    print("Patch 3b: Call O-PEaR gradient accumulation before optimizer step")
    ok = patch_file(
        dp_actor_path,
        '                grad_norm = self._optimizer_step()',
        '                # O-PEaR: accumulate contrastive loss gradients with GRPO gradients\n'
        '                if opear_accumulate_gradients is not None:\n'
        '                    opear_accumulate_gradients(self, data, metrics)\n'
        '                grad_norm = self._optimizer_step()',
        "dp_actor.py: call O-PEaR before optimizer step",
    )
    success = success and ok
```

- [ ] **Step 2: Run the patch script to verify it applies cleanly**

```bash
source verl071/.venv/bin/activate
python verl071/patches/apply_verl_patches.py
```

Expected output should show `OK` for all patches (or `SKIP` if already applied).

- [ ] **Step 3: Verify the patched dp_actor.py**

```bash
grep -A3 "opear_accumulate_gradients" verl071/.venv/lib/python3.12/site-packages/verl/workers/actor/dp_actor.py
```

Expected: shows the import and the call site.

- [ ] **Step 4: Commit**

```bash
git add verl071/patches/apply_verl_patches.py
git commit -m "feat(opear): add verl patch for combined O-PEaR + GRPO gradients"
```

---

### Task 3: Simplify `extensions.py` — remove actor patching

**Files:**
- Modify: `verl071/opear/extensions.py`

Remove `_extend_actor()` and `_opear_step()` — the actor-side logic now lives in the verl patch + `actor_hook.py`. Keep `_extend_trainer()` which patches `RayPPOTrainer.__init__` and `_update_actor` (this works fine in the TaskRunner process).

- [ ] **Step 1: Rewrite extensions.py**

Replace the entire file with:

```python
"""O-PEaR extensions for verl's RayPPOTrainer.

Extends the trainer via method wrapping to inject contrastive data generation
into the training loop. The actor-side O-PEaR loss is handled by a static
verl patch (see patches/apply_verl_patches.py), not by runtime patching.

Call apply() inside the Ray worker process (e.g. in OPEaRTaskRunner.run())
before RayPPOTrainer is instantiated.
"""
import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def apply():
    """Patch RayPPOTrainer to enable O-PEaR contrastive data generation.

    Safe to call multiple times (no-op after first).
    """
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    _extend_trainer()
    logger.info("O-PEaR trainer extensions applied")


def _extend_trainer():
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    _orig_init = RayPPOTrainer.__init__
    _orig_update_actor = RayPPOTrainer._update_actor

    def __init__(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.opear_enabled = self.config.algorithm.get("opear", {}).get("enable", False)
        if self.opear_enabled:
            from verl071.opear.guide import OPEaRGuide
            cfg = self.config.algorithm.opear
            self.opear_guide = OPEaRGuide(
                model=cfg.get("guide_model", "gpt-5.4-nano"),
                beta=cfg.get("beta", 0.5),
            )
            self.opear_lambda = cfg.get("lambda_coef", 0.5)
            self.opear_alpha = cfg.get("alpha", 0.5)
            print(f"[O-PEaR] enabled: lambda={self.opear_lambda}, "
                  f"alpha={self.opear_alpha}, beta={cfg.get('beta', 0.5)}")

    def _update_actor(self, batch):
        if getattr(self, "opear_enabled", False):
            try:
                _generate_contrastive_data(self, batch)
            except Exception as e:
                print(f"[O-PEaR] contrastive generation failed: {e}")
                import traceback
                traceback.print_exc()
        return _orig_update_actor(self, batch)

    RayPPOTrainer.__init__ = __init__
    RayPPOTrainer._update_actor = _update_actor


def _generate_contrastive_data(trainer, batch):
    """Generate contrastive pairs and attach to batch.meta_info."""
    from verl071.opear.data import reconstruct_trajectories, tokenize_contrastive_responses

    trajectories = reconstruct_trajectories(batch, trainer.tokenizer)
    if not trajectories:
        print("[O-PEaR] no trajectories reconstructed, skipping")
        return

    traj_uids = [t["traj_uid"] for t in trajectories]
    selected_uids = trainer.opear_guide.select_rollouts(
        traj_uids,
        trainer.config.actor_rollout_ref.rollout.n,
    )
    selected = [t for t in trajectories if t["traj_uid"] in set(selected_uids)]
    if not selected:
        print("[O-PEaR] no trajectories selected, skipping")
        return

    pairs = trainer.opear_guide.generate_contrastive_batch(selected)
    opear_data = tokenize_contrastive_responses(
        selected, pairs, batch, trainer.tokenizer,
        max_response_length=batch.batch["responses"].shape[-1],
    )
    if opear_data is not None:
        batch.meta_info["opear_data"] = opear_data
        batch.meta_info["opear_alpha"] = trainer.opear_alpha
        batch.meta_info["opear_lambda"] = trainer.opear_lambda
        n = sum(1 for p in pairs if p is not None)
        print(f"[O-PEaR] {n}/{len(selected)} valid contrastive pairs")
    else:
        print("[O-PEaR] no valid contrastive pairs")
```

- [ ] **Step 2: Commit**

```bash
git add verl071/opear/extensions.py
git commit -m "refactor(opear): remove actor monkey-patching, use verl patch instead"
```

---

### Task 4: Add nest_asyncio to guide.py

**Files:**
- Modify: `verl071/opear/guide.py`

- [ ] **Step 1: Add nest_asyncio.apply() at module level**

At the top of `guide.py`, after the existing imports (after `from verl071.opear.prompts import ...`), add:

```python
import nest_asyncio
nest_asyncio.apply()
```

This ensures `asyncio.run()` works inside Ray workers that already have a running event loop.

- [ ] **Step 2: Commit**

```bash
git add verl071/opear/guide.py
git commit -m "fix(opear): add nest_asyncio for Ray worker event loop compatibility"
```

---

### Task 5: End-to-end verification

**Files:** None (testing only)

- [ ] **Step 1: Ensure patches are applied**

```bash
source verl071/.venv/bin/activate
python verl071/patches/apply_verl_patches.py
```

- [ ] **Step 2: Verify the import works in the venv**

```bash
source verl071/.venv/bin/activate
python -c "
from verl.workers.actor.dp_actor import DataParallelPPOActor
import inspect
src = inspect.getsource(DataParallelPPOActor.update_policy)
assert 'opear_accumulate_gradients' in src, 'Patch not applied!'
print('dp_actor.py patch verified')
"
```

- [ ] **Step 3: Launch a 4-GPU run and monitor for all 3 signals**

```bash
tmux new-session -d -s opear_test -c /workspace/home/lab/shiv/verl-agent \
  "source verl071/.venv/bin/activate && set -a && source .env && set +a && \
   export CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=offline && \
   python verl071/run_alfworld_opear.py 2>&1 | tee /tmp/opear_e2e.log"
```

Monitor with:
```bash
grep "O-PEaR\|Training Progress\|opear/" /tmp/opear_e2e.log | tail -10
```

**Three signals required (all must appear after step 1 completes):**
1. `[O-PEaR] enabled: lambda=0.5, alpha=0.5, beta=0.5` — trainer init worked
2. `[O-PEaR] X/Y valid contrastive pairs` — guide model generated pairs
3. `opear/scaled_loss` in metrics — O-PEaR loss computed and applied

- [ ] **Step 4: If all 3 signals confirmed, commit and push**

```bash
git push origin opear --force-with-lease
```
