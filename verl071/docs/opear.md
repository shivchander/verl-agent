# O-PEaR: Off-Policy Environment-aware Regularization

O-PEaR is a contrastive regularization method that augments reinforcement learning (Dr. GRPO) with environment-aware guidance. It uses a privileged guide model (GPT-5.4-nano) that has access to ground-truth PDDL facts from the environment to generate contrastive response pairs — one consistent with the facts (compliant), one contradicting them (violating). The policy is then trained to prefer the compliant responses, providing a learning signal that goes beyond sparse task rewards.

## Table of Contents

- [Motivation](#motivation)
- [Algorithm Overview](#algorithm-overview)
- [Architecture](#architecture)
- [Module Reference](#module-reference)
- [Data Flow](#data-flow)
- [Multi-Turn Token Structure](#multi-turn-token-structure)
- [Contrastive Sequence Rebuilding](#contrastive-sequence-rebuilding)
- [Gradient Scale Matching](#gradient-scale-matching)
- [Loss Function](#loss-function)
- [Hyperparameters](#hyperparameters)
- [WandB Metrics](#wandb-metrics)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
- [Testing](#testing)
- [Design Decisions and Lessons Learned](#design-decisions-and-lessons-learned)

---

## Motivation

Standard RL for embodied agents (like ALFWorld) relies on sparse binary rewards: the agent either completes the task or doesn't. This creates two problems:

1. **Credit assignment**: With multi-turn episodes of 10-50 actions, it's hard to determine which specific actions were good or bad.
2. **Hallucination**: The agent may develop reasoning patterns that happen to reach correct actions but are based on incorrect beliefs about the environment.

O-PEaR addresses both by introducing a dense, per-turn signal: "does this response reflect correct understanding of the environment state?" It uses a privileged guide model that can see the true environment state (PDDL facts) to generate contrastive pairs, teaching the policy to distinguish fact-consistent reasoning from plausible-but-wrong reasoning.

---

## Algorithm Overview

At each training step, O-PEaR augments the standard GRPO update:

```
1. GRPO rollouts:     Policy generates group_size rollouts per prompt
2. GRPO advantages:   Group-relative advantage estimation (no critic)
3. GRPO backward:     Standard policy gradient loss → ∇_GRPO accumulated

4. O-PEaR selection:  Pick floor(selection_ratio * group_size) rollouts per group
5. Reconstruction:    Decode selected rollouts into multi-turn trajectories
6. Guide generation:  GPT-5.4-nano rewrites each trajectory into compliant + violating versions
7. Tokenization:      Rebuild full sequences with guide rewrites (prompt + obs preserved)
8. Micro-batched forward: For each pair, compute log-probs under the policy (1 pair at a time)
9. Contrastive loss:  L_opear = -mean(log_sigmoid(beta * mean_gap))
10. O-PEaR backward:  scaled = lambda * L_opear / num_pairs → ∇_OPEAR accumulated

11. Optimizer step:   optimizer.step() uses ∇_GRPO + ∇_OPEAR combined gradients
```

The key insight is that steps 4-10 happen between GRPO's backward pass and the optimizer step, so both gradient contributions accumulate into the same parameter gradients before a single optimizer update.

---

## Architecture

```
run_alfworld_opear.py (config-driven launcher)
    │
    ▼
main_opear.py (OPEaRTaskRunner)
    │  applies extensions.py inside Ray worker
    ▼
verl RayPPOTrainer (patched)
    │
    ├── _update_actor() ← extensions.py wraps this
    │   │
    │   ├── 1. _generate_contrastive_data()
    │   │       ├── select_batch_positions()       [data.py]
    │   │       ├── reconstruct_trajectories()     [data.py]
    │   │       ├── guide.generate_contrastive_batch()  [guide.py]
    │   │       │       └── build_guide_prompt()   [prompts.py]
    │   │       │       └── parse_guide_response() [prompts.py]
    │   │       └── tokenize_contrastive_responses() [data.py]
    │   │               └── _build_contrastive_sequence()
    │   │
    │   └── 2. Original _update_actor (GRPO)
    │           │
    │           ├── GRPO forward + backward (16 micro-batches)
    │           ├── opear_accumulate_gradients()    [actor_hook.py]
    │           │       ├── for each pair (micro-batched):
    │           │       │   ├── _forward_micro_batch() (compliant)
    │           │       │   ├── _forward_micro_batch() (violating)
    │           │       │   ├── compute_opear_loss()   [loss.py]
    │           │       │   └── (lam * loss / N).backward()
    │           │       └── grad norms tracked
    │           └── optimizer.step()
    │
    └── verl dp_actor.py (patched to call actor_hook)
```

---

## Module Reference

### `loss.py` — Contrastive Loss

**`compute_opear_loss(compliant_log_probs, compliant_mask, violating_log_probs, violating_mask, beta=1.0)`**

- **Inputs**: `(N, response_len)` tensors for compliant/violating log-probs and masks
- **Normalization**: Per-token mean log-prob = `sum(lp * mask) / sum(mask)`
- **Gap**: `gap = compliant_mean_lp - violating_mean_lp` (positive = model prefers compliant)
- **Loss**: `L = -mean(log_sigmoid(beta * gap))` — bounded, DPO-style
- **Returns**: `(loss_tensor, metrics_dict)`

### `data.py` — Trajectory Reconstruction and Tokenization

**`select_batch_positions(batch, group_size, selection_ratio)`**
- Per-group selection using `uid` field to identify groups
- Picks `floor(selection_ratio * group_size)` rollouts per group (not globally)

**`reconstruct_trajectories(batch, tokenizer, batch_positions)`**
- Decodes selected rollouts into structured multi-turn trajectories
- Uses `response_mask` segments to separate individual assistant turns

**`tokenize_contrastive_responses(trajectories, contrastive_pairs, batch, tokenizer, max_response_length)`**
- Rebuilds full sequences: prompt + original observations + guide rewrites (no truncation)
- Pads all sequences to common length for batching

### `guide.py` — Guide Model Client

**`OPEaRGuide(model, max_completion_tokens, temperature, max_concurrent)`**
- Synchronous OpenAI client using `ThreadPoolExecutor` (avoids asyncio conflicts in Ray)
- Generates compliant + violating pairs in parallel, retries up to 3 times

### `prompts.py` — Prompt Templates and Parsing

- Compliant prompt: rewrite actions to be CONSISTENT with privileged facts
- Violating prompt: rewrite actions to sound PLAUSIBLE but CONTRADICT facts
- Critical: compliant reasoning must NOT mention privileged facts (prevents information leak)

### `actor_hook.py` — Gradient Accumulation Hook

**`opear_accumulate_gradients(actor, data, metrics)`**
- Called from verl's `dp_actor.py` between GRPO backward and `optimizer.step()`
- **Micro-batched**: processes one pair at a time (2 sequences) to avoid OOM
- Each per-pair loss scaled by `lambda / num_pairs` — no `loss_scale_factor`
- Tracks gradient norms: GRPO-only, O-PEaR-only (approximate), and combined

### `extensions.py` — Trainer Integration

**`apply()`** — patches `RayPPOTrainer` to generate contrastive data before each `_update_actor` call. Idempotent.

### `main_opear.py` — Entry Point

Subclasses verl's `TaskRunner` to call `extensions.apply()` inside the Ray worker.

### `run_alfworld_opear.py` — Config-Driven Runner

Reads YAML config, merges with defaults, launches training. Hardcoded to ALFWorld (interaction config, reward function). Supports `--gpus` override and `--dry-run`.

### `patches/apply_verl_patches.py` — verl Modifications

Four patches to the installed verl package:
1. Inject `_trajectory_info` into agent loop kwargs
2. Pass trajectory info into `interaction_kwargs`
3. Import `opear_accumulate_gradients` in `dp_actor.py`
4. Call it between GRPO backward and `optimizer.step()`

---

## Data Flow

```
Training batch (DataProto)
│  batch.batch: {input_ids, prompts, responses, response_mask, attention_mask}
│  batch.non_tensor_batch: {uid, facts_str}
│
├─ [extensions.py] _generate_contrastive_data()
│   ├─ select_batch_positions() → indices
│   ├─ reconstruct_trajectories() → structured turn data
│   ├─ guide.generate_contrastive_batch() → compliant/violating rewrites
│   └─ tokenize_contrastive_responses() → token tensors
│
├─ [dp_actor.py] GRPO forward/backward (16 micro-batches)
│   → ∇_GRPO accumulated
│
├─ [actor_hook.py] opear_accumulate_gradients()
│   for each pair i in 1..N:
│       ├─ forward compliant_i → log_probs_c
│       ├─ forward violating_i → log_probs_v
│       ├─ loss_i = -logsigmoid(beta * mean_gap_i)
│       └─ (lambda * loss_i / N).backward() → ∇_OPEAR accumulated
│
└─ [dp_actor.py] optimizer.step()
    → single update using ∇_GRPO + ∇_OPEAR
```

---

## Multi-Turn Token Structure

verl packs multi-turn conversations as a flat token sequence:

```
[prompt tokens] [asst1 tokens] [obs1 tokens] [asst2 tokens] [obs2 tokens] ... [pad]
                |---- response portion -----------------------------------------|

response_mask:  [1 1 1 ... 1]  [0 0 0 ... 0]  [1 1 1 ... 1]  [0 0 0 ... 0]  [0 0]
                 assistant 1    observation 1    assistant 2    observation 2    pad
```

- `response_mask = 1`: Model-generated tokens (assistant turns) — included in loss
- `response_mask = 0`: Environment observations + padding — excluded from loss
- `attention_mask = 1` for all real tokens (including observations), `0` for padding only

---

## Contrastive Sequence Rebuilding

When the guide rewrites assistant turns, the new text may be longer or shorter than the original. We rebuild the entire response from parts (no truncation):

```
Original:   [prompt] [asst1_orig (50 tok)] [obs1 (30 tok)] [asst2_orig (40 tok)] [pad]
Rebuilt:    [prompt] [asst1_new (70 tok)]  [obs1 (30 tok)] [asst2_new (55 tok)]  [pad]
```

Prompt and observation tokens are preserved verbatim. Only assistant tokens change.

---

## Gradient Scale Matching

O-PEaR uses **per-token-mean** log-probs inside logsigmoid, which naturally normalizes by sequence length (~800 tokens). GRPO uses **token-sum** divided by `loss_scale_factor` (15360). These produce per-token gradients of the same order of magnitude (~4-6e-6), so **no additional scaling factor is needed**.

```python
# actor_hook.py — no loss_sf division
scaled = lam * pair_loss / num_pairs
scaled.backward()
```

Why this works (per-token gradient comparison):
- GRPO: `A_t * ratio_t / (loss_sf * grad_accum) ≈ O(1) / (15360 * 16) ≈ 4e-6`
- O-PEaR: `sigmoid(-gap) * beta / (L * N) ≈ 0.3 * 0.1 / (800 * 64) ≈ 6e-7`

With `lambda=1.0`, O-PEaR gradients are ~2-5x GRPO. Lambda directly controls the relative weight.

**Previous bug**: The code previously divided by `loss_scale_factor` (15360), which over-normalized and made O-PEaR a no-op (grad norm ~1e-7 vs GRPO ~0.2). Combined with alpha=0.5 inside the sigmoid and beta=0.1, the total suppression was ~16,000x.

---

## Loss Function

```
mean_lp_c = sum(lp_c * mask_c) / sum(mask_c)   # per-token mean (frozen lengths)
mean_lp_v = sum(lp_v * mask_v) / sum(mask_v)
gap = mean_lp_c - mean_lp_v
L = -mean(log_sigmoid(beta * gap))
```

Properties:
- **Bounded**: Loss is in `[0, log(2) ≈ 0.693]`
- **Self-saturating**: As the model learns to distinguish pairs (gap >> 0), gradients vanish naturally
- **beta** controls saturation speed: higher beta = saturates faster. With beta=0.1 and typical gap ~0.8, the gradient multiplier is `sigmoid(-0.08) ≈ 0.48` (strong signal)
- **DPO-style**: Same functional form as Direct Preference Optimization
- **No alpha**: Both compliant and violating terms weighted equally in the gap

---

## Hyperparameters

| Parameter | Config Key | Default | Description |
|-----------|-----------|---------|-------------|
| **lambda** | `opear_lambda` / `algorithm.opear.lambda_coef` | 0.5 | Weight of O-PEaR relative to GRPO. At lambda=1.0 with beta=0.1, O-PEaR is ~2-5x GRPO. |
| **beta** | `opear_beta` / `algorithm.opear.beta` | 1.0 | Temperature for logsigmoid. Controls saturation speed and gradient magnitude. |
| **selection_ratio** | `opear_selection_ratio` / `algorithm.opear.selection_ratio` | 0.5 | Fraction of each group to select. `floor(ratio * group_size)` per group. |
| **guide_model** | `opear_guide_model` / `algorithm.opear.guide_model` | `"gpt-5.4-nano"` | OpenAI model for contrastive pair generation. |

### Recommended starting points

- **Conservative**: `lambda=0.1, beta=0.1` — O-PEaR is ~0.2-0.5x GRPO
- **Moderate**: `lambda=1.0, beta=0.1` — O-PEaR is ~2-5x GRPO
- **Strong**: `lambda=1.0, beta=1.0` — O-PEaR is ~19x GRPO

---

## WandB Metrics

All metrics are logged under the `opear/` prefix:

| Metric | Description |
|--------|-------------|
| `opear/loss` | Raw contrastive loss (before lambda scaling) |
| `opear/scaled_loss` | Loss after `lambda / num_pairs` scaling |
| `opear/compliant_logprob` | Mean per-token log-prob for compliant sequences |
| `opear/violating_logprob` | Mean per-token log-prob for violating sequences |
| `opear/logprob_gap` | Mean `lp_compliant - lp_violating` (should be positive and growing) |
| `opear/gap_std` | Standard deviation of the gap across pairs |
| `opear/gap_min` | Minimum gap in the batch (watch for negative values) |
| `opear/gap_max` | Maximum gap in the batch |
| `opear/num_pairs` | Number of valid contrastive pairs in this step |
| `opear/compliant_length` | Mean number of active tokens in compliant sequences |
| `opear/violating_length` | Mean number of active tokens in violating sequences |
| `opear/grad_norm` | Approximate gradient norm from O-PEaR alone |
| `opear/grpo_grad_norm` | Gradient norm from GRPO before O-PEaR backward |
| `opear/combined_grad_norm` | Combined gradient norm after both GRPO + O-PEaR |
| `opear/guide_time_s` | Wall-clock time for guide model API calls |
| `opear/num_segments` | Mean assistant turns per trajectory (multi-turn depth) |
| `opear/lambda` | Current lambda value |
| `opear/beta` | Current beta value |
| `opear/selection_ratio` | Current selection ratio |

### What to watch

- **`opear/logprob_gap`**: Should start near 0 and grow positive. If negative, the model prefers violating responses.
- **`opear/loss`**: Should start near `log(2) ≈ 0.693` and decrease toward 0 as the gap grows.
- **`opear/grad_norm` vs `opear/grpo_grad_norm`**: If O-PEaR >> GRPO, reduce lambda or beta. If O-PEaR ≈ 0, something is wrong.
- **`opear/guide_time_s`**: Typically 15-40s. If >> 60s, guide API may be rate-limited.
- **`opear/num_pairs`**: Should be > 0 every step. If 0, guide is failing consistently.

---

## Configuration

### YAML config files

Located in `verl071/configs/`. Example (`opear_logsigmoid_lambda1.yaml`):

```yaml
experiment_name: drgrpo_opear_logsigmoid_lambda1_qwen3_4b
gpus: "0,1,2,3"

# O-PEaR config
opear_enable: true
opear_lambda: 1.0
opear_beta: 1.0
opear_selection_ratio: 0.5
opear_guide_model: gpt-5.4-nano

# Model
model: Qwen/Qwen3-4B
tp: 2
gpu_memory_utilization: 0.5
max_model_len: 18432

# Data
group_size: 8
train_batch_size: 16
val_batch_size: 134
max_prompt_length: 2048
max_response_length: 15360
seed: 42

# Actor
lr: 1.0e-6
loss_agg_mode: seq-mean-token-sum-norm

# Algorithm
gamma: 0.95

# Agent loop
max_user_turns: 50

# Trainer
epochs: 250
save_freq: 30
test_freq: 5
val_before_train: true
logger: '["console","wandb"]'
project_name: opear_alfworld
```

---

## Running Experiments

### Prerequisites

```bash
# 1. Activate venv
cd /workspace/home/lab/shiv/verl-agent
source .venv/bin/activate

# 2. Apply verl patches (required after venv creation / verl reinstall)
python verl071/patches/apply_verl_patches.py

# 3. Ensure OPENAI_API_KEY is set (for guide model)
echo $OPENAI_API_KEY  # or set in .env file
```

### Launch training

```bash
# Single run
python verl071/run_alfworld_opear.py verl071/configs/opear_logsigmoid_lambda1.yaml

# Override GPUs
python verl071/run_alfworld_opear.py verl071/configs/opear_logsigmoid_lambda1.yaml --gpus 4,5,6,7

# Dry run (print command without executing)
python verl071/run_alfworld_opear.py verl071/configs/opear_logsigmoid_lambda1.yaml --dry-run
```

### Launch in tmux (production)

```bash
tmux new-session -d -s opear-logsig1 \
  "cd /workspace/home/lab/shiv/verl-agent && source .venv/bin/activate && \
   python verl071/run_alfworld_opear.py verl071/configs/opear_logsigmoid_lambda1.yaml \
   2>&1 | tee /tmp/opear-logsig1.log"
```

---

## Testing

### Unit tests

```bash
PYTHONPATH=. pytest verl071/tests/ -v
```

### Gradient scale verification (requires GPU)

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python verl071/tests/test_gradient_scale.py
```

Loads Qwen3-4B, computes actual gradient norms for GRPO-style vs O-PEaR-style losses, and compares magnitudes.

### End-to-end test (requires GPU + OpenAI API key)

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python verl071/tests/test_opear_e2e.py
```

---

## Design Decisions and Lessons Learned

### 1. Logsigmoid over unbounded loss
The original unbounded loss (`L = -mean(R)`) diverged to -36 by step 123, collapsing rewards. Logsigmoid is bounded in `[0, log(2)]` and self-saturates.

### 2. No loss_scale_factor for O-PEaR
O-PEaR uses per-token-mean log-probs which naturally normalize by sequence length (~800). GRPO uses token-sum / loss_sf (15360). Both produce per-token gradients of ~4-6e-6, so no additional scaling is needed. Previously, dividing by loss_sf made O-PEaR a no-op (~1e-7 grad norm vs GRPO's ~0.2).

### 3. Micro-batched forward passes
O-PEaR processes one contrastive pair at a time (2 sequences) to avoid OOM. GRPO uses micro_batch_size=1, so forwarding all 64 pairs at once would require ~14GB of activation memory beyond the available headroom. Micro-batching reduces this to ~3-4GB per pair.

### 4. free_cache_engine=True
vLLM's KV cache (~37GB per GPU) must be freed during training to make room for FSDP actor + O-PEaR forward passes. The cache is reallocated each rollout phase.

### 5. Rebuild sequences instead of in-place swap
The original approach truncated guide rewrites to fit original segment lengths, cutting off `<action>` tags. The current approach rebuilds sequences from parts with no truncation.

### 6. Per-group selection (not global)
Each task group contributes `floor(ratio * group_size)` trajectories independently.

### 7. Preventing privileged information leakage
The compliant prompt explicitly forbids mentioning PDDL facts in `<think>` tags. Reasoning must sound natural, as if the agent figured out the correct action from observations alone.

### 8. Synchronous guide client (not async)
Ray workers conflict with `asyncio.run()`. Using `ThreadPoolExecutor` avoids this while still enabling parallel API calls.

### 9. Beta controls both saturation and gradient magnitude
With per-token-mean gap ~0.8 and beta=0.1: `sigmoid(-0.08) ≈ 0.48` (strong signal). With beta=1.0: `sigmoid(-0.8) ≈ 0.31` (still strong). With beta=10: `sigmoid(-8) ≈ 0.0003` (saturated). Beta also linearly scales the gradient, so reducing beta from 1.0 to 0.1 reduces O-PEaR's contribution by ~10x.
