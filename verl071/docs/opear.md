# O-PEaR: Off-Policy Environment-aware Regularization

O-PEaR is a contrastive regularization method that augments reinforcement learning (Dr. GRPO) with environment-aware guidance. It uses a privileged guide model (GPT-5.4-nano) that has access to ground-truth PDDL facts from the environment to generate contrastive response pairs — one consistent with the facts (compliant), one contradicting them (violating). The policy is then trained to prefer the compliant responses, providing a learning signal that goes beyond sparse task rewards.

## Table of Contents

- [Motivation](#motivation)
- [Algorithm Overview](#algorithm-overview)
- [Architecture](#architecture)
- [Module Reference](#module-reference)
  - [loss.py — Contrastive Loss](#lossspy--contrastive-loss)
  - [data.py — Trajectory Reconstruction and Tokenization](#datapy--trajectory-reconstruction-and-tokenization)
  - [guide.py — Guide Model Client](#guidepy--guide-model-client)
  - [prompts.py — Prompt Templates and Parsing](#promptspy--prompt-templates-and-parsing)
  - [actor_hook.py — Gradient Accumulation Hook](#actor_hookpy--gradient-accumulation-hook)
  - [extensions.py — Trainer Integration](#extensionspy--trainer-integration)
  - [main_opear.py — Entry Point](#main_opearpy--entry-point)
  - [run_opear.py — Config-Driven Runner](#run_opearpy--config-driven-runner)
  - [patches/apply_verl_patches.py — verl Modifications](#patchesapply_verl_patchespy--verl-modifications)
- [Data Flow](#data-flow)
- [Multi-Turn Token Structure](#multi-turn-token-structure)
- [Contrastive Sequence Rebuilding](#contrastive-sequence-rebuilding)
- [Gradient Scale Matching](#gradient-scale-matching)
- [Loss Functions](#loss-functions)
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
8. Forward pass:      Compute log-probs for compliant and violating sequences under the policy
9. Contrastive loss:  L_opear = -mean(log_sigmoid(beta * (lp_compliant - lp_violating)))
10. O-PEaR backward:  scaled_loss = lambda * L_opear / loss_scale_factor → ∇_OPEAR accumulated

11. Optimizer step:   optimizer.step() uses ∇_GRPO + ∇_OPEAR combined gradients
```

The key insight is that steps 4-10 happen between GRPO's backward pass and the optimizer step, so both gradient contributions accumulate into the same parameter gradients before a single optimizer update.

---

## Architecture

```
run_opear.py (config-driven launcher)
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
    │           ├── GRPO forward + backward
    │           ├── opear_accumulate_gradients()    [actor_hook.py]
    │           │       ├── _forward_micro_batch() (compliant)
    │           │       ├── _forward_micro_batch() (violating)
    │           │       ├── compute_opear_loss()   [loss.py]
    │           │       └── scaled.backward()
    │           └── optimizer.step()
    │
    └── verl dp_actor.py (patched to call actor_hook)
```

---

## Module Reference

### `loss.py` — Contrastive Loss

**`compute_opear_loss(compliant_log_probs, compliant_mask, violating_log_probs, violating_mask, alpha, loss_type, beta)`**

Computes the O-PEaR regularization loss from per-token log-probabilities.

- **Inputs**: `(N, response_len)` tensors for compliant/violating log-probs and masks
- **Normalization**: Per-sequence mean log-prob = `sum(lp * mask) / sum(mask)` (length-invariant)
- **Gap**: `gap = compliant_norm_lp - violating_norm_lp` (positive = model prefers compliant)
- **Loss variants**:
  - `"unbounded"`: `L = -mean(alpha * lp_c - (1-alpha) * lp_v)` — original, R grows without limit
  - `"logsigmoid"`: `L = -mean(log_sigmoid(beta * gap))` — bounded [0, -log(2)], DPO-style
- **Returns**: `(loss_tensor, metrics_dict)`

### `data.py` — Trajectory Reconstruction and Tokenization

**`select_batch_positions(batch, group_size, selection_ratio)`**
- Per-group selection using `uid` field to identify groups
- Picks `floor(selection_ratio * group_size)` rollouts per group (not globally)
- Returns list of batch indices

**`_find_assistant_segments(response_mask)`**
- Finds contiguous `mask=1` regions in the response portion
- Returns `[(start, end), ...]` pairs within the response

**`reconstruct_trajectories(batch, tokenizer, batch_positions)`**
- Decodes selected rollouts into structured multi-turn trajectories
- Uses `response_mask` segments to separate individual assistant turns
- Extracts observations between turns, task description from prompt, PDDL facts
- Returns list of trajectory dicts with `turns`, `assistant_segments`, `facts`, etc.

**`_build_contrastive_sequence(prompt_ids, orig_response_ids, orig_response_mask, segments, rewrite_turns, tokenizer, pad_token_id, max_seq_len)`**
- Reassembles a full sequence: `[prompt] + [obs1] + [rewrite1] + [obs2] + [rewrite2] + ...`
- **No truncation** of guide rewrites (unlike the original in-place swap approach)
- Observations copied verbatim from original rollout
- Pads to `max_seq_len` for batching
- Returns dict with `input_ids`, `attention_mask`, `response_mask`

**`tokenize_contrastive_responses(trajectories, contrastive_pairs, batch, tokenizer, max_response_length)`**
- Orchestrates sequence building for all valid pairs
- Calls `_build_contrastive_sequence` for each compliant and violating variant
- Pads all sequences to common length with `pad_and_stack`
- Returns dict with `{compliant,violating}_{input_ids,attention_mask,response_mask}` tensors

### `guide.py` — Guide Model Client

**`OPEaRGuide(model, max_completion_tokens, temperature, max_concurrent)`**
- Synchronous OpenAI client (avoids asyncio/uvloop conflicts in Ray workers)
- Uses `ThreadPoolExecutor` for parallelism

**`generate_pair(turns, task_description, facts)`**
- Calls guide model twice in parallel (compliant + violating) for a single trajectory
- Retries up to 3 times per call with exponential backoff

**`generate_contrastive_batch(trajectories)`**
- Generates pairs for all selected trajectories concurrently (up to `max_concurrent=32`)
- Returns list of `Optional[dict]` parallel to input trajectories

### `prompts.py` — Prompt Templates and Parsing

**System prompts**: `COMPLIANT_SYSTEM` and `VIOLATING_SYSTEM`
- Compliant: "rewrite so actions are CONSISTENT with privileged facts"
- Violating: "rewrite so actions sound PLAUSIBLE but CONTRADICT the facts"
- Critical rule in compliant prompt: reasoning must NOT mention privileged facts (prevents information leak into `<think>` tags)

**`build_guide_prompt(turns, task_description, mode, facts)`**
- Constructs OpenAI chat messages with system prompt + formatted trajectory + facts

**`parse_guide_response(response_text, expected_turns)`**
- Parses `[TURN N]` markers from guide output
- Extracts `<think>...</think>` and `<action>...</action>` from each turn block
- Raises `ValueError` on malformed output (triggers retry in `_call_guide`)

### `actor_hook.py` — Gradient Accumulation Hook

**`opear_accumulate_gradients(actor, data, metrics)`**
- Called from verl's `dp_actor.py` between GRPO backward and `optimizer.step()`
- Extracts `opear_data` from `data.meta_info`
- Runs two forward passes through the actor model (compliant + violating sequences)
- Computes contrastive loss, scales by `lambda / loss_scale_factor`
- Calls `scaled.backward()` to accumulate gradients with existing GRPO gradients
- Tracks gradient norms: GRPO-only, OPEAR-only (approximate), and combined

### `extensions.py` — Trainer Integration

**`apply()`** (call once in Ray worker)
- Patches `RayPPOTrainer.__init__` to read O-PEaR config and instantiate `OPEaRGuide`
- Patches `RayPPOTrainer._update_actor` to call `_generate_contrastive_data` before original GRPO update
- Idempotent — safe to call multiple times

**`_generate_contrastive_data(trainer, batch)`**
- Orchestrates the full pipeline: select → reconstruct → guide → tokenize
- Attaches results to `batch.meta_info` for downstream use by `actor_hook.py`

### `main_opear.py` — Entry Point

**`OPEaRTaskRunner(TaskRunner)`**
- Subclasses verl's `TaskRunner` to call `extensions.apply()` inside the Ray worker
- Uses Hydra config with verl's `ppo_trainer` config schema

### `run_opear.py` — Config-Driven Runner

- Reads YAML config, merges with defaults, launches training
- Handles data preprocessing, environment variables, CUDA device assignment
- Supports `--gpus` override and `--dry-run` mode

### `patches/apply_verl_patches.py` — verl Modifications

Four patches applied to the installed verl package:

1. **agent_loop.py**: Injects `_trajectory_info` (global_step, rollout_n) into agent kwargs for deterministic game selection
2. **tool_agent_loop.py**: Passes trajectory info into `interaction_kwargs` for the environment
3. **dp_actor.py (import)**: Adds `try/except` import of `opear_accumulate_gradients`
4. **dp_actor.py (call site)**: Calls `opear_accumulate_gradients(self, data, metrics)` between GRPO backward and `optimizer.step()`

---

## Data Flow

### Step-by-step through one training iteration:

```
Training batch (DataProto)
│  batch.batch: {input_ids, prompts, responses, response_mask, attention_mask}
│  batch.non_tensor_batch: {uid, facts_str}
│
├─ [extensions.py] _generate_contrastive_data()
│   │
│   ├─ select_batch_positions(batch, group_size=8, selection_ratio=0.5)
│   │   → selects 4 rollouts per task group (indices into batch)
│   │
│   ├─ reconstruct_trajectories(batch, tokenizer, positions)
│   │   → decodes tokens into structured trajectories:
│   │     [{traj_uid, group_id, task_description, facts,
│   │       turns: [{role: "observation"/"assistant", content: "..."}],
│   │       assistant_segments: [(start, end), ...]}]
│   │
│   ├─ guide.generate_contrastive_batch(trajectories)
│   │   → calls GPT-5.4-nano for each trajectory (parallel)
│   │   → returns [{compliant: [{think, action}, ...],
│   │               violating: [{think, action}, ...]}, ...]
│   │
│   └─ tokenize_contrastive_responses(trajectories, pairs, batch, tokenizer)
│       → rebuilds full sequences with guide rewrites
│       → returns {compliant_input_ids, compliant_attention_mask,
│                  compliant_response_mask, violating_*, ...}
│
├─ [dp_actor.py] Standard GRPO forward/backward
│   → ∇_GRPO accumulated in model parameters
│
├─ [actor_hook.py] opear_accumulate_gradients()
│   ├─ Forward pass: compliant sequences → log_probs_c
│   ├─ Forward pass: violating sequences → log_probs_v
│   ├─ compute_opear_loss(lp_c, mask_c, lp_v, mask_v)
│   ├─ scaled = lambda * loss / loss_scale_factor
│   └─ scaled.backward()  → ∇_OPEAR accumulated with ∇_GRPO
│
└─ [dp_actor.py] optimizer.step()
    → single update using combined ∇_GRPO + ∇_OPEAR
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

The `_find_assistant_segments()` function identifies the `[start, end)` boundaries of each contiguous `mask=1` region, which correspond to individual assistant turns.

---

## Contrastive Sequence Rebuilding

When the guide rewrites assistant turns, the new text may be longer or shorter than the original. Rather than truncating rewrites to fit original segment lengths (which would cut off `<action>` tags), we rebuild the entire response sequence from parts:

```
Original:   [prompt] [asst1_orig (50 tok)] [obs1 (30 tok)] [asst2_orig (40 tok)] [pad]
                      ↓ guide rewrite        (preserved)     ↓ guide rewrite
Rebuilt:    [prompt] [asst1_new (70 tok)]  [obs1 (30 tok)] [asst2_new (55 tok)]  [pad]
```

Key properties:
- Prompt tokens: identical to original (unchanged)
- Observation tokens: copied verbatim from original rollout (unchanged)
- Assistant tokens: guide's full rewrite (no truncation)
- Response mask: `1` for rewritten assistant tokens, `0` for observation tokens
- All sequences padded to common length for batching

---

## Gradient Scale Matching

GRPO and O-PEaR losses must produce gradients of comparable magnitude for `lambda` to have intuitive semantics. The critical issue:

**GRPO path**: Inside verl's `agg_loss()`, the raw policy gradient loss is divided by `loss_scale_factor` (= `max_response_length` = 15360). This happens per micro-batch, so after gradient accumulation across 16 micro-batches, the effective scale is `raw_loss / loss_scale_factor`.

**O-PEaR path**: Runs once per mini-batch (not per micro-batch). Without correction, O-PEaR's raw loss (~0.5) would produce gradients ~20,000x larger than GRPO's (~0.000007).

**Fix**: `scaled = lambda * loss / loss_scale_factor`

This ensures `lambda=1` means "OPEAR gradients roughly equal to GRPO gradients." We divide by `loss_scale_factor` (not `gradient_accumulation`) because O-PEaR runs once per mini-batch, not once per micro-batch.

```
GRPO:  raw=0.1  → /15360 → 0.0000065 per mini-batch
OPEAR: raw=0.5  → *0.5/15360 → 0.0000163 (≈2.5x GRPO with lambda=0.5)
```

---

## Loss Functions

### Unbounded (original)

```
R = alpha * mean_lp_compliant - (1 - alpha) * mean_lp_violating
L = -mean(R)
```

Problem: As training progresses, the gap between compliant and violating log-probs grows without bound. The loss magnitude increases over time, eventually dominating GRPO and collapsing rewards.

### Logsigmoid (recommended)

```
gap = mean_lp_compliant - mean_lp_violating
L = -mean(log_sigmoid(beta * gap))
```

Properties:
- **Bounded**: Loss is in `[0, log(2) ≈ 0.693]`
- **Self-saturating**: As the model learns to distinguish pairs (gap >> 0), gradients vanish naturally
- **beta controls saturation speed**:
  - `beta=1.0`: Saturates when gap ≈ 5 (strong gradient signal early)
  - `beta=0.1`: Saturates when gap ≈ 50 (gentler, longer-lasting signal)
- **DPO-style**: Same functional form as Direct Preference Optimization

---

## Hyperparameters

| Parameter | Config Key | Default | Description |
|-----------|-----------|---------|-------------|
| **lambda** | `opear_lambda` / `algorithm.opear.lambda_coef` | 0.5 | Weight of O-PEaR loss relative to GRPO. After gradient scaling, lambda=1 means equal magnitude. |
| **alpha** | `opear_alpha` / `algorithm.opear.alpha` | 0.5 | Balance between compliant (up-weight) and violating (down-weight) terms. Only used for `unbounded` loss. |
| **selection_ratio** | `opear_selection_ratio` / `algorithm.opear.selection_ratio` | 0.5 | Fraction of each group to select for contrastive pair generation. `floor(ratio * group_size)` per group. |
| **loss_type** | `opear_loss_type` / `algorithm.opear.loss_type` | `"unbounded"` | `"unbounded"` or `"logsigmoid"`. Logsigmoid is recommended. |
| **beta** | `opear_loss_beta` / `algorithm.opear.loss_beta` | 1.0 | Temperature for logsigmoid loss. Lower = gentler saturation. Current experiments use 0.1. |
| **guide_model** | `opear_guide_model` / `algorithm.opear.guide_model` | `"gpt-5.4-nano"` | OpenAI model for contrastive pair generation. |

### Recommended starting points

- **Conservative**: `lambda=0.1, beta=0.1, loss_type=logsigmoid` — gentle regularization
- **Moderate**: `lambda=0.5, beta=0.1, loss_type=logsigmoid` — balanced
- **Aggressive**: `lambda=1.0, beta=0.1, loss_type=logsigmoid` — strong regularization

---

## WandB Metrics

All metrics are logged under the `opear/` prefix:

| Metric | Description |
|--------|-------------|
| `opear/loss` | Raw contrastive loss (before scaling) |
| `opear/scaled_loss` | Loss after `lambda / loss_scale_factor` scaling |
| `opear/compliant_logprob` | Mean normalized log-prob for compliant sequences |
| `opear/violating_logprob` | Mean normalized log-prob for violating sequences |
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
| `opear/lambda` | Current lambda value (constant) |
| `opear/alpha` | Current alpha value (constant) |
| `opear/loss_type` | 1.0 = logsigmoid, 0.0 = unbounded |
| `opear/loss_beta` | Current beta value (constant) |
| `opear/selection_ratio` | Current selection ratio (constant) |

### What to watch

- **`opear/logprob_gap`**: Should start near 0 and grow positive. If it stays negative, the model is preferring violating responses.
- **`opear/loss`**: For logsigmoid, should start near `log(2) ≈ 0.693` and decrease toward 0 as the gap grows.
- **`opear/grad_norm` vs `opear/grpo_grad_norm`**: If OPEAR grad norm >> GRPO, lambda is too high.
- **`opear/guide_time_s`**: Typically 2-10s. If >> 30s, guide API may be rate-limited.
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
opear_alpha: 0.5
opear_selection_ratio: 0.5
opear_guide_model: gpt-5.4-nano
opear_loss_type: logsigmoid
opear_loss_beta: 0.1

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

### Hydra overrides

The YAML config is translated to Hydra CLI overrides by `run_opear.py`. O-PEaR config is passed via the `+algorithm.opear.*` namespace (the `+` prefix adds new keys to the Hydra config).

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
python verl071/run_opear.py verl071/configs/opear_logsigmoid_lambda1.yaml

# Override GPUs
python verl071/run_opear.py verl071/configs/opear_logsigmoid_lambda1.yaml --gpus 4,5,6,7

# Dry run (print command without executing)
python verl071/run_opear.py verl071/configs/opear_logsigmoid_lambda1.yaml --dry-run
```

### Launch in tmux (production)

```bash
tmux new-session -d -s opear-logsig1 \
  "cd /workspace/home/lab/shiv/verl-agent && source .venv/bin/activate && \
   python verl071/run_opear.py verl071/configs/opear_logsigmoid_lambda1.yaml \
   2>&1 | tee /tmp/opear-logsig1.log"
```

### Environment variables set by runner

| Variable | Value | Purpose |
|----------|-------|---------|
| `CUDA_VISIBLE_DEVICES` | From config `gpus` | GPU assignment |
| `PYTHONPATH` | Prepends repo root | Module resolution |
| `TOKENIZERS_PARALLELISM` | `true` | HuggingFace tokenizer parallelism |
| `NCCL_DEBUG` | `WARN` | Reduce NCCL log noise |
| `VLLM_LOGGING_LEVEL` | `WARN` | Reduce vLLM log noise |
| `TMPDIR` | `/mnt/nvme0n1/tmp` | Avoid filling root `/tmp` |
| `RAY_TMPDIR` | `/mnt/nvme0n1/tmp` | Ray temp files on NVMe |

---

## Testing

### Unit tests

```bash
# All tests
PYTHONPATH=. pytest verl071/tests/ -v

# Specific modules
PYTHONPATH=. pytest verl071/tests/test_opear_loss.py -v       # Loss computation
PYTHONPATH=. pytest verl071/tests/test_opear_guide.py -v      # Guide + selection
PYTHONPATH=. pytest verl071/tests/test_opear_prompts.py -v    # Prompt building/parsing
PYTHONPATH=. pytest verl071/tests/test_opear_mock.py -v       # Mock integration
PYTHONPATH=. pytest verl071/tests/test_opear_data_integration.py -v  # Full data pipeline
```

### End-to-end test (requires GPU + OpenAI API key)

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python verl071/tests/test_opear_e2e.py
```

This loads Qwen3-4B on a single GPU, builds a realistic multi-turn batch, calls the real guide model, computes log-probs and contrastive losses, and logs everything to `opear_e2e_results.txt`.

---

## Design Decisions and Lessons Learned

### 1. Logsigmoid over unbounded loss
The original unbounded loss (`L = -mean(R)`) diverged to -36 by step 123, collapsing rewards. As the gap between compliant and violating log-probs grows, the unbounded loss magnitude increases without limit, eventually dominating GRPO. Logsigmoid is bounded in `[0, log(2)]` and self-saturates — gradients vanish as the model learns to distinguish pairs.

### 2. Gradient scale matching via loss_scale_factor
GRPO's internal `agg_loss()` divides raw loss by `loss_scale_factor` (= `max_response_length` = 15360). Without matching this division, OPEAR with lambda=0.5 produced gradients ~20,000x larger than GRPO. The fix: `scaled = lambda * loss / loss_scale_factor`.

### 3. Rebuild sequences instead of in-place swap
The original approach swapped guide rewrites into the exact token positions of the original assistant segments, truncating if the rewrite was longer. This cut off `<action>` tags and corrupted the contrastive signal. The current approach rebuilds the entire response from parts: `[obs1] + [rewrite1] + [obs2] + [rewrite2] + ...`, preserving full guide outputs.

### 4. Per-group selection (not global)
The original `select_rollouts` picked `floor(ratio * group_size)` trajectories globally. With 2 groups of 8, it might select 4 from group 1 and 0 from group 2. Now selection is per-group: each task group contributes `floor(ratio * group_size)` trajectories.

### 5. Preventing privileged information leakage
The compliant prompt originally said "reasoning should reflect knowledge consistent with privileged facts." The guide interpreted this literally, quoting PDDL facts in `<think>` tags. Fix: "reasoning must ONLY reference observable information, do NOT mention privileged facts." The reasoning should sound natural — as if the agent figured out the correct action from observations alone.

### 6. Synchronous guide client (not async)
Ray workers have their own event loops that conflict with `asyncio.run()` and `uvloop`. Using `ThreadPoolExecutor` instead of `asyncio` avoids these conflicts while still enabling parallel API calls.

### 7. Beta naming disambiguation
The codebase had two unrelated "beta" parameters: selection fraction and loss temperature. Renamed selection fraction to `selection_ratio` everywhere to eliminate ambiguity.

### 8. Response mask semantics
`response_mask = 1` means "model-generated token" (include in loss), `response_mask = 0` means "observation or padding" (exclude from loss). `attention_mask` is separate — it controls which tokens the transformer can attend to. Setting `attention_mask = 0` for padding tokens within the sequence would break causal attention; instead, padding tokens keep `attention_mask = 1` but are excluded from loss via `response_mask = 0`.
