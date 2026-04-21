# ALFWorld Dr. GRPO Training (verl 0.7.1)

Multi-turn reinforcement learning on ALFWorld text-based embodied tasks using Dr. GRPO (no reference model) via verl 0.7.1's native interaction system.

## Overview

This setup trains a Qwen3-4B model to solve ALFWorld household tasks using:

- **Dr. GRPO** — Group Relative Policy Optimization without a reference model, using token-level loss normalization instead of KL regularization
- **verl 0.7.1** — Uses the native `BaseInteraction` + `ToolAgentLoop` for multi-turn rollouts
- **ALFWorld** — Text-based embodied environment (TextWorld backend) with 3553 training games across 6 task types

### Key Features

- Rich system prompts with task description, admissible actions, and interaction history
- Action parsing from `<action>...</action>` tags with `<think>...</think>` reasoning
- Per-turn history buffer (configurable length)
- Reward: `10.0 × won` (binary task completion)
- No reference model needed (saves ~8GB GPU memory)

## Files

| File | Description |
|---|---|
| `run_alfworld.py` | **Dr. GRPO** training — no ref model, token-level normalization |
| `run_alfworld_grpo.py` | **Standard GRPO+KL** baseline — ref model + KL loss |
| `eval_alfworld.py` | Per-task evaluation with profiles (short/medium/long) |
| `alfworld_interaction.py` | `BaseInteraction` implementation wrapping ALFWorld |
| `alfworld_interaction_config.yaml` | Interaction config (max_steps, history_length) |
| `alfworld_reward.py` | Custom reward function reading turn_scores |
| `patches/apply_verl_patches.py` | Patch script for verl agent loop modifications |
| `requirements.txt` | Key dependencies |
| `requirements-lock.txt` | Full dependency lock |

## Setup

```bash
# Create venv
uv venv .venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install -r verl071/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Or for exact reproducibility:
uv pip install -r verl071/requirements-lock.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Verify
python -c "import torch, verl, vllm, flash_attn, alfworld; print('All imports OK')"
```

### ALFWorld Data

ALFWorld requires game data. If not already present:

```bash
export ALFWORLD_DATA=$HOME/alfworld_data
python -c "import alfworld; alfworld.download()"
```

The training config expects the ALFWorld config at:
`agent_system/environments/env_package/alfworld/configs/config_tw.yaml`

### verl Patches

Two small patches to verl's agent loop are required for deterministic game selection
(correct GRPO grouping). Apply them after installing verl:

```bash
python verl071/patches/apply_verl_patches.py
```

This injects `global_step` and `rollout_n` into the interaction kwargs so that all
rollouts in the same GRPO group play the same ALFWorld game.

## Running

```bash
# From the repo root (not the verl071/ directory)
cd /path/to/verl-agent

# Copy files to repo root (interaction must be importable)
cp verl071/alfworld_interaction.py .
cp verl071/alfworld_interaction_config.yaml .
cp verl071/alfworld_reward.py .
cp verl071/run_alfworld.py .

# Launch training (uses GPUs 4-7 by default)
python run_alfworld.py
```

### Configuration

Key hyperparameters in `run_alfworld.py`:

| Parameter | Value | Notes |
|---|---|---|
| Model | Qwen/Qwen3-4B | Can swap to Qwen2.5-1.5B for faster iteration |
| Algorithm | Dr. GRPO | No ref model, token-level normalization |
| Group size | 8 | Rollouts per prompt for advantage estimation |
| Batch size | 16 | Unique prompts per step (128 total rollouts) |
| Response budget | 8192 tokens | Total across all turns |
| Max turns | 50 | Per-episode interaction limit |
| History length | 2 | Recent (obs, action) pairs in prompt |
| Learning rate | 1e-6 | |
| Epochs | 150 | ~7.5 hours on 4×H100 |
| TP | 2 | Tensor parallelism for vLLM |
| Reward | 10.0 × won | Binary task completion |

### GPU Requirements

- **4× H100 80GB** (or similar)
- FSDP for training, TP=2 for vLLM inference
- Peak memory: ~19GB per GPU (no ref model)
- `CUDA_VISIBLE_DEVICES` configurable in `run_alfworld.py`

### Dr. GRPO Settings

From [verl docs](https://verl.readthedocs.io/en/latest/algo/grpo.html#drgrpo):

```
algorithm.norm_adv_by_std_in_grpo=False       # No std normalization
actor.use_kl_loss=False                        # No KL loss (drops ref model)
actor.loss_agg_mode=seq-mean-token-sum-norm    # Token-level normalization
actor.loss_scale_factor=8192                   # Consistent normalization
algorithm.use_kl_in_reward=False               # No KL in reward
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  verl 0.7.1                     │
│  ┌───────────┐  ┌──────────────────────────┐    │
│  │   FSDP    │  │     ToolAgentLoop        │    │
│  │  Actor    │  │  ┌────────────────────┐  │    │
│  │  (train)  │  │  │ AlfWorldInteraction│  │    │
│  └───────────┘  │  │  - env management  │  │    │
│                 │  │  - prompt building  │  │    │
│  ┌───────────┐  │  │  - action parsing  │  │    │
│  │   vLLM    │  │  │  - reward (10×won) │  │    │
│  │  (infer)  │  │  └────────────────────┘  │    │
│  │   TP=2    │  └──────────────────────────┘    │
│  └───────────┘                                  │
│                 ┌──────────────────────────┐     │
│                 │    NaiveRewardManager    │     │
│                 │  + alfworld_reward.py    │     │
│                 └──────────────────────────┘     │
└─────────────────────────────────────────────────┘
```

### Interaction Flow

1. **Start**: `start_interaction()` creates ALFWorld game env
2. **Turn 0**: Returns initial observation prompt (no env step)
3. **Turn N**: Model generates `<think>...</think><action>...</action>`
4. **Parse**: Extract action from tags, validate format
5. **Step**: Execute action in ALFWorld, get next observation
6. **Prompt**: Build next prompt with task, history, admissible actions
7. **Repeat** until task solved (reward=10.0) or max turns/tokens reached

### Prompt Template

First turn (no history):
```
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {observation}
Your admissible actions are: [{actions}].

Now it's your turn to take an action.
You should first reason step-by-step... <think> </think> tags.
...choose an admissible action... <action> </action> tags.
```

Subsequent turns (with history):
```
You are an expert agent... Your task is to: {task}
Prior to this step, you have taken {N} step(s). Recent history: {history}
You are now at step {N+1} and your current observation is: {observation}
Your admissible actions are: [{actions}].
...
```

## Evaluation

Per-task evaluation on ALFWorld valid_seen (140 games) and valid_unseen (134 games).

Three eval profiles control per-turn token budget and total context:

| Profile | Per-turn tokens | Total context | Use case |
|---|---|---|---|
| `short` | 512 | 16k | Fast eval, constrained generation |
| `medium` | 2048 | 16k | Default, balanced |
| `long` | 2048 | 40k | Matches training, best for comparison |

### Running Evals

**Recommended approach** — pre-start vLLM server for best performance:

```bash
# 1. Convert FSDP checkpoint to HF format
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/.../global_step_150/actor \
    --target_dir /path/to/hf_model

# 2. Start vLLM server
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model /path/to/hf_model \
    --port 8100 --dtype bfloat16 \
    --max-model-len 16384 \   # 40960 for long profile
    --gpu-memory-utilization 0.9 &

# 3. Run eval with --no-server
python verl071/eval_alfworld.py \
    --checkpoint /path/to/hf_model \
    --split unseen --seed 123 \
    --no-server --api-base http://localhost:8100/v1 \
    --max-concurrent 8 \
    --profile medium \
    --output eval_results.json
```

Key flags:
- `--profile short|medium|long`: eval profile (default: medium)
- `--max-concurrent 8`: parallel games via multiprocessing
- `--no-server --api-base URL`: use pre-started vLLM (recommended)
- `--full`: both splits × 3 seeds
- `--temperature 0.4`: matches training val settings

All evals save full trajectories (raw model output, parsed action, observation per turn)
alongside per-task success rates and turn counts.

### Results: Dr. GRPO vs Lambda01 at Step 150 (eval_out_of_distribution, seed 123)

| Profile | Dr. GRPO step 150 | Lambda01 step 150 |
|---|---|---|
| **Short** (512/turn, 16k ctx) | 41.8% (33.6 avg steps) | 70.9% (23.4 avg steps) |
| **Medium** (2k/turn, 16k ctx) | 70.1% (22.8 avg steps) | 70.9% (21.6 avg steps) |
| **Long** (2k/turn, 40k ctx) | 71.6% (22.3 avg steps) | **80.6%** (21.4 avg steps) |

## Differences from Original verl-agent (0.3.1)

| Aspect | Original (0.3.1) | This Setup (0.7.1) |
|---|---|---|
| verl version | 0.3.1 (vendored) | 0.7.1 (pip) |
| Multi-turn | Custom `rollout_loop.py` | Native `ToolAgentLoop` |
| Environment | Ray remote actors per env | `BaseInteraction` wrapper |
| Response budget | 512 per turn | 8192 total across turns |
| Algorithm | GRPO + KL | Dr. GRPO (no ref model) |
| Model | Qwen2.5-1.5B | Qwen3-4B |
| vLLM | 0.12.0 | 0.15.1 |
