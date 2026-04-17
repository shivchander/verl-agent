"""Run ALFWorld Dr. GRPO + O-PEaR training via verl 0.7.1.

Setup:
  - Dr. GRPO: no reference model, token-level loss normalization
  - O-PEaR: off-policy environment-aware regularization via GPT-5.4-nano guide
  - TP=2 for more vLLM memory headroom
  - System prompts with task description, history, admissible actions
  - Action extraction from <action>...</action> tags
  - Reward: 10.0 * won (task completion)
  - group_size=8, 150 epochs
"""
import os
import subprocess
import sys

from dotenv import load_dotenv
load_dotenv()

GROUP_SIZE = 8
TRAIN_DATA_SIZE = 16
VAL_DATA_SIZE = 134  # matches eval_out_of_distribution split (134 unseen games)

# Prepare data
print("Preparing data...")
subprocess.run(
    [sys.executable, "-m", "examples.data_preprocess.prepare",
     "--mode", "text",
     "--train_data_size", str(TRAIN_DATA_SIZE),
     "--val_data_size", str(VAL_DATA_SIZE)],
    env={**os.environ, "PYTHONPATH": os.getcwd()},
    check=True,
)

INTERACTION_CONFIG = os.path.join(os.getcwd(), "alfworld_interaction_config.yaml")
REWARD_FN = os.path.join(os.getcwd(), "alfworld_reward.py")

cmd = [
    sys.executable, "-m", "verl071.main_opear",
    # Algorithm — Dr. GRPO (no ref model, token-level normalization)
    "algorithm.adv_estimator=grpo",
    "algorithm.use_kl_in_reward=False",
    "algorithm.norm_adv_by_std_in_grpo=False",
    "algorithm.gamma=0.95",
    # O-PEaR: Off-Policy Environment-aware Regularization
    "+algorithm.opear.enable=True",
    "+algorithm.opear.lambda_coef=0.5",
    "+algorithm.opear.alpha=0.5",
    "+algorithm.opear.beta=0.5",
    "+algorithm.opear.guide_model=gpt-5.4-nano",
    # Data
    f"data.train_files={os.path.expanduser('~/data/verl-agent/text/train.parquet')}",
    f"data.val_files={os.path.expanduser('~/data/verl-agent/text/test.parquet')}",
    f"data.train_batch_size={TRAIN_DATA_SIZE}",
    f"data.val_batch_size={VAL_DATA_SIZE}",
    "data.max_prompt_length=2048",
    "data.max_response_length=15360",
    "data.filter_overlong_prompts=True",
    "data.truncation=error",
    "data.return_raw_chat=True",
    "data.seed=42",
    # Model
    "actor_rollout_ref.model.path=Qwen/Qwen3-4B",
    "actor_rollout_ref.model.use_remove_padding=True",
    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
    # Actor — Dr. GRPO: no KL loss, no ref model
    "actor_rollout_ref.actor.optim.lr=1e-6",
    f"actor_rollout_ref.actor.ppo_mini_batch_size={TRAIN_DATA_SIZE}",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.actor.use_kl_loss=False",
    "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm",
    "actor_rollout_ref.actor.loss_scale_factor=15360",
    "actor_rollout_ref.actor.entropy_coeff=0",
    "actor_rollout_ref.actor.fsdp_config.param_offload=False",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
    # Rollout (vLLM) — TP=2 for memory headroom
    "actor_rollout_ref.rollout.name=vllm",
    "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
    "actor_rollout_ref.rollout.max_model_len=18432",
    "actor_rollout_ref.rollout.load_format=safetensors",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2",
    "actor_rollout_ref.rollout.enable_chunked_prefill=False",
    "actor_rollout_ref.rollout.enforce_eager=False",
    "actor_rollout_ref.rollout.free_cache_engine=False",
    # Validation sampling
    "actor_rollout_ref.rollout.val_kwargs.temperature=0.4",
    "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
    # Agent loop — tool_agent with ALFWorld interaction
    "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
    f"actor_rollout_ref.rollout.multi_turn.interaction_config_path={INTERACTION_CONFIG}",
    "actor_rollout_ref.rollout.multi_turn.max_user_turns=50",
    # Group size for GRPO advantage estimation
    f"actor_rollout_ref.rollout.n={GROUP_SIZE}",
    # Ref model — disabled for Dr. GRPO
    "actor_rollout_ref.ref.fsdp_config.param_offload=True",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2",
    # Reward — custom function that reads turn_scores from interaction
    f"reward.custom_reward_function.path={REWARD_FN}",
    "reward.custom_reward_function.name=compute_score",
    # Trainer
    "trainer.critic_warmup=0",
    "trainer.n_gpus_per_node=4",
    "trainer.nnodes=1",
    "trainer.total_epochs=250",
    "trainer.save_freq=50",
    "trainer.test_freq=5",
    "trainer.val_before_train=True",
    'trainer.logger=["console","wandb"]',
    "trainer.project_name=verl_agent_alfworld",
    "trainer.experiment_name=drgrpo_opear_qwen3_4b",
]

env = os.environ.copy()
env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")
env["TOKENIZERS_PARALLELISM"] = "true"
env["NCCL_DEBUG"] = "WARN"
env["VLLM_LOGGING_LEVEL"] = "WARN"
env["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

print(f"Launching verl ALFWorld Dr. GRPO + O-PEaR training")
print(f"  Model: Qwen3-4B | GPUs: {env['CUDA_VISIBLE_DEVICES']} | TP=2")
print(f"  Dr. GRPO: no ref model, token-level normalization")
print(f"  O-PEaR: lambda=0.5, alpha=0.5, beta=0.5, guide=gpt-5.4-nano")
print(f"  Batch: {TRAIN_DATA_SIZE} x {GROUP_SIZE} group = {TRAIN_DATA_SIZE * GROUP_SIZE} rollouts/step")
print(f"  Epochs: 250 | Response budget: 15360 | Max steps: 50 | gamma: 0.95")

proc = subprocess.Popen(
    cmd,
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

for line in proc.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()

proc.wait()
print(f"\nExited with code: {proc.returncode}")
