"""Quick 1-GPU smoke test for O-PEaR + DrGRPO.

Runs 2 epochs with batch_size=2 to verify the full pipeline connects.
"""
import os
import subprocess
import sys

from dotenv import load_dotenv
load_dotenv()

GROUP_SIZE = 2
TRAIN_DATA_SIZE = 4
VAL_DATA_SIZE = 4

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
    # Algorithm
    "algorithm.adv_estimator=grpo",
    "algorithm.use_kl_in_reward=False",
    "algorithm.norm_adv_by_std_in_grpo=False",
    "algorithm.gamma=0.95",
    # O-PEaR
    "+algorithm.opear.enable=True",
    "+algorithm.opear.lambda_coef=0.5",
    "+algorithm.opear.alpha=0.5",
    "+algorithm.opear.beta=0.5",
    "+algorithm.opear.guide_model=gpt-5.4-nano",
    # Data (small)
    f"data.train_files={os.path.expanduser('~/data/verl-agent/text/train.parquet')}",
    f"data.val_files={os.path.expanduser('~/data/verl-agent/text/test.parquet')}",
    f"data.train_batch_size={TRAIN_DATA_SIZE}",
    f"data.val_batch_size={VAL_DATA_SIZE}",
    "data.max_prompt_length=1024",
    "data.max_response_length=1024",
    "data.filter_overlong_prompts=True",
    "data.truncation=error",
    "data.return_raw_chat=True",
    "data.seed=42",
    # Model
    "actor_rollout_ref.model.path=Qwen/Qwen3-4B",
    "actor_rollout_ref.model.use_remove_padding=True",
    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
    # Actor
    "actor_rollout_ref.actor.optim.lr=1e-6",
    f"actor_rollout_ref.actor.ppo_mini_batch_size={TRAIN_DATA_SIZE}",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.actor.use_kl_loss=False",
    "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm",
    "actor_rollout_ref.actor.loss_scale_factor=1024",
    "actor_rollout_ref.actor.entropy_coeff=0",
    "actor_rollout_ref.actor.fsdp_config.param_offload=True",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
    # Rollout — TP=1, 1 GPU
    "actor_rollout_ref.rollout.name=vllm",
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.3",
    "actor_rollout_ref.rollout.max_model_len=2048",
    "actor_rollout_ref.rollout.load_format=safetensors",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2",
    "actor_rollout_ref.rollout.enable_chunked_prefill=False",
    "actor_rollout_ref.rollout.enforce_eager=True",
    "actor_rollout_ref.rollout.free_cache_engine=True",
    # Validation
    "actor_rollout_ref.rollout.val_kwargs.temperature=0.4",
    "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
    # Agent loop
    "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
    f"actor_rollout_ref.rollout.multi_turn.interaction_config_path={INTERACTION_CONFIG}",
    "actor_rollout_ref.rollout.multi_turn.max_user_turns=5",
    # Group size
    f"actor_rollout_ref.rollout.n={GROUP_SIZE}",
    # Ref model (disabled)
    "actor_rollout_ref.ref.fsdp_config.param_offload=True",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2",
    # Reward
    f"reward.custom_reward_function.path={REWARD_FN}",
    "reward.custom_reward_function.name=compute_score",
    # Trainer — 1 GPU, 2 epochs, no val
    "trainer.critic_warmup=0",
    "trainer.n_gpus_per_node=1",
    "trainer.nnodes=1",
    "trainer.total_epochs=2",
    "trainer.save_freq=999",
    "trainer.test_freq=999",
    "trainer.val_before_train=False",
    'trainer.logger=["console"]',
    "trainer.project_name=opear_smoke_test",
    "trainer.experiment_name=smoke_1gpu",
]

env = os.environ.copy()
env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")
env["TOKENIZERS_PARALLELISM"] = "true"
env["NCCL_DEBUG"] = "WARN"
env["VLLM_LOGGING_LEVEL"] = "WARN"
env["CUDA_VISIBLE_DEVICES"] = "0"

print("Launching 1-GPU O-PEaR smoke test")
print("  batch=2, group=2, epochs=2, max_turns=5, response_budget=2048")

proc = subprocess.Popen(
    cmd, env=env,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1,
)
for line in proc.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()

proc.wait()
print(f"\nExited with code: {proc.returncode}")
