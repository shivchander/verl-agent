"""Launch two DrGRPO + O-PEaR runs in parallel with different lambda values.

Run A (GPUs 0-3): lambda=0.1  (conservative regularization)
Run B (GPUs 4-7): lambda=1.0  (aggressive regularization)
"""
import os
import subprocess
import sys
import time

from dotenv import load_dotenv
load_dotenv()

GROUP_SIZE = 8
TRAIN_DATA_SIZE = 16
VAL_DATA_SIZE = 134

# Prepare data once
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

RUNS = [
    {"gpus": "0,1,2,3", "lambda": 0.1, "name": "opear_lambda01"},
    {"gpus": "4,5,6,7", "lambda": 1.0, "name": "opear_lambda10"},
]


def build_cmd(run_cfg):
    return [
        sys.executable, "-m", "verl.trainer.main_ppo",
        # Algorithm — Dr. GRPO
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "algorithm.norm_adv_by_std_in_grpo=False",
        "algorithm.gamma=0.95",
        # O-PEaR
        "algorithm.opear.enable=True",
        f"algorithm.opear.lambda_coef={run_cfg['lambda']}",
        "algorithm.opear.alpha=0.5",
        "algorithm.opear.beta=0.5",
        "algorithm.opear.guide_model=gpt-5.4-nano",
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
        # Actor
        "actor_rollout_ref.actor.optim.lr=1e-6",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={TRAIN_DATA_SIZE}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm",
        "actor_rollout_ref.actor.loss_scale_factor=15360",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        # Rollout (vLLM)
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
        "actor_rollout_ref.rollout.max_model_len=18432",
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        "actor_rollout_ref.rollout.enforce_eager=False",
        "actor_rollout_ref.rollout.free_cache_engine=False",
        # Validation
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.4",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        # Agent loop
        "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
        f"actor_rollout_ref.rollout.multi_turn.interaction_config_path={INTERACTION_CONFIG}",
        "actor_rollout_ref.rollout.multi_turn.max_user_turns=50",
        f"actor_rollout_ref.rollout.n={GROUP_SIZE}",
        # Ref model (disabled)
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2",
        # Reward
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
        'trainer.logger=["console"]',
        "trainer.project_name=verl_agent_alfworld",
        f"trainer.experiment_name=drgrpo_{run_cfg['name']}_qwen3_4b",
    ]


def main():
    print("=" * 60)
    print("  O-PEaR Sweep: 2 parallel runs")
    print("=" * 60)
    for r in RUNS:
        print(f"  {r['name']}: GPUs {r['gpus']} | lambda={r['lambda']}")
    print()

    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = os.getcwd() + ":" + base_env.get("PYTHONPATH", "")
    base_env["TOKENIZERS_PARALLELISM"] = "true"
    base_env["NCCL_DEBUG"] = "WARN"
    base_env["VLLM_LOGGING_LEVEL"] = "WARN"

    log_dir = os.path.join(os.getcwd(), "verl071", "logs")
    os.makedirs(log_dir, exist_ok=True)

    procs = []
    for run_cfg in RUNS:
        env = base_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = run_cfg["gpus"]

        log_path = os.path.join(log_dir, f"{run_cfg['name']}.log")
        log_file = open(log_path, "w")

        cmd = build_cmd(run_cfg)
        print(f"Launching {run_cfg['name']} on GPUs {run_cfg['gpus']} → {log_path}")
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=log_file, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        procs.append((run_cfg, proc, log_file, log_path))

    print(f"\nBoth runs launched. Tailing logs...")
    print(f"  tail -f verl071/logs/opear_lambda01.log")
    print(f"  tail -f verl071/logs/opear_lambda10.log")
    print()

    # Wait for both
    for run_cfg, proc, log_file, log_path in procs:
        proc.wait()
        log_file.close()
        print(f"{run_cfg['name']} exited with code {proc.returncode}")


if __name__ == "__main__":
    main()
