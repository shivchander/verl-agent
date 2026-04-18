"""Unified OPEAR runner — reads experiment config from YAML, launches training.

Usage (from repo root):
    python verl071/run_opear.py verl071/configs/opear_logsigmoid.yaml
    python verl071/run_opear.py verl071/configs/opear_lambda05.yaml --gpus 0,1,2,3
    python verl071/run_opear.py verl071/configs/opear_test.yaml --dry-run
"""
import argparse
import os
import subprocess
import sys

import yaml
from dotenv import load_dotenv

load_dotenv()

DEFAULTS = {
    # Data
    "group_size": 8,
    "train_batch_size": 16,
    "val_batch_size": 134,
    "max_prompt_length": 2048,
    "max_response_length": 15360,
    "seed": 42,
    # Model
    "model": "Qwen/Qwen3-4B",
    "tp": 2,
    "gpu_memory_utilization": 0.5,
    "max_model_len": 18432,
    # Actor
    "lr": 1e-6,
    "micro_batch_size_per_gpu": 1,
    "loss_agg_mode": "seq-mean-token-sum-norm",
    # Algorithm
    "gamma": 0.95,
    # O-PEaR
    "opear_enable": True,
    "opear_lambda": 0.5,
    "opear_alpha": 0.5,
    "opear_beta": 0.5,
    "opear_guide_model": "gpt-5.4-nano",
    "opear_loss_type": "unbounded",
    "opear_loss_beta": 1.0,
    # Agent loop
    "max_user_turns": 50,
    # Trainer
    "epochs": 250,
    "save_freq": 30,
    "test_freq": 5,
    "val_before_train": True,
    "logger": '["console","wandb"]',
    "project_name": "verl_agent_alfworld",
    # Infra
    "gpus": "4,5,6,7",
    "n_gpus_per_node": 4,
}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        overrides = yaml.safe_load(f) or {}
    cfg = {**DEFAULTS, **overrides}
    return cfg


def build_cmd(cfg: dict) -> list[str]:
    interaction_config = os.path.join(os.getcwd(), "alfworld_interaction_config.yaml")
    reward_fn = os.path.join(os.getcwd(), "alfworld_reward.py")
    loss_scale = cfg["max_response_length"]

    cmd = [
        sys.executable, "-m", "verl071.main_opear",
        # Algorithm
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "algorithm.norm_adv_by_std_in_grpo=False",
        f"algorithm.gamma={cfg['gamma']}",
        # O-PEaR
        f"+algorithm.opear.enable={cfg['opear_enable']}",
        f"+algorithm.opear.lambda_coef={cfg['opear_lambda']}",
        f"+algorithm.opear.alpha={cfg['opear_alpha']}",
        f"+algorithm.opear.beta={cfg['opear_beta']}",
        f"+algorithm.opear.guide_model={cfg['opear_guide_model']}",
        f"+algorithm.opear.loss_type={cfg['opear_loss_type']}",
        f"+algorithm.opear.loss_beta={cfg['opear_loss_beta']}",
        # Data
        f"data.train_files={os.path.expanduser('~/data/verl-agent/text/train.parquet')}",
        f"data.val_files={os.path.expanduser('~/data/verl-agent/text/test.parquet')}",
        f"data.train_batch_size={cfg['train_batch_size']}",
        f"data.val_batch_size={cfg['val_batch_size']}",
        f"data.max_prompt_length={cfg['max_prompt_length']}",
        f"data.max_response_length={cfg['max_response_length']}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.return_raw_chat=True",
        f"data.seed={cfg['seed']}",
        # Model
        f"actor_rollout_ref.model.path={cfg['model']}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        # Actor
        f"actor_rollout_ref.actor.optim.lr={cfg['lr']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg['train_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={cfg['micro_batch_size_per_gpu']}",
        "actor_rollout_ref.actor.use_kl_loss=False",
        f"actor_rollout_ref.actor.loss_agg_mode={cfg['loss_agg_mode']}",
        f"actor_rollout_ref.actor.loss_scale_factor={loss_scale}",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        # Rollout (vLLM)
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={cfg['tp']}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={cfg['gpu_memory_utilization']}",
        f"actor_rollout_ref.rollout.max_model_len={cfg['max_model_len']}",
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        "actor_rollout_ref.rollout.enforce_eager=False",
        "actor_rollout_ref.rollout.free_cache_engine=False",
        # Validation sampling
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.4",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        # Agent loop
        "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
        f"actor_rollout_ref.rollout.multi_turn.interaction_config_path={interaction_config}",
        f"actor_rollout_ref.rollout.multi_turn.max_user_turns={cfg['max_user_turns']}",
        f"actor_rollout_ref.rollout.n={cfg['group_size']}",
        # Ref model (disabled for Dr. GRPO)
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2",
        # Reward
        f"reward.custom_reward_function.path={reward_fn}",
        "reward.custom_reward_function.name=compute_score",
        # Trainer
        "trainer.critic_warmup=0",
        f"trainer.n_gpus_per_node={cfg['n_gpus_per_node']}",
        "trainer.nnodes=1",
        f"trainer.total_epochs={cfg['epochs']}",
        f"trainer.save_freq={cfg['save_freq']}",
        f"trainer.test_freq={cfg['test_freq']}",
        f"trainer.val_before_train={str(cfg['val_before_train']).lower()}",
        f"trainer.logger={cfg['logger']}",
        f"trainer.project_name={cfg['project_name']}",
        f"trainer.experiment_name={cfg['experiment_name']}",
    ]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="OPEAR training runner")
    parser.add_argument("config", help="Path to experiment YAML config")
    parser.add_argument("--gpus", help="Override CUDA_VISIBLE_DEVICES (e.g. 0,1,2,3)")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.gpus:
        cfg["gpus"] = args.gpus

    # Prepare data
    print("Preparing data...")
    subprocess.run(
        [sys.executable, "-m", "examples.data_preprocess.prepare",
         "--mode", "text",
         "--train_data_size", str(cfg["train_batch_size"]),
         "--val_data_size", str(cfg["val_batch_size"])],
        env={**os.environ, "PYTHONPATH": os.getcwd()},
        check=True,
    )

    cmd = build_cmd(cfg)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")
    env["TOKENIZERS_PARALLELISM"] = "true"
    env["NCCL_DEBUG"] = "WARN"
    env["VLLM_LOGGING_LEVEL"] = "WARN"
    env["CUDA_VISIBLE_DEVICES"] = cfg["gpus"]
    env["TMPDIR"] = "/mnt/nvme0n1/tmp"
    env["RAY_TMPDIR"] = "/mnt/nvme0n1/tmp"

    print(f"Launching O-PEaR training: {cfg['experiment_name']}")
    print(f"  Model: {cfg['model']} | GPUs: {cfg['gpus']} | TP={cfg['tp']}")
    print(f"  O-PEaR: lambda={cfg['opear_lambda']}, loss={cfg['opear_loss_type']}, "
          f"beta={cfg['opear_loss_beta']}, guide={cfg['opear_guide_model']}")
    print(f"  Batch: {cfg['train_batch_size']} x {cfg['group_size']} = "
          f"{cfg['train_batch_size'] * cfg['group_size']} rollouts/step")
    print(f"  Epochs: {cfg['epochs']} | Save: every {cfg['save_freq']} steps")

    if args.dry_run:
        print("\n[dry-run] Command:")
        print(" \\\n  ".join(cmd))
        return

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


if __name__ == "__main__":
    main()
