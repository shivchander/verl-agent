"""Run a single O-PEaR experiment from a YAML config.

Usage:
    python verl071/run_experiment.py verl071/experiments/opear_lambda05.yaml
"""
import os
import subprocess
import sys

import yaml
from dotenv import load_dotenv
load_dotenv()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.yaml>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    name = cfg["experiment_name"]
    gpus = cfg["gpus"]
    o = cfg["opear"]
    t = cfg["training"]

    GROUP_SIZE = t["group_size"]
    TRAIN_DATA_SIZE = t["train_batch_size"]
    VAL_DATA_SIZE = t["val_batch_size"]

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

    # Build command — matches run_alfworld.py exactly, plus O-PEaR
    # Use ++ prefix for all overrides to handle keys not in base Hydra schema
    def ov(k, v):
        """Hydra override that creates key if missing."""
        return f"++{k}={v}"

    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        # Algorithm
        ov("algorithm.adv_estimator", "grpo"),
        ov("algorithm.use_kl_in_reward", False),
        ov("algorithm.norm_adv_by_std_in_grpo", False),
        ov("algorithm.gamma", cfg["algorithm"]["gamma"]),
        # O-PEaR
        ov("algorithm.opear.enable", o["enable"]),
        ov("algorithm.opear.lambda_coef", o["lambda_coef"]),
        ov("algorithm.opear.alpha", o["alpha"]),
        ov("algorithm.opear.beta", o["beta"]),
        ov("algorithm.opear.guide_model", o["guide_model"]),
        # Data
        ov("data.train_files", os.path.expanduser("~/data/verl-agent/text/train.parquet")),
        ov("data.val_files", os.path.expanduser("~/data/verl-agent/text/test.parquet")),
        ov("data.train_batch_size", TRAIN_DATA_SIZE),
        ov("data.val_batch_size", VAL_DATA_SIZE),
        ov("data.max_prompt_length", 2048),
        ov("data.max_response_length", t["max_response_length"]),
        ov("data.filter_overlong_prompts", True),
        ov("data.truncation", "error"),
        ov("data.return_raw_chat", True),
        ov("data.seed", 42),
        # Model
        ov("actor_rollout_ref.model.path", cfg["model"]["path"]),
        ov("actor_rollout_ref.model.use_remove_padding", True),
        ov("actor_rollout_ref.model.enable_gradient_checkpointing", True),
        # Actor
        ov("actor_rollout_ref.actor.optim.lr", t["lr"]),
        ov("actor_rollout_ref.actor.ppo_mini_batch_size", TRAIN_DATA_SIZE),
        ov("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", 1),
        ov("actor_rollout_ref.actor.use_kl_loss", False),
        ov("actor_rollout_ref.actor.loss_agg_mode", "seq-mean-token-sum-norm"),
        ov("actor_rollout_ref.actor.loss_scale_factor", t["max_response_length"]),
        ov("actor_rollout_ref.actor.entropy_coeff", 0),
        ov("actor_rollout_ref.actor.fsdp_config.param_offload", False),
        ov("actor_rollout_ref.actor.fsdp_config.optimizer_offload", False),
        # Rollout
        ov("actor_rollout_ref.rollout.name", "vllm"),
        ov("actor_rollout_ref.rollout.tensor_model_parallel_size", cfg["rollout"]["tensor_parallel_size"]),
        ov("actor_rollout_ref.rollout.gpu_memory_utilization", cfg["rollout"]["gpu_memory_utilization"]),
        ov("actor_rollout_ref.rollout.max_model_len", cfg["rollout"]["max_model_len"]),
        ov("actor_rollout_ref.rollout.load_format", "safetensors"),
        ov("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu", 2),
        ov("actor_rollout_ref.rollout.enable_chunked_prefill", False),
        ov("actor_rollout_ref.rollout.enforce_eager", False),
        ov("actor_rollout_ref.rollout.free_cache_engine", False),
        ov("actor_rollout_ref.rollout.val_kwargs.temperature", cfg["rollout"]["temperature"]),
        ov("actor_rollout_ref.rollout.val_kwargs.do_sample", True),
        # Agent loop
        ov("actor_rollout_ref.rollout.agent.default_agent_loop", "tool_agent"),
        ov("actor_rollout_ref.rollout.multi_turn.interaction_config_path", INTERACTION_CONFIG),
        ov("actor_rollout_ref.rollout.multi_turn.max_user_turns", cfg["rollout"]["max_turns"]),
        ov("actor_rollout_ref.rollout.n", GROUP_SIZE),
        # Ref model (disabled)
        ov("actor_rollout_ref.ref.fsdp_config.param_offload", True),
        ov("actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu", 2),
        # Reward
        ov("reward.custom_reward_function.path", REWARD_FN),
        ov("reward.custom_reward_function.name", "compute_score"),
        # Trainer
        ov("trainer.critic_warmup", 0),
        ov("trainer.n_gpus_per_node", 4),
        ov("trainer.nnodes", 1),
        ov("trainer.total_epochs", t["total_epochs"]),
        ov("trainer.save_freq", t["save_freq"]),
        ov("trainer.test_freq", t["test_freq"]),
        ov("trainer.val_before_train", t.get("val_before_train", True)),
        '++trainer.logger=["console","wandb"]',
        ov("trainer.project_name", "verl_agent_alfworld"),
        ov("trainer.experiment_name", f"drgrpo_{name}_qwen3_4b"),
    ]

    project_root = os.getcwd()
    env = os.environ.copy()
    # Only add project root for agent_system and verl071 imports,
    # NOT for verl itself (use the installed venv version to avoid vllm mismatch)
    env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")
    env["TOKENIZERS_PARALLELISM"] = "true"
    env["NCCL_DEBUG"] = "WARN"
    env["VLLM_LOGGING_LEVEL"] = "WARN"
    env["CUDA_VISIBLE_DEVICES"] = gpus

    print(f"Launching {name} on GPUs {gpus}")
    print(f"  O-PEaR: lambda={o['lambda_coef']}, alpha={o['alpha']}, beta={o['beta']}")
    print(f"  Model: {cfg['model']['path']} | Epochs: {t['total_epochs']}")

    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

    proc.wait()
    print(f"\n{name} exited with code {proc.returncode}")


if __name__ == "__main__":
    main()
