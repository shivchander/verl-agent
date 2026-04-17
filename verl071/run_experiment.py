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


def build_cmd(cfg):
    t = cfg["training"]
    r = cfg["rollout"]
    o = cfg["opear"]

    interaction_config = os.path.join(os.getcwd(), "alfworld_interaction_config.yaml")
    reward_fn = os.path.join(os.getcwd(), "alfworld_reward.py")

    return [
        sys.executable, "-m", "verl.trainer.main_ppo",
        # Algorithm
        f"algorithm.adv_estimator={cfg['algorithm']['adv_estimator']}",
        f"algorithm.use_kl_in_reward={cfg['algorithm']['use_kl_in_reward']}",
        f"algorithm.norm_adv_by_std_in_grpo={cfg['algorithm']['norm_adv_by_std_in_grpo']}",
        f"algorithm.gamma={cfg['algorithm']['gamma']}",
        # O-PEaR
        f"algorithm.opear.enable={o['enable']}",
        f"algorithm.opear.lambda_coef={o['lambda_coef']}",
        f"algorithm.opear.alpha={o['alpha']}",
        f"algorithm.opear.beta={o['beta']}",
        f"algorithm.opear.guide_model={o['guide_model']}",
        # Data
        f"data.train_files={os.path.expanduser('~/data/verl-agent/text/train.parquet')}",
        f"data.val_files={os.path.expanduser('~/data/verl-agent/text/test.parquet')}",
        f"data.train_batch_size={t['train_batch_size']}",
        f"data.val_batch_size={t['val_batch_size']}",
        f"data.max_prompt_length={t['max_prompt_length']}",
        f"data.max_response_length={t['max_response_length']}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.return_raw_chat=True",
        "data.seed=42",
        # Model
        f"actor_rollout_ref.model.path={cfg['model']['path']}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        # Actor
        f"actor_rollout_ref.actor.optim.lr={t['lr']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={t['train_batch_size']}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm",
        f"actor_rollout_ref.actor.loss_scale_factor={t['max_response_length']}",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        # Rollout
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={r['tensor_parallel_size']}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={r['gpu_memory_utilization']}",
        f"actor_rollout_ref.rollout.max_model_len={r['max_model_len']}",
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        "actor_rollout_ref.rollout.enforce_eager=False",
        "actor_rollout_ref.rollout.free_cache_engine=False",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={r['temperature']}",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        # Agent loop
        "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
        f"actor_rollout_ref.rollout.multi_turn.interaction_config_path={interaction_config}",
        f"actor_rollout_ref.rollout.multi_turn.max_user_turns={r['max_turns']}",
        f"actor_rollout_ref.rollout.n={t['group_size']}",
        # Ref model (disabled)
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2",
        # Reward
        f"reward.custom_reward_function.path={reward_fn}",
        "reward.custom_reward_function.name=compute_score",
        # Trainer
        "trainer.critic_warmup=0",
        "trainer.n_gpus_per_node=4",
        "trainer.nnodes=1",
        f"trainer.total_epochs={t['total_epochs']}",
        f"trainer.save_freq={t['save_freq']}",
        f"trainer.test_freq={t['test_freq']}",
        f"trainer.val_before_train={t.get('val_before_train', True)}",
        'trainer.logger=["console","wandb"]',
        "trainer.project_name=verl_agent_alfworld",
        f"trainer.experiment_name=drgrpo_{cfg['experiment_name']}_qwen3_4b",
    ]


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.yaml>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    name = cfg["experiment_name"]
    gpus = cfg["gpus"]

    # Prepare data
    t = cfg["training"]
    print("Preparing data...")
    subprocess.run(
        [sys.executable, "-m", "examples.data_preprocess.prepare",
         "--mode", "text",
         "--train_data_size", str(t["train_batch_size"]),
         "--val_data_size", str(t["val_batch_size"])],
        env={**os.environ, "PYTHONPATH": os.getcwd()},
        check=True,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")
    env["TOKENIZERS_PARALLELISM"] = "true"
    env["NCCL_DEBUG"] = "WARN"
    env["VLLM_LOGGING_LEVEL"] = "WARN"
    env["CUDA_VISIBLE_DEVICES"] = gpus

    print(f"Launching {name} on GPUs {gpus}")
    print(f"  O-PEaR: lambda={cfg['opear']['lambda_coef']}, alpha={cfg['opear']['alpha']}, beta={cfg['opear']['beta']}")
    print(f"  Model: {cfg['model']['path']} | Epochs: {t['total_epochs']}")

    cmd = build_cmd(cfg)
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
