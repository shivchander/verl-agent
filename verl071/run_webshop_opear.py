"""Unified O-PEaR runner for WebShop — reads experiment config from YAML, launches training.

Usage (from repo root):
    python verl071/run_webshop_opear.py verl071/configs/opear_webshop.yaml
    python verl071/run_webshop_opear.py verl071/configs/opear_webshop.yaml --gpus 0,1,2,3
    python verl071/run_webshop_opear.py verl071/configs/opear_webshop.yaml --dry-run
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
    "val_batch_size": 128,
    "max_prompt_length": 2048,
    "max_response_length": 4096,
    "seed": 42,
    # Model
    "model": "Qwen/Qwen3-4B",
    "tp": 2,
    "gpu_memory_utilization": 0.5,
    "max_model_len": 8192,
    "free_cache_engine": True,
    # Actor
    "lr": 1e-6,
    "micro_batch_size_per_gpu": 1,
    "loss_agg_mode": "seq-mean-token-sum-norm",
    # Algorithm
    "gamma": 0.95,
    # O-PEaR
    "opear_enable": True,
    "opear_lambda": 0.1,
    "opear_beta": 0.1,
    "opear_selection_ratio": 0.5,
    "opear_margin": 0.0,
    "opear_guide_model": "gpt-5.4-nano",
    # Agent loop
    "max_user_turns": 15,
    # Trainer
    "epochs": 250,
    "save_freq": 50,
    "test_freq": 5,
    "val_before_train": True,
    "logger": '["console","wandb"]',
    "project_name": "opear_webshop",
    "experiment_name": "drgrpo_opear_webshop_qwen3_4b",
    # Infra
    "gpus": "4,5,6,7",
    "n_gpus_per_node": 4,
}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        overrides = yaml.safe_load(f) or {}
    cfg = {**DEFAULTS, **overrides}
    return cfg


def prepare_data(cfg: dict):
    """Inline data preparation for WebShop.

    Cannot reuse examples.data_preprocess.prepare because it hardcodes
    interaction_kwargs.name='alfworld'. WebShop needs name='webshop' so the
    tool_agent loop dispatches to WebShopInteraction.
    """
    # Fix HF cache for data prep if inherited path is not writable
    _hf_cache = os.environ.get("HF_HOME", "")
    if _hf_cache and not os.access(os.path.dirname(_hf_cache), os.W_OK):
        _fallback = os.path.expanduser("~/.cache/huggingface")
        os.environ["HF_HOME"] = _fallback
        os.environ["TRANSFORMERS_CACHE"] = _fallback

    DATA_DIR = os.path.expanduser("~/data/verl-agent/text")
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Preparing data...")
    import datasets
    dataset = datasets.load_dataset("hiyouga/geometry3k")

    def make_map_fn(split):
        def process_fn(example, idx):
            return {
                "data_source": "text",
                "prompt": [{"role": "user",
                            "content": "You are starting a new task in the WebShop environment. Please wait for the observation."}],
                "ability": "agent",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "interaction_kwargs": {
                        "name": "webshop",
                        "split": split,
                        "index": idx,
                    },
                },
            }
        return process_fn

    train_ds = dataset["train"].select(range(cfg["train_batch_size"]))
    test_ds = dataset["test"].select(range(cfg["val_batch_size"]))
    train_ds = train_ds.map(make_map_fn("train"), with_indices=True, remove_columns=train_ds.column_names)
    test_ds = test_ds.map(make_map_fn("test"), with_indices=True, remove_columns=test_ds.column_names)
    train_ds.to_parquet(os.path.join(DATA_DIR, "train.parquet"))
    test_ds.to_parquet(os.path.join(DATA_DIR, "test.parquet"))
    print(f"Data written to {DATA_DIR} (train={len(train_ds)}, test={len(test_ds)})")


def build_cmd(cfg: dict) -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    interaction_config = os.path.join(script_dir, "webshop_interaction_config.yaml")
    reward_fn = os.path.join(script_dir, "webshop_reward.py")
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
        f"+algorithm.opear.beta={cfg['opear_beta']}",
        f"+algorithm.opear.selection_ratio={cfg['opear_selection_ratio']}",
        f"+algorithm.opear.margin={cfg['opear_margin']}",
        f"+algorithm.opear.guide_model={cfg['opear_guide_model']}",
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
        f"actor_rollout_ref.rollout.free_cache_engine={cfg['free_cache_engine']}",
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
    parser = argparse.ArgumentParser(description="O-PEaR WebShop training runner")
    parser.add_argument("config", help="Path to experiment YAML config")
    parser.add_argument("--gpus", help="Override CUDA_VISIBLE_DEVICES (e.g. 0,1,2,3)")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.gpus:
        cfg["gpus"] = args.gpus

    # Prepare data
    prepare_data(cfg)

    cmd = build_cmd(cfg)

    env = os.environ.copy()
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    env["PYTHONPATH"] = SCRIPT_DIR + ":" + PROJECT_ROOT + ":" + env.get("PYTHONPATH", "")

    # Java for pyserini (WebShop search engine)
    if "JAVA_HOME" not in env:
        env["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-17.0.18.0.8-2.el9.x86_64"

    # Override caches if inherited paths are not writable
    hf_cache_dir = env.get("HF_HOME", "")
    if hf_cache_dir and not os.access(os.path.dirname(hf_cache_dir), os.W_OK):
        hf_cache_dir = os.path.expanduser("~/.cache/huggingface")
        env["HF_HOME"] = hf_cache_dir
        env["TRANSFORMERS_CACHE"] = hf_cache_dir
    vllm_cache_dir = env.get("VLLM_CACHE_ROOT", "")
    if vllm_cache_dir and not os.access(os.path.dirname(vllm_cache_dir), os.W_OK):
        env["VLLM_CACHE_ROOT"] = os.path.expanduser("~/.cache/vllm")

    env["TOKENIZERS_PARALLELISM"] = "true"
    env["NCCL_DEBUG"] = "WARN"
    env["VLLM_LOGGING_LEVEL"] = "WARN"
    env["CUDA_VISIBLE_DEVICES"] = cfg["gpus"]
    env["TMPDIR"] = "/mnt/nvme0n1/tmp"
    env["RAY_TMPDIR"] = "/mnt/nvme0n1/tmp"

    print(f"Launching O-PEaR WebShop training: {cfg['experiment_name']}")
    print(f"  Model: {cfg['model']} | GPUs: {cfg['gpus']} | TP={cfg['tp']}")
    print(f"  O-PEaR: lambda={cfg['opear_lambda']}, beta={cfg['opear_beta']}, guide={cfg['opear_guide_model']}")
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
