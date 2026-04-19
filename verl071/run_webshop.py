"""Run WebShop Dr. GRPO training via verl 0.7.1.

Setup:
  - Dr. GRPO: no reference model, token-level loss normalization
  - TP=2 for more vLLM memory headroom
  - System prompts with task description, history, available actions
  - Action extraction from <action>...</action> tags
  - Reward: 10.0 * won (task completion)
  - group_size=8, 250 epochs, max 15 steps per episode
"""
import os
import subprocess
import sys

GROUP_SIZE = 8
TRAIN_DATA_SIZE = 16
VAL_DATA_SIZE = 128  # test goals (index 0-500)

# --------------- Inline data preparation --------------- #
# Cannot reuse examples.data_preprocess.prepare because it hardcodes
# interaction_kwargs.name='alfworld'. WebShop needs name='webshop' so the
# tool_agent loop dispatches to WebShopInteraction.

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

train_ds = dataset["train"].select(range(TRAIN_DATA_SIZE))
test_ds = dataset["test"].select(range(VAL_DATA_SIZE))
train_ds = train_ds.map(make_map_fn("train"), with_indices=True, remove_columns=train_ds.column_names)
test_ds = test_ds.map(make_map_fn("test"), with_indices=True, remove_columns=test_ds.column_names)
train_ds.to_parquet(os.path.join(DATA_DIR, "train.parquet"))
test_ds.to_parquet(os.path.join(DATA_DIR, "test.parquet"))
print(f"Data written to {DATA_DIR} (train={len(train_ds)}, test={len(test_ds)})")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTERACTION_CONFIG = os.path.join(SCRIPT_DIR, "webshop_interaction_config.yaml")
REWARD_FN = os.path.join(SCRIPT_DIR, "webshop_reward.py")

cmd = [
    sys.executable, "-m", "verl.trainer.main_ppo",
    # Algorithm -- Dr. GRPO (no ref model, token-level normalization)
    "algorithm.adv_estimator=grpo",
    "algorithm.use_kl_in_reward=False",
    "algorithm.norm_adv_by_std_in_grpo=False",
    "algorithm.gamma=0.95",
    # Data
    f"data.train_files={os.path.expanduser('~/data/verl-agent/text/train.parquet')}",
    f"data.val_files={os.path.expanduser('~/data/verl-agent/text/test.parquet')}",
    f"data.train_batch_size={TRAIN_DATA_SIZE}",
    f"data.val_batch_size={VAL_DATA_SIZE}",
    "data.max_prompt_length=2048",
    "data.max_response_length=4096",
    "data.filter_overlong_prompts=True",
    "data.truncation=error",
    "data.return_raw_chat=True",
    "data.seed=42",
    # Model
    "actor_rollout_ref.model.path=Qwen/Qwen3-4B",
    "actor_rollout_ref.model.use_remove_padding=True",
    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
    # Actor -- Dr. GRPO: no KL loss, no ref model
    "actor_rollout_ref.actor.optim.lr=1e-6",
    f"actor_rollout_ref.actor.ppo_mini_batch_size={TRAIN_DATA_SIZE}",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.actor.use_kl_loss=False",
    "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm",
    "actor_rollout_ref.actor.loss_scale_factor=4096",
    "actor_rollout_ref.actor.entropy_coeff=0",
    "actor_rollout_ref.actor.fsdp_config.param_offload=False",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
    # Rollout (vLLM) -- TP=2 for memory headroom
    "actor_rollout_ref.rollout.name=vllm",
    "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
    "actor_rollout_ref.rollout.max_model_len=8192",
    "actor_rollout_ref.rollout.load_format=safetensors",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2",
    "actor_rollout_ref.rollout.enable_chunked_prefill=False",
    "actor_rollout_ref.rollout.enforce_eager=False",
    "actor_rollout_ref.rollout.free_cache_engine=False",
    # Validation sampling
    "actor_rollout_ref.rollout.val_kwargs.temperature=0.4",
    "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
    # Agent loop -- tool_agent with WebShop interaction
    "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
    f"actor_rollout_ref.rollout.multi_turn.interaction_config_path={INTERACTION_CONFIG}",
    "actor_rollout_ref.rollout.multi_turn.max_user_turns=15",
    # Group size for GRPO advantage estimation
    f"actor_rollout_ref.rollout.n={GROUP_SIZE}",
    # Ref model -- disabled for Dr. GRPO
    "actor_rollout_ref.ref.fsdp_config.param_offload=True",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2",
    # Reward -- custom function that reads turn_scores from interaction
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
    "trainer.project_name=verl_agent_webshop",
    "trainer.experiment_name=drgrpo_qwen3_4b",
]

env = os.environ.copy()
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
env["PYTHONPATH"] = SCRIPT_DIR + ":" + PROJECT_ROOT + ":" + env.get("PYTHONPATH", "")
# Override caches if inherited paths are not writable
hf_cache_dir = env.get("HF_HOME", "")
if hf_cache_dir and not os.access(os.path.dirname(hf_cache_dir), os.W_OK):
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface")
    env["HF_HOME"] = hf_cache_dir
    env["TRANSFORMERS_CACHE"] = hf_cache_dir
vllm_cache_dir = env.get("VLLM_CACHE_ROOT", "")
if vllm_cache_dir and not os.access(os.path.dirname(vllm_cache_dir), os.W_OK):
    env["VLLM_CACHE_ROOT"] = os.path.expanduser("~/.cache/vllm")
# Java for pyserini (WebShop search engine)
if "JAVA_HOME" not in env:
    env["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-17.0.18.0.8-2.el9.x86_64"
env["TOKENIZERS_PARALLELISM"] = "true"
env["NCCL_DEBUG"] = "WARN"
env["VLLM_LOGGING_LEVEL"] = "WARN"
env["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

print(f"Launching verl WebShop Dr. GRPO training")
print(f"  Model: Qwen3-4B | GPUs: {env['CUDA_VISIBLE_DEVICES']} | TP=2")
print(f"  Dr. GRPO: no ref model, token-level normalization")
print(f"  Batch: {TRAIN_DATA_SIZE} x {GROUP_SIZE} group = {TRAIN_DATA_SIZE * GROUP_SIZE} rollouts/step")
print(f"  Epochs: 250 | Response budget: 4096 | Max steps: 15 | gamma: 0.95")

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
