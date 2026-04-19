"""ALFWorld per-task evaluation for trained checkpoints.

Loads a Qwen3-4B + LoRA adapter via vLLM and evaluates on ALFWorld
valid_seen (140 games) and valid_unseen (134 games) splits, reporting
per-task-type and overall success rates.

Usage:
    python verl071/eval_alfworld.py \
        --checkpoint /path/to/lora_adapter \
        --split unseen \
        --seed 123

    # Full eval (both splits, 3 seeds):
    python verl071/eval_alfworld.py \
        --checkpoint /path/to/lora_adapter \
        --full
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

# --------------- Prompt Templates (same as training) --------------- #

ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

TASK_TYPES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]

SPLIT_MAP = {
    "seen": "eval_in_distribution",
    "unseen": "eval_out_of_distribution",
    "eval_in_distribution": "eval_in_distribution",
    "eval_out_of_distribution": "eval_out_of_distribution",
}


def extract_action(text: str) -> str:
    """Extract action from <action>...</action> tags."""
    text_lower = text.lower()
    start = text_lower.find("<action>")
    end = text_lower.find("</action>")
    if start == -1 or end == -1:
        return text_lower[-30:]
    return text_lower[start + 8:end].strip()


def format_admissible(actions):
    return "\n ".join(f"'{s}'" for s in actions if s != "help")


def extract_task(obs: str) -> str:
    marker = "Your task is to: "
    idx = obs.find(marker)
    if idx != -1:
        return obs[idx + len(marker):].strip()
    return ""


def get_task_type(gamefile: str) -> str:
    """Extract task type from gamefile path."""
    folder = gamefile.split("/")[-3] if "/" in gamefile else gamefile
    for tt in TASK_TYPES:
        if folder.startswith(tt):
            return tt
    return "unknown"


def build_prompt(obs, admissible, task, history, step_count, history_length=2):
    """Build observation prompt matching training format."""
    adm_str = format_admissible(admissible)
    if step_count == 0 or history_length <= 0:
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=obs,
            admissible_actions=adm_str,
        )
    recent = history[-history_length:]
    start_idx = max(0, len(history) - history_length)
    lines = []
    for j, (h_obs, h_action) in enumerate(recent):
        sn = start_idx + j + 1
        lines.append(f"[Observation {sn}: '{h_obs}', Action {sn}: '{h_action}']")
    return ALFWORLD_TEMPLATE.format(
        task_description=task,
        step_count=step_count,
        history_length=len(recent),
        action_history="\n".join(lines),
        current_step=step_count + 1,
        current_observation=obs,
        admissible_actions=adm_str,
    )


def run_episode(client, model_name, game_env, max_steps=50, history_length=2,
                temperature=0.4, lora_name=None):
    """Run a single ALFWorld episode, return (won, steps, task_type)."""
    obs, info = game_env.reset()
    obs_text = obs[0] if isinstance(obs, (list, tuple)) else obs

    admissible = info.get("admissible_commands", [])
    if isinstance(admissible, (list, tuple)) and admissible and isinstance(admissible[0], list):
        admissible = admissible[0]

    gamefile = info.get("extra.gamefile", [""])[0] if isinstance(info.get("extra.gamefile"), list) else info.get("extra.gamefile", "")
    task_type = get_task_type(gamefile)
    task = extract_task(obs_text)
    history = []

    for step in range(max_steps):
        prompt = build_prompt(obs_text, admissible, task, history, step, history_length)

        messages = [{"role": "user", "content": prompt}]

        extra_body = {}
        if lora_name:
            extra_body["model"] = lora_name

        try:
            response = client.chat.completions.create(
                model=lora_name or model_name,
                messages=messages,
                max_tokens=512,
                temperature=temperature,
                top_p=1.0,
            )
            raw_action = response.choices[0].message.content or ""
        except Exception as e:
            print(f"    API error at step {step}: {e}")
            raw_action = ""

        action = extract_action(raw_action)

        # Step env
        try:
            next_obs, scores, dones, next_info = game_env.step([action])
        except Exception as e:
            print(f"    Env error at step {step}: {e}")
            break

        next_obs_text = next_obs[0] if isinstance(next_obs, (list, tuple)) else str(next_obs)
        done = dones[0] if isinstance(dones, (list, tuple)) else bool(dones)

        for k in list(next_info.keys()):
            if isinstance(next_info[k], (list, tuple)) and len(next_info[k]) == 1:
                next_info[k] = next_info[k][0]

        won = bool(next_info.get("won", False))

        history.append((obs_text, action))
        obs_text = next_obs_text

        admissible = next_info.get("admissible_commands", admissible)
        if isinstance(admissible, (list, tuple)) and admissible and isinstance(admissible[0], list):
            admissible = admissible[0]

        if done or won:
            return won, step + 1, task_type

    return False, max_steps, task_type


def evaluate_split(client, model_name, split_name, alf_config_path, seed=123,
                   max_steps=50, temperature=0.4, lora_name=None):
    """Evaluate all games in a split, return per-task results."""
    from alfworld.agents.environment import get_environment
    import yaml

    with open(alf_config_path) as f:
        config = yaml.safe_load(f)

    alf_split = SPLIT_MAP.get(split_name, split_name)
    tw_env = get_environment("AlfredTWEnv")(config, train_eval=alf_split)
    game_env = tw_env.init_env(batch_size=1)
    game_env.seed(seed)

    num_games = tw_env.num_games
    results = []

    print(f"\nEvaluating {num_games} games on {alf_split} (seed={seed})")
    print("-" * 60)

    for i in range(num_games):
        won, steps, task_type = run_episode(
            client, model_name, game_env,
            max_steps=max_steps, temperature=temperature,
            lora_name=lora_name,
        )
        results.append({"won": won, "steps": steps, "task_type": task_type})
        status = "PASS" if won else "FAIL"
        print(f"  [{i+1:3d}/{num_games}] {status} in {steps:2d} steps — {task_type}")

    game_env.close()

    # Aggregate
    per_type = defaultdict(lambda: {"total": 0, "won": 0, "steps": []})
    for r in results:
        tt = r["task_type"]
        per_type[tt]["total"] += 1
        if r["won"]:
            per_type[tt]["won"] += 1
        per_type[tt]["steps"].append(r["steps"])

    overall_won = sum(r["won"] for r in results)
    overall_total = len(results)

    return {
        "split": alf_split,
        "seed": seed,
        "overall": {"won": overall_won, "total": overall_total,
                     "rate": overall_won / overall_total if overall_total else 0},
        "per_type": {
            tt: {"won": d["won"], "total": d["total"],
                 "rate": d["won"] / d["total"] if d["total"] else 0,
                 "avg_steps": sum(d["steps"]) / len(d["steps"]) if d["steps"] else 0}
            for tt, d in per_type.items()
        },
    }


def print_results(results_list):
    """Print formatted results table."""
    # Aggregate across seeds
    per_type_agg = defaultdict(lambda: {"won": 0, "total": 0})
    overall_agg = {"won": 0, "total": 0}

    for r in results_list:
        overall_agg["won"] += r["overall"]["won"]
        overall_agg["total"] += r["overall"]["total"]
        for tt, d in r["per_type"].items():
            per_type_agg[tt]["won"] += d["won"]
            per_type_agg[tt]["total"] += d["total"]

    split = results_list[0]["split"]
    seeds = [r["seed"] for r in results_list]

    print(f"\n{'='*60}")
    print(f"Results: {split} | Seeds: {seeds}")
    print(f"{'='*60}")
    print(f"{'Task Type':<40} {'Won':>5} {'Total':>6} {'Rate':>7}")
    print(f"{'-'*60}")

    for tt in TASK_TYPES:
        if tt in per_type_agg:
            d = per_type_agg[tt]
            rate = d["won"] / d["total"] if d["total"] else 0
            print(f"{tt:<40} {d['won']:>5} {d['total']:>6} {rate:>6.1%}")

    rate = overall_agg["won"] / overall_agg["total"] if overall_agg["total"] else 0
    print(f"{'-'*60}")
    print(f"{'OVERALL':<40} {overall_agg['won']:>5} {overall_agg['total']:>6} {rate:>6.1%}")
    print()


def main():
    parser = argparse.ArgumentParser(description="ALFWorld per-task evaluation")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to LoRA adapter checkpoint")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B",
                        help="Base model name (default: Qwen/Qwen3-4B)")
    parser.add_argument("--split", default="unseen", choices=["seen", "unseen"],
                        help="Eval split (default: unseen)")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed (default: 123)")
    parser.add_argument("--full", action="store_true",
                        help="Run full eval: both splits, 3 seeds")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max steps per episode (default: 50)")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Sampling temperature (default: 0.4)")
    parser.add_argument("--api-base", default="http://localhost:8000/v1",
                        help="vLLM API base URL")
    parser.add_argument("--lora-name", default="alfworld-adapter",
                        help="LoRA adapter name for vLLM (default: alfworld-adapter)")
    parser.add_argument("--alf-config",
                        default=os.path.join(os.path.dirname(__file__), "..",
                                             "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"),
                        help="ALFWorld config path")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel size for vLLM (default: 1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="vLLM server port (default: 8000)")
    parser.add_argument("--gpu", default=None,
                        help="CUDA_VISIBLE_DEVICES for vLLM server")
    parser.add_argument("--no-server", action="store_true",
                        help="Don't start vLLM server (assume already running)")
    args = parser.parse_args()

    # Start vLLM server if needed
    server_proc = None
    if not args.no_server:
        import subprocess
        vllm_cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", args.base_model,
            "--enable-lora",
            "--lora-modules", f"{args.lora_name}={args.checkpoint}",
            "--max-lora-rank", "128",
            "--port", str(args.port),
            "--dtype", "bfloat16",
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.9",
            "--tensor-parallel-size", str(args.tp),
        ]
        env = os.environ.copy()
        if args.gpu:
            env["CUDA_VISIBLE_DEVICES"] = args.gpu

        print(f"Starting vLLM server on port {args.port}...")
        server_proc = subprocess.Popen(
            vllm_cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

        # Wait for server to be ready
        import urllib.request
        api_base = f"http://localhost:{args.port}/v1"
        for i in range(120):
            try:
                urllib.request.urlopen(f"{api_base}/models", timeout=2)
                print(f"vLLM server ready after {i+1}s")
                break
            except Exception:
                time.sleep(1)
        else:
            print("ERROR: vLLM server did not start within 120s")
            server_proc.kill()
            sys.exit(1)
    else:
        api_base = args.api_base

    try:
        from openai import OpenAI
        client = OpenAI(base_url=api_base, api_key="dummy")

        if args.full:
            splits = ["seen", "unseen"]
            seeds = [123, 456, 789]
        else:
            splits = [args.split]
            seeds = [args.seed]

        all_results = {}
        for split in splits:
            split_results = []
            for seed in seeds:
                r = evaluate_split(
                    client, args.base_model, split, args.alf_config,
                    seed=seed, max_steps=args.max_steps,
                    temperature=args.temperature, lora_name=args.lora_name,
                )
                split_results.append(r)
            print_results(split_results)
            all_results[split] = split_results

        if args.output:
            # Convert for JSON serialization
            serializable = {}
            for split, results in all_results.items():
                serializable[split] = results
            with open(args.output, "w") as f:
                json.dump(serializable, f, indent=2, default=str)
            print(f"Results saved to {args.output}")

    finally:
        if server_proc:
            print("Shutting down vLLM server...")
            server_proc.kill()
            server_proc.wait()


if __name__ == "__main__":
    main()
