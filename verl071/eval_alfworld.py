"""ALFWorld per-task evaluation for trained checkpoints.

Runs games in parallel using multiprocessing for high throughput.
Saves full trajectories (think/action/observation per turn) alongside
per-task success rates and turn counts.

Three eval profiles:
  --profile short   : 512 per turn, 16k total context
  --profile medium  : 2048 per turn, 16k total context
  --profile long    : 2048 per turn, 40k total context (matches training)

Usage:
    # Quick eval (unseen, 1 seed, medium profile)
    python verl071/eval_alfworld.py \
        --checkpoint /path/to/hf_model \
        --split unseen --seed 123 --gpu 0

    # Full eval with training-matched settings
    python verl071/eval_alfworld.py \
        --checkpoint /path/to/hf_model \
        --full --profile long --gpu 0

    # With existing vLLM server
    python verl071/eval_alfworld.py \
        --checkpoint /path/to/hf_model \
        --full --no-server --api-base http://localhost:8100/v1
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

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
}

PROFILES = {
    "short":  {"max_tokens_per_turn": 512,  "max_model_len": 16384},
    "medium": {"max_tokens_per_turn": 2048, "max_model_len": 16384},
    "long":   {"max_tokens_per_turn": 2048, "max_model_len": 40960},
}


def extract_action(text: str) -> str:
    t = text.lower()
    s, e = t.find("<action>"), t.find("</action>")
    if s == -1 or e == -1:
        return t[-30:]
    return t[s + 8:e].strip()


def format_admissible(actions):
    return "\n ".join(f"'{s}'" for s in actions if s != "help")


def extract_task(obs: str) -> str:
    marker = "Your task is to: "
    idx = obs.find(marker)
    return obs[idx + len(marker):].strip() if idx != -1 else ""


def get_task_type(gamefile: str) -> str:
    folder = gamefile.split("/")[-3] if "/" in gamefile else gamefile
    for tt in TASK_TYPES:
        if folder.startswith(tt):
            return tt
    return "unknown"


def build_prompt(obs, admissible, task, history, step_count, history_length=2):
    adm_str = format_admissible(admissible)
    if step_count == 0 or history_length <= 0:
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=obs, admissible_actions=adm_str)

    recent = history[-history_length:]
    start_idx = max(0, len(history) - history_length)
    lines = [f"[Observation {start_idx+j+1}: '{h_obs}', Action {start_idx+j+1}: '{h_act}']"
             for j, (h_obs, h_act) in enumerate(recent)]
    return ALFWORLD_TEMPLATE.format(
        task_description=task, step_count=step_count,
        history_length=len(recent), action_history="\n".join(lines),
        current_step=step_count + 1, current_observation=obs,
        admissible_actions=adm_str)


def run_episode(client, model_name, game_env, game_idx, max_steps=50,
                temperature=0.4, max_tokens_per_turn=2048, total_games=0):
    """Run a single ALFWorld episode with full conversation history.

    Returns result dict with won, steps, task_type, and full trajectory.
    """
    obs, info = game_env.reset()
    obs_text = obs[0] if isinstance(obs, (list, tuple)) else obs

    admissible = info.get("admissible_commands", [])
    if isinstance(admissible, (list, tuple)) and admissible and isinstance(admissible[0], list):
        admissible = admissible[0]

    gamefile = info.get("extra.gamefile", [""])[0] if isinstance(info.get("extra.gamefile"), list) else info.get("extra.gamefile", "")
    task_type = get_task_type(gamefile)
    task = extract_task(obs_text)
    history = []
    trajectory = []

    # Record initial observation
    trajectory.append({
        "step": 0, "type": "initial_observation",
        "observation": obs_text, "admissible_commands": admissible,
    })

    # First model response (matches training's placeholder first turn)
    messages = [
        {"role": "user", "content": "You are starting a new task in the ALFRED Embodied Environment. Please wait for the observation."}
    ]
    try:
        response = client.chat.completions.create(
            model=model_name, messages=messages,
            max_tokens=64, temperature=temperature)
        first_response = response.choices[0].message.content or ""
    except Exception:
        first_response = "I will wait for the observation."
    messages.append({"role": "assistant", "content": first_response})

    # Initial observation prompt
    initial_prompt = build_prompt(obs_text, admissible, task, history, 0)
    messages.append({"role": "user", "content": initial_prompt})

    for step in range(max_steps):
        # Model generates with full conversation context
        try:
            response = client.chat.completions.create(
                model=model_name, messages=messages,
                max_tokens=max_tokens_per_turn, temperature=temperature)
            raw_action = response.choices[0].message.content or ""
        except Exception as e:
            trajectory.append({
                "step": step + 1, "type": "api_error", "error": str(e),
            })
            raw_action = ""

        messages.append({"role": "assistant", "content": raw_action})
        action = extract_action(raw_action)

        # Step environment
        try:
            next_obs, scores, dones, next_info = game_env.step([action])
        except Exception as e:
            trajectory.append({
                "step": step + 1, "type": "env_error",
                "raw_response": raw_action, "parsed_action": action,
                "error": str(e),
            })
            break

        next_obs_text = next_obs[0] if isinstance(next_obs, (list, tuple)) else str(next_obs)
        done = dones[0] if isinstance(dones, (list, tuple)) else bool(dones)

        for k in list(next_info.keys()):
            if isinstance(next_info[k], (list, tuple)) and len(next_info[k]) == 1:
                next_info[k] = next_info[k][0]

        won = bool(next_info.get("won", False))

        trajectory.append({
            "step": step + 1, "type": "turn",
            "raw_response": raw_action,
            "parsed_action": action,
            "observation": next_obs_text,
            "won": won,
        })

        if done or won:
            return {
                "won": won, "steps": step + 1, "task_type": task_type,
                "game_idx": game_idx, "trajectory": trajectory,
            }

        # Build next observation and add to conversation
        history.append((obs_text, action))
        obs_text = next_obs_text

        admissible = next_info.get("admissible_commands", admissible)
        if isinstance(admissible, (list, tuple)) and admissible and isinstance(admissible[0], list):
            admissible = admissible[0]

        next_prompt = build_prompt(obs_text, admissible, task, history, step + 1)
        messages.append({"role": "user", "content": next_prompt})

    return {
        "won": False, "steps": max_steps, "task_type": task_type,
        "game_idx": game_idx, "trajectory": trajectory,
    }


def _run_game_worker(args):
    """Worker function for multiprocessing — runs one game in its own process."""
    (game_idx, game_file, alf_config_path, alf_split, seed, max_steps,
     temperature, max_tokens_per_turn, model_name, api_base_url, total_games) = args

    import textworld
    import textworld.gym
    import yaml
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    from openai import OpenAI

    with open(alf_config_path) as f:
        config = yaml.safe_load(f)

    domain_randomization = config["env"]["domain_randomization"] if alf_split == "train" else False
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])

    alfred_demangler = AlfredDemangler(shuffle=domain_randomization)
    wrappers = [alfred_demangler, AlfredInfos]
    env_id = textworld.gym.register_games(
        [game_file], request_infos, batch_size=1, asynchronous=False,
        max_episode_steps=max_steps, wrappers=wrappers)
    env = textworld.gym.make(env_id)
    env.seed(seed)

    client = OpenAI(base_url=api_base_url, api_key="dummy")

    try:
        result = run_episode(
            client, model_name, env, game_idx,
            max_steps=max_steps, temperature=temperature,
            max_tokens_per_turn=max_tokens_per_turn,
            total_games=total_games)
        status = "PASS" if result["won"] else "FAIL"
        print(f"  [{game_idx+1:3d}/{total_games}] {status} in {result['steps']:2d} steps — {result['task_type']}")
        return result
    except Exception as e:
        print(f"  [{game_idx+1:3d}/{total_games}] ERROR — {e}")
        return {
            "won": False, "steps": 0, "task_type": get_task_type(game_file),
            "game_idx": game_idx, "trajectory": [{"step": 0, "type": "error", "error": str(e)}],
        }
    finally:
        env.close()


def evaluate_split(client, model_name, split_name, alf_config_path, seed=123,
                   max_steps=50, temperature=0.4, max_tokens_per_turn=2048,
                   max_concurrent=16):
    """Evaluate all games in a split with parallel execution."""
    from alfworld.agents.environment import get_environment
    import yaml
    import textworld

    with open(alf_config_path) as f:
        config = yaml.safe_load(f)

    alf_split = SPLIT_MAP.get(split_name, split_name)
    tw_env = get_environment("AlfredTWEnv")(config, train_eval=alf_split)
    num_games = tw_env.num_games

    print(f"\nEvaluating {num_games} games on {alf_split} "
          f"(seed={seed}, concurrent={max_concurrent}, "
          f"max_tokens/turn={max_tokens_per_turn})")
    print("-" * 60)

    game_files = list(tw_env.game_files)
    game_args = [
        (i, game_files[i], alf_config_path, alf_split, seed, max_steps,
         temperature, max_tokens_per_turn, model_name,
         f"{client.base_url}", num_games)
        for i in range(num_games)
    ]

    t0 = time.time()

    with Pool(processes=max_concurrent) as pool:
        results = pool.map(_run_game_worker, game_args)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s ({elapsed/num_games:.1f}s/game)")

    # Aggregate
    per_type = defaultdict(lambda: {"total": 0, "won": 0, "total_steps": 0, "won_steps": 0})
    for r in results:
        if r:
            tt = r["task_type"]
            per_type[tt]["total"] += 1
            per_type[tt]["total_steps"] += r["steps"]
            if r["won"]:
                per_type[tt]["won"] += 1
                per_type[tt]["won_steps"] += r["steps"]

    overall_won = sum(1 for r in results if r and r["won"])
    overall_steps = sum(r["steps"] for r in results if r)
    won_steps = sum(r["steps"] for r in results if r and r["won"])

    return {
        "split": alf_split,
        "seed": seed,
        "elapsed_s": elapsed,
        "max_tokens_per_turn": max_tokens_per_turn,
        "overall": {
            "won": overall_won, "total": num_games,
            "rate": overall_won / num_games if num_games else 0,
            "avg_steps": overall_steps / num_games if num_games else 0,
            "avg_steps_won": won_steps / overall_won if overall_won else 0,
        },
        "per_type": {
            tt: {
                "won": d["won"], "total": d["total"],
                "rate": d["won"] / d["total"] if d["total"] else 0,
                "avg_steps": d["total_steps"] / d["total"] if d["total"] else 0,
                "avg_steps_won": d["won_steps"] / d["won"] if d["won"] else 0,
            }
            for tt, d in per_type.items()
        },
        "trajectories": results,
    }


def print_results(results_list):
    """Print formatted results table."""
    per_type_agg = defaultdict(lambda: {"won": 0, "total": 0, "total_steps": 0, "won_steps": 0})
    overall_agg = {"won": 0, "total": 0, "total_steps": 0, "won_steps": 0}

    for r in results_list:
        overall_agg["won"] += r["overall"]["won"]
        overall_agg["total"] += r["overall"]["total"]
        for tt, d in r["per_type"].items():
            per_type_agg[tt]["won"] += d["won"]
            per_type_agg[tt]["total"] += d["total"]
            per_type_agg[tt]["total_steps"] += d["avg_steps"] * d["total"]
            if d["won"] > 0:
                per_type_agg[tt]["won_steps"] += d["avg_steps_won"] * d["won"]

    split = results_list[0]["split"]
    seeds = [r["seed"] for r in results_list]
    total_time = sum(r["elapsed_s"] for r in results_list)
    max_tok = results_list[0]["max_tokens_per_turn"]

    print(f"\n{'='*75}")
    print(f"Results: {split} | Seeds: {seeds} | "
          f"max_tokens/turn: {max_tok} | Time: {total_time:.0f}s")
    print(f"{'='*75}")
    print(f"{'Task Type':<40} {'Won':>5} {'Total':>6} {'Rate':>7} {'AvgSteps':>9} {'WonSteps':>9}")
    print(f"{'-'*75}")

    for tt in TASK_TYPES:
        if tt in per_type_agg:
            d = per_type_agg[tt]
            rate = d["won"] / d["total"] if d["total"] else 0
            avg = d["total_steps"] / d["total"] if d["total"] else 0
            avg_w = d["won_steps"] / d["won"] if d["won"] else 0
            print(f"{tt:<40} {d['won']:>5} {d['total']:>6} {rate:>6.1%} {avg:>8.1f} {avg_w:>8.1f}")

    o = overall_agg
    rate = o["won"] / o["total"] if o["total"] else 0
    print(f"{'-'*75}")
    print(f"{'OVERALL':<40} {o['won']:>5} {o['total']:>6} {rate:>6.1%}")
    print()


def main():
    parser = argparse.ArgumentParser(description="ALFWorld per-task evaluation")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint (full HF model or LoRA adapter)")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B",
                        help="Base model for LoRA mode (ignored for full models)")
    parser.add_argument("--lora", action="store_true",
                        help="Checkpoint is a LoRA adapter (default: full model)")
    parser.add_argument("--split", default="unseen", choices=["seen", "unseen"],
                        help="Eval split (default: unseen)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--full", action="store_true",
                        help="Run full eval: both splits, 3 seeds")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--profile", default="medium",
                        choices=["short", "medium", "long"],
                        help="Eval profile: short (512/16k), medium (2k/16k), long (2k/40k)")
    parser.add_argument("--max-tokens-per-turn", type=int, default=None,
                        help="Override per-turn token limit (default: from profile)")
    parser.add_argument("--max-concurrent", type=int, default=16,
                        help="Max parallel games (default: 16)")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--lora-name", default="alfworld-adapter")
    parser.add_argument("--alf-config",
                        default=os.path.join(os.path.dirname(__file__), "..",
                                             "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"))
    parser.add_argument("--output", default=None,
                        help="Save results + trajectories to JSON file")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", default=None,
                        help="CUDA_VISIBLE_DEVICES for vLLM server")
    parser.add_argument("--no-server", action="store_true",
                        help="Don't start vLLM server (assume already running)")
    args = parser.parse_args()

    # Resolve profile
    profile = PROFILES[args.profile]
    max_tokens_per_turn = args.max_tokens_per_turn or profile["max_tokens_per_turn"]
    max_model_len = profile["max_model_len"]

    # Determine model name for API
    model_name = args.lora_name if args.lora else args.checkpoint

    # Start vLLM server if needed
    server_proc = None
    if not args.no_server:
        import subprocess
        if args.lora:
            vllm_cmd = [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", args.base_model,
                "--enable-lora",
                "--lora-modules", f"{args.lora_name}={args.checkpoint}",
                "--max-lora-rank", "128",
                "--port", str(args.port),
                "--dtype", "bfloat16",
                "--max-model-len", str(max_model_len),
                "--gpu-memory-utilization", "0.9",
                "--tensor-parallel-size", str(args.tp),
            ]
        else:
            vllm_cmd = [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", args.checkpoint,
                "--port", str(args.port),
                "--dtype", "bfloat16",
                "--max-model-len", str(max_model_len),
                "--gpu-memory-utilization", "0.9",
                "--tensor-parallel-size", str(args.tp),
            ]

        env = os.environ.copy()
        if args.gpu:
            env["CUDA_VISIBLE_DEVICES"] = args.gpu

        print(f"Starting vLLM server on port {args.port} "
              f"(max_model_len={max_model_len})...")
        server_proc = subprocess.Popen(vllm_cmd, env=env,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        import urllib.request
        api_base = f"http://localhost:{args.port}/v1"
        for i in range(180):
            try:
                urllib.request.urlopen(f"{api_base}/models", timeout=2)
                print(f"vLLM server ready after {i+1}s")
                break
            except Exception:
                time.sleep(1)
        else:
            print("ERROR: vLLM server did not start within 180s")
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

        print(f"\nProfile: {args.profile} "
              f"(max_tokens/turn={max_tokens_per_turn}, "
              f"max_model_len={max_model_len})")

        all_results = {}
        for split in splits:
            split_results = []
            for seed in seeds:
                r = evaluate_split(
                    client, model_name, split, args.alf_config,
                    seed=seed, max_steps=args.max_steps,
                    temperature=args.temperature,
                    max_tokens_per_turn=max_tokens_per_turn,
                    max_concurrent=args.max_concurrent)
                split_results.append(r)
            print_results(split_results)
            all_results[split] = split_results

        if args.output:
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"Results + trajectories saved to {args.output}")

    finally:
        if server_proc:
            print("Shutting down vLLM server...")
            server_proc.kill()
            server_proc.wait()


if __name__ == "__main__":
    main()
