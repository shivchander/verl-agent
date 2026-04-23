"""ALFWorld evaluation using LEAP's prompting format.

Uses LEAP's prompt template and single-turn conversation structure
with our eval infrastructure (profiles, trajectories, per-task stats).

This enables fair comparison with LEAP-trained checkpoints that expect
the REASON:/ACTION: output format and LEAP's specific prompt structure.

Usage:
    python verl071/eval_alfworld_leap.py \
        --checkpoint /path/to/model_or_adapter \
        --split unseen --seed 123 --gpu 0 --profile medium

    # LoRA adapter
    python verl071/eval_alfworld_leap.py \
        --checkpoint /path/to/lora_adapter --lora --gpu 0
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

# --------------- LEAP Prompt Template --------------- #

LEAP_TEMPLATE = """You are an intelligent assistant named ALFRED in a text-based interactive game called TextWorld. Your objective is to complete the given tasks by reasoning through the information provided and taking appropriate actions.

Your task is the following:
{task}

Below is the history of previous observations and actions:
{observation_action_history}


Given the history of previous observation and action above, a reminder that your task is:
{task}

You are given as input the current observation and the list of possible candidate_actions:
{{
    "observation": "{observation}",
    "candidate_actions": {candidate_actions}
}}


Your goal is to generate the action to take at this time step (chosen from candidate_actions) along with the reason for taking the action.

Please follow these general instructions:
* You MUST choose action from the list of candidate_actions.
* If "observation": "Nothing happens.", it is because you chose an invalid action not from the list of candidate_actions in the previous timestep.
* Oftentimes the task requires you interact with objects not present in your observation. You must search the environment to locate the objective.
* Consult the history of previous observations and actions to see what actions you have tried already so as to not repeat your actions.
* Do NOT repeat the same action as the last action in your observation_action_history. It's going to yield the same result.
* Make sure action is VERBATIM copied from the list of candidate_actions.

You need to generate a response in the following format. Please issue only a single action at a time.
REASON:
Rationale for what action to take next based on the task and previous history. In your reason, consult candidate_actions to precisely state VERBATIM which action you will do.
ACTION:
The action to be taken, chosen ONLY from candidate_actions"""

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


def extract_task(obs: str) -> str:
    m = re.search(r"(?<=\n\nYour task is to: ).*", obs)
    if m:
        return m.group(0).strip()
    marker = "Your task is to: "
    idx = obs.find(marker)
    return obs[idx + len(marker):].strip() if idx != -1 else ""


def get_task_type(gamefile: str) -> str:
    folder = gamefile.split("/")[-3] if "/" in gamefile else gamefile
    for tt in TASK_TYPES:
        if folder.startswith(tt):
            return tt
    return "unknown"


def parse_reason_action(text: str) -> tuple[str, str]:
    """Parse REASON:/ACTION: from model output (LEAP format)."""
    reason_match = re.search(r"REASON:\s*(.*?)\s*ACTION:", text, re.DOTALL)
    action_match = re.search(r"ACTION:\s*([^\n]+)", text)
    reason = reason_match.group(1).strip() if reason_match else text
    action = action_match.group(1).strip() if action_match else ""
    action = action.lower()
    action = re.sub(r'[^a-z0-9 /]', '', action)
    return reason, action


def build_leap_prompt(task, observation, candidate_actions, history):
    """Build LEAP-style prompt with full history."""
    if history:
        hist_str = json.dumps(history, indent=2)
    else:
        hist_str = "[]"

    return LEAP_TEMPLATE.format(
        task=task,
        observation_action_history=hist_str,
        observation=observation,
        candidate_actions=json.dumps(candidate_actions),
    )


def run_episode(client, model_name, game_env, game_idx, max_steps=50,
                temperature=0.3, max_tokens_per_turn=2048, total_games=0):
    """Run a single ALFWorld episode using LEAP's prompting format.

    Key differences from our standard eval:
    - Single-turn prompts (full history baked into one user message)
    - REASON:/ACTION: output format
    - Action string cleaning (lowercase, strip non-alphanumeric)
    - No placeholder first turn
    """
    obs, info = game_env.reset()
    obs_text = obs[0] if isinstance(obs, (list, tuple)) else obs

    admissible = info.get("admissible_commands", [])
    if isinstance(admissible, (list, tuple)) and admissible and isinstance(admissible[0], list):
        admissible = admissible[0]

    gamefile = info.get("extra.gamefile", [""])[0] if isinstance(info.get("extra.gamefile"), list) else info.get("extra.gamefile", "")
    task_type = get_task_type(gamefile)
    task = extract_task(obs_text)
    history = []  # list of {"observation": ..., "action": ...}
    trajectory = []

    trajectory.append({
        "step": 0, "type": "initial_observation",
        "observation": obs_text, "admissible_commands": admissible,
    })

    for step in range(max_steps):
        prompt = build_leap_prompt(task, obs_text, admissible, history)

        # Single-turn: one user message per step
        messages = [{"role": "user", "content": prompt}]

        try:
            response = client.chat.completions.create(
                model=model_name, messages=messages,
                max_tokens=max_tokens_per_turn, temperature=temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            raw_response = response.choices[0].message.content or ""
        except Exception as e:
            trajectory.append({"step": step + 1, "type": "api_error", "error": str(e)})
            raw_response = ""

        reason, action = parse_reason_action(raw_response)

        # Step environment
        try:
            next_obs, scores, dones, next_info = game_env.step([action])
        except Exception as e:
            trajectory.append({
                "step": step + 1, "type": "env_error",
                "raw_response": raw_response, "parsed_action": action,
                "reason": reason, "error": str(e),
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
            "raw_response": raw_response,
            "reason": reason,
            "parsed_action": action,
            "observation": next_obs_text,
            "won": won,
        })

        # Update history (LEAP style: observation + action, no reason)
        history.append({"observation": obs_text, "action": action})

        if done or won:
            return {
                "won": won, "steps": step + 1, "task_type": task_type,
                "game_idx": game_idx, "trajectory": trajectory,
            }

        obs_text = next_obs_text
        admissible = next_info.get("admissible_commands", admissible)
        if isinstance(admissible, (list, tuple)) and admissible and isinstance(admissible[0], list):
            admissible = admissible[0]

    return {
        "won": False, "steps": max_steps, "task_type": task_type,
        "game_idx": game_idx, "trajectory": trajectory,
    }


def _run_game_worker(args):
    """Worker function for multiprocessing."""
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
                   max_steps=50, temperature=0.3, max_tokens_per_turn=2048,
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
          f"max_tokens/turn={max_tokens_per_turn}, temp={temperature})")
    print(f"Prompt format: LEAP (single-turn, REASON:/ACTION:)")
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
        "prompt_format": "leap",
        "temperature": temperature,
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
    per_type_agg = defaultdict(lambda: {"won": 0, "total": 0, "total_steps": 0, "won_steps": 0})
    overall_agg = {"won": 0, "total": 0}

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
    print(f"Results (LEAP format): {split} | Seeds: {seeds} | "
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
    parser = argparse.ArgumentParser(description="ALFWorld eval with LEAP prompting")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--split", default="unseen", choices=["seen", "unseen"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--profile", default="medium", choices=["short", "medium", "long"])
    parser.add_argument("--max-tokens-per-turn", type=int, default=None)
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--lora-name", default="leap-adapter")
    parser.add_argument("--alf-config",
                        default=os.path.join(os.path.dirname(__file__), "..",
                                             "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--no-server", action="store_true")
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    max_tokens_per_turn = args.max_tokens_per_turn or profile["max_tokens_per_turn"]
    max_model_len = profile["max_model_len"]

    model_name = args.lora_name if args.lora else args.checkpoint

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

        print(f"Starting vLLM server on port {args.port} (max_model_len={max_model_len})...")
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
        print(f"Prompt format: LEAP | Temperature: {args.temperature}")

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
