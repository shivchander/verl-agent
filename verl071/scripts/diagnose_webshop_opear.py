"""Diagnostic: run 1 O-PEaR step on WebShop with full response logging.

Reconstructs exactly what happens during training:
1. Run 4 rollouts (1 task x group_size=4) through WebShop
2. Show the model's actual responses
3. Call the O-PEaR guide to generate contrastive pairs
4. Print compliant vs violating responses side by side
5. Compute and display the log-prob gap

Usage (from repo root):
    CUDA_VISIBLE_DEVICES=0,1 python verl071/scripts/diagnose_webshop_opear.py
"""

import os
import sys
import json
import time

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERL071_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(VERL071_DIR)
WEBSHOP_DIR = os.path.join(PROJECT_ROOT, "agent_system", "environments", "env_package", "webshop", "webshop")

sys.path.insert(0, VERL071_DIR)
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Fix HF/vLLM cache if inherited path is not writable
for var in ("HF_HOME", "TRANSFORMERS_CACHE", "VLLM_CACHE_ROOT"):
    val = os.environ.get(var, "")
    if val and not os.access(os.path.dirname(val), os.W_OK):
        if "vllm" in var.lower():
            os.environ[var] = os.path.expanduser("~/.cache/vllm")
        else:
            os.environ[var] = os.path.expanduser("~/.cache/huggingface")

# vLLM first, then WebShop (CUDA fork issue)
from vllm import LLM, SamplingParams

if os.path.isdir(WEBSHOP_DIR):
    sys.path.insert(0, WEBSHOP_DIR)
import gym
from web_agent_site.envs import WebAgentTextEnv

from webshop_interaction import (
    extract_action, extract_task, format_obs,
    format_available_actions, format_goal_facts,
    WEBSHOP_TEMPLATE_NO_HIS, WEBSHOP_TEMPLATE,
    MAX_OBS_CHARS,
)
from opear.prompts import build_guide_prompt

HISTORY_LENGTH = 2

def build_prompt(task, obs_text, avail_actions, history, step_count):
    if step_count == 0 or HISTORY_LENGTH <= 0:
        return WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=task,
            current_observation=obs_text,
            available_actions=avail_actions,
        )
    recent = history[-HISTORY_LENGTH:]
    start_idx = max(0, len(history) - HISTORY_LENGTH)
    lines = [f"[Observation {start_idx+j+1}: '{h_obs}', Action {start_idx+j+1}: '{h_act}']"
             for j, (h_obs, h_act) in enumerate(recent)]
    prompt = WEBSHOP_TEMPLATE.format(
        task_description=task, step_count=step_count,
        history_length=len(recent), action_history="\n".join(lines),
        current_step=step_count + 1, current_observation=obs_text,
        available_actions=avail_actions,
    )
    if len(prompt) > MAX_OBS_CHARS:
        prompt = WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=task, current_observation=obs_text,
            available_actions=avail_actions,
        )
    return prompt

# ---- Config ----
MODEL = os.environ.get("DIAG_MODEL", "Qwen/Qwen3-4B")
TP = int(os.environ.get("DIAG_TP", "2"))
GOAL_IDX = int(os.environ.get("DIAG_GOAL", "500"))  # first train goal
MAX_STEPS = 5  # just 5 steps to see the pattern
TEMPERATURE = 0.4


def run_episode_verbose(llm, sp, env, goal_idx):
    """Run one episode and return full trajectory."""
    obs, info = env.reset(session=goal_idx)
    info = dict(info or {})
    info["available_actions"] = env.get_available_actions()

    task = extract_task(obs)
    obs_text = format_obs(obs, task)
    history = []

    # Get privileged facts
    goal = env.server.goals[goal_idx]
    facts = format_goal_facts(goal)

    print(f"\n{'='*80}")
    print(f"TASK: {task}")
    print(f"\nPRIVILEGED FACTS (used by O-PEaR guide, NOT seen by model):")
    print(f"  {facts}")
    print(f"{'='*80}")

    trajectory_turns = []

    for step in range(MAX_STEPS):
        avail = format_available_actions(info["available_actions"])
        prompt = build_prompt(task, obs_text, avail, history, step)

        messages = [{"role": "user", "content": prompt}]
        outputs = llm.chat(messages, sampling_params=sp)
        response = outputs[0].outputs[0].text
        finish = outputs[0].outputs[0].finish_reason

        action_result = extract_action(response)
        action = action_result[0] if isinstance(action_result, tuple) else action_result

        print(f"\n--- Step {step} (finish={finish}, {len(response)} chars) ---")
        print(f"OBSERVATION: {obs_text[:200]}...")
        print(f"RESPONSE:\n{response[:600]}")
        if len(response) > 600:
            print(f"  ... ({len(response) - 600} more chars)")
        print(f"EXTRACTED ACTION: {action}")

        trajectory_turns.append({
            "step": step,
            "observation": obs_text,
            "response": response,
            "action": action,
            "finish_reason": finish,
        })

        history.append((obs_text, action))
        obs, reward, done, info = env.step(action)
        info = dict(info or {})
        if not done:
            info["available_actions"] = env.get_available_actions()
        obs_text = format_obs(obs, task)

        if done:
            print(f"\n*** DONE! reward={reward} ***")
            break

    return {
        "task": task,
        "facts": facts,
        "turns": trajectory_turns,
        "reward": reward if done else 0.0,
    }


def demo_contrastive_pairs(trajectory):
    """Show what O-PEaR guide would generate for this trajectory."""
    facts = trajectory["facts"]
    turns = trajectory["turns"]

    # Build the trajectory text (assistant turns joined)
    assistant_segments = [t["response"] for t in turns]

    print(f"\n{'='*80}")
    print("O-PEaR CONTRASTIVE PAIR GENERATION")
    print(f"{'='*80}")

    # Show what goes to the guide model
    for mode in ["compliant", "violating"]:
        prompt = build_guide_prompt(
            facts=facts,
            assistant_segments=assistant_segments,
            mode=mode,
        )
        print(f"\n--- {mode.upper()} GUIDE PROMPT ---")
        print(f"System: {prompt[0]['content'][:300]}...")
        print(f"User: {prompt[1]['content'][:500]}...")

    # Actually call the guide if OPENAI_API_KEY is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        from openai import OpenAI
        client = OpenAI()
        guide_model = "gpt-4.1-nano"

        for mode in ["compliant", "violating"]:
            prompt = build_guide_prompt(
                facts=facts,
                assistant_segments=assistant_segments,
                mode=mode,
            )
            print(f"\n--- {mode.upper()} GUIDE RESPONSE ---")
            try:
                resp = client.chat.completions.create(
                    model=guide_model,
                    messages=prompt,
                    max_tokens=2048,
                    temperature=0.7,
                )
                text = resp.choices[0].message.content
                print(text[:800])
                if len(text) > 800:
                    print(f"  ... ({len(text) - 800} more chars)")
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        print("\n[OPENAI_API_KEY not set -- skipping actual guide calls]")


def main():
    print(f"Loading model: {MODEL} (TP={TP})")
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TP,
        gpu_memory_utilization=0.5,
        max_model_len=8192,
        trust_remote_code=True,
    )
    sp = SamplingParams(temperature=TEMPERATURE, max_tokens=2048)

    print("Loading WebShop environment...")
    env = gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=None)
    print(f"WebShop loaded: {len(env.server.goals)} goals")

    # Run one episode
    traj = run_episode_verbose(llm, sp, env, GOAL_IDX)

    # Show contrastive pair generation
    demo_contrastive_pairs(traj)

    env.close()

    # Save full output
    output_path = "/tmp/opear_webshop_diagnostic.json"
    with open(output_path, "w") as f:
        json.dump(traj, f, indent=2, default=str)
    print(f"\nFull trajectory saved to {output_path}")


if __name__ == "__main__":
    main()
