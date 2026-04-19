"""Evaluate a model on WebShop following the official protocol.

Runs 500 test episodes (goals 0-499) in parallel batches, reports:
  - Task Score (avg reward, 0-100 scale)
  - Success Rate % (reward == 1.0)

Usage:
  python eval_webshop.py --model_path Qwen/Qwen3-4B
  python eval_webshop.py --model_path checkpoints/.../huggingface
  python eval_webshop.py --model_path Qwen/Qwen3-4B --batch_size 32
"""

import argparse
import os
import sys
import time

# WebShop path -- imported lazily AFTER vLLM init to avoid CUDA fork issues
WEBSHOP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "agent_system", "environments", "env_package", "webshop", "webshop"
)

from vllm import LLM, SamplingParams

# --------------- Prompt Templates --------------- #

TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

HISTORY_LENGTH = 2
MAX_OBS_CHARS = 13000


def extract_action(text: str) -> str:
    text_lower = text.lower()
    start = text_lower.find("<action>")
    end = text_lower.find("</action>")
    if start == -1 or end == -1:
        return text_lower[-20:]
    return text_lower[start + len("<action>"):end].strip()


def extract_task(obs: str) -> str:
    parts = obs.split(" [SEP] ")
    if len(parts) >= 3 and parts[1] == "Instruction:":
        return parts[2]
    return obs


def format_obs(obs: str, task: str) -> str:
    parts = obs.split(" [SEP] ")
    try:
        idx = parts.index(task)
        return " [SEP] ".join(f"'{p}'" for p in parts[idx + 1:])
    except ValueError:
        return obs


def format_available_actions(avail: dict) -> str:
    actions = []
    if avail.get("has_search_bar"):
        actions.append("search[<your query>]")
    for txt in avail.get("clickables", []):
        actions.append(f"click[{txt}]")
    return "\n".join(f"'{s}'," for s in actions)


def build_prompt(task, obs_text, avail_actions, history, step_count):
    if step_count == 0 or HISTORY_LENGTH <= 0:
        return TEMPLATE_NO_HIS.format(
            task_description=task,
            current_observation=obs_text,
            available_actions=avail_actions,
        )

    recent = history[-HISTORY_LENGTH:]
    start_idx = max(0, len(history) - HISTORY_LENGTH)
    lines = []
    for j, (h_obs, h_action) in enumerate(recent):
        step_num = start_idx + j + 1
        lines.append(f"[Observation {step_num}: '{h_obs}', Action {step_num}: '{h_action}']")
    action_history = "\n".join(lines)

    prompt = TEMPLATE.format(
        task_description=task,
        step_count=step_count,
        history_length=len(recent),
        action_history=action_history,
        current_step=step_count + 1,
        current_observation=obs_text,
        available_actions=avail_actions,
    )

    if len(prompt) > MAX_OBS_CHARS:
        prompt = TEMPLATE_NO_HIS.format(
            task_description=task,
            current_observation=obs_text,
            available_actions=avail_actions,
        )
    return prompt


class EpisodeState:
    """Tracks state for one WebShop episode."""
    __slots__ = ("goal_idx", "env", "task", "obs_text", "history",
                 "step_count", "done", "reward", "_avail")

    def __init__(self, goal_idx, env):
        self.goal_idx = goal_idx
        self.env = env
        self.step_count = 0
        self.done = False
        self.reward = 0.0
        self.history = []

        obs, info = env.reset(session=goal_idx)
        info = dict(info or {})
        info["available_actions"] = env.get_available_actions()
        self.task = extract_task(obs)
        self.obs_text = format_obs(obs, self.task)
        self._avail = info["available_actions"]

    def get_prompt(self):
        avail = format_available_actions(self._avail)
        return build_prompt(self.task, self.obs_text, avail, self.history, self.step_count)

    def step(self, action):
        self.history.append((self.obs_text, action))
        self.step_count += 1

        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})
        if not done:
            info["available_actions"] = self.env.get_available_actions()
        self.obs_text = format_obs(obs, self.task)
        self._avail = info.get("available_actions", {})

        if done:
            self.done = True
            self.reward = reward
        elif self.step_count >= 15:
            self.done = True
            self.reward = 0.0


def run_batched_eval(llm, sampling_params, env_factory, goal_indices, batch_size, max_steps):
    """Run evaluation with batched parallel episodes."""
    scores = {}  # goal_idx -> reward
    act_counts = {}  # goal_idx -> num actions
    pending = list(goal_indices)  # goals not yet started
    active = []  # currently running EpisodeStates

    start_time = time.time()
    step_count = 0

    while pending or active:
        # Fill batch with new episodes
        while len(active) < batch_size and pending:
            goal_idx = pending.pop(0)
            env = env_factory()
            ep = EpisodeState(goal_idx, env)
            active.append(ep)

        if not active:
            break

        # Build prompts for all active episodes
        prompts = []
        for ep in active:
            messages = [{"role": "user", "content": ep.get_prompt()}]
            prompts.append(messages)

        # Batched inference
        outputs = llm.chat(prompts, sampling_params=sampling_params)

        # Process responses and step environments
        newly_done = []
        for i, (ep, output) in enumerate(zip(active, outputs)):
            response = output.outputs[0].text
            action = extract_action(response)
            ep.step(action)

            if ep.done:
                scores[ep.goal_idx] = ep.reward
                act_counts[ep.goal_idx] = ep.step_count
                ep.env.close()
                newly_done.append(i)

        # Remove completed episodes (reverse order to preserve indices)
        for i in sorted(newly_done, reverse=True):
            active.pop(i)

        step_count += 1
        completed = len(scores)
        total = len(goal_indices)

        if completed > 0 and (completed % 10 == 0 or step_count % 5 == 0):
            elapsed = time.time() - start_time
            avg = sum(scores.values()) / completed
            wins = sum(1 for s in scores.values() if s == 1.0)
            print(f"[{completed}/{total}] avg_task_score={avg*100:.1f} | "
                  f"success_rate={wins}/{completed} ({wins/completed*100:.1f}%) | "
                  f"active={len(active)} | {elapsed:.0f}s")

    return scores, act_counts


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on WebShop (official protocol, batched)")
    parser.add_argument("--model_path", type=str, required=True, help="HF model path or checkpoint dir")
    parser.add_argument("--num_episodes", type=int, default=500, help="Number of test episodes (default: 500)")
    parser.add_argument("--max_steps", type=int, default=15, help="Max steps per episode")
    parser.add_argument("--batch_size", type=int, default=32, help="Parallel episodes per batch")
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--gpu_memory", type=float, default=0.5, help="GPU memory utilization")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    print(f"Config: {args.num_episodes} episodes, batch={args.batch_size}, "
          f"max {args.max_steps} steps, temp={args.temperature}, TP={args.tp}")

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=8192,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=2048,
    )

    print("Loading WebShop environment...")
    if os.path.isdir(WEBSHOP_DIR) and WEBSHOP_DIR not in sys.path:
        sys.path.insert(0, WEBSHOP_DIR)
    import gym
    from web_agent_site.envs import WebAgentTextEnv  # noqa: triggers gym.register

    # Pre-load products once (shared across all envs via module-level cache)
    _warmup = gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=None)
    num_goals = len(_warmup.server.goals)
    _warmup.close()
    print(f"WebShop loaded: {num_goals} goals")

    def env_factory():
        return gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=None)

    goal_indices = list(range(args.num_episodes))

    start_time = time.time()
    scores, act_counts = run_batched_eval(llm, sampling_params, env_factory, goal_indices,
                                           args.batch_size, args.max_steps)
    total_time = time.time() - start_time

    # Sort by goal index for consistent output
    ordered_scores = [scores[i] for i in range(args.num_episodes)]
    ordered_acts = [act_counts[i] for i in range(args.num_episodes)]
    avg_score = sum(ordered_scores) / len(ordered_scores)
    avg_acts = sum(ordered_acts) / len(ordered_acts)
    num_success = sum(1 for s in ordered_scores if s == 1.0)

    print("\n" + "=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Episodes: {len(ordered_scores)} | Batch size: {args.batch_size}")
    print(f"Score:        {avg_score * 100:.2f} / 100")
    print(f"#Act:         {avg_acts:.2f}")
    print(f"Success Rate: {num_success}/{len(ordered_scores)} ({num_success/len(ordered_scores)*100:.2f}%)")
    print(f"Time: {total_time:.0f}s ({total_time/len(ordered_scores):.1f}s/episode)")
    print("=" * 60)


if __name__ == "__main__":
    main()
