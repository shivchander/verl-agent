"""WebShop interaction for verl 0.7.1's ToolAgentLoop.

Wraps WebShop text-based environments as a verl BaseInteraction.
Matches the original verl-agent functionality:
  - Deterministic goal selection: rollouts in the same GRPO group play the
    same goal (based on prompt index + epoch counter)
  - Cycles through all goals across epochs
  - Train goals: index 500+, Test goals: index 0-500
  - Rich prompts with task description, history, available actions
  - Action extraction from <action>...</action> tags
  - Reward: 10.0 * won (task completion)
"""

import logging
import os
import re
import sys
import threading
from typing import Any, Optional
from uuid import uuid4

import gym
from verl.interactions.base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# --------------- Prompt Templates (from verl-agent) --------------- #

WEBSHOP_TEMPLATE_NO_HIS = """
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

WEBSHOP_TEMPLATE = """
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

# Max chars for observation prompt before falling back to no-history
MAX_OBS_CHARS = 13000


def extract_action(text: str) -> tuple[str, bool]:
    """Extract action from <action>...</action> tags."""
    text_lower = text.lower()
    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = text_lower.find(start_tag)
    end_idx = text_lower.find(end_tag)

    if start_idx == -1 or end_idx == -1:
        return text_lower[-20:], False

    action = text_lower[start_idx + len(start_tag):end_idx].strip()
    has_think = "<think>" in text_lower and "</think>" in text_lower
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))

    return action, has_think and not has_chinese


def format_available_actions(avail: dict) -> str:
    """Format available actions as search[query] / click[text] list."""
    actions = []

    if avail.get("has_search_bar"):
        actions.append("search[<your query>]")

    for txt in avail.get("clickables", []):
        actions.append(f"click[{txt}]")

    return "\n".join(f"'{s}'," for s in actions)


def extract_task(obs: str) -> str:
    """Extract task description from WebShop initial observation.

    WebShop observations are formatted as:
        WebShop [SEP] Instruction: [SEP] <task> [SEP] ...
    """
    parts = obs.split(" [SEP] ")
    if len(parts) >= 3 and parts[1] == "Instruction:":
        return parts[2]
    return obs


def format_obs(obs: str, task: str) -> str:
    """Remove the task prefix from a WebShop observation.

    Returns everything after the task description in the [SEP]-delimited
    observation string.
    """
    parts = obs.split(" [SEP] ")
    try:
        idx = parts.index(task)
        return " [SEP] ".join(f"'{p}'" for p in parts[idx + 1:])
    except ValueError:
        return obs


class WebShopInteraction(BaseInteraction):
    """WebShop as a verl interaction.

    Goal selection is deterministic based on (prompt_index, call_count) so
    that all rollouts in the same GRPO group play the same game.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.max_steps = config.get("max_steps", 15)
        self.history_length = config.get("history_length", 2)
        self.envs = {}  # instance_id -> episode state

        # WebShop env kwargs
        self._env_kwargs = {
            "observation_mode": "text",
            "num_products": None,
        }
        webshop_path = config.get("webshop_path")
        if webshop_path:
            self._env_kwargs["file_path"] = os.path.join(webshop_path, "items_shuffle.json")
            self._env_kwargs["attr_path"] = os.path.join(webshop_path, "items_ins_v2_1000.json")

        # Goal indices per split (lazy loaded)
        self._goal_indices: dict[str, list[int]] = {}
        self._lock = threading.Lock()

    def _ensure_webshop_on_path(self):
        """Add WebShop to sys.path and register the gym environment."""
        webshop_dir = os.path.join(
            os.path.dirname(__file__),
            "agent_system", "environments", "env_package", "webshop", "webshop"
        )
        if not os.path.isdir(webshop_dir):
            # Try from project root
            webshop_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "agent_system", "environments", "env_package", "webshop", "webshop"
            )
        if os.path.isdir(webshop_dir) and webshop_dir not in sys.path:
            sys.path.insert(0, webshop_dir)
        # Import triggers gym.register() for WebAgentTextEnv-v0
        from web_agent_site.envs import WebAgentTextEnv  # noqa: F401

    def _load_goal_indices(self, split: str) -> list[int]:
        """Load and cache goal index list for a split."""
        with self._lock:
            if split not in self._goal_indices:
                self._ensure_webshop_on_path()
                env = gym.make("WebAgentTextEnv-v0", **self._env_kwargs)
                num_goals = len(env.server.goals)
                if split in ("test", "val", "eval"):
                    indices = list(range(500))
                else:
                    indices = list(range(500, num_goals))
                self._goal_indices[split] = indices
                env.close()
                logger.info(f"Loaded {len(indices)} goals for split={split} (total={num_goals})")
            return self._goal_indices[split]

    def _select_goal_index(self, split: str, prompt_index: int, global_step: int = 0) -> int:
        """Select a goal index deterministically.

        All rollouts with the same (prompt_index, global_step) get the same
        goal. Different global_steps cycle through different goals.
        """
        indices = self._load_goal_indices(split)
        num_goals = len(indices)
        idx = (prompt_index + global_step * 16) % num_goals
        return indices[idx]

    def _create_env(self):
        """Create a single WebShop environment instance."""
        self._ensure_webshop_on_path()
        env = gym.make("WebAgentTextEnv-v0", **self._env_kwargs)
        return env

    def _build_observation_prompt(self, env_state: dict) -> str:
        """Build the full observation prompt with history and available actions."""
        obs_text = env_state["current_obs"]
        avail_actions = format_available_actions(env_state["available_actions"])
        history = env_state["history"]
        step_count = env_state["step_count"]

        if step_count == 0 or self.history_length <= 0:
            return WEBSHOP_TEMPLATE_NO_HIS.format(
                task_description=env_state["task"],
                current_observation=obs_text,
                available_actions=avail_actions,
            )
        else:
            recent = history[-self.history_length:]
            history_lines = []
            start_idx = max(0, len(history) - self.history_length)
            for j, (h_obs, h_action) in enumerate(recent):
                step_num = start_idx + j + 1
                history_lines.append(
                    f"[Observation {step_num}: '{h_obs}', Action {step_num}: '{h_action}']"
                )
            action_history = "\n".join(history_lines)

            prompt = WEBSHOP_TEMPLATE.format(
                task_description=env_state["task"],
                step_count=step_count,
                history_length=len(recent),
                action_history=action_history,
                current_step=step_count + 1,
                current_observation=obs_text,
                available_actions=avail_actions,
            )

            # Fall back to no-history if prompt is too long
            if len(prompt) > MAX_OBS_CHARS:
                logger.warning(f"Prompt too long ({len(prompt)} chars), dropping history")
                prompt = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=env_state["task"],
                    current_observation=obs_text,
                    available_actions=avail_actions,
                )

            return prompt

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Start a new WebShop episode.

        Goal selection is deterministic: all rollouts with the same
        prompt index play the same goal (correct GRPO grouping).
        """
        if instance_id is None:
            instance_id = str(uuid4())

        split = kwargs.get("split", "train")
        prompt_index = kwargs.get("index", 0)
        if isinstance(prompt_index, str):
            prompt_index = int(prompt_index)
        global_step = kwargs.get("global_step", 0)

        try:
            goal_idx = self._select_goal_index(split, prompt_index, global_step)
            env = self._create_env()
            obs, info = env.reset(session=goal_idx)
            info = dict(info or {})
            info["available_actions"] = env.get_available_actions()

            task = extract_task(obs)
            obs_formatted = format_obs(obs, task)

            self.envs[instance_id] = {
                "env": env,
                "step_count": 0,
                "done": False,
                "task": task,
                "current_obs": obs_formatted,
                "available_actions": info["available_actions"],
                "history": [],
                "goal_idx": goal_idx,
            }
            print(
                f"[WebShop] instance={instance_id[:8]} index={prompt_index} "
                f"goal_idx={goal_idx} task={task[:60]}..."
            )
        except Exception as e:
            logger.error(f"Failed to start WebShop episode: {e}")
            self.envs[instance_id] = {
                "env": None,
                "step_count": 0,
                "done": True,
                "task": "",
                "current_obs": f"Error: {e}",
                "available_actions": {},
                "history": [],
                "goal_idx": -1,
            }

        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """Process the agent's action and return environment observation."""
        env_state = self.envs.get(instance_id)
        if env_state is None or env_state["done"] or env_state["env"] is None:
            return True, "Episode ended.", 0.0, {}

        # On the first call, return the initial observation without stepping
        if not env_state.get("initialized"):
            env_state["initialized"] = True
            initial_prompt = self._build_observation_prompt(env_state)
            return False, initial_prompt, 0.0, {"step": 0, "initial": True}

        # Extract the last assistant message as the raw action
        raw_action = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                raw_action = msg.get("content", "").strip()
                break

        if not raw_action:
            return True, "No action provided.", 0.0, {}

        action, is_valid = extract_action(raw_action)

        env = env_state["env"]
        env_state["step_count"] += 1

        available_actions = env_state["available_actions"]

        try:
            obs, reward, done, info = env.step(action)
            info = dict(info or {})
            if not done:
                available_actions = env.get_available_actions()

            task_score = reward  # raw WebShop reward (0-1 partial credit)

            # Binary win: reward == 1.0 and done
            if done and reward == 1.0:
                won = 1.0
                reward_val = 10.0
            else:
                won = 0.0
                reward_val = 0.0

            obs_formatted = format_obs(obs, env_state["task"])

        except Exception as e:
            logger.error(f"WebShop step error: {e}")
            obs_formatted = f"Error: {e}"
            reward_val = 0.0
            task_score = 0.0
            won = 0.0
            done = True
            is_valid = False

        env_state["history"].append((env_state["current_obs"], action))
        env_state["current_obs"] = obs_formatted
        env_state["available_actions"] = available_actions
        env_state["done"] = done or env_state["step_count"] >= self.max_steps
        should_terminate = env_state["done"]

        next_prompt = self._build_observation_prompt(env_state)

        return should_terminate, next_prompt, reward_val, {
            "step": env_state["step_count"],
            "done": done,
            "is_valid": is_valid,
            "won": won,
            "task_score": task_score,
        }

    async def finalize_interaction(self, instance_id: str = None, **kwargs) -> None:
        """Clean up episode state and close per-episode env."""
        if instance_id is None:
            return
        env_state = self.envs.pop(instance_id, None)
        if env_state and env_state.get("env"):
            try:
                env_state["env"].close()
            except Exception:
                pass
