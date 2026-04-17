"""ALFWorld interaction for verl 0.7.1's ToolAgentLoop.

Wraps ALFWorld text-based environments as a verl BaseInteraction.
Matches the original verl-agent functionality:
  - Rich prompts with task description, history, admissible actions
  - Action extraction from <action>...</action> tags
  - History buffer for recent observations and actions
  - Reward: 10.0 * won (task completion)
"""

import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

from verl.interactions.base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# --------------- Prompt Templates (from verl-agent) --------------- #

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


def extract_action(text: str) -> tuple[str, bool]:
    """Extract action from <action>...</action> tags.

    Returns (action_text, is_valid).
    """
    text_lower = text.lower()
    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = text_lower.find(start_tag)
    end_idx = text_lower.find(end_tag)

    if start_idx == -1 or end_idx == -1:
        return text_lower[-30:], False

    action = text_lower[start_idx + len(start_tag):end_idx].strip()

    # Check for <think>...</think> tags
    has_think = "<think>" in text_lower and "</think>" in text_lower

    # Check for Chinese characters (invalid)
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))

    is_valid = has_think and not has_chinese
    return action, is_valid


def format_admissible(actions: list[str]) -> str:
    """Format admissible actions, excluding 'help'."""
    return "\n ".join(f"'{s}'" for s in actions if s != "help")


class AlfWorldInteraction(BaseInteraction):
    """ALFWorld text game as a verl interaction."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.max_steps = config.get("max_steps", 50)
        self.history_length = config.get("history_length", 2)
        self.envs = {}  # instance_id -> env state

        # ALFWorld setup
        self._alf_config_path = config.get(
            "alf_config_path",
            os.path.join(os.path.dirname(__file__),
                         "agent_system/environments/env_package/alfworld/configs/config_tw.yaml")
        )
        self._tw_env = None  # Lazy-initialized shared AlfredTWEnv

    def _get_tw_env(self):
        """Lazy-init the AlfredTWEnv (shared across episodes)."""
        if self._tw_env is None:
            from alfworld.agents.environment import get_environment
            import yaml

            with open(self._alf_config_path) as f:
                alf_config = yaml.safe_load(f)

            AlfredTWEnv = get_environment("AlfredTWEnv")
            self._tw_env = AlfredTWEnv(alf_config, train_eval="train")
            logger.info("ALFWorld AlfredTWEnv initialized")
        return self._tw_env

    def _extract_task(self, initial_obs: str) -> str:
        """Extract task description from initial observation."""
        marker = "Your task is to: "
        idx = initial_obs.find(marker)
        if idx != -1:
            return initial_obs[idx + len(marker):].strip()
        return ""

    def _build_observation_prompt(self, env_state: dict) -> str:
        """Build the full observation prompt with history and admissible actions."""
        obs_text = env_state["current_obs"]
        admissible = format_admissible(env_state["admissible_commands"])
        history = env_state["history"]
        step_count = env_state["step_count"]

        if step_count == 0 or self.history_length <= 0:
            # First step — no history
            return ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=obs_text,
                admissible_actions=admissible,
            )
        else:
            # Subsequent steps — include history
            recent = history[-self.history_length:]
            history_lines = []
            start_idx = max(0, len(history) - self.history_length)
            for j, (h_obs, h_action) in enumerate(recent):
                step_num = start_idx + j + 1
                history_lines.append(
                    f"[Observation {step_num}: '{h_obs}', Action {step_num}: '{h_action}']"
                )
            action_history = "\n".join(history_lines)

            return ALFWORLD_TEMPLATE.format(
                task_description=env_state["task"],
                step_count=step_count,
                history_length=len(recent),
                action_history=action_history,
                current_step=step_count + 1,
                current_observation=obs_text,
                admissible_actions=admissible,
            )

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Start a new ALFWorld episode."""
        if instance_id is None:
            instance_id = str(uuid4())

        try:
            tw_env = self._get_tw_env()
            game_env = tw_env.init_env(batch_size=1)
            obs, info = game_env.reset()
            initial_obs = obs[0] if isinstance(obs, (list, tuple)) else obs

            # Extract admissible commands
            admissible = info.get("admissible_commands", [])
            if isinstance(admissible, (list, tuple)) and admissible and isinstance(admissible[0], list):
                admissible = admissible[0]

            # Extract task description from initial observation
            task = self._extract_task(initial_obs)

            self.envs[instance_id] = {
                "game_env": game_env,
                "step_count": 0,
                "done": False,
                "task": task,
                "current_obs": initial_obs,
                "admissible_commands": admissible,
                "history": [],  # list of (obs, action) tuples
            }
            logger.info(f"Started ALFWorld episode {instance_id}, task: {task[:80]}...")
        except Exception as e:
            logger.error(f"Failed to create ALFWorld env: {e}")
            self.envs[instance_id] = {
                "game_env": None,
                "step_count": 0,
                "done": True,
                "task": "",
                "current_obs": f"Error: {e}",
                "admissible_commands": [],
                "history": [],
            }

        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """Process the agent's action and return environment observation.

        Returns:
            (should_terminate, observation_prompt, reward, extra_info)
        """
        env_state = self.envs.get(instance_id)
        if env_state is None or env_state["done"] or env_state["game_env"] is None:
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

        # Parse action from <action> tags
        action, is_valid = extract_action(raw_action)

        # Step the environment
        game_env = env_state["game_env"]
        env_state["step_count"] += 1

        try:
            obs, scores, dones, info = game_env.step([action])
            obs_text = obs[0] if isinstance(obs, (list, tuple)) else str(obs)
            done = dones[0] if isinstance(dones, (list, tuple)) else bool(dones)

            # Unpack info (ALFWorld wraps values in lists)
            for k in list(info.keys()):
                if isinstance(info[k], (list, tuple)) and len(info[k]) == 1:
                    info[k] = info[k][0]

            # Reward: 10.0 * won
            won = float(info.get("won", False))
            reward_val = 10.0 * won

            # Update admissible commands
            admissible = info.get("admissible_commands", env_state["admissible_commands"])
            if isinstance(admissible, (list, tuple)) and admissible and isinstance(admissible[0], list):
                admissible = admissible[0]

        except Exception as e:
            logger.error(f"ALFWorld step error: {e}")
            obs_text = f"Error: {e}"
            reward_val = 0.0
            done = True
            is_valid = False
            admissible = env_state["admissible_commands"]

        # Store history (previous obs + action taken)
        env_state["history"].append((env_state["current_obs"], action))

        # Update state
        env_state["current_obs"] = obs_text
        env_state["admissible_commands"] = admissible
        env_state["done"] = done or env_state["step_count"] >= self.max_steps
        should_terminate = env_state["done"]

        # Build the next observation prompt (with history, admissible actions, etc.)
        next_prompt = self._build_observation_prompt(env_state)

        return should_terminate, next_prompt, reward_val, {
            "step": env_state["step_count"],
            "done": done,
            "is_valid": is_valid,
            "won": won if not done else 0.0,
        }

    async def finalize_interaction(self, instance_id: str = None, **kwargs) -> None:
        """Clean up the episode."""
        if instance_id is None:
            return
        env_state = self.envs.pop(instance_id, None)
        if env_state and env_state.get("game_env"):
            try:
                env_state["game_env"].close()
            except Exception:
                pass
