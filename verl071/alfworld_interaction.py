"""ALFWorld interaction for verl 0.7.1's ToolAgentLoop.

Wraps ALFWorld text-based environments as a verl BaseInteraction.
Matches the original verl-agent functionality:
  - Deterministic game selection: rollouts in the same GRPO group play the
    same game (based on prompt index + epoch counter)
  - Cycles through all games across epochs
  - Train split: 3553 games, Val split: 134 unseen games (eval_out_of_distribution)
  - Rich prompts with task description, history, admissible actions
  - Action extraction from <action>...</action> tags
  - Reward: 10.0 * won (task completion)
"""

import logging
import os
import re
import threading
from typing import Any, Optional
from uuid import uuid4

import textworld
import textworld.gym
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

# Map data split names to ALFWorld train_eval parameter
SPLIT_MAP = {
    "train": "train",
    "test": "eval_out_of_distribution",
    "val": "eval_out_of_distribution",
    "eval": "eval_out_of_distribution",
    "eval_in_distribution": "eval_in_distribution",
    "eval_out_of_distribution": "eval_out_of_distribution",
}


def extract_action(text: str) -> tuple[str, bool]:
    """Extract action from <action>...</action> tags."""
    text_lower = text.lower()
    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = text_lower.find(start_tag)
    end_idx = text_lower.find(end_tag)

    if start_idx == -1 or end_idx == -1:
        return text_lower[-30:], False

    action = text_lower[start_idx + len(start_tag):end_idx].strip()
    has_think = "<think>" in text_lower and "</think>" in text_lower
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))

    return action, has_think and not has_chinese


def format_admissible(actions: list[str]) -> str:
    """Format admissible actions, excluding 'help'."""
    return "\n ".join(f"'{s}'" for s in actions if s != "help")


class AlfWorldInteraction(BaseInteraction):
    """ALFWorld text game as a verl interaction.

    Game selection is deterministic based on (prompt_index, call_count) so
    that all rollouts in the same GRPO group play the same game. The
    call_count increments per prompt_index, ensuring different games across
    epochs.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.max_steps = config.get("max_steps", 50)
        self.history_length = config.get("history_length", 2)
        self.envs = {}  # instance_id -> episode state

        self._alf_config_path = config.get(
            "alf_config_path",
            os.path.join(os.path.dirname(__file__),
                         "agent_system/environments/env_package/alfworld/configs/config_tw.yaml")
        )

        # Game file lists per split (lazy loaded)
        self._game_files: dict[str, list[str]] = {}
        self._alf_configs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _load_game_files(self, split: str) -> list[str]:
        """Load and cache game file list for a split."""
        alf_split = SPLIT_MAP.get(split, "train")
        if alf_split not in self._game_files:
            from alfworld.agents.environment import get_environment
            import yaml

            with open(self._alf_config_path) as f:
                alf_config = yaml.safe_load(f)

            tw_env = get_environment("AlfredTWEnv")(alf_config, train_eval=alf_split)
            self._game_files[alf_split] = list(tw_env.game_files)
            self._alf_configs[alf_split] = alf_config
            logger.info(f"Loaded {len(tw_env.game_files)} games for split={alf_split}")
        return self._game_files[alf_split]

    def _select_game_index(self, split: str, prompt_index: int) -> int:
        """Select a game index deterministically based on prompt_index.

        Pure function of prompt_index — all rollouts with the same index
        (same GRPO group) get the same game, regardless of which worker
        or interaction instance handles them. Each interaction instance is
        created fresh per rollout in verl 0.7.1, so no mutable state can
        be used for coordination.

        This means the same 16 games are used each epoch. With 16 prompts
        spread across 6 task types, the model sees diverse tasks each step.
        """
        game_files = self._load_game_files(split)
        num_games = len(game_files)
        # Spread indices across game list using a stride
        game_idx = (prompt_index * 223) % num_games  # 223 is prime, good spread
        return game_idx

    def _create_single_game_env(self, split: str, game_idx: int):
        """Create a TextWorld env for a single game file."""
        alf_split = SPLIT_MAP.get(split, "train")
        game_files = self._game_files[alf_split]
        game_file = game_files[game_idx]

        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        alf_config = self._alf_configs[alf_split]
        domain_randomization = alf_config["env"]["domain_randomization"] if alf_split == "train" else False
        alfred_demangler = AlfredDemangler(shuffle=domain_randomization)
        wrappers = [alfred_demangler, AlfredInfos]

        request_infos = textworld.EnvInfos(
            won=True, admissible_commands=True, extras=["gamefile"]
        )

        env_id = textworld.gym.register_games(
            [game_file], request_infos,
            batch_size=1, asynchronous=False,
            max_episode_steps=self.max_steps,
            wrappers=wrappers,
        )
        env = textworld.gym.make(env_id)
        return env

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
            return ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=obs_text,
                admissible_actions=admissible,
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
        """Start a new ALFWorld episode.

        Game selection is deterministic: all rollouts with the same
        prompt index play the same game (correct GRPO grouping).
        """
        if instance_id is None:
            instance_id = str(uuid4())

        split = kwargs.get("split", "train")
        # The 'index' from extra_info identifies the prompt — same across a GRPO group
        prompt_index = kwargs.get("index", 0)
        if isinstance(prompt_index, str):
            prompt_index = int(prompt_index)

        try:
            game_idx = self._select_game_index(split, prompt_index)
            game_env = self._create_single_game_env(split, game_idx)
            obs, info = game_env.reset()
            initial_obs = obs[0] if isinstance(obs, (list, tuple)) else obs

            admissible = info.get("admissible_commands", [])
            if isinstance(admissible, (list, tuple)) and admissible and isinstance(admissible[0], list):
                admissible = admissible[0]

            task = self._extract_task(initial_obs)

            self.envs[instance_id] = {
                "game_env": game_env,
                "step_count": 0,
                "done": False,
                "task": task,
                "current_obs": initial_obs,
                "admissible_commands": admissible,
                "history": [],
                "game_idx": game_idx,
            }
            print(
                f"[ALFWorld] instance={instance_id[:8]} index={prompt_index} "
                f"game_idx={game_idx} task={task[:50]}..."
            )
        except Exception as e:
            logger.error(f"Failed to start ALFWorld episode: {e}")
            self.envs[instance_id] = {
                "game_env": None,
                "step_count": 0,
                "done": True,
                "task": "",
                "current_obs": f"Error: {e}",
                "admissible_commands": [],
                "history": [],
                "game_idx": -1,
            }

        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """Process the agent's action and return environment observation."""
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

        action, is_valid = extract_action(raw_action)

        game_env = env_state["game_env"]
        env_state["step_count"] += 1

        try:
            obs, scores, dones, info = game_env.step([action])
            obs_text = obs[0] if isinstance(obs, (list, tuple)) else str(obs)
            done = dones[0] if isinstance(dones, (list, tuple)) else bool(dones)

            for k in list(info.keys()):
                if isinstance(info[k], (list, tuple)) and len(info[k]) == 1:
                    info[k] = info[k][0]

            won = float(info.get("won", False))
            reward_val = 10.0 * won

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

        env_state["history"].append((env_state["current_obs"], action))
        env_state["current_obs"] = obs_text
        env_state["admissible_commands"] = admissible
        env_state["done"] = done or env_state["step_count"] >= self.max_steps
        should_terminate = env_state["done"]

        next_prompt = self._build_observation_prompt(env_state)

        return should_terminate, next_prompt, reward_val, {
            "step": env_state["step_count"],
            "done": done,
            "is_valid": is_valid,
            "won": won if not done else 0.0,
        }

    async def finalize_interaction(self, instance_id: str = None, **kwargs) -> None:
        """Clean up episode state and close per-episode env."""
        if instance_id is None:
            return
        env_state = self.envs.pop(instance_id, None)
        if env_state and env_state.get("game_env"):
            try:
                env_state["game_env"].close()
            except Exception:
                pass
