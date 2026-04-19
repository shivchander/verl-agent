"""WebShop reward function for verl GRPO training.

Reward: 10.0 * won (task completion) - invalid_action_penalty.
The interaction returns per-turn rewards via turn_scores.
Invalid actions (missing <think>/<action> tags) are penalized.
"""

import re

INVALID_ACTION_PENALTY = 0.1


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Compute WebShop reward from turn_scores + format penalty.

    - Base reward: max(turn_scores) -- 10.0 if task completed, 0.0 otherwise
    - Penalty: -0.1 per invalid action format in the response
    """
    base_reward = 0.0
    if extra_info and isinstance(extra_info, dict):
        turn_scores = extra_info.get("turn_scores", [])
        if turn_scores and isinstance(turn_scores, list):
            base_reward = float(max(turn_scores))

    # Check response format for <think>...</think> and <action>...</action>
    penalty = 0.0
    if solution_str:
        text = solution_str.lower()
        has_think = "<think>" in text and "</think>" in text
        has_action = "<action>" in text and "</action>" in text
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', solution_str))

        if not has_think or not has_action or has_chinese:
            penalty = INVALID_ACTION_PENALTY

    return base_reward - penalty
