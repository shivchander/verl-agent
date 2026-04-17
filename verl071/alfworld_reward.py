"""ALFWorld reward function for verl GRPO training.

Matches the original verl-agent reward: 10.0 * won (task completion).
The interaction returns per-turn rewards via turn_scores. We take the
max turn score as the final episode reward.
"""


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Compute ALFWorld reward from turn_scores.

    The interaction stores per-turn rewards (10.0 * won) in turn_scores.
    The max score represents whether the task was ever completed.
    """
    if extra_info and isinstance(extra_info, dict):
        turn_scores = extra_info.get("turn_scores", [])
        if turn_scores and isinstance(turn_scores, list):
            return float(max(turn_scores))

    return 0.0
