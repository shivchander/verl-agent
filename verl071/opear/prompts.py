"""O-PEaR guide model prompt templates and parsing utilities.

The guide model (GPT-5.4-nano) rewrites student assistant responses given
privileged PDDL facts from the ALFWorld environment. It produces two kinds
of alternative responses:

- Compliant: consistent with the ground-truth facts
- Violating: plausible from observations alone but contradicting the facts

Each rewritten response uses the exact format:
    <think>...</think><action>...</action>
"""

import re
from typing import Literal

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

COMPLIANT_SYSTEM = """\
You are an expert rewriting assistant for the ALFRED Embodied Environment.

You will be given:
1. A task description.
2. A multi-turn trajectory of the student agent, consisting of observations \
and the student's assistant responses.
3. Privileged ground-truth facts from the environment's PDDL state.

Your job is to rewrite ONLY the assistant responses so that they are \
CONSISTENT with the privileged facts. Do not alter observations.

Rules:
- Each rewritten response MUST use EXACTLY this format (no extra text):
    <think>YOUR REASONING HERE</think><action>YOUR ACTION HERE</action>
- The action MUST be chosen from the admissible actions listed in the \
corresponding observation.
- Your reasoning in <think> should reflect knowledge that is consistent \
with the privileged facts.
- Produce exactly one rewritten response per assistant turn.
- Label each rewritten response with a [TURN N] marker (1-indexed).

Output format (one block per assistant turn):
[TURN 1]
<think>...</think><action>...</action>
[TURN 2]
<think>...</think><action>...</action>
...
"""

VIOLATING_SYSTEM = """\
You are an expert rewriting assistant for the ALFRED Embodied Environment.

You will be given:
1. A task description.
2. A multi-turn trajectory of the student agent, consisting of observations \
and the student's assistant responses.
3. Privileged ground-truth facts from the environment's PDDL state.

Your job is to rewrite ONLY the assistant responses so that they sound \
PLAUSIBLE given the observations alone, but actually CONTRADICT the \
privileged facts. Do not alter observations.

Rules:
- Each rewritten response MUST use EXACTLY this format (no extra text):
    <think>YOUR REASONING HERE</think><action>YOUR ACTION HERE</action>
- The action MUST be chosen from the admissible actions listed in the \
corresponding observation, but should lead away from the correct solution.
- Your reasoning in <think> should sound convincing but reflect beliefs \
that contradict the privileged facts.
- Produce exactly one rewritten response per assistant turn.
- Label each rewritten response with a [TURN N] marker (1-indexed).

Output format (one block per assistant turn):
[TURN 1]
<think>...</think><action>...</action>
[TURN 2]
<think>...</think><action>...</action>
...
"""


# ---------------------------------------------------------------------------
# Trajectory formatting
# ---------------------------------------------------------------------------

def format_trajectory(turns: list[dict]) -> str:
    """Format a multi-turn trajectory for the guide prompt.

    Args:
        turns: list of dicts, each with keys:
            - "role": one of "observation", "assistant"
            - "content": the text content of that turn

    Returns:
        A formatted string representation of the trajectory, with each turn
        clearly labeled by role.
    """
    lines: list[str] = []
    for i, turn in enumerate(turns, 1):
        role = turn["role"].upper()
        content = turn["content"].strip()
        lines.append(f"[{role} {i}]\n{content}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Guide prompt construction
# ---------------------------------------------------------------------------

def build_guide_prompt(
    turns: list[dict],
    task_description: str,
    mode: Literal["compliant", "violating"],
    facts: str = "",
) -> list[dict]:
    """Build chat messages for the OpenAI API guide call.

    Args:
        turns: multi-turn trajectory (see format_trajectory).
        task_description: the ALFWorld task description string.
        mode: "compliant" or "violating".
        facts: privileged PDDL facts from the environment state.

    Returns:
        A list of message dicts suitable for the OpenAI chat completions API,
        with "role" and "content" keys.
    """
    if mode == "compliant":
        system_prompt = COMPLIANT_SYSTEM
    elif mode == "violating":
        system_prompt = VIOLATING_SYSTEM
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Must be 'compliant' or 'violating'.")

    trajectory_text = format_trajectory(turns)

    user_content = (
        f"Task: {task_description}\n\n"
        f"Privileged facts:\n{facts}\n\n"
        f"Trajectory:\n{trajectory_text}\n\n"
        f"Rewrite the assistant responses according to your instructions."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# Pattern matches [TURN N] markers, tolerant of whitespace variations.
_TURN_MARKER_RE = re.compile(r"\[TURN\s+(\d+)\]", re.IGNORECASE)


def parse_guide_response(
    response_text: str,
    expected_turns: int,
) -> list[dict]:
    """Parse the guide model output into per-turn responses.

    Splits on ``[TURN N]`` markers and validates that each block contains
    both ``<think>`` and ``<action>`` tags.

    Args:
        response_text: raw text output from the guide model.
        expected_turns: the number of assistant turns we expect.

    Returns:
        A list of dicts, one per turn, each with keys:
            - "turn": 1-indexed turn number
            - "raw": the full text block for this turn
            - "think": content inside <think>...</think>
            - "action": content inside <action>...</action>

    Raises:
        ValueError: if the number of parsed turn blocks does not match
            ``expected_turns``, or if any block is missing required tags.
    """
    # Find all [TURN N] positions
    markers = list(_TURN_MARKER_RE.finditer(response_text))

    if len(markers) != expected_turns:
        raise ValueError(
            f"Expected {expected_turns} turn(s) but found {len(markers)} "
            f"[TURN N] marker(s) in guide response."
        )

    results: list[dict] = []

    for idx, marker in enumerate(markers):
        turn_num = int(marker.group(1))
        start = marker.end()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(response_text)
        block = response_text[start:end].strip()

        # Extract <think>...</think>
        think_match = re.search(
            r"<think>(.*?)</think>", block, re.DOTALL | re.IGNORECASE
        )
        if think_match is None:
            raise ValueError(
                f"[TURN {turn_num}] is missing <think>...</think> tags."
            )

        # Extract <action>...</action>
        action_match = re.search(
            r"<action>(.*?)</action>", block, re.DOTALL | re.IGNORECASE
        )
        if action_match is None:
            raise ValueError(
                f"[TURN {turn_num}] is missing <action>...</action> tags."
            )

        results.append({
            "turn": turn_num,
            "raw": block,
            "think": think_match.group(1).strip(),
            "action": action_match.group(1).strip(),
        })

    return results
