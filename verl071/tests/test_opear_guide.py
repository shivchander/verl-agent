"""Tests for verl071.opear.guide module.

Includes a live API test against GPT-5.4-nano with a 1-turn ALFWorld
trajectory, plus unit tests for select_rollouts.
"""

import sys
import os

import pytest

# Ensure the verl071 package is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Load .env so the API key is available.
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from verl071.opear.guide import OPEaRGuide


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TURNS_1 = [
    {
        "role": "observation",
        "content": (
            "You are in the middle of a room. Looking quickly around you, "
            "you see a countertop 1, a cabinet 1, a stoveburner 1, "
            "a fridge 1, and a sinkbasin 1.\n"
            "Your task is to: put a clean mug in the cabinet.\n"
            "Admissible actions: go to countertop 1, go to cabinet 1, "
            "go to stoveburner 1, go to fridge 1, go to sinkbasin 1, look"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "<think>I need to find a mug first. Let me check the "
            "countertop.</think><action>go to countertop 1</action>"
        ),
    },
]

TASK_DESC_1 = "put a clean mug in the cabinet"

FACTS_1 = (
    "mug_1 is on countertop_1\n"
    "mug_1 is dirty\n"
    "cabinet_1 is closed\n"
    "sinkbasin_1 is available"
)


# ---------------------------------------------------------------------------
# Live API test
# ---------------------------------------------------------------------------

class TestOPEaRGuideLive:
    """Live API test with GPT-5.4-nano."""

    @pytest.fixture
    def guide(self):
        return OPEaRGuide(
            model="gpt-5.4-nano",
            max_completion_tokens=4096,
            temperature=0.7,
            max_concurrent=4,
        )

    def test_generate_pair_single_turn(self, guide):
        """Call GPT-5.4-nano with a 1-turn trajectory and verify structure."""
        result = guide.generate_contrastive_batch([
            {
                "turns": SAMPLE_TURNS_1,
                "task_description": TASK_DESC_1,
                "facts": FACTS_1,
            }
        ])

        assert len(result) == 1
        pair = result[0]

        # Pair should not be None
        assert pair is not None, "API call returned None -- check OPENAI_API_KEY and connectivity"

        # Must have both keys
        assert "compliant" in pair
        assert "violating" in pair

        # Each should contain exactly 1 parsed turn (1 assistant turn in input)
        assert len(pair["compliant"]) == 1
        assert len(pair["violating"]) == 1

        # Each parsed turn should have think and action keys with non-empty strings
        for mode in ("compliant", "violating"):
            turn_dict = pair[mode][0]
            assert "think" in turn_dict, f"{mode} missing 'think' key"
            assert "action" in turn_dict, f"{mode} missing 'action' key"
            assert isinstance(turn_dict["think"], str)
            assert isinstance(turn_dict["action"], str)
            assert len(turn_dict["think"]) > 0, f"{mode} 'think' is empty"
            assert len(turn_dict["action"]) > 0, f"{mode} 'action' is empty"


# ---------------------------------------------------------------------------
# Unit tests for select_rollouts
# ---------------------------------------------------------------------------

class TestSelectRollouts:
    @pytest.fixture
    def guide(self):
        # Use a dummy key -- select_rollouts does not call the API
        orig_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key"
        g = OPEaRGuide(model="gpt-5.4-nano", beta=0.5)
        yield g
        # Restore original key
        if orig_key is not None:
            os.environ["OPENAI_API_KEY"] = orig_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)

    def test_basic_selection(self, guide):
        uids = list(range(10))
        selected = guide.select_rollouts(uids, group_size=8)
        # floor(0.5 * 8) = 4
        assert len(selected) == 4
        assert all(uid in uids for uid in selected)

    def test_no_duplicates(self, guide):
        uids = list(range(20))
        selected = guide.select_rollouts(uids, group_size=10)
        assert len(selected) == len(set(selected))

    def test_fewer_uids_than_needed(self, guide):
        uids = [0, 1]
        selected = guide.select_rollouts(uids, group_size=10)
        # floor(0.5 * 10) = 5, but only 2 available
        assert len(selected) == 2

    def test_zero_group_size(self, guide):
        uids = list(range(5))
        selected = guide.select_rollouts(uids, group_size=0)
        assert selected == []

    def test_empty_uids(self, guide):
        selected = guide.select_rollouts([], group_size=8)
        assert selected == []
