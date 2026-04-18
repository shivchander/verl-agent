"""Tests for verl071.opear.guide module and data.select_batch_positions.

Includes a live API test against GPT-5.4-nano with a 1-turn ALFWorld
trajectory, plus unit tests for per-group batch position selection.
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

class TestSelectBatchPositions:
    """Tests for data.select_batch_positions (per-group rollout selection)."""

    def _make_batch(self, uids):
        """Create a minimal mock batch with the given uid list."""
        from types import SimpleNamespace
        return SimpleNamespace(non_tensor_batch={"uid": uids})

    def test_basic_per_group_selection(self):
        from verl071.opear.data import select_batch_positions
        # 4 tasks, 3 rollouts each = 12 total
        uids = ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
        batch = self._make_batch(uids)
        selected = select_batch_positions(batch, group_size=3, selection_ratio=0.5)
        # floor(0.5 * 3) = 1 per group, 4 groups = 4 total
        assert len(selected) == 4
        # One from each group
        groups_hit = {uids[i] for i in selected}
        assert groups_hit == {"a", "b", "c", "d"}

    def test_beta_half_group8(self):
        from verl071.opear.data import select_batch_positions
        # 16 tasks, 8 rollouts each = 128 total (matches real config)
        uids = [str(i) for i in range(16) for _ in range(8)]
        batch = self._make_batch(uids)
        selected = select_batch_positions(batch, group_size=8, selection_ratio=0.5)
        # floor(0.5 * 8) = 4 per group, 16 groups = 64 total
        assert len(selected) == 64
        assert len(set(selected)) == 64  # no duplicates

    def test_no_duplicates(self):
        from verl071.opear.data import select_batch_positions
        uids = ["a"] * 8 + ["b"] * 8
        batch = self._make_batch(uids)
        selected = select_batch_positions(batch, group_size=8, selection_ratio=0.5)
        assert len(selected) == len(set(selected))

    def test_small_group(self):
        from verl071.opear.data import select_batch_positions
        # Group has fewer members than k_per_group
        uids = ["a", "a", "b", "b"]
        batch = self._make_batch(uids)
        selected = select_batch_positions(batch, group_size=8, selection_ratio=0.5)
        # floor(0.5 * 8) = 4, but each group only has 2
        assert len(selected) == 4  # 2 from each group

    def test_empty_batch(self):
        from verl071.opear.data import select_batch_positions
        batch = self._make_batch([])
        selected = select_batch_positions(batch, group_size=8, selection_ratio=0.5)
        assert selected == []

    def test_no_uid_field(self):
        from verl071.opear.data import select_batch_positions
        from types import SimpleNamespace
        batch = SimpleNamespace(non_tensor_batch={})
        selected = select_batch_positions(batch, group_size=8, selection_ratio=0.5)
        assert selected == []
