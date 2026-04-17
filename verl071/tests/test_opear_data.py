"""Tests for O-PEaR data packing (reconstruct_trajectories, tokenize_contrastive_responses).

Covers:
- reconstruct_trajectories: grouping by traj_uid, active mask filtering,
  task description extraction, facts extraction, turn structure
- tokenize_contrastive_responses: correct shapes, mask values, None-pair
  skipping, parsed-dict and string inputs, multiple trajectories, empty result
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

# Ensure the verl071 package is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from verl071.opear.data import reconstruct_trajectories, tokenize_contrastive_responses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_tokenizer(token_count: int = 10):
    """Create a mock tokenizer that returns a fixed number of token IDs."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.encode = MagicMock(
        side_effect=lambda text, add_special_tokens=False: list(
            range(1, token_count + 1)
        )
    )
    tokenizer.decode = MagicMock(
        side_effect=lambda ids, skip_special_tokens=True: "decoded text"
    )
    return tokenizer


def _make_mock_batch(
    n_rows: int,
    prompt_len: int,
    response_len: int,
    traj_uids: list[str],
    active_masks: list[bool] | None = None,
    facts: list[str] | None = None,
):
    """Create a mock DataProto-like batch object."""
    total_len = prompt_len + response_len
    batch = MagicMock()
    batch.batch = {
        "prompts": torch.ones(n_rows, prompt_len, dtype=torch.long),
        "responses": torch.ones(n_rows, response_len, dtype=torch.long) * 2,
        "attention_mask": torch.ones(n_rows, total_len, dtype=torch.long),
    }
    non_tensor = {
        "traj_uid": np.array(traj_uids, dtype=object),
    }
    if active_masks is not None:
        non_tensor["active_masks"] = np.array(active_masks, dtype=object)
    if facts is not None:
        non_tensor["facts_str"] = np.array(facts, dtype=object)

    batch.non_tensor_batch = non_tensor
    return batch


# ---------------------------------------------------------------------------
# reconstruct_trajectories tests
# ---------------------------------------------------------------------------

class TestReconstructTrajectories:
    """Test grouping and decoding of per-turn batch into trajectories."""

    def test_single_trajectory_single_turn(self):
        tokenizer = _make_mock_tokenizer()
        batch = _make_mock_batch(
            n_rows=1,
            prompt_len=10,
            response_len=5,
            traj_uids=["uid-a"],
            active_masks=[True],
        )

        result = reconstruct_trajectories(batch, tokenizer)

        assert len(result) == 1
        traj = result[0]
        assert traj["traj_uid"] == "uid-a"
        assert traj["turn_indices"] == [0]
        # Two entries per turn: observation + assistant
        assert len(traj["turns"]) == 2
        assert traj["turns"][0]["role"] == "observation"
        assert traj["turns"][1]["role"] == "assistant"

    def test_two_trajectories(self):
        tokenizer = _make_mock_tokenizer()
        batch = _make_mock_batch(
            n_rows=3,
            prompt_len=10,
            response_len=5,
            traj_uids=["uid-a", "uid-a", "uid-b"],
            active_masks=[True, True, True],
        )

        result = reconstruct_trajectories(batch, tokenizer)

        assert len(result) == 2
        uids = {t["traj_uid"] for t in result}
        assert uids == {"uid-a", "uid-b"}
        # uid-a has 2 active turns -> 4 turn entries (obs+asst each)
        traj_a = [t for t in result if t["traj_uid"] == "uid-a"][0]
        assert traj_a["turn_indices"] == [0, 1]
        assert len(traj_a["turns"]) == 4

    def test_inactive_turns_filtered(self):
        tokenizer = _make_mock_tokenizer()
        batch = _make_mock_batch(
            n_rows=3,
            prompt_len=10,
            response_len=5,
            traj_uids=["uid-a", "uid-a", "uid-a"],
            active_masks=[True, False, True],
        )

        result = reconstruct_trajectories(batch, tokenizer)

        assert len(result) == 1
        traj = result[0]
        assert traj["turn_indices"] == [0, 2]  # index 1 was inactive
        assert len(traj["turns"]) == 4

    def test_all_inactive_skipped(self):
        tokenizer = _make_mock_tokenizer()
        batch = _make_mock_batch(
            n_rows=2,
            prompt_len=10,
            response_len=5,
            traj_uids=["uid-a", "uid-a"],
            active_masks=[False, False],
        )

        result = reconstruct_trajectories(batch, tokenizer)
        assert len(result) == 0

    def test_empty_batch(self):
        batch = MagicMock()
        batch.non_tensor_batch = {"traj_uid": np.array([])}
        tokenizer = _make_mock_tokenizer()

        result = reconstruct_trajectories(batch, tokenizer)
        assert result == []

    def test_task_description_extracted(self):
        """Task description is extracted from 'Your task is to: ...' in observation."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.decode = MagicMock(
            return_value="You are an agent. Your task is to: put a clean mug in the cabinet\nMore text here."
        )

        batch = _make_mock_batch(
            n_rows=1,
            prompt_len=10,
            response_len=5,
            traj_uids=["uid-a"],
            active_masks=[True],
        )

        result = reconstruct_trajectories(batch, tokenizer)

        assert len(result) == 1
        assert result[0]["task_description"] == "put a clean mug in the cabinet"

    def test_task_description_default(self):
        """Falls back to default when marker is not found."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.decode = MagicMock(return_value="No task marker here.")

        batch = _make_mock_batch(
            n_rows=1,
            prompt_len=10,
            response_len=5,
            traj_uids=["uid-a"],
            active_masks=[True],
        )

        result = reconstruct_trajectories(batch, tokenizer)
        assert result[0]["task_description"] == "Complete the task."

    def test_facts_from_first_active_turn(self):
        tokenizer = _make_mock_tokenizer()
        batch = _make_mock_batch(
            n_rows=2,
            prompt_len=10,
            response_len=5,
            traj_uids=["uid-a", "uid-a"],
            active_masks=[True, True],
            facts=["mug_1 on counter; lamp_1 on desk", "plate_1 on table"],
        )

        result = reconstruct_trajectories(batch, tokenizer)

        assert len(result) == 1
        assert result[0]["facts"] == "mug_1 on counter; lamp_1 on desk"

    def test_no_active_masks_defaults_to_all_active(self):
        """When active_masks is not in non_tensor_batch, all turns are active."""
        tokenizer = _make_mock_tokenizer()
        batch = _make_mock_batch(
            n_rows=2,
            prompt_len=10,
            response_len=5,
            traj_uids=["uid-a", "uid-a"],
            active_masks=None,  # not provided
        )

        result = reconstruct_trajectories(batch, tokenizer)

        assert len(result) == 1
        assert result[0]["turn_indices"] == [0, 1]


# ---------------------------------------------------------------------------
# tokenize_contrastive_responses tests
# ---------------------------------------------------------------------------

class TestTokenizeContrastiveBasic:
    """Test basic tokenization with correct shapes and mask values."""

    def test_basic_shapes(self):
        tokenizer = _make_mock_tokenizer(token_count=10)

        prompt_len = 20
        max_response_length = 15
        batch = _make_mock_batch(
            n_rows=2,
            prompt_len=prompt_len,
            response_len=max_response_length,
            traj_uids=["uid-a"],
        )

        trajectories = [
            {
                "traj_uid": "uid-a",
                "task_description": "test",
                "turns": [
                    {"role": "observation", "content": "obs"},
                    {"role": "assistant", "content": "resp"},
                ],
                "turn_indices": [0],
            },
        ]

        contrastive_pairs = [
            {
                "compliant": [
                    {"turn": 1, "raw": "...", "think": "good", "action": "go"}
                ],
                "violating": [
                    {"turn": 1, "raw": "...", "think": "bad", "action": "stay"}
                ],
            },
        ]

        result = tokenize_contrastive_responses(
            trajectories, contrastive_pairs, batch, tokenizer, max_response_length
        )

        assert result is not None
        assert result["compliant_input_ids"].shape == (
            1,
            prompt_len + max_response_length,
        )
        assert result["violating_input_ids"].shape == (
            1,
            prompt_len + max_response_length,
        )
        assert result["compliant_attention_mask"].shape == (
            1,
            prompt_len + max_response_length,
        )
        assert result["compliant_response_mask"].shape == (1, max_response_length)
        assert result["violating_response_mask"].shape == (1, max_response_length)

    def test_response_mask_values(self):
        """Response mask: 10 ones (real tokens) + 5 zeros (padding)."""
        tokenizer = _make_mock_tokenizer(token_count=10)

        prompt_len = 20
        max_response_length = 15
        batch = _make_mock_batch(
            n_rows=1,
            prompt_len=prompt_len,
            response_len=max_response_length,
            traj_uids=["uid-a"],
        )

        trajectories = [
            {
                "traj_uid": "uid-a",
                "task_description": "test",
                "turns": [
                    {"role": "observation", "content": "obs"},
                    {"role": "assistant", "content": "resp"},
                ],
                "turn_indices": [0],
            },
        ]

        contrastive_pairs = [
            {
                "compliant": [
                    {"turn": 1, "raw": "...", "think": "good", "action": "go"}
                ],
                "violating": [
                    {"turn": 1, "raw": "...", "think": "bad", "action": "stay"}
                ],
            },
        ]

        result = tokenize_contrastive_responses(
            trajectories, contrastive_pairs, batch, tokenizer, max_response_length
        )

        assert result is not None
        # 10 real tokens, 5 padding
        assert result["compliant_response_mask"][0, :10].sum().item() == 10
        assert result["compliant_response_mask"][0, 10:].sum().item() == 0
        assert result["violating_response_mask"][0, :10].sum().item() == 10
        assert result["violating_response_mask"][0, 10:].sum().item() == 0

    def test_attention_mask_values(self):
        """Attention mask: 1s for prompt + real response, 0s for padding."""
        tokenizer = _make_mock_tokenizer(token_count=10)

        prompt_len = 20
        max_response_length = 15
        batch = _make_mock_batch(
            n_rows=1,
            prompt_len=prompt_len,
            response_len=max_response_length,
            traj_uids=["uid-a"],
        )

        trajectories = [
            {
                "traj_uid": "uid-a",
                "task_description": "test",
                "turns": [
                    {"role": "observation", "content": "obs"},
                    {"role": "assistant", "content": "resp"},
                ],
                "turn_indices": [0],
            },
        ]

        contrastive_pairs = [
            {
                "compliant": [
                    {"turn": 1, "raw": "...", "think": "r", "action": "a"}
                ],
                "violating": [
                    {"turn": 1, "raw": "...", "think": "r", "action": "a"}
                ],
            },
        ]

        result = tokenize_contrastive_responses(
            trajectories, contrastive_pairs, batch, tokenizer, max_response_length
        )

        assert result is not None
        attn = result["compliant_attention_mask"][0]
        # Prompt portion: all 1s (our mock batch has all-ones attention mask)
        assert attn[:prompt_len].sum().item() == prompt_len
        # Response portion: 10 ones + 5 zeros
        assert attn[prompt_len : prompt_len + 10].sum().item() == 10
        assert attn[prompt_len + 10 :].sum().item() == 0


class TestTokenizeContrastiveNonePairs:
    """Test that None pairs are skipped."""

    def test_none_pair_skipped(self):
        tokenizer = _make_mock_tokenizer(token_count=3)

        batch = _make_mock_batch(
            n_rows=2,
            prompt_len=10,
            response_len=10,
            traj_uids=["a", "b"],
        )

        trajectories = [
            {
                "traj_uid": "a",
                "turns": [
                    {"role": "observation", "content": "obs"},
                    {"role": "assistant", "content": "resp"},
                ],
                "turn_indices": [0],
                "task_description": "x",
            },
            {
                "traj_uid": "b",
                "turns": [
                    {"role": "observation", "content": "obs"},
                    {"role": "assistant", "content": "resp"},
                ],
                "turn_indices": [1],
                "task_description": "y",
            },
        ]
        pairs = [
            None,
            {
                "compliant": [
                    {"turn": 1, "raw": "c", "think": "c", "action": "c"}
                ],
                "violating": [
                    {"turn": 1, "raw": "v", "think": "v", "action": "v"}
                ],
            },
        ]

        result = tokenize_contrastive_responses(
            trajectories, pairs, batch, tokenizer, 10
        )
        assert result is not None
        assert result["compliant_input_ids"].shape[0] == 1  # only one valid pair

    def test_all_none_returns_none(self):
        tokenizer = _make_mock_tokenizer()

        batch = _make_mock_batch(
            n_rows=2,
            prompt_len=10,
            response_len=10,
            traj_uids=["a", "b"],
        )

        trajectories = [
            {"traj_uid": "a", "turns": [], "turn_indices": [0], "task_description": "x"},
            {"traj_uid": "b", "turns": [], "turn_indices": [1], "task_description": "y"},
        ]
        pairs = [None, None]

        result = tokenize_contrastive_responses(
            trajectories, pairs, batch, tokenizer, 10
        )
        assert result is None


class TestTokenizeContrastiveMultiTrajectory:
    """Test multiple trajectories with mixed success/failure."""

    def test_mixed_none_and_valid(self):
        """Two out of three trajectories have valid pairs."""
        tokenizer = _make_mock_tokenizer(token_count=5)

        batch = _make_mock_batch(
            n_rows=3,
            prompt_len=8,
            response_len=10,
            traj_uids=["a", "b", "c"],
        )

        trajectories = [
            {"traj_uid": "a", "turns": [{"role": "observation", "content": "o"}, {"role": "assistant", "content": "r"}], "turn_indices": [0], "task_description": "ta"},
            {"traj_uid": "b", "turns": [{"role": "observation", "content": "o"}, {"role": "assistant", "content": "r"}], "turn_indices": [1], "task_description": "tb"},
            {"traj_uid": "c", "turns": [{"role": "observation", "content": "o"}, {"role": "assistant", "content": "r"}], "turn_indices": [2], "task_description": "tc"},
        ]

        pairs = [
            {"compliant": [{"turn": 1, "raw": "", "think": "t1", "action": "a1"}],
             "violating": [{"turn": 1, "raw": "", "think": "t1v", "action": "a1v"}]},
            None,
            {"compliant": [{"turn": 1, "raw": "", "think": "t3", "action": "a3"}],
             "violating": [{"turn": 1, "raw": "", "think": "t3v", "action": "a3v"}]},
        ]

        result = tokenize_contrastive_responses(
            trajectories, pairs, batch, tokenizer, 10
        )

        assert result is not None
        assert result["compliant_input_ids"].shape[0] == 2
        assert result["violating_input_ids"].shape[0] == 2

    def test_multi_turn_trajectory(self):
        """A trajectory with 2 turns produces 2 entries."""
        tokenizer = _make_mock_tokenizer(token_count=4)

        batch = _make_mock_batch(
            n_rows=2,
            prompt_len=8,
            response_len=10,
            traj_uids=["a", "a"],
        )

        trajectories = [
            {
                "traj_uid": "a",
                "task_description": "test",
                "turns": [
                    {"role": "observation", "content": "obs1"},
                    {"role": "assistant", "content": "resp1"},
                    {"role": "observation", "content": "obs2"},
                    {"role": "assistant", "content": "resp2"},
                ],
                "turn_indices": [0, 1],
            },
        ]

        pairs = [
            {
                "compliant": [
                    {"turn": 1, "raw": "", "think": "c1", "action": "ca1"},
                    {"turn": 2, "raw": "", "think": "c2", "action": "ca2"},
                ],
                "violating": [
                    {"turn": 1, "raw": "", "think": "v1", "action": "va1"},
                    {"turn": 2, "raw": "", "think": "v2", "action": "va2"},
                ],
            },
        ]

        result = tokenize_contrastive_responses(
            trajectories, pairs, batch, tokenizer, 10
        )

        assert result is not None
        assert result["compliant_input_ids"].shape[0] == 2
        assert result["violating_input_ids"].shape[0] == 2


class TestTokenizeContrastiveStringInput:
    """Test that plain string inputs work alongside parsed dicts."""

    def test_plain_string_turns(self):
        tokenizer = _make_mock_tokenizer(token_count=10)

        prompt_len = 20
        max_response_length = 15
        batch = _make_mock_batch(
            n_rows=1,
            prompt_len=prompt_len,
            response_len=max_response_length,
            traj_uids=["uid-a"],
        )

        trajectories = [
            {
                "traj_uid": "uid-a",
                "task_description": "test",
                "turns": [
                    {"role": "observation", "content": "obs"},
                    {"role": "assistant", "content": "resp"},
                ],
                "turn_indices": [0],
            },
        ]

        # Use plain strings instead of parsed dicts
        contrastive_pairs = [
            {
                "compliant": ["<think>good</think><action>go</action>"],
                "violating": ["<think>bad</think><action>stay</action>"],
            },
        ]

        result = tokenize_contrastive_responses(
            trajectories, contrastive_pairs, batch, tokenizer, max_response_length
        )

        assert result is not None
        assert result["compliant_input_ids"].shape == (
            1,
            prompt_len + max_response_length,
        )
        assert result["compliant_response_mask"][0, :10].sum().item() == 10
        assert result["compliant_response_mask"][0, 10:].sum().item() == 0


class TestTokenizeContrastiveTruncation:
    """Test that responses longer than max_response_length are truncated."""

    def test_truncation(self):
        # Tokenizer returns 10 tokens, but max_response_length is 6
        tokenizer = _make_mock_tokenizer(token_count=10)
        max_response_length = 6

        batch = _make_mock_batch(
            n_rows=1,
            prompt_len=8,
            response_len=max_response_length,
            traj_uids=["uid-a"],
        )

        trajectories = [
            {
                "traj_uid": "uid-a",
                "task_description": "test",
                "turns": [
                    {"role": "observation", "content": "obs"},
                    {"role": "assistant", "content": "resp"},
                ],
                "turn_indices": [0],
            },
        ]

        contrastive_pairs = [
            {
                "compliant": [
                    {"turn": 1, "raw": "", "think": "long", "action": "text"}
                ],
                "violating": [
                    {"turn": 1, "raw": "", "think": "long", "action": "text"}
                ],
            },
        ]

        result = tokenize_contrastive_responses(
            trajectories, contrastive_pairs, batch, tokenizer, max_response_length
        )

        assert result is not None
        assert result["compliant_response_mask"].shape == (1, max_response_length)
        # All 6 slots filled with real tokens (truncated from 10 -> 6)
        assert result["compliant_response_mask"][0].sum().item() == 6
        # No padding
        assert result["compliant_response_mask"][0].min().item() == 1.0


class TestTokenizeContrastiveParsedDictFormat:
    """Verify that parsed dicts are reconstructed into the correct text format."""

    def test_reconstructed_text_format(self):
        """The reconstructed text should be <think>...</think><action>...</action>."""
        call_log = []
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        def _capture_encode(text, add_special_tokens=False):
            call_log.append(text)
            return [1, 2, 3]

        tokenizer.encode = _capture_encode

        batch = _make_mock_batch(
            n_rows=1,
            prompt_len=5,
            response_len=10,
            traj_uids=["uid-a"],
        )

        trajectories = [
            {
                "traj_uid": "uid-a",
                "task_description": "test",
                "turns": [
                    {"role": "observation", "content": "obs"},
                    {"role": "assistant", "content": "resp"},
                ],
                "turn_indices": [0],
            },
        ]

        contrastive_pairs = [
            {
                "compliant": [
                    {
                        "turn": 1,
                        "raw": "raw text",
                        "think": "I see a mug on the counter",
                        "action": "take mug 1 from countertop 1",
                    }
                ],
                "violating": [
                    {
                        "turn": 1,
                        "raw": "raw text",
                        "think": "The mug is probably in the cabinet",
                        "action": "go to cabinet 1",
                    }
                ],
            },
        ]

        tokenize_contrastive_responses(
            trajectories, contrastive_pairs, batch, tokenizer, 10
        )

        assert len(call_log) == 2
        assert (
            call_log[0]
            == "<think>I see a mug on the counter</think><action>take mug 1 from countertop 1</action>"
        )
        assert (
            call_log[1]
            == "<think>The mug is probably in the cabinet</think><action>go to cabinet 1</action>"
        )


class TestTokenizeContrastiveEdgeCases:
    """Edge cases for tokenize_contrastive_responses."""

    def test_empty_trajectories(self):
        tokenizer = _make_mock_tokenizer()
        batch = _make_mock_batch(
            n_rows=0,
            prompt_len=5,
            response_len=5,
            traj_uids=[],
        )

        result = tokenize_contrastive_responses([], [], batch, tokenizer, 10)
        assert result is None

    def test_fewer_guide_turns_than_batch_turns(self):
        """When the guide produced fewer turns, only the matching subset is used."""
        tokenizer = _make_mock_tokenizer(token_count=4)

        batch = _make_mock_batch(
            n_rows=3,
            prompt_len=8,
            response_len=10,
            traj_uids=["a", "a", "a"],
        )

        trajectories = [
            {
                "traj_uid": "a",
                "task_description": "test",
                "turns": [
                    {"role": "observation", "content": "o1"},
                    {"role": "assistant", "content": "r1"},
                    {"role": "observation", "content": "o2"},
                    {"role": "assistant", "content": "r2"},
                    {"role": "observation", "content": "o3"},
                    {"role": "assistant", "content": "r3"},
                ],
                "turn_indices": [0, 1, 2],  # 3 batch turns
            },
        ]

        # Guide only returned 2 turns instead of 3
        pairs = [
            {
                "compliant": [
                    {"turn": 1, "raw": "", "think": "c1", "action": "ca1"},
                    {"turn": 2, "raw": "", "think": "c2", "action": "ca2"},
                ],
                "violating": [
                    {"turn": 1, "raw": "", "think": "v1", "action": "va1"},
                    {"turn": 2, "raw": "", "think": "v2", "action": "va2"},
                ],
            },
        ]

        result = tokenize_contrastive_responses(
            trajectories, pairs, batch, tokenizer, 10
        )

        assert result is not None
        # Only 2 entries, not 3
        assert result["compliant_input_ids"].shape[0] == 2

    def test_pad_token_id_none_falls_back_to_zero(self):
        """If tokenizer.pad_token_id is None, pad with 0."""
        tokenizer = _make_mock_tokenizer(token_count=3)
        tokenizer.pad_token_id = None

        batch = _make_mock_batch(
            n_rows=1,
            prompt_len=5,
            response_len=10,
            traj_uids=["a"],
        )

        trajectories = [
            {
                "traj_uid": "a",
                "task_description": "test",
                "turns": [
                    {"role": "observation", "content": "obs"},
                    {"role": "assistant", "content": "resp"},
                ],
                "turn_indices": [0],
            },
        ]

        pairs = [
            {
                "compliant": [{"turn": 1, "raw": "", "think": "t", "action": "a"}],
                "violating": [{"turn": 1, "raw": "", "think": "t", "action": "a"}],
            },
        ]

        result = tokenize_contrastive_responses(
            trajectories, pairs, batch, tokenizer, 10
        )

        assert result is not None
        # Padding tokens (indices 3..9) should be 0
        resp_portion = result["compliant_input_ids"][0, 5:]  # after prompt
        assert resp_portion[3:].sum().item() == 0
