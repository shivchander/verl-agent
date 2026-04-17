"""Tests for verl071.opear.prompts module.

Covers:
- Prompt construction for both compliant and violating modes
- format_trajectory output
- Response parsing: single turn, multi turn
- Error cases: wrong turn count, missing tags
- Whitespace tolerance in parsing
"""

import sys
import os
import pytest

# Ensure the verl071 package is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from verl071.opear.prompts import (
    COMPLIANT_SYSTEM,
    VIOLATING_SYSTEM,
    format_trajectory,
    build_guide_prompt,
    parse_guide_response,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TURNS = [
    {"role": "observation", "content": "You are in the kitchen. You see a countertop."},
    {"role": "assistant", "content": "<think>I should look for the mug.</think><action>go to countertop 1</action>"},
    {"role": "observation", "content": "On the countertop 1, you see a mug 1."},
    {"role": "assistant", "content": "<think>I found the mug. Pick it up.</think><action>take mug 1 from countertop 1</action>"},
]

TASK_DESC = "put a clean mug in the cabinet"

FACTS = "mug_1 is on countertop_1\ncabinet_1 is closed"


# ---------------------------------------------------------------------------
# System prompt content tests
# ---------------------------------------------------------------------------

class TestSystemPrompts:
    def test_compliant_mentions_consistent(self):
        assert "CONSISTENT" in COMPLIANT_SYSTEM

    def test_compliant_mentions_format(self):
        assert "<think>" in COMPLIANT_SYSTEM
        assert "<action>" in COMPLIANT_SYSTEM
        assert "[TURN" in COMPLIANT_SYSTEM

    def test_violating_mentions_contradict(self):
        assert "CONTRADICT" in VIOLATING_SYSTEM

    def test_violating_mentions_plausible(self):
        assert "PLAUSIBLE" in VIOLATING_SYSTEM

    def test_violating_mentions_format(self):
        assert "<think>" in VIOLATING_SYSTEM
        assert "<action>" in VIOLATING_SYSTEM
        assert "[TURN" in VIOLATING_SYSTEM

    def test_prompts_are_different(self):
        assert COMPLIANT_SYSTEM != VIOLATING_SYSTEM


# ---------------------------------------------------------------------------
# format_trajectory tests
# ---------------------------------------------------------------------------

class TestFormatTrajectory:
    def test_single_turn(self):
        turns = [{"role": "observation", "content": "You see a desk."}]
        result = format_trajectory(turns)
        assert "[OBSERVATION 1]" in result
        assert "You see a desk." in result

    def test_multi_turn(self):
        result = format_trajectory(SAMPLE_TURNS)
        assert "[OBSERVATION 1]" in result
        assert "[ASSISTANT 2]" in result
        assert "[OBSERVATION 3]" in result
        assert "[ASSISTANT 4]" in result

    def test_content_preserved(self):
        result = format_trajectory(SAMPLE_TURNS)
        assert "go to countertop 1" in result
        assert "take mug 1 from countertop 1" in result

    def test_empty_turns(self):
        result = format_trajectory([])
        assert result == ""

    def test_whitespace_stripped(self):
        turns = [{"role": "observation", "content": "  some text  \n  "}]
        result = format_trajectory(turns)
        assert "some text" in result


# ---------------------------------------------------------------------------
# build_guide_prompt tests
# ---------------------------------------------------------------------------

class TestBuildGuidePrompt:
    def test_compliant_mode(self):
        messages = build_guide_prompt(SAMPLE_TURNS, TASK_DESC, "compliant", FACTS)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == COMPLIANT_SYSTEM
        assert messages[1]["role"] == "user"

    def test_violating_mode(self):
        messages = build_guide_prompt(SAMPLE_TURNS, TASK_DESC, "violating", FACTS)
        assert len(messages) == 2
        assert messages[0]["content"] == VIOLATING_SYSTEM

    def test_user_message_contains_task(self):
        messages = build_guide_prompt(SAMPLE_TURNS, TASK_DESC, "compliant", FACTS)
        user_content = messages[1]["content"]
        assert TASK_DESC in user_content

    def test_user_message_contains_facts(self):
        messages = build_guide_prompt(SAMPLE_TURNS, TASK_DESC, "compliant", FACTS)
        user_content = messages[1]["content"]
        assert "mug_1 is on countertop_1" in user_content
        assert "cabinet_1 is closed" in user_content

    def test_user_message_contains_trajectory(self):
        messages = build_guide_prompt(SAMPLE_TURNS, TASK_DESC, "compliant", FACTS)
        user_content = messages[1]["content"]
        assert "[OBSERVATION 1]" in user_content
        assert "go to countertop 1" in user_content

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            build_guide_prompt(SAMPLE_TURNS, TASK_DESC, "invalid", FACTS)

    def test_empty_facts(self):
        messages = build_guide_prompt(SAMPLE_TURNS, TASK_DESC, "compliant", "")
        user_content = messages[1]["content"]
        assert "Privileged facts:" in user_content


# ---------------------------------------------------------------------------
# parse_guide_response tests -- single turn
# ---------------------------------------------------------------------------

class TestParseGuideResponseSingleTurn:
    def test_basic_single_turn(self):
        response = (
            "[TURN 1]\n"
            "<think>The mug is on the countertop.</think>"
            "<action>go to countertop 1</action>"
        )
        result = parse_guide_response(response, expected_turns=1)
        assert len(result) == 1
        assert result[0]["turn"] == 1
        assert result[0]["think"] == "The mug is on the countertop."
        assert result[0]["action"] == "go to countertop 1"

    def test_single_turn_with_whitespace(self):
        response = (
            "  [TURN  1]  \n\n"
            "  <think>  reasoning here  </think>  \n"
            "  <action>  look  </action>  \n"
        )
        result = parse_guide_response(response, expected_turns=1)
        assert len(result) == 1
        assert result[0]["think"] == "reasoning here"
        assert result[0]["action"] == "look"

    def test_single_turn_multiline_think(self):
        response = (
            "[TURN 1]\n"
            "<think>Line one.\n"
            "Line two.\n"
            "Line three.</think>"
            "<action>open drawer 1</action>"
        )
        result = parse_guide_response(response, expected_turns=1)
        assert "Line one." in result[0]["think"]
        assert "Line three." in result[0]["think"]


# ---------------------------------------------------------------------------
# parse_guide_response tests -- multi turn
# ---------------------------------------------------------------------------

class TestParseGuideResponseMultiTurn:
    def test_two_turns(self):
        response = (
            "[TURN 1]\n"
            "<think>First reasoning.</think><action>go to shelf 1</action>\n\n"
            "[TURN 2]\n"
            "<think>Second reasoning.</think><action>take book 1 from shelf 1</action>"
        )
        result = parse_guide_response(response, expected_turns=2)
        assert len(result) == 2
        assert result[0]["turn"] == 1
        assert result[0]["action"] == "go to shelf 1"
        assert result[1]["turn"] == 2
        assert result[1]["action"] == "take book 1 from shelf 1"

    def test_three_turns(self):
        response = (
            "[TURN 1]\n"
            "<think>A</think><action>look</action>\n"
            "[TURN 2]\n"
            "<think>B</think><action>go to desk 1</action>\n"
            "[TURN 3]\n"
            "<think>C</think><action>open drawer 1</action>\n"
        )
        result = parse_guide_response(response, expected_turns=3)
        assert len(result) == 3
        assert [r["think"] for r in result] == ["A", "B", "C"]
        assert result[2]["action"] == "open drawer 1"

    def test_raw_block_preserved(self):
        response = (
            "[TURN 1]\n"
            "<think>reason</think><action>act</action>\n"
            "[TURN 2]\n"
            "<think>reason2</think><action>act2</action>"
        )
        result = parse_guide_response(response, expected_turns=2)
        assert "<think>reason</think>" in result[0]["raw"]
        assert "<action>act2</action>" in result[1]["raw"]


# ---------------------------------------------------------------------------
# parse_guide_response tests -- error cases
# ---------------------------------------------------------------------------

class TestParseGuideResponseErrors:
    def test_wrong_turn_count_too_few(self):
        response = (
            "[TURN 1]\n"
            "<think>Only one.</think><action>look</action>"
        )
        with pytest.raises(ValueError, match="Expected 2 turn"):
            parse_guide_response(response, expected_turns=2)

    def test_wrong_turn_count_too_many(self):
        response = (
            "[TURN 1]\n<think>A</think><action>a</action>\n"
            "[TURN 2]\n<think>B</think><action>b</action>\n"
            "[TURN 3]\n<think>C</think><action>c</action>"
        )
        with pytest.raises(ValueError, match="Expected 2 turn"):
            parse_guide_response(response, expected_turns=2)

    def test_missing_think_tag(self):
        response = (
            "[TURN 1]\n"
            "<action>go to desk 1</action>"
        )
        with pytest.raises(ValueError, match="missing <think>"):
            parse_guide_response(response, expected_turns=1)

    def test_missing_action_tag(self):
        response = (
            "[TURN 1]\n"
            "<think>I should go to the desk.</think>"
        )
        with pytest.raises(ValueError, match="missing <action>"):
            parse_guide_response(response, expected_turns=1)

    def test_no_markers_at_all(self):
        response = "<think>No markers.</think><action>look</action>"
        with pytest.raises(ValueError, match="Expected 1 turn.*found 0"):
            parse_guide_response(response, expected_turns=1)

    def test_missing_tags_in_second_turn(self):
        response = (
            "[TURN 1]\n"
            "<think>OK</think><action>look</action>\n"
            "[TURN 2]\n"
            "I forgot the tags."
        )
        with pytest.raises(ValueError, match="TURN 2.*missing <think>"):
            parse_guide_response(response, expected_turns=2)

    def test_zero_expected_turns_empty_response(self):
        result = parse_guide_response("", expected_turns=0)
        assert result == []


# ---------------------------------------------------------------------------
# Whitespace robustness
# ---------------------------------------------------------------------------

class TestWhitespaceRobustness:
    def test_case_insensitive_turn_marker(self):
        response = (
            "[turn 1]\n"
            "<think>reason</think><action>look</action>"
        )
        result = parse_guide_response(response, expected_turns=1)
        assert result[0]["action"] == "look"

    def test_extra_spaces_in_marker(self):
        response = (
            "[TURN   1]\n"
            "<think>reason</think><action>look</action>"
        )
        result = parse_guide_response(response, expected_turns=1)
        assert result[0]["turn"] == 1

    def test_case_insensitive_tags(self):
        response = (
            "[TURN 1]\n"
            "<Think>reason</Think><Action>look</Action>"
        )
        result = parse_guide_response(response, expected_turns=1)
        assert result[0]["think"] == "reason"
        assert result[0]["action"] == "look"
