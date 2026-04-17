"""Tests for PDDL facts capture from TextWorld environments.

Verifies that textworld.EnvInfos(facts=True) returns PDDL propositions
on reset and step, and that the serialization logic produces valid strings.
"""

from __future__ import annotations

import pytest
import textworld
import textworld.gym


@pytest.fixture(scope="module")
def game_file(tmp_path_factory):
    """Generate a simple TextWorld game file for testing."""
    options = textworld.GameOptions()
    options.nb_rooms = 1
    options.nb_objects = 2
    options.quest_length = 1
    tmp_dir = tmp_path_factory.mktemp("tw_games")
    options.path = str(tmp_dir / "test_game")
    game = textworld.generator.make_game(options)
    return textworld.generator.compile_game(game, options)


class TestFactsCaptureReset:
    """Verify facts are returned on environment reset."""

    def test_reset_returns_facts_key(self, game_file):
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True)
        env_id = textworld.gym.register_games([game_file], request_infos, max_episode_steps=50)
        env = textworld.gym.make(env_id)
        try:
            _, infos = env.reset()
            assert "facts" in infos, "infos should contain 'facts' key after reset"
        finally:
            env.close()

    def test_reset_facts_are_nonempty(self, game_file):
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True)
        env_id = textworld.gym.register_games([game_file], request_infos, max_episode_steps=50)
        env = textworld.gym.make(env_id)
        try:
            _, infos = env.reset()
            assert len(infos["facts"]) > 0, "facts should be non-empty on reset"
        finally:
            env.close()

    def test_reset_facts_are_propositions(self, game_file):
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True)
        env_id = textworld.gym.register_games([game_file], request_infos, max_episode_steps=50)
        env = textworld.gym.make(env_id)
        try:
            _, infos = env.reset()
            from textworld.logic import Proposition
            for fact in infos["facts"]:
                assert isinstance(fact, Proposition), (
                    f"Each fact should be a Proposition, got {type(fact)}"
                )
        finally:
            env.close()


class TestFactsCaptureStep:
    """Verify facts are returned and may change after a step."""

    def test_step_returns_facts(self, game_file):
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True)
        env_id = textworld.gym.register_games([game_file], request_infos, max_episode_steps=50)
        env = textworld.gym.make(env_id)
        try:
            _, infos = env.reset()
            cmds = infos["admissible_commands"]
            assert len(cmds) > 0, "Need at least one admissible command"
            _, _, _, step_infos = env.step(cmds[0])
            assert "facts" in step_infos, "infos should contain 'facts' after step"
            assert len(step_infos["facts"]) > 0, "facts should be non-empty after step"
        finally:
            env.close()

    def test_facts_are_serializable(self, game_file):
        """Facts can be serialized to a semicolon-separated string."""
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True)
        env_id = textworld.gym.register_games([game_file], request_infos, max_episode_steps=50)
        env = textworld.gym.make(env_id)
        try:
            _, infos = env.reset()
            facts_str = "; ".join(str(f) for f in infos["facts"])
            assert len(facts_str) > 0, "Serialized facts string should be non-empty"
            assert "(" in facts_str, "Serialized facts should contain predicate notation"
        finally:
            env.close()


class TestFactsCaptureBatched:
    """Verify facts work with batched environments (as used by ALFWorld)."""

    def test_batched_reset_facts(self, game_file):
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True)
        env_id = textworld.gym.register_games(
            [game_file], request_infos, max_episode_steps=50, batch_size=2
        )
        env = textworld.gym.make(env_id)
        try:
            _, infos = env.reset()
            assert "facts" in infos, "Batched infos should contain 'facts'"
            assert len(infos["facts"]) == 2, "Should have facts for each env in batch"
            for i, facts in enumerate(infos["facts"]):
                assert len(facts) > 0, f"Facts for env {i} should be non-empty"
        finally:
            env.close()

    def test_batched_step_facts(self, game_file):
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True)
        env_id = textworld.gym.register_games(
            [game_file], request_infos, max_episode_steps=50, batch_size=2
        )
        env = textworld.gym.make(env_id)
        try:
            _, infos = env.reset()
            cmds = infos["admissible_commands"]
            # Pick first admissible command for each env
            actions = [c[0] for c in cmds]
            _, _, _, step_infos = env.step(actions)
            assert "facts" in step_infos, "Batched step infos should contain 'facts'"
            assert len(step_infos["facts"]) == 2
        finally:
            env.close()


class TestFactsSerialization:
    """Test the serialization logic used in AlfWorldEnvironmentManager.step()."""

    def test_serialize_nonempty_facts(self, game_file):
        """Reproduce the serialization from env_manager.py."""
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True)
        env_id = textworld.gym.register_games([game_file], request_infos, max_episode_steps=50)
        env = textworld.gym.make(env_id)
        try:
            _, infos = env.reset()
            raw_facts = infos.get("facts", [])
            assert raw_facts, "Should have facts for serialization test"
            facts_str = "; ".join(str(f) for f in raw_facts)
            # Each fact should produce a string like "predicate(arg1, arg2)"
            parts = facts_str.split("; ")
            assert len(parts) == len(raw_facts)
            for part in parts:
                assert "(" in part and ")" in part, (
                    f"Each serialized fact should look like a predicate, got: {part}"
                )
        finally:
            env.close()

    def test_serialize_empty_facts(self):
        """Empty facts list produces empty string."""
        raw_facts = []
        if raw_facts:
            facts_str = "; ".join(str(f) for f in raw_facts)
        else:
            facts_str = ""
        assert facts_str == ""
