"""Mock end-to-end O-PEaR test with real ALFWorld + real GPT-5.4-nano API.

Run standalone: PYTHONPATH=. python verl071/tests/test_opear_mock.py
(pytest will skip this test if OPENAI_API_KEY is not set to a real key)
"""
import asyncio
import os
import random
import sys

import pytest
import torch
import yaml
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

_api_key = os.environ.get("OPENAI_API_KEY", "")
_has_real_key = _api_key.startswith("sk-proj-") or _api_key.startswith("sk-live-")


def run_alfworld_episode(max_steps=3):
    """Run a short ALFWorld episode and collect trajectory data with facts."""
    import textworld
    import textworld.gym

    alf_config_path = os.path.join(
        os.path.dirname(__file__),
        "../../agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
    )
    with open(alf_config_path) as f:
        alf_config = yaml.safe_load(f)

    from alfworld.agents.environment import get_environment
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    tw_env = get_environment("AlfredTWEnv")(alf_config, train_eval="train")
    game_files = list(tw_env.game_files)

    request_infos = textworld.EnvInfos(
        won=True, admissible_commands=True, facts=True, extras=["gamefile"]
    )
    env_id = textworld.gym.register_games(
        [game_files[0]],
        request_infos,
        batch_size=1,
        asynchronous=False,
        max_episode_steps=50,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
    )
    env = textworld.gym.make(env_id)
    obs, info = env.reset()
    initial_obs = obs[0] if isinstance(obs, (list, tuple)) else obs

    # Extract task
    marker = "Your task is to: "
    idx = initial_obs.find(marker)
    task_description = initial_obs[idx + len(marker):].strip() if idx != -1 else "unknown"

    # Build turns in the format the guide expects
    turns = []
    facts_str = "; ".join(str(f) for f in info.get("facts", []))
    obs_text = initial_obs

    for step in range(max_steps):
        admissible = info.get("admissible_commands", [[]])[0]
        admissible_str = ", ".join(f"'{a}'" for a in admissible if a != "help")

        # Add observation turn
        turns.append({"role": "observation", "content": obs_text})

        # Mock student response
        action = random.choice([a for a in admissible if a != "help"]) if admissible else "look"
        student_response = f"<think>I will try: {action}</think><action>{action}</action>"
        turns.append({"role": "assistant", "content": student_response})

        obs, scores, dones, info = env.step([action])
        obs_text = obs[0] if isinstance(obs, (list, tuple)) else str(obs)
        facts_str = "; ".join(str(f) for f in info.get("facts", []))

        if dones[0]:
            break

    env.close()
    return task_description, turns, facts_str


@pytest.mark.skipif(not _has_real_key, reason="OPENAI_API_KEY not set to a real key")
def test_mock_e2e():
    print("=" * 60)
    print("O-PEaR Mock End-to-End Test")
    print("=" * 60)

    # Step 1: Run ALFWorld episode
    print("\n[1/5] Running ALFWorld episode...")
    task_description, turns, facts_str = run_alfworld_episode(max_steps=2)
    num_assistant_turns = sum(1 for t in turns if t["role"] == "assistant")
    print(f"  Task: {task_description[:80]}")
    print(f"  Total turns: {len(turns)} ({num_assistant_turns} assistant)")
    for i, t in enumerate(turns):
        print(f"  {t['role']}: {t['content'][:80]}...")
    print(f"  Facts sample: {facts_str[:120]}...")

    # Step 2: Call GPT-5.4-nano guide
    print("\n[2/5] Calling GPT-5.4-nano guide model...")
    from verl071.opear.guide import OPEaRGuide

    guide = OPEaRGuide(model="gpt-5.4-nano", max_completion_tokens=2048)
    result = asyncio.run(guide.generate_pair(turns, task_description, facts_str))

    assert result is not None, "Guide returned None!"
    print(f"  Compliant turns: {len(result['compliant'])}")
    print(f"  Violating turns: {len(result['violating'])}")

    # Step 3: Verify format
    print("\n[3/5] Verifying response format...")
    for label, responses in [("compliant", result["compliant"]), ("violating", result["violating"])]:
        for i, resp in enumerate(responses):
            assert resp.get("think"), f"{label} turn {i+1}: empty think"
            assert resp.get("action"), f"{label} turn {i+1}: empty action"
            print(f"  {label} turn {i+1}: OK")
            print(f"    think: {resp['think'][:60]}...")
            print(f"    action: {resp['action']}")
    print("  Format verification PASSED")

    # Step 4: Tokenize
    print("\n[4/5] Testing tokenization...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

    for label, responses in [("compliant", result["compliant"]), ("violating", result["violating"])]:
        for i, resp in enumerate(responses):
            text = f"<think>{resp['think']}</think><action>{resp['action']}</action>"
            tokens = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"  {label} turn {i+1}: {len(tokens)} tokens")
            assert "<think>" in decoded.lower() or "think" in decoded.lower(), f"Round-trip lost think"
            assert "<action>" in decoded.lower() or "action" in decoded.lower(), f"Round-trip lost action"
    print("  Tokenization round-trip PASSED")

    # Step 5: Compute mock loss
    print("\n[5/5] Computing mock O-PEaR loss...")
    from verl071.opear.loss import compute_opear_loss

    N = len(result["compliant"])
    L = 50
    c_lp = torch.randn(N, L) * 0.5 - 1.0
    v_lp = torch.randn(N, L) * 0.5 - 2.0
    mask = torch.ones(N, L)

    loss, metrics = compute_opear_loss(c_lp, mask, v_lp, mask, alpha=0.5)
    print(f"  O-PEaR loss: {loss.item():.4f}")
    print(f"  Compliant logprob: {metrics['opear/compliant_logprob']:.4f}")
    print(f"  Violating logprob: {metrics['opear/violating_logprob']:.4f}")
    print(f"  R mean: {metrics['opear/R_mean']:.4f}")

    print("\n" + "=" * 60)
    print("MOCK END-TO-END TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_mock_e2e()
