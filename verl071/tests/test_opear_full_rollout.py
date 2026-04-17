"""Full O-PEaR rollout test: real model generation + guide + loss computation.

Runs Qwen3-4B on a real ALFWorld game, captures facts, calls GPT-5.4-nano
for compliant/violating rewrites, computes actual log-probs, and calculates
the O-PEaR loss. Logs all inputs/outputs for inspection.
"""
import asyncio
import json
import os
import random
import sys

import torch
import yaml
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

SEPARATOR = "=" * 70


def log_section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def run_alfworld_episode_with_model(model, tokenizer, device, max_steps=3):
    """Run ALFWorld episode with real Qwen3-4B model inference."""
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
    game_idx = random.randint(0, min(100, len(game_files) - 1))
    env_id = textworld.gym.register_games(
        [game_files[game_idx]],
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

    # ALFWorld prompt templates (inlined to avoid verl.interactions import)
    ALFWORLD_TEMPLATE_NO_HIS = (
        "\nYou are an expert agent operating in the ALFRED Embodied Environment.\n"
        "Your current observation is: {current_observation}\n"
        "Your admissible actions of the current situation are: [{admissible_actions}].\n\n"
        "Now it's your turn to take an action.\n"
        "You should first reason step-by-step about the current situation. "
        "This reasoning process MUST be enclosed within <think> </think> tags.\n"
        "Once you've finished your reasoning, you should choose an admissible action "
        "for current step and present it within <action> </action> tags.\n"
    )
    ALFWORLD_TEMPLATE = (
        "\nYou are an expert agent operating in the ALFRED Embodied Environment. "
        "Your task is to: {task_description}\n"
        "Prior to this step, you have already taken {step_count} step(s). "
        "Below are the most recent {history_length} observations and the corresponding "
        "actions you took: {action_history}\n"
        "You are now at step {current_step} and your current observation is: {current_observation}\n"
        "Your admissible actions of the current situation are: [{admissible_actions}].\n\n"
        "Now it's your turn to take an action.\n"
        "You should first reason step-by-step about the current situation. "
        "This reasoning process MUST be enclosed within <think> </think> tags.\n"
        "Once you've finished your reasoning, you should choose an admissible action "
        "for current step and present it within <action> </action> tags.\n"
    )
    def format_admissible(actions):
        return "\n ".join(f"'{s}'" for s in actions if s != "help")

    turns = []  # for guide model
    all_facts = []
    model_inputs_outputs = []  # for logging

    obs_text = initial_obs
    history = []

    for step in range(max_steps):
        admissible = info.get("admissible_commands", [[]])[0]
        raw_facts = info.get("facts", [])
        # Unwrap batched facts (batch_size=1)
        if raw_facts and isinstance(raw_facts[0], list):
            raw_facts = raw_facts[0]
        # Filter to useful instance-level facts only
        _USEFUL = {'inreceptacle', 'holds',
                    'isclean', 'ishot', 'iscool', 'issliced',
                    'opened'}
        useful = [f for f in raw_facts if hasattr(f, 'name') and f.name in _USEFUL]
        facts_str = "; ".join(
            f"{f.name}({', '.join(v.name.strip() for v in f.arguments)})"
            for f in useful
        )
        all_facts.append(facts_str)
        admissible_str = format_admissible(admissible)

        # Build prompt using the same templates as training
        if step == 0:
            prompt_text = ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=obs_text,
                admissible_actions=admissible_str,
            )
        else:
            recent = history[-2:]
            history_lines = []
            start_idx = max(0, len(history) - 2)
            for j, (h_obs, h_action) in enumerate(recent):
                step_num = start_idx + j + 1
                history_lines.append(
                    f"[Observation {step_num}: '{h_obs}', Action {step_num}: '{h_action}']"
                )
            action_history = "\n".join(history_lines)
            prompt_text = ALFWORLD_TEMPLATE.format(
                task_description=task_description,
                step_count=step,
                history_length=len(recent),
                action_history=action_history,
                current_step=step + 1,
                current_observation=obs_text,
                admissible_actions=admissible_str,
            )

        # Generate with Qwen3-4B
        chat = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        response_ids = output_ids[0, input_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        # Log model input/output
        model_inputs_outputs.append({
            "step": step,
            "prompt": prompt_text,
            "response": response_text,
            "facts": facts_str,
            "admissible": [a for a in admissible if a != "help"],
        })

        # Add to guide turns
        turns.append({"role": "observation", "content": prompt_text})
        turns.append({"role": "assistant", "content": response_text})

        # Extract action and step environment
        # Extract action from <action>...</action> tags
        resp_lower = response_text.lower()
        a_start = resp_lower.find("<action>")
        a_end = resp_lower.find("</action>")
        if a_start != -1 and a_end != -1:
            action = resp_lower[a_start + len("<action>"):a_end].strip()
        else:
            action = resp_lower[-30:]

        history.append((obs_text, action))

        obs, scores, dones, info = env.step([action])
        obs_text = obs[0] if isinstance(obs, (list, tuple)) else str(obs)

        if dones[0]:
            break

    env.close()

    # Use facts from the last step
    combined_facts = "; ".join(all_facts)

    return task_description, turns, combined_facts, model_inputs_outputs


def compute_sequence_logprob(model, tokenizer, prompt_text, response_text, device):
    """Compute log-probability of response_text given prompt_text under the model."""
    chat = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": response_text},
    ]
    full_text = tokenizer.apply_chat_template(chat, tokenize=False)
    full_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)

    # Find where the response starts
    prompt_chat = [{"role": "user", "content": prompt_text}]
    prompt_text_templated = tokenizer.apply_chat_template(
        prompt_chat, add_generation_prompt=True, tokenize=False
    )
    prompt_ids = tokenizer.encode(prompt_text_templated, return_tensors="pt")
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:]

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Only response portion (after prompt)
    resp_start = prompt_len - 1  # -1 because of shift
    resp_log_probs = token_log_probs[0, resp_start:]
    resp_len = resp_log_probs.shape[0]

    avg_log_prob = resp_log_probs.sum().item() / max(resp_len, 1)
    return avg_log_prob, resp_log_probs, resp_len


def main():
    log_section("O-PEaR Full Rollout Test")
    print("Model: Qwen/Qwen3-4B")
    print("Guide: GPT-5.4-nano")

    # ── Step 1: Load model ──
    log_section("Step 1: Loading Qwen3-4B")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded on {device}, dtype={model.dtype}")

    # ── Step 2: Run ALFWorld episode with model ──
    log_section("Step 2: Running ALFWorld Episode with Qwen3-4B")
    task_description, turns, facts_str, model_io = run_alfworld_episode_with_model(
        model, tokenizer, device, max_steps=3
    )
    num_assistant = sum(1 for t in turns if t["role"] == "assistant")
    print(f"  Task: {task_description}")
    print(f"  Turns: {len(turns)} ({num_assistant} assistant)")

    for io in model_io:
        print(f"\n  ── Turn {io['step']} ──")
        print(f"  PROMPT (first 300 chars):")
        print(f"    {io['prompt'][:300]}...")
        print(f"  MODEL RESPONSE:")
        print(f"    {io['response'][:500]}")
        print(f"  ADMISSIBLE: {io['admissible'][:8]}...")
        print(f"  FACTS (first 200 chars): {io['facts'][:200]}...")

    # ── Step 3: Call GPT-5.4-nano guide ──
    log_section("Step 3: Calling GPT-5.4-nano Guide Model")
    from verl071.opear.guide import OPEaRGuide
    from verl071.opear.prompts import build_guide_prompt

    guide = OPEaRGuide(model="gpt-5.4-nano", max_completion_tokens=4096)

    # Log the actual prompts sent to the guide
    compliant_msgs = build_guide_prompt(turns, task_description, "compliant", facts_str)
    violating_msgs = build_guide_prompt(turns, task_description, "violating", facts_str)

    print("\n  ── COMPLIANT PROMPT (system) ──")
    print(f"    {compliant_msgs[0]['content'][:300]}...")
    print(f"\n  ── COMPLIANT PROMPT (user, first 500 chars) ──")
    print(f"    {compliant_msgs[1]['content'][:500]}...")
    print(f"\n  ── VIOLATING PROMPT (system) ──")
    print(f"    {violating_msgs[0]['content'][:300]}...")

    result = asyncio.run(guide.generate_pair(turns, task_description, facts_str))
    assert result is not None, "Guide returned None!"

    print(f"\n  Guide returned {len(result['compliant'])} compliant, {len(result['violating'])} violating turns")

    for label, responses in [("COMPLIANT", result["compliant"]), ("VIOLATING", result["violating"])]:
        print(f"\n  ── {label} RESPONSES ──")
        for i, resp in enumerate(responses):
            print(f"    Turn {i+1}:")
            print(f"      Think: {resp['think'][:200]}")
            print(f"      Action: {resp['action']}")

    # ── Step 4: Compute log-probs under student model ──
    log_section("Step 4: Computing Log-Probabilities Under Qwen3-4B")

    # We need to compute log-probs for:
    # a) Original student responses
    # b) Compliant responses
    # c) Violating responses

    all_logprobs = {"original": [], "compliant": [], "violating": []}
    all_resp_lengths = {"original": [], "compliant": [], "violating": []}

    for i, io in enumerate(model_io):
        prompt = io["prompt"]
        orig_response = io["response"]

        # Original
        avg_lp, _, resp_len = compute_sequence_logprob(
            model, tokenizer, prompt, orig_response, device
        )
        all_logprobs["original"].append(avg_lp)
        all_resp_lengths["original"].append(resp_len)

        # Compliant
        if i < len(result["compliant"]):
            c_resp = result["compliant"][i]
            c_text = f"<think>{c_resp['think']}</think><action>{c_resp['action']}</action>"
            avg_lp_c, _, resp_len_c = compute_sequence_logprob(
                model, tokenizer, prompt, c_text, device
            )
            all_logprobs["compliant"].append(avg_lp_c)
            all_resp_lengths["compliant"].append(resp_len_c)

        # Violating
        if i < len(result["violating"]):
            v_resp = result["violating"][i]
            v_text = f"<think>{v_resp['think']}</think><action>{v_resp['action']}</action>"
            avg_lp_v, _, resp_len_v = compute_sequence_logprob(
                model, tokenizer, prompt, v_text, device
            )
            all_logprobs["violating"].append(avg_lp_v)
            all_resp_lengths["violating"].append(resp_len_v)

    print("\n  Per-turn average log-probabilities:")
    print(f"  {'Turn':<6} {'Original':>12} {'Compliant':>12} {'Violating':>12}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12}")
    for i in range(len(all_logprobs["original"])):
        orig = all_logprobs["original"][i]
        comp = all_logprobs["compliant"][i] if i < len(all_logprobs["compliant"]) else float("nan")
        viol = all_logprobs["violating"][i] if i < len(all_logprobs["violating"]) else float("nan")
        print(f"  {i+1:<6} {orig:>12.4f} {comp:>12.4f} {viol:>12.4f}")

    print(f"\n  Response lengths (tokens):")
    for label in ["original", "compliant", "violating"]:
        lens = all_resp_lengths[label]
        print(f"    {label}: {lens}")

    # ── Step 5: Compute O-PEaR loss ──
    log_section("Step 5: Computing O-PEaR Loss")
    from verl071.opear.loss import compute_opear_loss

    N = min(len(all_logprobs["compliant"]), len(all_logprobs["violating"]))
    if N == 0:
        print("  No valid pairs to compute loss!")
        return

    # Build tensors from actual log-probs
    # Use the max response length for padding
    max_len = max(
        max(all_resp_lengths["compliant"][:N]),
        max(all_resp_lengths["violating"][:N]),
    )

    c_lp_tensor = torch.zeros(N, max_len)
    c_mask_tensor = torch.zeros(N, max_len)
    v_lp_tensor = torch.zeros(N, max_len)
    v_mask_tensor = torch.zeros(N, max_len)

    for i in range(N):
        # Recompute full token-level log-probs for compliant
        c_resp = result["compliant"][i]
        c_text = f"<think>{c_resp['think']}</think><action>{c_resp['action']}</action>"
        _, c_token_lp, c_len = compute_sequence_logprob(
            model, tokenizer, model_io[i]["prompt"], c_text, device
        )
        c_len = min(c_len, max_len)
        c_lp_tensor[i, :c_len] = c_token_lp[:c_len].cpu()
        c_mask_tensor[i, :c_len] = 1.0

        # Violating
        v_resp = result["violating"][i]
        v_text = f"<think>{v_resp['think']}</think><action>{v_resp['action']}</action>"
        _, v_token_lp, v_len = compute_sequence_logprob(
            model, tokenizer, model_io[i]["prompt"], v_text, device
        )
        v_len = min(v_len, max_len)
        v_lp_tensor[i, :v_len] = v_token_lp[:v_len].cpu()
        v_mask_tensor[i, :v_len] = 1.0

    loss, metrics = compute_opear_loss(
        c_lp_tensor, c_mask_tensor, v_lp_tensor, v_mask_tensor, alpha=0.5
    )

    print(f"  O-PEaR Loss:          {metrics['opear/loss']:.6f}")
    print(f"  Compliant avg logprob: {metrics['opear/compliant_logprob']:.6f}")
    print(f"  Violating avg logprob: {metrics['opear/violating_logprob']:.6f}")
    print(f"  R mean:                {metrics['opear/R_mean']:.6f}")
    print(f"  Num pairs:             {metrics['opear/num_pairs']}")

    print(f"\n  Interpretation:")
    if metrics['opear/compliant_logprob'] > metrics['opear/violating_logprob']:
        print(f"    Student already assigns HIGHER prob to compliant than violating.")
        print(f"    Loss is negative => regularizer is already satisfied.")
    else:
        print(f"    Student assigns HIGHER prob to violating than compliant.")
        print(f"    Loss is positive => regularizer will push model toward compliant behavior.")

    log_section("FULL ROLLOUT TEST COMPLETE")

    # Save full results to JSON for inspection
    output_path = os.path.join(os.path.dirname(__file__), "full_rollout_results.json")
    save_data = {
        "task": task_description,
        "model_rollout": [],
        "compliant_responses": [],
        "violating_responses": [],
        "logprobs": {
            "original": all_logprobs["original"],
            "compliant": all_logprobs["compliant"],
            "violating": all_logprobs["violating"],
        },
        "response_lengths": {
            "original": all_resp_lengths["original"],
            "compliant": all_resp_lengths["compliant"],
            "violating": all_resp_lengths["violating"],
        },
        "opear_metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
        "facts": facts_str[:2000],
    }
    for io in model_io:
        save_data["model_rollout"].append({
            "step": io["step"],
            "prompt": io["prompt"],
            "response": io["response"],
            "admissible": io["admissible"],
            # NOTE: facts are NOT shown to the student model.
            # They are captured here only for reference / debugging.
        })
    for resp in result["compliant"]:
        save_data["compliant_responses"].append({
            "think": resp["think"],
            "action": resp["action"],
        })
    for resp in result["violating"]:
        save_data["violating_responses"].append({
            "think": resp["think"],
            "action": resp["action"],
        })

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
