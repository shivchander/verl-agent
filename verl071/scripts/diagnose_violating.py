"""Diagnostic script: Call the guide model on sample trajectories and inspect outputs.

Usage:
    PYTHONPATH=. python verl071/scripts/diagnose_violating.py [--n-samples 3]

Loads the latest checkpoint's training data, reconstructs trajectories,
calls the guide model, and prints compliant vs violating side-by-side.
"""

import argparse
import json
import os
import sys
import random

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from verl071.opear.prompts import build_guide_prompt, parse_guide_response, format_trajectory
from verl071.opear.guide import OPEaRGuide


def make_sample_trajectories():
    """Create realistic ALFWorld trajectories for testing guide quality.

    These are based on common ALFWorld patterns to test violating generation.
    """
    samples = [
        {
            "task_description": "put a clean lettuce in countertop.",
            "facts": (
                "(objectType lettuce1 LettuceType)\n"
                "(atLocation lettuce1 fridge1)\n"
                "(isClean lettuce1)\n"
                "(receptacleType countertop1 CounterTopType)\n"
                "(atLocation countertop1 kitchen)\n"
                "(objectType sinkbasin1 SinkBasinType)\n"
                "(inReceptacle apple1 countertop1)\n"
            ),
            "turns": [
                {"role": "observation", "content": (
                    "You are in the middle of a room. Looking quickly around you, "
                    "you see a cabinet 1, a cabinet 2, a countertop 1, a fridge 1, "
                    "a garbagecan 1, a sinkbasin 1, and a stoveburner 1.\n\n"
                    "Your task is to: put a clean lettuce in countertop."
                )},
                {"role": "assistant", "content": (
                    "<think>I need to find a lettuce first. Let me check common "
                    "locations like the fridge or countertop.</think>"
                    "<action>go to fridge 1</action>"
                )},
                {"role": "observation", "content": (
                    "You arrive at fridge 1. The fridge 1 is closed.\n\n"
                    "Admissible actions: open fridge 1, go to cabinet 1, "
                    "go to cabinet 2, go to countertop 1, go to sinkbasin 1"
                )},
                {"role": "assistant", "content": (
                    "<think>The fridge is closed. I should open it to look for "
                    "the lettuce inside.</think>"
                    "<action>open fridge 1</action>"
                )},
                {"role": "observation", "content": (
                    "You open the fridge 1. The fridge 1 is open. In it, you see "
                    "a lettuce 1 and a potato 1.\n\n"
                    "Admissible actions: take lettuce 1 from fridge 1, "
                    "take potato 1 from fridge 1, close fridge 1, "
                    "go to countertop 1, go to sinkbasin 1"
                )},
                {"role": "assistant", "content": (
                    "<think>I can see the lettuce in the fridge. I should take it "
                    "and then clean it if needed.</think>"
                    "<action>take lettuce 1 from fridge 1</action>"
                )},
            ],
        },
        {
            "task_description": "heat some egg and put it in garbagecan.",
            "facts": (
                "(objectType egg1 EggType)\n"
                "(atLocation egg1 countertop1)\n"
                "(receptacleType microwave1 MicrowaveType)\n"
                "(receptacleType garbagecan1 GarbageCanType)\n"
                "(isHot egg1 false)\n"
                "(objectType pan1 PanType)\n"
                "(atLocation pan1 stoveburner1)\n"
            ),
            "turns": [
                {"role": "observation", "content": (
                    "You are in the middle of a room. Looking quickly around you, "
                    "you see a cabinet 1, a countertop 1, a garbagecan 1, "
                    "a microwave 1, a stoveburner 1, and a toaster 1.\n\n"
                    "Your task is to: heat some egg and put it in garbagecan."
                )},
                {"role": "assistant", "content": (
                    "<think>I need to find an egg, heat it, then put it in the "
                    "garbagecan. Let me check the countertop first.</think>"
                    "<action>go to countertop 1</action>"
                )},
                {"role": "observation", "content": (
                    "On the countertop 1, you see an egg 1, a knife 1, and a "
                    "saltshaker 1.\n\n"
                    "Admissible actions: take egg 1 from countertop 1, "
                    "take knife 1 from countertop 1, "
                    "take saltshaker 1 from countertop 1, "
                    "go to microwave 1, go to stoveburner 1, go to garbagecan 1"
                )},
                {"role": "assistant", "content": (
                    "<think>Found the egg on the countertop. I'll take it and "
                    "then go heat it in the microwave.</think>"
                    "<action>take egg 1 from countertop 1</action>"
                )},
            ],
        },
        {
            "task_description": "look at bowl under the desklamp.",
            "facts": (
                "(objectType bowl1 BowlType)\n"
                "(atLocation bowl1 drawer1)\n"
                "(objectType desklamp1 DeskLampType)\n"
                "(atLocation desklamp1 desk1)\n"
                "(isToggled desklamp1 false)\n"
                "(objectType pen1 PenType)\n"
                "(atLocation pen1 desk1)\n"
            ),
            "turns": [
                {"role": "observation", "content": (
                    "You are in the middle of a room. Looking quickly around you, "
                    "you see a bed 1, a desk 1, a drawer 1, a drawer 2, "
                    "a garbagecan 1, and a shelf 1.\n\n"
                    "Your task is to: look at bowl under the desklamp."
                )},
                {"role": "assistant", "content": (
                    "<think>I need to find a bowl and examine it under a desklamp. "
                    "Let me check the desk first since desklamps are usually there.</think>"
                    "<action>go to desk 1</action>"
                )},
                {"role": "observation", "content": (
                    "On the desk 1, you see a desklamp 1, a pen 1, and a "
                    "creditcard 1.\n\n"
                    "Admissible actions: use desklamp 1, take pen 1 from desk 1, "
                    "take creditcard 1 from desk 1, go to drawer 1, go to drawer 2, "
                    "go to bed 1, go to shelf 1"
                )},
                {"role": "assistant", "content": (
                    "<think>I see the desklamp but no bowl here. I need to find the "
                    "bowl first. Let me check the drawers.</think>"
                    "<action>go to drawer 1</action>"
                )},
            ],
        },
    ]
    return samples


def print_comparison(sample, pair, idx):
    """Print compliant vs violating side-by-side for analysis."""
    print(f"\n{'='*80}")
    print(f"SAMPLE {idx}: {sample['task_description']}")
    print(f"{'='*80}")

    print(f"\n--- PRIVILEGED FACTS ---")
    print(sample["facts"].strip())

    print(f"\n--- ORIGINAL TRAJECTORY ---")
    for turn in sample["turns"]:
        role = turn["role"].upper()
        content = turn["content"][:200]
        suffix = "..." if len(turn["content"]) > 200 else ""
        print(f"  [{role}] {content}{suffix}")

    compliant = pair["compliant"]
    violating = pair["violating"]

    n_turns = min(len(compliant), len(violating))

    for t in range(n_turns):
        print(f"\n--- TURN {t+1} ---")
        print(f"  COMPLIANT think: {compliant[t]['think']}")
        print(f"  COMPLIANT action: {compliant[t]['action']}")
        print(f"  VIOLATING think: {violating[t]['think']}")
        print(f"  VIOLATING action: {violating[t]['action']}")

        # Analysis
        same_action = compliant[t]["action"].strip() == violating[t]["action"].strip()
        print(f"  >> Same action? {same_action}")
        if not same_action:
            print(f"  >> Actions differ: '{compliant[t]['action'].strip()}' vs '{violating[t]['action'].strip()}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--model", type=str, default="gpt-5.4-nano")
    args = parser.parse_args()

    guide = OPEaRGuide(model=args.model, temperature=0.7)
    samples = make_sample_trajectories()[:args.n_samples]

    print(f"Generating contrastive pairs for {len(samples)} samples using {args.model}...")

    for i, sample in enumerate(samples):
        pair = guide.generate_pair(
            turns=sample["turns"],
            task_description=sample["task_description"],
            facts=sample["facts"],
        )
        if pair is None:
            print(f"\nSample {i+1}: FAILED to generate pair")
            continue

        print_comparison(sample, pair, i + 1)

    # Also print the raw prompts for inspection
    print(f"\n{'='*80}")
    print("RAW VIOLATING PROMPT (for reference)")
    print(f"{'='*80}")
    msgs = build_guide_prompt(
        samples[0]["turns"],
        samples[0]["task_description"],
        "violating",
        samples[0]["facts"],
    )
    print(f"\n[SYSTEM]\n{msgs[0]['content']}\n")
    print(f"[USER]\n{msgs[1]['content']}")


if __name__ == "__main__":
    main()
