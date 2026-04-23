"""Create stratified subsets of ALFWorld training games.

Generates JSON files containing game indices for data-constrained experiments.
Each subset maintains the same task-type distribution as the full dataset.

Usage:
    python scripts/create_data_subsets.py [--seed 42] [--output_dir data/subsets]
"""

import argparse
import json
import os
import random
from collections import defaultdict

TASK_TYPES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]

SUBSET_PERCENTAGES = [1, 10, 25, 50]


def classify_game(path: str) -> str:
    for tt in TASK_TYPES:
        if tt in path:
            return tt
    return "unknown"


def create_stratified_subsets(game_files: list[str], seed: int = 42) -> dict:
    """Create stratified subsets preserving task-type distribution."""
    rng = random.Random(seed)

    # Group indices by task type
    type_indices = defaultdict(list)
    for i, g in enumerate(game_files):
        tt = classify_game(g)
        type_indices[tt].append(i)

    # Shuffle within each type for random selection
    for tt in type_indices:
        rng.shuffle(type_indices[tt])

    total = len(game_files)
    subsets = {}

    for pct in SUBSET_PERCENTAGES:
        target_total = max(len(TASK_TYPES), int(total * pct / 100))
        selected = []

        for tt in TASK_TYPES:
            n = max(1, round(len(type_indices[tt]) * pct / 100))
            selected.extend(type_indices[tt][:n])

        selected = sorted(set(selected))

        # Verify distribution
        type_counts = defaultdict(int)
        for idx in selected:
            tt = classify_game(game_files[idx])
            type_counts[tt] += 1

        subsets[pct] = {
            "percentage": pct,
            "total_games": len(selected),
            "full_dataset_size": total,
            "seed": seed,
            "task_type_counts": dict(type_counts),
            "indices": selected,
        }

    return subsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    # Load ALFWorld game files
    import yaml
    from alfworld.agents.environment import get_environment

    config_path = os.path.join(
        os.path.dirname(__file__), "..",
        "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
    )
    # Try local path first, fall back to parent repo
    if not os.path.exists(config_path):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    env = get_environment("AlfredTWEnv")(config, train_eval="train")
    game_files = list(env.game_files)
    print(f"Total training games: {len(game_files)}")

    subsets = create_stratified_subsets(game_files, seed=args.seed)

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "subsets")
    os.makedirs(args.output_dir, exist_ok=True)

    for pct, data in subsets.items():
        fname = f"train_subset_{pct}pct_seed{args.seed}.json"
        path = os.path.join(args.output_dir, fname)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {pct}%: {data['total_games']} games -> {path}")
        for tt in TASK_TYPES:
            n = data["task_type_counts"].get(tt, 0)
            print(f"    {tt}: {n}")

    print("\nDone.")


if __name__ == "__main__":
    main()
