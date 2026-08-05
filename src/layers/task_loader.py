import json
import numpy as np
import os
from pathlib import Path

# ============================================================
# ARC TASK LOADER
# Layer 1 of the ARC Solver Pipeline
#
# What this does:
# - Reads the ARC JSON files from the data folder
# - Returns training pairs and test inputs as NumPy arrays
# - A NumPy array is just a grid of numbers (0-9 for colors)
# ============================================================

def _find_data_dir():
    """
    Locate the ARC json files without hardcoding a machine.

    Order: an explicit ARC_DATA_DIR override, then any Kaggle input
    directory containing arc-agi files (the competition dataset is
    mounted read-only under /kaggle/input/<name>/ and the name changes
    between competition years), then the repo's own data/ folder.
    """
    override = os.environ.get("ARC_DATA_DIR")
    if override and os.path.isdir(override):
        return override

    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        for entry in sorted(os.listdir(kaggle_input)):
            candidate = os.path.join(kaggle_input, entry)
            if not os.path.isdir(candidate):
                continue
            if any(f.startswith("arc-agi") for f in os.listdir(candidate)):
                return candidate

    return str(Path(__file__).resolve().parent.parent.parent / "data")


DATA_PATH = _find_data_dir()

def load_task(task_id, split="training"):
    """
    Load a single task by its ID.
    
    split = "training" uses arc-agi_training_challenges.json
    split = "evaluation" uses arc-agi_evaluation_challenges.json
    split = "test" uses arc-agi_test_challenges.json
    
    Returns a dict with:
        "train": list of {"input": np.array, "output": np.array}
        "test":  list of {"input": np.array} (no output - that's what we predict)
        "id":    the task ID string
    """
    if split == "training":
        challenges_file = os.path.join(DATA_PATH, "arc-agi_training_challenges.json")
    elif split == "evaluation":
        challenges_file = os.path.join(DATA_PATH, "arc-agi_evaluation_challenges.json")
    elif split == "test":
        challenges_file = os.path.join(DATA_PATH, "arc-agi_test_challenges.json")
    else:
        raise ValueError(f"Unknown split: {split}. Use 'training', 'evaluation', or 'test'.")

    with open(challenges_file, "r") as f:
        all_tasks = json.load(f)

    if task_id not in all_tasks:
        raise KeyError(f"Task '{task_id}' not found in {split} challenges.")

    raw_task = all_tasks[task_id]

    # Convert training pairs to NumPy arrays
    train_pairs = []
    for pair in raw_task["train"]:
        train_pairs.append({
            "input":  np.array(pair["input"],  dtype=np.int32),
            "output": np.array(pair["output"], dtype=np.int32)
        })

    # Convert test inputs to NumPy arrays (no output available)
    test_inputs = []
    for item in raw_task["test"]:
        test_inputs.append({
            "input": np.array(item["input"], dtype=np.int32)
        })

    return {
        "id":    task_id,
        "train": train_pairs,
        "test":  test_inputs
    }


def load_all_tasks(split="training"):
    """
    Load every task from a split into a dict keyed by task ID.
    
    Returns:
        dict of { task_id: task_dict }
    """
    if split == "training":
        challenges_file = os.path.join(DATA_PATH, "arc-agi_training_challenges.json")
    elif split == "evaluation":
        challenges_file = os.path.join(DATA_PATH, "arc-agi_evaluation_challenges.json")
    elif split == "test":
        challenges_file = os.path.join(DATA_PATH, "arc-agi_test_challenges.json")
    else:
        raise ValueError(f"Unknown split: {split}")

    with open(challenges_file, "r") as f:
        all_tasks = json.load(f)

    loaded = {}
    for task_id, raw_task in all_tasks.items():

        train_pairs = []
        for pair in raw_task["train"]:
            train_pairs.append({
                "input":  np.array(pair["input"],  dtype=np.int32),
                "output": np.array(pair["output"], dtype=np.int32)
            })

        test_inputs = []
        for item in raw_task["test"]:
            test_inputs.append({
                "input": np.array(item["input"], dtype=np.int32)
            })

        loaded[task_id] = {
            "id":    task_id,
            "train": train_pairs,
            "test":  test_inputs
        }

    return loaded


def load_solutions(split="training"):
    """
    Load the ground truth solutions for training or evaluation tasks.
    Only useful for measuring your accuracy during development.
    
    Returns:
        dict of { task_id: list of output grids (np.array) }
    """
    if split == "training":
        solutions_file = os.path.join(DATA_PATH, "arc-agi_training_solutions.json")
    elif split == "evaluation":
        solutions_file = os.path.join(DATA_PATH, "arc-agi_evaluation_solutions.json")
    else:
        raise ValueError(f"No solutions available for split: {split}")

    with open(solutions_file, "r") as f:
        raw_solutions = json.load(f)

    solutions = {}
    for task_id, grids in raw_solutions.items():
        solutions[task_id] = [np.array(g, dtype=np.int32) for g in grids]

    return solutions


def get_task_ids(split="training"):
    """
    Returns a list of all task IDs in a split.
    Useful for iterating over all tasks.
    """
    if split == "training":
        challenges_file = os.path.join(DATA_PATH, "arc-agi_training_challenges.json")
    elif split == "evaluation":
        challenges_file = os.path.join(DATA_PATH, "arc-agi_evaluation_challenges.json")
    elif split == "test":
        challenges_file = os.path.join(DATA_PATH, "arc-agi_test_challenges.json")
    else:
        raise ValueError(f"Unknown split: {split}")

    with open(challenges_file, "r") as f:
        all_tasks = json.load(f)

    return list(all_tasks.keys())


# ============================================================
# QUICK TEST - run this file directly to verify it works
# In Command Prompt: py C:\arc_solver\src\layers\task_loader.py
# ============================================================

if __name__ == "__main__":
    print("Loading all training tasks...")
    tasks = load_all_tasks("training")
    ids = get_task_ids("training")

    print(f"Total training tasks loaded: {len(tasks)}")
    print(f"First 5 task IDs: {ids[:5]}")

    # Look at the first task in detail
    first_id = ids[0]
    first_task = tasks[first_id]

    print(f"\nTask ID: {first_id}")
    print(f"Number of training pairs: {len(first_task['train'])}")
    print(f"Number of test inputs: {len(first_task['test'])}")

    # Show the first training pair
    pair = first_task["train"][0]
    print(f"\nFirst training pair:")
    print(f"  Input shape:  {pair['input'].shape}  (rows x cols)")
    print(f"  Output shape: {pair['output'].shape}")
    print(f"\nInput grid:\n{pair['input']}")
    print(f"\nOutput grid:\n{pair['output']}")

    print("\nTask loader is working correctly.")

# ============================================================
# DEV / HOLDOUT SPLIT
#
# Every form, atom, cost heuristic and threshold in this solver was
# chosen by looking at training results, and scored on those same 1000
# tasks. That makes the training number optimistic even as a measure of
# "generalises to unseen tasks of this difficulty" -- the designer has
# seen every task, so the solver is fitted to them whether or not any
# single rule is.
#
# The fix is a split within training: tune on DEV, and keep HOLDOUT as a
# clean estimate. Deterministic (sorted ids, fixed seed) so the same
# tasks land in the same side on every machine and every run -- a split
# that drifts is worse than none, because a gain could just be a
# reshuffle.
#
# IMPORTANT: this split was created on 2026-07-30, AFTER the solver had
# already been tuned against all 1000 tasks. The first holdout number is
# therefore already contaminated and is NOT a clean estimate of anything.
# It becomes meaningful only for decisions taken from here on.
# ============================================================

HOLDOUT_FRACTION = 0.25
SPLIT_SEED = 20260730


def get_dev_holdout_split(split="training",
                          holdout_fraction=HOLDOUT_FRACTION,
                          seed=SPLIT_SEED):
    """
    Partition a split's task ids into (dev, holdout), deterministically.

    Sorting before shuffling matters: dict iteration order would make the
    partition depend on the JSON's key order rather than on the seed.
    """
    import random

    ids = sorted(get_task_ids(split))
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)

    cut = int(len(shuffled) * holdout_fraction)
    holdout = sorted(shuffled[:cut])
    dev = sorted(shuffled[cut:])
    return dev, holdout
