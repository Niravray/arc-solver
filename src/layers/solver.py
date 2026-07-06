import numpy as np
import sys
import json
import os

sys.path.append(r"C:\arc_solver\src\layers")
sys.path.append(r"C:\arc_solver\src\primitives")

from task_loader import load_all_tasks, get_task_ids, load_solutions
from object_extractor import extract_objects, get_background_color
from property_encoder import encode_all_objects
from relationship_graph import build_relationship_graph
from difference_engine import compute_difference
from verifier import (
    Hypothesis, verify_hypothesis, verify_all,
    generate_hypotheses, get_top_predictions
)

# ============================================================
# BEAM SEARCH + MAIN SOLVER
# Layer 8 of the ARC Solver Pipeline
#
# What this does:
# - Ties all layers together into one solve() function
# - For each task: generates hypotheses, verifies them,
#   applies the best ones to the test input
# - Returns 2 predictions per task (ARC allows 2 attempts)
# - Runs the full pipeline on all tasks and measures accuracy
# ============================================================


def solve_task(task):
    """
    Solve a single ARC task.

    task = task dict with "train" and "test" keys

    Returns a list of up to 2 predicted output grids.
    If the system cannot solve the task, returns a fallback prediction.
    """
    # Step 1: Generate hypotheses based on diff analysis
    hypotheses = generate_hypotheses(task)

    # Step 2: Verify hypotheses against all training pairs
    verified = verify_all(hypotheses, task)

    # Step 3: Get test input
    test_input = task["test"][0]["input"]

    # Step 4: Apply top verified hypotheses to test input
    if verified:
        predictions = get_top_predictions(verified, test_input, task, n=2)
        if predictions:
            # Pad to 2 predictions if we only have 1
            while len(predictions) < 2:
                predictions.append(predictions[0])
            return predictions[:2], verified[0].name

    # Step 5: Fallback — return the test input unchanged and a blank grid
    # This is our "I don't know" answer
    fallback = test_input.copy()
    blank = np.zeros_like(test_input)
    return [fallback, blank], "fallback"


def solve_all_tasks(tasks, task_ids, solutions=None, verbose=True):
    """
    Run the solver on all tasks.

    tasks      = dict of task_id -> task dict
    task_ids   = list of task IDs to solve
    solutions  = dict of task_id -> correct output grids (optional, for scoring)
    verbose    = print progress

    Returns a results dict with predictions and accuracy.
    """
    results = {}
    correct = 0
    total   = 0

    for i, task_id in enumerate(task_ids):
        task = tasks[task_id]

        # Solve the task
        predictions, method = solve_task(task)

        # Store predictions
        results[task_id] = {
            "predictions": predictions,
            "method":      method,
            "correct":     False
        }

        # Score if solutions available
        if solutions and task_id in solutions:
            correct_outputs = solutions[task_id]
            for pred in predictions:
                for correct_out in correct_outputs:
                    if np.array_equal(pred, correct_out):
                        results[task_id]["correct"] = True
                        correct += 1
                        break
                if results[task_id]["correct"]:
                    break

        total += 1

        if verbose and (i + 1) % 50 == 0:
            acc = correct / total * 100 if total > 0 else 0
            print(f"  Progress: {i+1}/{len(task_ids)} tasks | "
                  f"Solved: {correct}/{total} ({acc:.1f}%)")

    accuracy = correct / total if total > 0 else 0

    return {
        "results":   results,
        "correct":   correct,
        "total":     total,
        "accuracy":  accuracy,
        "solved_ids": [tid for tid, r in results.items() if r["correct"]],
        "unsolved_ids": [tid for tid, r in results.items() if not r["correct"]]
    }


def score_on_training(tasks, task_ids, limit=None):
    """
    Run the solver on training tasks where we have solutions.
    Measures real accuracy.

    limit = only test on first N tasks (faster for debugging)
    """
    print("Loading training solutions...")
    solutions = load_solutions("training")

    if limit:
        task_ids = task_ids[:limit]

    print(f"Running solver on {len(task_ids)} training tasks...")
    print()

    report = solve_all_tasks(tasks, task_ids, solutions, verbose=True)

    print(f"\n{'='*50}")
    print(f"RESULTS: {report['correct']}/{report['total']} tasks solved")
    print(f"ACCURACY: {report['accuracy']*100:.2f}%")
    print(f"{'='*50}")

    if report["solved_ids"]:
        print(f"\nSolved tasks ({len(report['solved_ids'])}):")
        for tid in report["solved_ids"]:
            method = report["results"][tid]["method"]
            print(f"  {tid} — {method}")

    return report


def format_submission(results, task_ids):
    """
    Format predictions into the Kaggle submission format.

    Kaggle expects:
    {
        "task_id": {
            "attempt_1": [[row1], [row2], ...],
            "attempt_2": [[row1], [row2], ...]
        },
        ...
    }
    """
    submission = {}

    for task_id in task_ids:
        if task_id not in results["results"]:
            continue

        preds = results["results"][task_id]["predictions"]

        # Convert NumPy arrays to lists for JSON
        attempt_1 = preds[0].tolist() if len(preds) > 0 else [[0]]
        attempt_2 = preds[1].tolist() if len(preds) > 1 else attempt_1

        submission[task_id] = {
            "attempt_1": attempt_1,
            "attempt_2": attempt_2
        }

    return submission


def save_submission(submission, path=r"C:\arc_solver\submission.json"):
    """
    Save the submission to a JSON file.
    """
    with open(path, "w") as f:
        json.dump(submission, f)
    print(f"Submission saved to {path}")
    print(f"Tasks in submission: {len(submission)}")


def analyze_failures(report, tasks, limit=10):
    """
    Analyze why tasks failed.
    Prints the diff signature for each failed task to help
    identify which primitives are missing.
    """
    print(f"\nFailure analysis (first {limit} unsolved tasks):")
    print("-" * 50)

    for task_id in report["unsolved_ids"][:limit]:
        task = tasks[task_id]
        pair = task["train"][0]
        diff = compute_difference(pair["input"], pair["output"])

        print(f"Task {task_id}:")
        print(f"  Signature: {diff['signature']}")
        print(f"  Input:  {diff['input_shape']} | "
              f"Output: {diff['output_shape']}")
        print(f"  Objects: {diff['input_object_count']} -> "
              f"{diff['output_object_count']}")
        print(f"  Method tried: {report['results'][task_id]['method']}")
        print()


# ============================================================
# QUICK TEST
# py C:\arc_solver\src\layers\solver.py
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("ARC SOLVER — FULL PIPELINE TEST")
    print("=" * 50)
    print()

    print("Loading training tasks...")
    tasks = load_all_tasks("training")
    ids   = get_task_ids("training")
    print(f"Loaded {len(tasks)} tasks.")
    print()

    # Run on first 100 tasks for a quick baseline score
    print("Running on first 100 tasks...")
    report = score_on_training(tasks, ids, limit=100)

    # Analyze failures
    analyze_failures(report, tasks, limit=10)

    # Save submission format (for training tasks as a test)
    print("\nGenerating submission format...")
    test_tasks  = load_all_tasks("test")
    test_ids    = get_task_ids("test")
    test_report = solve_all_tasks(test_tasks, test_ids, verbose=False)
    submission  = format_submission(test_report, test_ids)
    save_submission(submission)

    print("\nFull pipeline complete.")
    print("Next step: submit submission.json to Kaggle.")