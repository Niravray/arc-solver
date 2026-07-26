import numpy as np # type: ignore
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

    NOTE: only ever looks at task["test"][0] -- kept exactly as-is for
    the local scoring path below (score_on_training / solve_all_tasks),
    which several other functions depend on. For the actual Kaggle
    submission, which needs a prediction for EVERY test input a task
    has, use solve_task_all_test_inputs() + build_kaggle_submission()
    instead, since that's the only path that formats correctly and
    covers every test input a task has.
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


def solve_task_all_test_inputs(task):
    """
    Same solving logic as solve_task(), but applied to EVERY test input
    in the task instead of just test[0]. Kept as a separate function
    rather than changing solve_task() itself, so the existing local
    scoring path (solve_all_tasks / score_on_training) stays exactly as
    it was -- no reason to add risk there right before a submission
    deadline when only the actual Kaggle output needs this.

    Returns a list of (predictions, method) tuples, one per test input,
    in the same order as task["test"].
    """
    hypotheses = generate_hypotheses(task)
    verified   = verify_all(hypotheses, task)

    results = []
    for test_item in task["test"]:
        test_input = test_item["input"]

        if verified:
            predictions = get_top_predictions(verified, test_input, task, n=2)
            if predictions:
                while len(predictions) < 2:
                    predictions.append(predictions[0])
                results.append((predictions[:2], verified[0].name))
                continue

        fallback = test_input.copy()
        blank = np.zeros_like(test_input)
        results.append(([fallback, blank], "fallback"))

    return results


def build_kaggle_submission(tasks, task_ids, verbose=False):
    """
    Build a submission dict in the exact format Kaggle requires:

    {
        "task_id": [
            {"attempt_1": [[...]], "attempt_2": [[...]]},
            ...  (one entry per test input the task has)
        ],
        ...
    }

    Every task_id in task_ids is guaranteed to appear in the output,
    since Kaggle requires every task_id in the input challenges file
    to also be present in submission.json. This is the function to use
    for the actual Kaggle submission.
    """
    submission = {}

    for i, task_id in enumerate(task_ids):
        task = tasks[task_id]
        per_test_results = solve_task_all_test_inputs(task)

        entries = []
        for predictions, method in per_test_results:
            entries.append({
                "attempt_1": predictions[0].tolist(),
                "attempt_2": predictions[1].tolist(),
            })
        submission[task_id] = entries

        if verbose and (i + 1) % 50 == 0:
            print(f"  Built {i+1}/{len(task_ids)} submission entries")

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
    Prints the diff signature for each failed task, plus how many
    hypotheses were actually generated and tried for it, to help
    identify which primitives are missing versus which tasks simply
    need more than one primitive combined.
    """
    print(f"\nFailure analysis (first {limit} unsolved tasks):")
    print("-" * 50)

    for task_id in report["unsolved_ids"][:limit]:
        task = tasks[task_id]
        pair = task["train"][0]
        diff = compute_difference(pair["input"], pair["output"])

        hyps = generate_hypotheses(task)

        print(f"Task {task_id}:")
        print(f"  Signature: {diff['signature']}")
        print(f"  Input:  {diff['input_shape']} | "
              f"Output: {diff['output_shape']}")
        print(f"  Objects: {diff['input_object_count']} -> "
              f"{diff['output_object_count']}")
        print(f"  Hypotheses tried: {len(hyps)}")
        print(f"  Method tried: {report['results'][task_id]['method']}")
        print()


# ============================================================
# FULL EVALUATION MODE
#
# Everything below is evaluation/reporting only -- it calls solve_task,
# generate_hypotheses, and verify_all exactly as they already exist and
# does not change what any of them compute. This exists because
# solve_all_tasks() above doesn't expose per-task timing or isolate
# per-task exceptions (one bad task would currently kill that whole
# loop), and both are needed for a 1000-task run. solve_all_tasks and
# score_on_training are untouched by any of this.
# ============================================================

def _diagnose_failure(task, method):
    """
    For a task solve_task() got wrong, figure out more than solve_task()
    itself exposes: how many hypotheses were even generated, and whether
    NOTHING verified against training (true fallback) versus something
    DID verify on every training pair but still didn't match the test
    solution (overfit to the training examples -- a meaningfully
    different failure mode). Re-runs generate_hypotheses/verify_all,
    the same public functions solve_task() already calls internally --
    this doesn't introduce a second solving path, just re-derives
    diagnostic detail solve_task()'s return value doesn't carry.
    """
    hypotheses = generate_hypotheses(task)
    verify_all(hypotheses, task)  # populates .verified/.fail_reason on each

    if method == "fallback":
        reason = (f"No hypothesis verified against all training pairs "
                  f"({len(hypotheses)} generated).")
    else:
        reason = (f"Hypothesis '{method}' verified on every training pair "
                  f"but the resulting test prediction did not match the "
                  f"expected solution.")

    return len(hypotheses), reason


def run_full_evaluation(tasks, task_ids, solutions, progress_every=50):
    """
    Evaluation-only pass over every task in task_ids. Solves each one via
    the existing solve_task() (unchanged), times it, isolates exceptions
    per-task so one broken task can't kill the run, and records enough
    detail on every failure to actually debug it afterward.

    Returns (results, total_duration_seconds) where results is a list of
    per-task dicts, one per task_id, in task_ids order.
    """
    import time
    import traceback

    results = []
    total_start = time.perf_counter()
    n = len(task_ids)

    for i, task_id in enumerate(task_ids):
        task = tasks[task_id]
        task_start = time.perf_counter()

        record = {
            "task_id":          task_id,
            "correct":          False,
            "method":           None,
            "duration_seconds": None,
            "error":            None,
            "hypotheses_tried": None,
            "failure_reason":   None,
            "predicted_output": None,
            "expected_output":  None,
        }

        try:
            predictions, method = solve_task(task)
            record["duration_seconds"] = time.perf_counter() - task_start
            record["method"] = method

            correct_outputs = solutions.get(task_id)
            is_correct = False
            if correct_outputs:
                for pred in predictions:
                    for correct_out in correct_outputs:
                        if np.array_equal(pred, correct_out):
                            is_correct = True
                            break
                    if is_correct:
                        break
            record["correct"] = is_correct

            if not is_correct:
                try:
                    n_hyps, reason = _diagnose_failure(task, method)
                    record["hypotheses_tried"] = n_hyps
                    record["failure_reason"] = reason
                except Exception:
                    record["failure_reason"] = (
                        "Could not compute failure diagnostics: "
                        + traceback.format_exc()
                    )

                record["predicted_output"] = (
                    predictions[0].tolist() if predictions else None
                )
                record["expected_output"] = (
                    correct_outputs[0].tolist() if correct_outputs else None
                )

                print(f"FAILED  {task_id}")
                print(f"  Predicted output: {record['predicted_output']}")
                print(f"  Expected output:  {record['expected_output']}")
                print(f"  Hypotheses tried:  {record['hypotheses_tried']}")
                print(f"  Reason: {record['failure_reason']}")

        except Exception:
            record["duration_seconds"] = time.perf_counter() - task_start
            record["error"] = traceback.format_exc()
            print(f"EXCEPTION on task {task_id}:")
            print(record["error"])

        results.append(record)

        if (i + 1) % progress_every == 0 or (i + 1) == n:
            pct = (i + 1) / n * 100
            print(f"Processed {i+1}/{n} ({pct:.1f}%)")

    total_duration = time.perf_counter() - total_start
    return results, total_duration


def print_evaluation_summary(results, total_duration):
    """
    Prints the summary block: totals, accuracy, timing, slowest/fastest
    10 tasks. Pure reporting over the results list -- no solving.
    """
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    errored = sum(1 for r in results if r["error"] is not None)
    accuracy = (correct / total * 100) if total else 0.0

    timed = [r for r in results if r["duration_seconds"] is not None]
    avg_time = (sum(r["duration_seconds"] for r in timed) / len(timed)) if timed else 0.0

    slowest = sorted(timed, key=lambda r: r["duration_seconds"], reverse=True)[:10]
    fastest = sorted(timed, key=lambda r: r["duration_seconds"])[:10]

    print("\n" + "=" * 60)
    print("FULL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total tasks evaluated:   {total}")
    print(f"Tasks solved correctly:  {correct}")
    print(f"Overall accuracy:        {accuracy:.2f}%")
    print(f"Tasks that raised exceptions: {errored}")
    print(f"Total execution time:    {total_duration:.2f}s "
          f"({total_duration/60:.2f} min)")
    print(f"Average time per task:   {avg_time:.4f}s")

    print("\nSlowest 10 tasks:")
    for r in slowest:
        print(f"  {r['task_id']}: {r['duration_seconds']:.4f}s")

    print("\nFastest 10 tasks:")
    for r in fastest:
        print(f"  {r['task_id']}: {r['duration_seconds']:.4f}s")

    if errored:
        print(f"\n{errored} tasks raised exceptions "
              f"(full tracebacks are in the saved results file):")
        for r in results:
            if r["error"] is not None:
                print(f"  {r['task_id']}")


def save_evaluation_results(results, path=r"C:\arc_solver\logs\full_evaluation_results.json"):
    """
    Saves the full per-task results list as JSON for later analysis --
    grids, timings, failure reasons, tracebacks, all of it, not just the
    summary printed to console.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull evaluation results saved to {path}")


def run_full_training_evaluation():
    """
    Entry point for the full 1000-task training evaluation described in
    the requirements: loads every training task, evaluates all of them
    through the unchanged solve_task(), prints the detailed summary and
    per-failure diagnostics, and saves everything to JSON.
    """
    print("=" * 60)
    print("FULL TRAINING EVALUATION")
    print("=" * 60)

    print("Loading all training tasks...")
    tasks    = load_all_tasks("training")
    task_ids = get_task_ids("training")
    print(f"Loaded {len(tasks)} tasks.\n")

    print("Loading training solutions...")
    solutions = load_solutions("training")

    print(f"Evaluating {len(task_ids)} tasks...\n")
    results, total_duration = run_full_evaluation(tasks, task_ids, solutions)

    print_evaluation_summary(results, total_duration)
    save_evaluation_results(results)

    return results


# ============================================================
# QUICK TEST
# py C:\arc_solver\src\layers\solver.py
# py C:\arc_solver\src\layers\solver.py --full-eval
# ============================================================

if __name__ == "__main__":
    if "--full-eval" in sys.argv:
        # New evaluation mode -- does not touch anything below in the
        # default path.
        run_full_training_evaluation()
    else:
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

        # Build the actual Kaggle submission -- build_kaggle_submission
        # gives every test input in every task a correctly-formatted
        # prediction, unlike solve_all_tasks which only covers test[0].
        print("\nGenerating submission format...")
        test_tasks = load_all_tasks("test")
        test_ids   = get_task_ids("test")
        submission = build_kaggle_submission(test_tasks, test_ids, verbose=True)
        save_submission(submission)

        print("\nFull pipeline complete.")
        print("Next step: submit submission.json to Kaggle.")