import os
import sys
from pathlib import Path

# Portable path bootstrap. Kaggle runs this on Linux under /kaggle, so
# absolute Windows paths cannot be baked in anywhere -- locations are
# derived from this file instead.
_SRC_DIR = Path(__file__).resolve().parent.parent
for _p in (_SRC_DIR / "layers", _SRC_DIR / "primitives"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np # type: ignore
import sys
import json
import os
import collections


from task_loader import (
    get_dev_holdout_split,
    get_task_ids,
    load_all_tasks,
    load_solutions,
)
from object_extractor import extract_objects, get_background_color
from property_encoder import encode_all_objects
from relationship_graph import build_relationship_graph
from difference_engine import compute_difference
from verifier import (
    Hypothesis, verify_hypothesis, verify_all,
    generate_hypotheses, get_top_predictions
)
from synthesis import synthesize_program

# ============================================================
# BEAM SEARCH + MAIN SOLVER
# Layer 8 of the ARC Solver Pipeline
#
# What this does:
# - Ties all layers together into one solve() function
# - For each task: asks both solvers for an answer, applies them to the
#   test input, and returns 2 predictions (ARC allows 2 attempts)
# - Runs the full pipeline on all tasks and measures accuracy
#
# TWO SOLVERS, NOT ONE
#
# There are two independent sources of answers here and they fail in
# different places, which is exactly why both are kept:
#
#   synthesis.py  INDUCES a rule per task, solving the holes in a small
#                 language from the observed change. Measured on the
#                 training set: 31 correct at 89% precision.
#   verifier.py   SWEEPS a prebuilt list of complete rules and keeps
#                 whatever verifies. Measured: 19 correct.
#
# They overlap on exactly ONE task out of 50, so the union is nearly the
# sum. Since ARC allows two attempts per test input, they do not have to
# compete: attempt 1 is the induced rule, attempt 2 is the sweep's best,
# and a task counts as solved if either is right. That is why this is a
# merge rather than a replacement, and why the sweep stays even though
# it is the weaker and less precise of the two.
#
# Synthesis goes first on precision: when it commits to a rule it is
# right 89% of the time, against a sweep hypothesis that verifies on the
# training pairs far more often than it generalizes.
# ============================================================


def _synthesis_prediction(program, test_input):
    """Apply an induced rule to a test grid, or None if it does not apply."""
    if program is None:
        return None
    try:
        return program.apply(test_input, get_background_color(test_input))
    except Exception:
        return None


def _merge_predictions(candidates, test_input):
    """
    Build exactly two attempts from the answers both solvers offered,
    dropping duplicates and padding when only one distinct answer exists.

    Deduplicating matters: if both solvers land on the same grid, using
    it for both attempts wastes the second one, so a distinct runner-up
    is preferred over a repeat.
    """
    merged = []
    for grid in candidates:
        if grid is None:
            continue
        if not any(np.array_equal(grid, kept) for kept in merged):
            merged.append(grid)
        if len(merged) == 2:
            break

    if not merged:
        return [test_input.copy(), np.zeros_like(test_input)]
    while len(merged) < 2:
        merged.append(merged[0])
    return merged[:2]


def solve_task(task):
    """
    Solve a single ARC task using both solvers.

    Returns (predictions, method) where predictions is a list of exactly
    2 grids. If neither solver produces anything, falls back to the test
    input unchanged plus a blank grid -- the "I don't know" answer.

    NOTE: only ever looks at task["test"][0], which is what the local
    scoring path (score_on_training / solve_all_tasks) needs. For the
    Kaggle submission, which needs a prediction for EVERY test input a
    task has, use solve_task_all_test_inputs() + build_kaggle_submission().
    """
    test_input = task["test"][0]["input"]

    # Source 1: induce a rule for this specific task.
    try:
        program, operator_name = synthesize_program(task)
    except Exception:
        program, operator_name = None, None
    induced = _synthesis_prediction(program, test_input)

    # An induced rule that also explains every training pair is a better
    # second attempt than an unrelated sweep hypothesis: the pairs cannot
    # distinguish it from the winner, so it is a genuinely different
    # reading of the same evidence rather than a worse one.
    induced_alternates = []
    for alternate in getattr(program, "alternates", []) or []:
        grid = _synthesis_prediction(alternate, test_input)
        if grid is not None:
            induced_alternates.append(grid)

    # Source 2: sweep the prebuilt hypothesis list.
    swept = []
    sweep_name = None
    try:
        verified = verify_all(generate_hypotheses(task), task)
        if verified:
            swept = get_top_predictions(verified, test_input, task, n=2)
            sweep_name = verified[0].name
    except Exception:
        swept = []

    predictions = _merge_predictions(
        [induced] + induced_alternates + list(swept), test_input)

    if induced is not None and sweep_name:
        method = f"synthesis:{operator_name}+sweep:{sweep_name}"
    elif induced is not None:
        method = f"synthesis:{operator_name}"
    elif sweep_name:
        method = sweep_name
    else:
        method = "fallback"

    return predictions, method


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

    Both the induced rule and the verified sweep hypotheses are derived
    ONCE from the training pairs and then applied to each test input --
    neither depends on which test grid it is being asked about, so
    re-deriving per grid would just repeat the same work.
    """
    try:
        program, operator_name = synthesize_program(task)
    except Exception:
        program, operator_name = None, None

    try:
        verified = verify_all(generate_hypotheses(task), task)
    except Exception:
        verified = []

    results = []
    for test_item in task["test"]:
        test_input = test_item["input"]

        induced = _synthesis_prediction(program, test_input)
        induced_alternates = []
        for alternate in getattr(program, "alternates", []) or []:
            grid = _synthesis_prediction(alternate, test_input)
            if grid is not None:
                induced_alternates.append(grid)

        swept, sweep_name = [], None
        if verified:
            try:
                swept = get_top_predictions(verified, test_input, task, n=2)
                sweep_name = verified[0].name
            except Exception:
                swept = []

        predictions = _merge_predictions(
            [induced] + induced_alternates + list(swept), test_input)

        if induced is not None and sweep_name:
            method = f"synthesis:{operator_name}+sweep:{sweep_name}"
        elif induced is not None:
            method = f"synthesis:{operator_name}"
        elif sweep_name:
            method = sweep_name
        else:
            method = "fallback"

        results.append((predictions, method))

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


def _default_submission_path():
    """Kaggle scores whatever lands in /kaggle/working/submission.json."""
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/submission.json"
    return str(Path(__file__).resolve().parent.parent.parent / "submission.json")


def save_submission(submission, path=None):
    """
    Save the submission to a JSON file.
    """
    path = path or _default_submission_path()
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
            "matched_attempt":  None,
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
                # Which ATTEMPT matched is the attribution that matters:
                # attempt 1 is the induced rule whenever synthesis produced
                # one, so this is what separates the two solvers' credit
                # rather than guessing from the method string.
                for slot, pred in enumerate(predictions, start=1):
                    for correct_out in correct_outputs:
                        if np.array_equal(pred, correct_out):
                            is_correct = True
                            record["matched_attempt"] = slot
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


def save_evaluation_results(results, path=None):
    """
    Saves the full per-task results list as JSON for later analysis --
    grids, timings, failure reasons, tracebacks, all of it, not just the
    summary printed to console.
    """
    path = path or str(Path(__file__).resolve().parent.parent.parent
                       / "logs" / "full_evaluation_results.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull evaluation results saved to {path}")


def evaluate_split(split, limit=None, verbose=False):
    """
    Run the full solver over one split and return (records, duration).

    `split` is "training" or "evaluation" -- both have ground-truth
    solutions. Progress printing is off by default so a dual run does
    not bury its own summary.
    """
    tasks     = load_all_tasks(split)
    task_ids  = get_task_ids(split)
    solutions = load_solutions(split)
    if limit:
        task_ids = task_ids[:limit]

    return run_full_evaluation(
        tasks, task_ids, solutions,
        progress_every=(50 if verbose else 10 ** 9),
    )


def _split_statistics(records):
    """Technical breakdown of one split's records."""
    total   = len(records)
    correct = [r for r in records if r["correct"]]
    errored = [r for r in records if r["error"] is not None]

    # Attribution. Attempt 1 holds the induced rule whenever synthesis
    # produced one, so the attempt that matched says which solver earned
    # the task -- the method string alone cannot, since it lists both
    # sources whenever both offered an answer.
    induced_offered = [r for r in records
                       if (r["method"] or "").startswith("synthesis:")]
    by_solver = collections.Counter()
    for r in correct:
        method = r["method"] or ""
        if method.startswith("synthesis:") and r["matched_attempt"] == 1:
            by_solver["synthesis"] += 1
        elif method == "fallback":
            by_solver["fallback (lucky)"] += 1
        else:
            by_solver["sweep"] += 1

    by_operator = collections.Counter()
    for r in correct:
        method = r["method"] or ""
        if method.startswith("synthesis:") and r["matched_attempt"] == 1:
            by_operator["synthesis:" + method.split("+")[0][10:]] += 1
        else:
            head = method.split("+sweep:")[-1] if "+sweep:" in method else method
            by_operator["sweep:" + head.split("_")[0]] += 1

    timed = [r["duration_seconds"] for r in records
             if r["duration_seconds"] is not None]

    return {
        "total":            total,
        "correct":          len(correct),
        "accuracy":         (len(correct) / total * 100) if total else 0.0,
        "errored":          len(errored),
        "induced_offered":  len(induced_offered),
        "induced_precision": (by_solver["synthesis"] / len(induced_offered) * 100)
                             if induced_offered else 0.0,
        "by_solver":        by_solver,
        "by_operator":      by_operator,
        "runtime":          sum(timed),
        "mean_task":        (sum(timed) / len(timed)) if timed else 0.0,
        "solved_ids":       [r["task_id"] for r in correct],
    }


def print_technical_report(stats_by_split):
    """
    Print accuracy AND the technical detail behind it, for every split.

    Both splits are always reported together. Measuring only on training
    once hid the fact that the solver scored 0 on the evaluation split --
    a headline accuracy with no attribution behind it cannot distinguish
    "proposed the wrong rule" from "proposed no rule at all", and those
    call for completely different work.
    """
    print("\n" + "=" * 64)
    print("TECHNICAL EVALUATION REPORT")
    print("=" * 64)

    print(f"\n{'split':<12}{'tasks':>7}{'solved':>8}{'accuracy':>11}"
          f"{'errors':>8}{'runtime':>10}")
    print("-" * 64)
    for split, s in stats_by_split.items():
        print(f"{split:<12}{s['total']:>7}{s['correct']:>8}"
              f"{s['accuracy']:>10.2f}%{s['errored']:>8}"
              f"{s['runtime']:>9.0f}s")

    for split, s in stats_by_split.items():
        print(f"\n--- {split} ---")
        print(f"  solver attribution     : {dict(s['by_solver']) or '{}'}")
        print(f"  induction offered rules: {s['induced_offered']}"
              f"   precision {s['induced_precision']:.0f}%")
        print(f"  mean time per task     : {s['mean_task']:.3f}s")
        if s["by_operator"]:
            print("  solved by:")
            for name, count in s["by_operator"].most_common():
                print(f"      {count:3d}  {name}")
        else:
            print("  solved by: nothing -- no rule was produced on any task")


def run_dual_evaluation(limit=None):
    """
    Evaluate BOTH splits and print one technical report.

    This is the default reporting path. The training split is the one we
    iterate against because it gives usable signal; the evaluation split
    is the honest measure of whether that signal transfers, and it is
    reported alongside so a training gain is never mistaken for a real
    one.
    """
    stats_by_split = {}

    print("Evaluating training split...")
    training_records, _ = evaluate_split("training", limit=limit)
    save_evaluation_results(training_records)

    # Partition the SAME records rather than running training twice --
    # the split is a reporting boundary, not a different computation.
    dev_ids, holdout_ids = get_dev_holdout_split("training")
    dev_set, holdout_set = set(dev_ids), set(holdout_ids)
    stats_by_split["training-dev"] = _split_statistics(
        [r for r in training_records if r["task_id"] in dev_set])
    stats_by_split["training-holdout"] = _split_statistics(
        [r for r in training_records if r["task_id"] in holdout_set])
    stats_by_split["training-all"] = _split_statistics(training_records)

    print("Evaluating evaluation split...")
    evaluation_records, _ = evaluate_split("evaluation", limit=limit)
    stats_by_split["evaluation"] = _split_statistics(evaluation_records)

    print_technical_report(stats_by_split)
    return stats_by_split


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
        # Reports BOTH splits. Training is what we iterate against;
        # evaluation is the honest check that the iteration transfers.
        run_dual_evaluation()
    elif "--training-only" in sys.argv:
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