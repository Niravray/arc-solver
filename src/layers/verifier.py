import numpy as np
import sys

sys.path.append(r"C:\arc_solver\src\layers")
sys.path.append(r"C:\arc_solver\src\primitives")

from task_loader import load_all_tasks, get_task_ids
from object_extractor import extract_objects, get_background_color
from property_encoder import encode_all_objects
from difference_engine import compute_difference
from spatial_color import (
    tile_grid, color_indicator_rule, rotate_90, rotate_180,
    mirror_horizontal, mirror_vertical, recolor_all, swap_colors,
    remove_smallest, recolor_largest, scale_up, crop_to_content
)

# ============================================================
# VERIFICATION LOOP
# Layer 7 of the ARC Solver Pipeline
#
# This is the most important component.
#
# What this does:
# - Takes a hypothesis (a rule describing a transformation)
# - Applies it to EVERY training input
# - Compares the result pixel-by-pixel to the actual training output
# - Returns True ONLY if ALL training pairs match exactly
# - If ANY pair fails — hypothesis is rejected immediately
#
# Zero tolerance. No exceptions.
# This is what separates 40% systems from 85%+ systems.
# ============================================================


class Hypothesis:
    """
    A hypothesis is a candidate rule that might explain a task.
    
    It has:
        name        : human readable name of the rule
        apply_fn    : function that takes (input_grid, objects, diff) 
                      and returns a predicted output grid
        confidence  : float 0-1, how confident we are this is right
        description : plain English description of the rule
    """
    def __init__(self, name, apply_fn, confidence=0.5, description=""):
        self.name        = name
        self.apply_fn    = apply_fn
        self.confidence  = confidence
        self.description = description
        self.verified    = False
        self.fail_reason = None

    def __repr__(self):
        return (f"Hypothesis(name={self.name}, "
                f"confidence={self.confidence:.2f}, "
                f"verified={self.verified})")


def verify_hypothesis(hypothesis, task):
    """
    Test a hypothesis against all training pairs in a task.
    
    hypothesis = a Hypothesis object
    task       = task dict with "train" list of input/output pairs
    
    Returns True if hypothesis passes ALL training pairs exactly.
    Returns False if ANY training pair fails.
    """
    train_pairs = task["train"]

    for i, pair in enumerate(train_pairs):
        input_grid  = pair["input"]
        output_grid = pair["output"]

        # Extract objects and diff for this pair
        bg      = get_background_color(input_grid)
        objects = encode_all_objects(extract_objects(input_grid, bg), input_grid.shape)
        diff    = compute_difference(input_grid, output_grid)

        # Apply the hypothesis
        try:
            predicted = hypothesis.apply_fn(input_grid, objects, diff)
        except Exception as e:
            hypothesis.verified    = False
            hypothesis.fail_reason = f"Exception on pair {i+1}: {e}"
            return False

        # If hypothesis returned None it cannot be applied
        if predicted is None:
            hypothesis.verified    = False
            hypothesis.fail_reason = f"Returned None on pair {i+1}"
            return False

        # Check shape matches
        if predicted.shape != output_grid.shape:
            hypothesis.verified    = False
            hypothesis.fail_reason = (f"Shape mismatch on pair {i+1}: "
                                      f"predicted {predicted.shape} "
                                      f"vs expected {output_grid.shape}")
            return False

        # Check pixel-perfect match
        if not np.array_equal(predicted, output_grid):
            hypothesis.verified    = False
            hypothesis.fail_reason = f"Pixel mismatch on pair {i+1}"
            return False

    # All pairs passed
    hypothesis.verified    = True
    hypothesis.fail_reason = None
    return True


def verify_all(hypotheses, task):
    """
    Test a list of hypotheses against a task.
    Returns a list of hypotheses that passed verification,
    sorted by confidence (highest first).
    """
    verified = []

    for hyp in hypotheses:
        if verify_hypothesis(hyp, task):
            verified.append(hyp)

    # Sort by confidence
    verified.sort(key=lambda h: h.confidence, reverse=True)
    return verified


def apply_to_test(hypothesis, test_input, task):
    """
    Apply a verified hypothesis to the test input.
    
    Returns the predicted output grid, or None if it fails.
    """
    bg      = get_background_color(test_input)
    objects = encode_all_objects(
        extract_objects(test_input, bg), test_input.shape
    )

    # Use the first training pair's diff as reference
    # (gives the hypothesis context about expected output size etc.)
    ref_diff = compute_difference(
        task["train"][0]["input"],
        task["train"][0]["output"]
    )

    try:
        predicted = hypothesis.apply_fn(test_input, objects, ref_diff)
        return predicted
    except Exception:
        return None


def get_top_predictions(verified_hypotheses, test_input, task, n=2):
    """
    Get the top N predictions for a test input.
    ARC allows 2 predictions per task — we submit the top 2.
    
    Returns a list of up to N predicted grids.
    """
    predictions = []

    for hyp in verified_hypotheses[:n]:
        pred = apply_to_test(hyp, test_input, task)
        if pred is not None:
            predictions.append(pred)

    return predictions


# ============================================================
# HYPOTHESIS GENERATORS
# These functions look at a diff and generate candidate hypotheses
# to try. This is the baseline version — simple rules only.
# ============================================================

def generate_hypotheses(task):
    """
    Generate all candidate hypotheses for a task.
    
    Looks at the diff from the first training pair and
    generates hypotheses based on what changed.
    
    Returns a list of Hypothesis objects.
    """
    hypotheses = []

    # Get diff from first training pair
    first_pair = task["train"][0]
    diff = compute_difference(first_pair["input"], first_pair["output"])
    sig  = diff["signature"]

    # ---- TILING HYPOTHESES ----
    # If the output is larger than the input, try tiling
    if diff["size_increased"] or diff["size_changed"]:
        for tr in [2, 3, 4, 5]:
            for tc in [2, 3, 4, 5]:
                tile_r, tile_c = tr, tc
                hypotheses.append(Hypothesis(
                    name        = f"tile_{tr}x{tc}",
                    apply_fn    = lambda grid, objs, d, r=tile_r, c=tile_c: (
                        tile_grid(grid, r, c)
                    ),
                    confidence  = 0.8 if diff.get("is_tiling") else 0.3,
                    description = f"Tile input {tr}x{tc} times"
                ))

    # ---- ROTATION HYPOTHESES ----
    # Always worth trying if size didn't change
    if not diff["size_changed"]:
        hypotheses.append(Hypothesis(
            name        = "rotate_90_cw",
            apply_fn    = lambda grid, objs, d: rotate_90(grid, clockwise=True),
            confidence  = 0.4,
            description = "Rotate 90 degrees clockwise"
        ))
        hypotheses.append(Hypothesis(
            name        = "rotate_180",
            apply_fn    = lambda grid, objs, d: rotate_180(grid),
            confidence  = 0.4,
            description = "Rotate 180 degrees"
        ))
        hypotheses.append(Hypothesis(
            name        = "rotate_90_ccw",
            apply_fn    = lambda grid, objs, d: rotate_90(grid, clockwise=False),
            confidence  = 0.4,
            description = "Rotate 90 degrees counter-clockwise"
        ))

    # ---- MIRROR HYPOTHESES ----
    if not diff["size_changed"]:
        hypotheses.append(Hypothesis(
            name        = "mirror_horizontal",
            apply_fn    = lambda grid, objs, d: mirror_horizontal(grid),
            confidence  = 0.4,
            description = "Mirror left-right"
        ))
        hypotheses.append(Hypothesis(
            name        = "mirror_vertical",
            apply_fn    = lambda grid, objs, d: mirror_vertical(grid),
            confidence  = 0.4,
            description = "Mirror top-bottom"
        ))

    # ---- COLOR INDICATOR HYPOTHESIS ----
    # If objects disappeared and something was recolored
    if "OBJECTS_REMOVED" in sig and "OBJECTS_RECOLORED" in sig:
        hypotheses.append(Hypothesis(
            name        = "color_indicator",
            apply_fn    = lambda grid, objs, d: color_indicator_rule(
                grid, objs, d["input_background"]
            ),
            confidence  = 0.9,
            description = "Smallest object recolors largest, then disappears"
        ))

    # ---- SCALE UP HYPOTHESES ----
    if diff["size_increased"]:
        for factor in [2, 3]:
            f = factor
            hypotheses.append(Hypothesis(
                name        = f"scale_up_{f}",
                apply_fn    = lambda grid, objs, d, fac=f: scale_up(grid, fac),
                confidence  = 0.5,
                description = f"Scale up by factor {f}"
            ))

    return hypotheses


# ============================================================
# QUICK TEST
# py C:\arc_solver\src\layers\verifier.py
# ============================================================

if __name__ == "__main__":
    print("Loading training tasks...")
    tasks = load_all_tasks("training")
    ids   = get_task_ids("training")
    print(f"Loaded {len(tasks)} tasks.\n")

    # ---- Test on color indicator task ----
    print("--- Testing color indicator hypothesis on task 009d5c81 ---")
    task = tasks["009d5c81"]

    hypotheses = generate_hypotheses(task)
    print(f"Generated {len(hypotheses)} hypotheses")
    for h in hypotheses:
        print(f"  {h.name} (confidence={h.confidence})")

    verified = verify_all(hypotheses, task)
    print(f"\nVerified hypotheses: {len(verified)}")
    for h in verified:
        print(f"  PASS: {h.name} — {h.description}")

    # Show failed hypotheses
    failed = [h for h in hypotheses if not h.verified]
    print(f"Failed hypotheses: {len(failed)}")
    for h in failed[:3]:
        print(f"  FAIL: {h.name} — {h.fail_reason}")

    # ---- Run on first 20 tasks ----
    print("\n--- Running on first 20 tasks ---")
    solved   = 0
    unsolved = 0

    for i in range(20):
        task_id = ids[i]
        task    = tasks[task_id]

        hypotheses = generate_hypotheses(task)
        verified   = verify_all(hypotheses, task)

        if verified:
            solved += 1
            best = verified[0]
            print(f"Task {i+1} ({task_id}): SOLVED — {best.name}")
        else:
            unsolved += 1
            print(f"Task {i+1} ({task_id}): unsolved "
                  f"({len(hypotheses)} hypotheses tried)")

    print(f"\nResults: {solved} solved, {unsolved} unsolved out of 20 tasks")
    print(f"Baseline accuracy: {solved/20*100:.1f}%")