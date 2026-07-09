import numpy as np # type: ignore
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
    scale_up, crop_to_content,
    remove_object, move_object_to_edge, recolor_object,
    SELECTION_PREDICATES
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

TILE_FACTORS = [2, 3, 4, 5]


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
# HYPOTHESIS COMPOSITION
#
# Every hypothesis above (and every single-step one below) calls
# exactly one primitive. That structurally can't solve a task whose
# signature has multiple flags at once (e.g. OBJECTS_REMOVED and
# OBJECTS_MOVED together) -- no single primitive removes AND moves.
#
# A "stage" here is a plain (grid, objs, diff) -> grid function --
# the exact same shape as a Hypothesis.apply_fn. _compose() chains
# two of them, re-extracting objects from the intermediate grid
# before the second stage runs. That re-extraction matters: after
# stage 1 removes or moves something, "the smallest object" for
# stage 2 needs to mean smallest of what's THERE NOW, not smallest
# of the original pre-stage-1 objects.
#
# These use small factory functions (_remove_stage, _move_to_edge_stage,
# etc.) rather than the inline-lambda-with-default-args style used
# below, because a composed hypothesis needs two independently-bound
# selectors/params instead of one, and that gets hard to read as a
# single nested lambda. Each factory call creates its own local
# closure, which sidesteps the late-binding-loop-variable bug the
# same way the default-arg trick does elsewhere in this file.
# ============================================================

def _compose(stage1_fn, stage2_fn):
    def composed(grid, objs, diff):
        intermediate = stage1_fn(grid, objs, diff)
        if intermediate is None:
            return None
        bg = get_background_color(intermediate)
        new_objs = encode_all_objects(
            extract_objects(intermediate, bg), intermediate.shape
        )
        return stage2_fn(intermediate, new_objs, diff)
    return composed


def _remove_stage(sel_fn):
    def stage(grid, objs, diff):
        obj = sel_fn(objs)
        if obj is None:
            return None
        return remove_object(grid, obj, diff["input_background"])
    return stage


def _move_to_edge_stage(sel_fn, direction):
    def stage(grid, objs, diff):
        obj = sel_fn(objs)
        if obj is None:
            return None
        return move_object_to_edge(grid, obj, direction, diff["input_background"])
    return stage


def _recolor_stage(sel_fn, color):
    def stage(grid, objs, diff):
        obj = sel_fn(objs)
        if obj is None:
            return None
        return recolor_object(grid, obj, color)
    return stage


def _tile_stage(tr, tc):
    def stage(grid, objs, diff):
        return tile_grid(grid, tr, tc)
    return stage


def _scale_stage(factor):
    def stage(grid, objs, diff):
        return scale_up(grid, factor)
    return stage


def _crop_stage():
    def stage(grid, objs, diff):
        return crop_to_content(grid, diff["input_background"])
    return stage


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

    # Colors actually observed in this pair, used by every recolor-family
    # hypothesis below (single-step and composed) so color guesses stay
    # bounded instead of sweeping all 10 colors blind. Computed once here
    # since several blocks below need the same sets.
    input_colors  = set(int(v) for v in np.unique(first_pair["input"]))
    output_colors = set(int(v) for v in np.unique(first_pair["output"]))
    new_colors       = output_colors - input_colors
    candidate_colors = new_colors if new_colors else output_colors

    # ---- TILING HYPOTHESES ----
    # If the output is larger than the input, try tiling
    if diff["size_increased"] or diff["size_changed"]:
        for tr in TILE_FACTORS:
            for tc in TILE_FACTORS:
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

    # ---- CROP TO CONTENT HYPOTHESIS ----
    if diff["size_changed"] and not diff["size_increased"]:
        hypotheses.append(Hypothesis(
            name        = "crop_to_content",
            apply_fn    = lambda grid, objs, d: crop_to_content(
                grid, d["input_background"]
            ),
            confidence  = 0.5,
            description = "Crop the grid down to its non-background content"
        ))

    # ---- OBJECT-SELECTOR HYPOTHESES ----
    if "OBJECTS_REMOVED" in sig:
        for sel_name, sel_fn in SELECTION_PREDICATES.items():
            hypotheses.append(Hypothesis(
                name        = f"remove_{sel_name}",
                apply_fn    = lambda grid, objs, d, sel=sel_fn: (
                    remove_object(grid, obj, d["input_background"])
                    if (obj := sel(objs)) is not None else None
                ),
                confidence  = 0.6,
                description = f"Remove the object selected by {sel_name}"
            ))

    if not diff["size_changed"]:
        for direction in ["up", "down", "left", "right"]:
            for sel_name, sel_fn in SELECTION_PREDICATES.items():
                hypotheses.append(Hypothesis(
                    name        = f"move_{sel_name}_to_edge_{direction}",
                    apply_fn    = lambda grid, objs, d, sel=sel_fn, dirn=direction: (
                        move_object_to_edge(grid, obj, dirn, d["input_background"])
                        if (obj := sel(objs)) is not None else None
                    ),
                    confidence  = 0.5,
                    description = f"Move the object selected by {sel_name} to the {direction} edge"
                ))

    if "OBJECTS_RECOLORED" in sig:
        for sel_name, sel_fn in SELECTION_PREDICATES.items():
            for color in candidate_colors:
                hypotheses.append(Hypothesis(
                    name        = f"recolor_{sel_name}_to_{color}",
                    apply_fn    = lambda grid, objs, d, sel=sel_fn, c=color: (
                        recolor_object(grid, obj, c)
                        if (obj := sel(objs)) is not None else None
                    ),
                    confidence  = 0.5,
                    description = f"Recolor the object selected by {sel_name} to color {color}"
                ))

        for old_c in input_colors:
            for new_c in output_colors:
                if old_c == new_c:
                    continue
                hypotheses.append(Hypothesis(
                    name        = f"recolor_all_{old_c}_to_{new_c}",
                    apply_fn    = lambda grid, objs, d, oc=old_c, nc=new_c: (
                        recolor_all(grid, oc, nc)
                    ),
                    confidence  = 0.4,
                    description = f"Recolor every {old_c} pixel to {new_c}"
                ))

        sorted_input_colors = sorted(input_colors)
        for i in range(len(sorted_input_colors)):
            for j in range(i + 1, len(sorted_input_colors)):
                ca, cb = sorted_input_colors[i], sorted_input_colors[j]
                hypotheses.append(Hypothesis(
                    name        = f"swap_colors_{ca}_{cb}",
                    apply_fn    = lambda grid, objs, d, a=ca, b=cb: (
                        swap_colors(grid, a, b)
                    ),
                    confidence  = 0.4,
                    description = f"Swap colors {ca} and {cb}"
                ))

    # ---- COMPOSED: remove an object, then move another to an edge ----
    # Targets tasks like OBJECTS_REMOVED|OBJECTS_MOVED together, which
    # neither remove_{selector} nor move_{selector}_to_edge_* can solve
    # alone -- each only performs one of the two actions.
    if "OBJECTS_REMOVED" in sig and "OBJECTS_MOVED" in sig:
        for rem_name, rem_sel in SELECTION_PREDICATES.items():
            for direction in ["up", "down", "left", "right"]:
                for mov_name, mov_sel in SELECTION_PREDICATES.items():
                    hypotheses.append(Hypothesis(
                        name        = (f"remove_{rem_name}_then_"
                                       f"move_{mov_name}_to_edge_{direction}"),
                        apply_fn    = _compose(
                            _remove_stage(rem_sel),
                            _move_to_edge_stage(mov_sel, direction)
                        ),
                        confidence  = 0.5,
                        description = (f"Remove object selected by {rem_name}, "
                                       f"then move object selected by {mov_name} "
                                       f"to the {direction} edge")
                    ))

    # ---- COMPOSED: resize the grid, then recolor a selected object ----
    # Targets tasks with SIZE_CHANGE and OBJECTS_RECOLORED together.
    # recolor_object never changes grid shape and none of the resize
    # primitives touch color, so this combination is otherwise
    # unreachable. The resize side reuses the same tile/scale/crop
    # options already generated above as single-step hypotheses.
    if diff["size_changed"] and "OBJECTS_RECOLORED" in sig:
        resize_stages = [("crop_to_content", _crop_stage())]
        for tr in TILE_FACTORS:
            for tc in TILE_FACTORS:
                resize_stages.append((f"tile_{tr}x{tc}", _tile_stage(tr, tc)))
        for factor in [2, 3]:
            resize_stages.append((f"scale_up_{factor}", _scale_stage(factor)))

        for resize_name, resize_stage in resize_stages:
            for sel_name, sel_fn in SELECTION_PREDICATES.items():
                for color in candidate_colors:
                    hypotheses.append(Hypothesis(
                        name        = f"{resize_name}_then_recolor_{sel_name}_to_{color}",
                        apply_fn    = _compose(
                            resize_stage,
                            _recolor_stage(sel_fn, color)
                        ),
                        confidence  = 0.5,
                        description = (f"{resize_name}, then recolor the object "
                                       f"selected by {sel_name} to color {color}")
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