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
    scale_up, crop_to_content,
    remove_object, move_object_to_edge, move_object, recolor_object,
    shift_grid, apply_gravity, recolor_by_mapping, fill_enclosed_regions,
    SELECTION_PREDICATES
)
from relational import move_until_touching, connect_aligned_objects
from relational_selectors import nearest_objects, farthest_objects, aligned_objects
from relationship_graph import build_relationship_graph

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

# Graph-based target selection, relative to an anchor object chosen by
# SELECTION_PREDICATES, rather than another absolute-property selector.
# Both return every object tied for the extreme distance from the anchor;
# taking candidates[0] picks the lowest id among ties, matching the
# deterministic tie-break SELECTION_PREDICATES itself uses.
RELATIONAL_TARGET_SELECTORS = {
    "nearest_neighbor":  nearest_objects,
    "farthest_neighbor": farthest_objects,
}


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


def _precompute_pair_contexts(task):
    """
    Compute (input_grid, output_grid, objects, diff) once per training
    pair. Every hypothesis tried against this task needs the exact same
    four things for a given pair -- only apply_fn differs between
    hypotheses -- so this used to get recomputed once per hypothesis per
    pair for no reason. With hypothesis counts now running into the
    hundreds on tasks where several signature flags overlap, that
    redundant recomputation was the actual cost, not the hypotheses
    themselves. Composition's own re-extraction inside _compose() is NOT
    touched by this -- that one is unavoidable, since the intermediate
    grid genuinely differs per hypothesis.
    """
    contexts = []
    for pair in task["train"]:
        input_grid  = pair["input"]
        output_grid = pair["output"]
        bg      = get_background_color(input_grid)
        objects = encode_all_objects(extract_objects(input_grid, bg), input_grid.shape)
        diff    = compute_difference(input_grid, output_grid)
        contexts.append((input_grid, output_grid, objects, diff))
    return contexts


def _verify_with_contexts(hypothesis, contexts):
    """
    Same pass/fail logic and fail_reason strings as before, just taking
    precomputed (input_grid, output_grid, objects, diff) tuples instead
    of recomputing them.
    """
    for i, (input_grid, output_grid, objects, diff) in enumerate(contexts):
        try:
            predicted = hypothesis.apply_fn(input_grid, objects, diff)
        except Exception as e:
            hypothesis.verified    = False
            hypothesis.fail_reason = f"Exception on pair {i+1}: {e}"
            return False

        if predicted is None:
            hypothesis.verified    = False
            hypothesis.fail_reason = f"Returned None on pair {i+1}"
            return False

        if predicted.shape != output_grid.shape:
            hypothesis.verified    = False
            hypothesis.fail_reason = (f"Shape mismatch on pair {i+1}: "
                                      f"predicted {predicted.shape} "
                                      f"vs expected {output_grid.shape}")
            return False

        if not np.array_equal(predicted, output_grid):
            hypothesis.verified    = False
            hypothesis.fail_reason = f"Pixel mismatch on pair {i+1}"
            return False

    hypothesis.verified    = True
    hypothesis.fail_reason = None
    return True


def verify_hypothesis(hypothesis, task):
    """
    Test a hypothesis against all training pairs in a task.
    
    hypothesis = a Hypothesis object
    task       = task dict with "train" list of input/output pairs
    
    Returns True if hypothesis passes ALL training pairs exactly.
    Returns False if ANY training pair fails.

    Unchanged behavior/signature from before -- this now just delegates
    to the shared-context helpers below, so single-hypothesis callers
    (like a diagnostic script testing one rule against one task) still
    work exactly as before. If you're testing MANY hypotheses against
    the same task, call verify_all instead of calling this in a loop --
    that's what actually avoids the redundant recomputation.
    """
    contexts = _precompute_pair_contexts(task)
    return _verify_with_contexts(hypothesis, contexts)


def verify_all(hypotheses, task):
    """
    Test a list of hypotheses against a task.
    Returns a list of hypotheses that passed verification,
    sorted by confidence (highest first).

    Computes each training pair's objects/diff ONCE (via
    _precompute_pair_contexts) and reuses it across every hypothesis,
    instead of recomputing per hypothesis per pair.
    """
    contexts = _precompute_pair_contexts(task)
    verified = []

    for hyp in hypotheses:
        if _verify_with_contexts(hyp, contexts):
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


def _matching_resize_stages(diff):
    """
    Resize stages whose statically-known output shape can actually match
    this pair's observed input->output shape, instead of crossing every
    tile/scale combination in TILE_FACTORS x TILE_FACTORS x [2, 3] against
    every selector and color regardless of fit.

    tile_grid(grid, tr, tc) always produces (in_rows*tr, in_cols*tc), and
    scale_up(grid, f) always produces (in_rows*f, in_cols*f) -- both fully
    determined before the stage ever runs. A tile/scale factor that
    doesn't match diff["row_scale"]/["col_scale"] can never pass
    verify_hypothesis's shape check on this pair, so generating it is
    wasted verification work -- this was the single biggest cost in the
    whole pipeline: resize_then_recolor crossed the full 19-stage list
    against 7 selectors and up to ~10 colors, up to ~1330 hypotheses on
    one task, and none of them ever verified on any task in a full run.

    crop_to_content's output shape depends on grid content, not a static
    formula, so it stays a candidate whenever the grid actually shrank.
    """
    stages = []
    row_scale, col_scale = diff["row_scale"], diff["col_scale"]

    if row_scale == int(row_scale) and col_scale == int(col_scale):
        tr, tc = int(row_scale), int(col_scale)
        if tr in TILE_FACTORS and tc in TILE_FACTORS:
            stages.append((f"tile_{tr}x{tc}", _tile_stage(tr, tc)))
        if tr == tc and tr in (2, 3):
            stages.append((f"scale_up_{tr}", _scale_stage(tr)))

    if diff["size_decreased"]:
        stages.append(("crop_to_content", _crop_stage()))

    return stages


def _make_cached_graph_builder():
    """
    Returns a (objs, grid) -> RelationshipGraph function that memoizes on
    id(grid).

    build_relationship_graph is O(n^2) in object count -- one task in the
    training set (0a1d4ef5) has 209 objects, so a single build is already
    non-trivial. Every relational hypothesis's apply_fn used to call
    build_relationship_graph directly, and verify_all calls apply_fn once
    per hypothesis per training pair -- with ~84 relational hypotheses,
    that's the identical graph for the identical (objs, grid) rebuilt from
    scratch ~84 times per pair, since objs/grid don't change across
    hypotheses within one pair (they come from the same precomputed
    context). That reintroduced the exact class of blowup the
    _matching_resize_stages fix above was for, just one layer down: a
    single generate_hypotheses() call for 0a1d4ef5 took over 3 hours
    before this cache existed.

    Scoped per generate_hypotheses() call (a fresh cache is created and
    closed over by that call's lambdas) rather than module-global, since
    id(grid) is only a valid cache key within one task's set of training
    pairs -- reusing it across tasks could alias unrelated grids whose
    ids happen to collide after garbage collection.
    """
    cache = {}

    def cached_graph(objs, grid):
        key = id(grid)
        if key not in cache:
            cache[key] = build_relationship_graph(objs, grid.shape)
        return cache[key]

    return cached_graph


def _infer_color_mapping(*grid_pairs):
    """
    Infer a single old_color -> new_color substitution table consistent
    with every pixel across all given (input_grid, output_grid) pairs, or
    None if no such consistent global mapping exists.

    This is the "derive the parameter from data" approach, not "sweep a
    bounded set of guesses": recolor_all_{old}_to_{new} can only ever
    express ONE color swap per hypothesis, so a task needing several
    colors remapped at once is structurally out of reach for it no
    matter how the sweep is bounded. Reading the whole table straight off
    the pixels is direct, and generalizes to any number of simultaneous
    swaps for the same cost as one.

    Pass a single pair to derive from training pair 1 alone. Pass every
    training pair to get the "tune again" version: strictly more
    evidence than pair 1 alone, catching colors pair 1 never happened to
    contain, and re-validating pair 1's own guess against everything
    else at the same time. Returns None (reject early) the moment two
    pairs disagree on what one color maps to, or any pair has a
    different shape than its output -- a global substitution table can't
    explain either case, so there's no point building a hypothesis that
    verify_all would just reject anyway.
    """
    mapping = {}
    for input_grid, output_grid in grid_pairs:
        if input_grid.shape != output_grid.shape:
            return None
        for old_color, new_color in zip(
            (int(v) for v in input_grid.flat),
            (int(v) for v in output_grid.flat),
        ):
            if old_color in mapping and mapping[old_color] != new_color:
                return None
            mapping[old_color] = new_color

    mapping = {k: v for k, v in mapping.items() if k != v}
    return mapping if mapping else None


def _mapping_name(mapping):
    return "_".join(f"{old}to{new}" for old, new in sorted(mapping.items()))


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
    cached_graph = _make_cached_graph_builder()

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

    # ---- UNIFORM SHIFT HYPOTHESIS ----
    # If every matched object in this pair moved by the exact same
    # (delta_row, delta_col), that's a rigid whole-scene pan rather than
    # a per-object move -- the vector is read directly off this pair's
    # diff (same "derive the parameter from data" approach already used
    # for tile/scale factors in _matching_resize_stages), so this is
    # exactly one hypothesis instead of a selector sweep. A uniform shift
    # can't also change grid size, so it's gated out when size_changed.
    if not diff["size_changed"] and diff["moved_objects"]:
        deltas = {
            (ch["delta_row"], ch["delta_col"])
            for _, _, ch in diff["moved_objects"]
        }
        if len(deltas) == 1:
            (shift_dr, shift_dc), = deltas
            if (shift_dr, shift_dc) != (0, 0):
                hypotheses.append(Hypothesis(
                    name        = f"shift_grid_{shift_dr}_{shift_dc}",
                    apply_fn    = lambda grid, objs, d, dr=shift_dr, dc=shift_dc: (
                        shift_grid(grid, dr, dc, d["input_background"])
                    ),
                    confidence  = 0.8,
                    description = (f"Shift the whole grid by "
                                   f"(row={shift_dr}, col={shift_dc})")
                ))

    # ---- GRAVITY: slide every object independently until it collides ----
    # Covers the movement pattern none of the selector-based move
    # hypotheses can reach: multiple objects each sliding by a DIFFERENT
    # amount, stopping individually at the boundary or at another object,
    # rather than one selected object moving by one shared amount/vector.
    # Only 4 hypotheses (one per direction) -- verify_all's shape/pixel
    # check does the actual disambiguation, same as the rotate/mirror
    # hypotheses above.
    if not diff["size_changed"] and "OBJECTS_MOVED" in sig:
        for direction in ["up", "down", "left", "right"]:
            hypotheses.append(Hypothesis(
                name        = f"gravity_{direction}",
                apply_fn    = lambda grid, objs, d, dirn=direction: (
                    apply_gravity(grid, objs, dirn, d["input_background"])
                ),
                confidence  = 0.6,
                description = (f"Slide every object {direction} until it hits "
                               f"the edge or another object")
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

    # ---- RELATIONAL: move one object until it touches another ----
    # Same trigger and selector reuse as the move-to-edge block above, but
    # the destination is another object's position rather than a grid
    # edge. move_until_touching only needs the two plain object dicts
    # SELECTION_PREDICATES already returns -- no RelationshipGraph needed
    # for this. Skips same-name selector pairs since those would usually
    # pick the same single object (moving something until it touches
    # itself is never valid).
    if not diff["size_changed"] and "OBJECTS_MOVED" in sig:
        for mov_name, mov_sel in SELECTION_PREDICATES.items():
            for tgt_name, tgt_sel in SELECTION_PREDICATES.items():
                if mov_name == tgt_name:
                    continue
                hypotheses.append(Hypothesis(
                    name        = f"move_{mov_name}_until_touching_{tgt_name}",
                    apply_fn    = lambda grid, objs, d, msel=mov_sel, tsel=tgt_sel: (
                        move_until_touching(
                            grid, moving, target, background=d["input_background"]
                        )
                        if (moving := msel(objs)) is not None
                        and (target := tsel(objs)) is not None
                        else None
                    ),
                    confidence  = 0.5,
                    description = (f"Move the object selected by {mov_name} until "
                                   f"it touches the object selected by {tgt_name}")
                ))

    # ---- RELATIONAL: move an object until it touches its nearest/farthest
    # neighbor ----
    # Same move_until_touching primitive as above, but the target is chosen
    # by actual graph distance from the anchor rather than by an unrelated
    # absolute-property selector. Covers tasks where "the object it moves
    # toward" isn't expressible as "the leftmost/largest/etc. object" on
    # its own -- e.g. two clusters of objects where only relative distance
    # picks out the right pair. Bounded to 7 anchors x 2 relations = 14
    # hypotheses when this fires, same order as the block above.
    if not diff["size_changed"] and "OBJECTS_MOVED" in sig:
        for mov_name, mov_sel in SELECTION_PREDICATES.items():
            for rel_name, rel_fn in RELATIONAL_TARGET_SELECTORS.items():
                hypotheses.append(Hypothesis(
                    name        = f"move_{mov_name}_until_touching_{rel_name}",
                    apply_fn    = lambda grid, objs, d, msel=mov_sel, rfn=rel_fn: (
                        move_until_touching(
                            grid, moving, target, background=d["input_background"]
                        )
                        if (moving := msel(objs)) is not None
                        and (candidates := rfn(
                            cached_graph(objs, grid), moving["id"]
                        ))
                        and (target := candidates[0]) is not None
                        else None
                    ),
                    confidence  = 0.5,
                    description = (f"Move the object selected by {mov_name} until "
                                   f"it touches its {rel_name}")
                ))

    # ---- RELATIONAL: connect two aligned objects with a drawn line ----
    # connect_aligned_objects draws new pixels between two objects sharing
    # a centroid row/column -- it can only ever explain tasks where new
    # object(s) appear without any existing pixel being touched, so it's
    # gated on OBJECTS_ADDED rather than OBJECTS_MOVED/RECOLORED. Colors
    # reuse the same candidate_colors set the recolor generators below use,
    # so this stays bounded (7 anchors x <=10 colors) instead of sweeping
    # all 10 colors blind.
    if not diff["size_changed"] and "OBJECTS_ADDED" in sig:
        for anchor_name, anchor_sel in SELECTION_PREDICATES.items():
            for color in candidate_colors:
                hypotheses.append(Hypothesis(
                    name        = f"connect_aligned_{anchor_name}_color_{color}",
                    apply_fn    = lambda grid, objs, d, asel=anchor_sel, c=color: (
                        connect_aligned_objects(grid, anchor, target, color=c)
                        if (anchor := asel(objs)) is not None
                        and (candidates := aligned_objects(
                            cached_graph(objs, grid), anchor["id"]
                        ))
                        and (target := candidates[0]) is not None
                        else None
                    ),
                    confidence  = 0.5,
                    description = (f"Draw a color-{color} line connecting the object "
                                   f"selected by {anchor_name} to its aligned neighbor")
                ))

    # ---- FILL ENCLOSED REGIONS ----
    # Classic ARC "flood fill the holes" pattern: any background pixel
    # not reachable from the grid border by a flood fill is walled in by
    # non-background content, and gets recolored. Only fires when a color
    # not already in the input appears in the output (gated on
    # COLORS_ADDED), reusing the same candidate_colors set as the recolor
    # generators, so this is at most len(candidate_colors) hypotheses.
    if not diff["size_changed"] and "COLORS_ADDED" in sig:
        for color in candidate_colors:
            hypotheses.append(Hypothesis(
                name        = f"fill_enclosed_regions_{color}",
                apply_fn    = lambda grid, objs, d, c=color: (
                    fill_enclosed_regions(grid, c, d["input_background"])
                ),
                confidence  = 0.6,
                description = f"Fill every enclosed background region with color {color}"
            ))

    # ---- INFERRED COLOR MAPPING (derive, don't sweep) ----
    # A full simultaneous N-color substitution table, read directly off
    # the pixels rather than swept combinatorially -- see
    # _infer_color_mapping's docstring for why the sweep-based recolor
    # hypotheses below structurally can't reach this. Two candidates:
    # one derived from pair 1 alone, one derived from every pair at once
    # (the "tune again" step -- strictly more evidence, catches colors
    # pair 1 didn't happen to contain). Both get handed to verify_all
    # exactly like every other hypothesis; only len(mapping) > 1 is kept
    # since a single-entry mapping is already covered by
    # recolor_all_{old}_to_{new} above at no extra cost.
    if "OBJECTS_RECOLORED" in sig or diff["color_count_changed"]:
        pair1_mapping = _infer_color_mapping(
            (first_pair["input"], first_pair["output"])
        )
        if pair1_mapping and len(pair1_mapping) > 1:
            hypotheses.append(Hypothesis(
                name        = f"color_mapping_pair1_{_mapping_name(pair1_mapping)}",
                apply_fn    = lambda grid, objs, d, m=pair1_mapping: (
                    recolor_by_mapping(grid, m)
                ),
                confidence  = 0.7,
                description = f"Apply color substitution {pair1_mapping} inferred from pair 1"
            ))

        all_pairs_mapping = _infer_color_mapping(
            *[(p["input"], p["output"]) for p in task["train"]]
        )
        if (all_pairs_mapping and len(all_pairs_mapping) > 1
                and all_pairs_mapping != pair1_mapping):
            hypotheses.append(Hypothesis(
                name        = f"color_mapping_all_pairs_{_mapping_name(all_pairs_mapping)}",
                apply_fn    = lambda grid, objs, d, m=all_pairs_mapping: (
                    recolor_by_mapping(grid, m)
                ),
                confidence  = 0.75,
                description = (f"Apply color substitution {all_pairs_mapping} "
                               f"inferred from all training pairs")
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
        resize_stages = _matching_resize_stages(diff)

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