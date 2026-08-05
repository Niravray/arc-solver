"""
Falsification test: the CORRESPONDENCE hypothesis.

    Are the objects that CHANGE characterised by a RELATION TO ANOTHER
    OBJECT in the same scene -- shape twin, twin-up-to-rotation,
    containment, scaled copy -- rather than by a property of the object
    alone?

This is the family docs/VOCABULARY_PROPOSAL.md Part 5 names as "what has
never been tested", and it is what src/layers/graph_difference.py already
computes while sitting unwired.

===========================================================================
PRE-REGISTRATION -- fixed before the first run, do not edit after
===========================================================================

WHAT IS MEASURED
  For every evaluation task, label each input object CHANGED (at least one
  of its cells has a different colour in the output) or UNCHANGED. Then ask
  whether some predicate of conjunction depth <= 2 -- the solver's real
  MAX_CONJUNCTION -- is exactly the characteristic function of the CHANGED
  set, simultaneously on EVERY training pair, under at least one scene
  interpretation.

  Three counts fall out:

    BASELINE-SEPARABLE       a predicate over the solver's EXISTING
                             non-relational atoms separates
    CORR-SEPARABLE           a predicate containing >= 1 correspondence
                             atom separates            (hypothesis, weak form)
    CORR-ONLY                CORR-SEPARABLE and NOT BASELINE-SEPARABLE
                                                       (hypothesis, decision form)

  CORR-ONLY is the decision-relevant number: correspondence has to explain
  tasks the current language cannot, not merely coincide on tasks it can.

  Reported alongside, not as a threshold: NEW-CORR-ONLY, which excludes
  `shape_is_unique`. That atom is ALREADY in build_atom_catalog -- the
  solver can already say "no other object has my shape" -- so the genuinely
  new content of this hypothesis is transform-twins, containment and
  scaled copies.

PRE-REGISTERED THRESHOLDS
  Primary  : CORR-ONLY     >= 8 of 120  -> hypothesis survives
  Secondary: CORR-SEPARABLE >= 8 of 120 -> weak form survives
  Below both: FALSIFIED.

  Why 8. Extension A used 5/32, Extension B 6/120. Separability is
  necessary, not sufficient: A's own control separated only 4 of 7 tasks
  the solver ALREADY solves, and separating the right objects still leaves
  finding the right action. Every "observable -> solved" attrition measured
  in this project has been 3-10x, so 8 separable tasks is optimistically
  2-4 solves. Wiring graph_difference.py into the induction loop is a real
  build; below 8 its expected yield is at or under the 1 task Extension B
  earned, which was already judged not worth building.

ADVERSARIAL CONTROL -- the B-v1 lesson, applied
  B v1 certified a broken detector because every control case was one the
  detector was built for. This control has a hard positive, a real
  negative, and a chance baseline. Both real cases were chosen by READING
  the grids, before the detector was run on them.

  POSITIVE (hard) -- training 7d1f7ee8.  MUST be detected.
    Two framed rectangles (colours 1 and 2) each hold a nested object;
    those nested objects are simplified (inner pattern collapsed to the
    frame's colour). A third nested-square object of the same general
    shape sits OUTSIDE any frame and is left alone -- a genuine decoy: it
    is not distinguishable from the changed objects by colour, size, or
    any single-object atom (confirmed by hand and by the instrument: no
    depth<=2 baseline conjunction separates this task at all). What
    decides it, verified by hand against every one of the object's 3
    training pairs, is exactly one relation -- containment:
        changed  <=>  corr:is_contained
    (An earlier candidate, 776ffc46, was tried first and dropped after
    reading it fully: its rule needs a frame chosen by SIZE, whose
    content is then read and matched against loose shapes -- a 3-step
    "derive, then use" chain, not a single correspondence atom. That is
    a real and different capability gap -- see CLAUDE.md's "two-level
    rules" line -- but it is not what this test is built to detect, and
    using it here would blame the wrong hypothesis for a miss.)
    If the detector misses 7d1f7ee8, the instrument is blind and no
    evaluation count it produces means anything.

  NEGATIVE (adversarial) -- chosen from training by the rule that it must
    (a) be decided by a NON-relational property, and (b) have the
    correspondence atoms genuinely ACTIVE (both truth values present) so
    that a spurious separation is possible. A negative on which the atoms
    are constant proves nothing -- that is the trivial pass B v1 got.
    MUST NOT be reported CORR-ONLY.

  CHANCE BASELINE -- permutation. Shuffle the CHANGED/UNCHANGED labels
    within each pair, preserving class sizes, and re-run the same test on
    the same scenes. Correspondence atoms stay active; only the thing they
    are supposed to explain is destroyed. If the real CORR-ONLY count is
    not clearly above the shuffled count, the count is noise regardless of
    its size.

WHAT IS *NOT* CLAIMED
  Separability is not solvability. A task can separate here and still be
  unsolvable, because the ACTION on the selected objects may be outside
  the language. This test can only falsify, never confirm.
===========================================================================
"""

import argparse
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
for _p in (_SRC / "layers", _SRC / "primitives"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

import synthesis
from object_extractor import (
    extract_by_color,
    extract_objects,
    get_background_color,
)
from property_encoder import encode_all_objects
from task_loader import load_all_tasks

POSITIVE_CONTROL = "7d1f7ee8"
THRESHOLD = 8

# Guards, mirroring the solver's own object budget. A scene with hundreds
# of objects is not one any selector was going to describe anyway.
MAX_OBJECTS_PER_PAIR = 120
MAX_TOTAL_OBJECTS = 400


# ==========================================================================
# Scene interpretations
# ==========================================================================

def interpretations(grid):
    """The readings the solver actually searches over."""
    background = get_background_color(grid)
    out = {
        "connected4": extract_objects(grid, background=background, connectivity=4),
        "connected8": extract_objects(grid, background=background, connectivity=8),
        "by_color": extract_by_color(grid, background=background),
        "no_background": extract_objects(
            grid, background=background, connectivity=4, include_background=True
        ),
    }
    try:
        panels = synthesis.extract_panels(grid, background)
        if panels:
            out["panels"] = panels
    except Exception:
        pass
    return out


def object_mask(obj):
    """Colour-blind occupancy mask of an object, cropped to its bbox."""
    mask = np.zeros((obj["height"], obj["width"]), dtype=bool)
    for r, c in obj["pixels"]:
        mask[r - obj["min_row"], c - obj["min_col"]] = True
    return mask


def mask_key(mask):
    return (mask.shape, mask.tobytes())


DIHEDRAL = (
    lambda m: np.rot90(m, 1),
    lambda m: np.rot90(m, 2),
    lambda m: np.rot90(m, 3),
    lambda m: np.fliplr(m),
    lambda m: np.flipud(m),
    lambda m: m.T,
    lambda m: np.rot90(m, 2).T,
)


def is_scaled_copy(small, large):
    """True if `large` is an integer k>=2 upscaling of `small`."""
    sh, sw = small.shape
    lh, lw = large.shape
    if sh == 0 or sw == 0 or lh % sh or lw % sw:
        return False
    k = lh // sh
    if k < 2 or lw // sw != k:
        return False
    return np.array_equal(np.kron(small, np.ones((k, k), dtype=bool)), large)


# ==========================================================================
# The correspondence atoms
# ==========================================================================

def correspondence_features(objects):
    """
    For each object, its relations to the OTHER objects in the same scene.

    Returns a list of {atom_name: bool}, parallel to `objects`.
    """
    masks = [object_mask(o) for o in objects]
    keys = [mask_key(m) for m in masks]
    shape_counts = Counter(keys)
    colour_shape_counts = Counter(
        (o["color"], k) for o, k in zip(objects, keys)
    )
    top = shape_counts.most_common()
    majority_key = None
    if top and top[0][1] >= 2 and (len(top) == 1 or top[0][1] > top[1][1]):
        majority_key = top[0][0]

    # Transform twins. Precompute every dihedral image of every mask once.
    variants = []
    for m in masks:
        seen = {mask_key(m)}
        for op in DIHEDRAL:
            seen.add(mask_key(op(m)))
        variants.append(seen)

    features = []
    for i, obj in enumerate(objects):
        others = [j for j in range(len(objects)) if j != i]

        has_twin = shape_counts[keys[i]] >= 2
        has_twin_same_colour = colour_shape_counts[(obj["color"], keys[i])] >= 2

        # Twin up to rotation/reflection, EXCLUDING the exact match, so the
        # atom carries information the plain twin atom does not.
        has_transform_twin = any(
            keys[j] in variants[i] and keys[j] != keys[i] for j in others
        )

        has_scaled = any(
            is_scaled_copy(masks[i], masks[j]) or is_scaled_copy(masks[j], masks[i])
            for j in others
        )

        contains = any(
            obj["min_row"] <= objects[j]["min_row"]
            and obj["max_row"] >= objects[j]["max_row"]
            and obj["min_col"] <= objects[j]["min_col"]
            and obj["max_col"] >= objects[j]["max_col"]
            and (
                obj["min_row"], obj["max_row"], obj["min_col"], obj["max_col"]
            ) != (
                objects[j]["min_row"], objects[j]["max_row"],
                objects[j]["min_col"], objects[j]["max_col"]
            )
            for j in others
        )
        contained = any(
            objects[j]["min_row"] <= obj["min_row"]
            and objects[j]["max_row"] >= obj["max_row"]
            and objects[j]["min_col"] <= obj["min_col"]
            and objects[j]["max_col"] >= obj["max_col"]
            and (
                obj["min_row"], obj["max_row"], obj["min_col"], obj["max_col"]
            ) != (
                objects[j]["min_row"], objects[j]["max_row"],
                objects[j]["min_col"], objects[j]["max_col"]
            )
            for j in others
        )

        features.append({
            "corr:has_shape_twin": has_twin,
            "corr:has_shape_twin_same_colour": has_twin_same_colour,
            "corr:has_transform_twin": has_transform_twin,
            "corr:has_twin_or_transform_twin": has_twin or has_transform_twin,
            "corr:has_scaled_copy": has_scaled,
            "corr:contains_object": contains,
            "corr:is_contained": contained,
            "corr:is_majority_shape": majority_key is not None
                                      and keys[i] == majority_key,
        })
    return features


# `shape_is_unique` is the one correspondence atom the solver ALREADY has.
# It is scored as correspondence (it is a cross-object relation) but tracked
# separately, because building nothing at all would still buy whatever it buys.
EXISTING_CORR_ATOMS = ("shape_is_unique", "not_shape_is_unique")


def is_correspondence(name):
    return name.startswith("corr:") or name.startswith("not_corr:") \
        or name in EXISTING_CORR_ATOMS


def is_new_correspondence(name):
    return is_correspondence(name) and name not in EXISTING_CORR_ATOMS


# ==========================================================================
# Building the bit-vector problem for one task under one interpretation
# ==========================================================================

def changed_flags(objects, input_grid, output_grid):
    """An object is CHANGED if any of its own cells differs in the output."""
    flags = []
    for obj in objects:
        flags.append(any(
            input_grid[r, c] != output_grid[r, c] for r, c in obj["pixels"]
        ))
    return flags


def build_problem(task, strategy):
    """
    Flatten every object of every training pair into one bit-indexed slot
    list, and evaluate every atom over it.

    Returns (target_mask, all_mask, {atom_name: mask}) or None if this
    interpretation cannot be scored.
    """
    per_pair_objects = []
    per_pair_changed = []

    for pair in task["train"]:
        grid_in, grid_out = pair["input"], pair["output"]
        if grid_in.shape != grid_out.shape:
            return None
        scenes = interpretations(grid_in)
        if strategy not in scenes:
            return None
        objects = scenes[strategy]
        if not objects or len(objects) > MAX_OBJECTS_PER_PAIR:
            return None
        objects = encode_all_objects(objects, grid_in.shape)
        per_pair_objects.append(objects)
        per_pair_changed.append(changed_flags(objects, grid_in, grid_out))

    total = sum(len(o) for o in per_pair_objects)
    if total == 0 or total > MAX_TOTAL_OBJECTS:
        return None

    # A predicate can only be tested where both classes occur somewhere.
    if not any(any(f) for f in per_pair_changed):
        return None
    if not any(not all(f) for f in per_pair_changed):
        return None
    if not any(any(f) and not all(f) for f in per_pair_changed):
        return None

    flat_objects = [o for objs in per_pair_objects for o in objs]
    catalog = synthesis.build_atom_catalog(flat_objects)

    corr_per_pair = [correspondence_features(objs) for objs in per_pair_objects]

    n = total
    all_mask = (1 << n) - 1
    target = 0
    bit = 0
    for changed in per_pair_changed:
        for flag in changed:
            if flag:
                target |= 1 << bit
            bit += 1

    atom_masks = {}

    def register(name, values):
        mask = 0
        for index, value in enumerate(values):
            if value:
                mask |= 1 << index
        atom_masks[name] = mask

    for name, fn in catalog:
        if name == "all":
            continue
        values = []
        for objs in per_pair_objects:
            for obj in objs:
                try:
                    values.append(bool(fn(obj, objs)))
                except Exception:
                    values.append(False)
        register(name, values)

    corr_names = sorted(corr_per_pair[0][0]) if corr_per_pair[0] else []
    for name in corr_names:
        values = []
        for feats in corr_per_pair:
            for feat in feats:
                values.append(feat[name])
        register(name, values)
        register("not_" + name, [not v for v in values])

    return target, all_mask, atom_masks


def dedupe(atom_masks, all_mask):
    """
    Collapse atoms with identical truth on this evidence, keeping the
    cheapest name. Two atoms that are the same bit vector are the same
    predicate here, and searching both doubles the work for nothing.
    """
    best = {}
    for name, mask in atom_masks.items():
        if mask == 0 or mask == all_mask:
            continue  # constant: separates nothing, cannot help a conjunction
        prior = best.get(mask)
        if prior is None or _rank(name) < _rank(prior):
            best[mask] = name
    return [(name, mask) for mask, name in best.items()]


def _rank(name):
    """Prefer structural names over measurement-pinned ones, as the solver does."""
    metric = name.lstrip("not_").startswith(synthesis.METRIC_ATOM_PREFIXES)
    return (1 if metric else 0, len(name), name)


def search_separator(target, all_mask, atoms, require=None, forbid_corr=False):
    """
    Find a conjunction of depth <= 2 whose truth set is exactly `target`.

    require='corr'  -> at least one correspondence atom must appear
    forbid_corr     -> no correspondence atom may appear
    """
    pool = atoms
    if forbid_corr:
        pool = [(n, m) for n, m in pool if not is_correspondence(n)]

    for name, mask in pool:
        if mask == target and (require != "corr" or is_correspondence(name)):
            return (name,)

    if require == "corr":
        corr = [(n, m) for n, m in pool if is_correspondence(n)]
        for name_a, mask_a in corr:
            if mask_a & target != target:
                continue  # a conjunct can only shrink the set
            for name_b, mask_b in pool:
                if name_b == name_a:
                    continue
                if mask_a & mask_b == target:
                    return tuple(sorted((name_a, name_b)))
        return None

    for (name_a, mask_a), (name_b, mask_b) in combinations(pool, 2):
        if mask_a & target != target or mask_b & target != target:
            continue
        if mask_a & mask_b == target:
            return tuple(sorted((name_a, name_b)))
    return None


# ==========================================================================
# Per-task verdict
# ==========================================================================

STRATEGIES = ("connected4", "connected8", "by_color", "no_background", "panels")


def classify_task(task, shuffle_rng=None):
    """
    Returns a dict describing which kind of separator, if any, exists for
    this task under any interpretation.
    """
    result = {
        "applicable": False,
        "baseline": None,
        "corr": None,
        "new_corr": None,
        "strategy": None,
    }

    for strategy in STRATEGIES:
        problem = build_problem(task, strategy)
        if problem is None:
            continue
        target, all_mask, atom_masks = problem
        result["applicable"] = True

        if shuffle_rng is not None:
            target = _shuffle_target(target, all_mask, shuffle_rng)

        atoms = dedupe(atom_masks, all_mask)

        baseline = search_separator(target, all_mask, atoms, forbid_corr=True)
        corr = search_separator(target, all_mask, atoms, require="corr")

        if corr is not None and result["corr"] is None:
            result["corr"] = corr
            result["strategy"] = strategy
            if all(is_new_correspondence(n) or not is_correspondence(n)
                   for n in corr) and any(is_new_correspondence(n) for n in corr):
                result["new_corr"] = corr
        if baseline is not None and result["baseline"] is None:
            result["baseline"] = baseline

        if result["corr"] is not None and result["baseline"] is not None:
            break

    return result


def _shuffle_target(target, all_mask, rng):
    """Permute the CHANGED labels, preserving how many there are."""
    n = all_mask.bit_length()
    bits = [(target >> i) & 1 for i in range(n)]
    rng.shuffle(bits)
    out = 0
    for i, b in enumerate(bits):
        if b:
            out |= 1 << i
    return out


# ==========================================================================
# Controls
# ==========================================================================

def run_positive_control(training):
    print("CONTROL 1 -- hard positive")
    print(f"  {POSITIVE_CONTROL}: nested objects inside a frame get simplified;")
    print("             a same-shaped decoy outside any frame does not.")
    task = training[POSITIVE_CONTROL]
    result = classify_task(task)
    if not result["applicable"]:
        print("  NOT APPLICABLE -- instrument cannot even label this task")
        return False
    if result["corr"] is None:
        print("  MISSED. The instrument is blind; the evaluation count is void.")
        return False
    print(f"  DETECTED  {' AND '.join(result['corr'])}   [{result['strategy']}]")
    print(f"  baseline-only separator: {result['baseline']}")
    return True


def find_negative_candidates(training, limit=25):
    """
    Shortlist training tasks where the correspondence atoms are ACTIVE and
    both classes are populated -- the only tasks where a negative control
    can actually fail. Choice among them is made by hand, not here.
    """
    out = []
    for task_id, task in sorted(training.items()):
        problem = build_problem(task, "connected4")
        if problem is None:
            continue
        target, all_mask, atom_masks = problem
        active = [
            name for name, mask in atom_masks.items()
            if name.startswith("corr:") and mask not in (0, all_mask)
        ]
        if len(active) >= 3:
            out.append((task_id, len(active)))
        if len(out) >= limit:
            break
    return out


def run_negative_control(training, task_id):
    print("CONTROL 2 -- adversarial negative")
    task = training[task_id]
    result = classify_task(task)
    verdict = result["corr"] is not None and result["baseline"] is None
    print(f"  {task_id}: corr={result['corr']}  baseline={result['baseline']}")
    if verdict:
        print("  FIRED. The instrument reports correspondence where the rule "
              "is not relational -- over-permissive.")
    else:
        print("  SILENT on CORR-ONLY, as required.")
    return not verdict


def run_permutation_control(evaluation, repeats=5, seed=20260805):
    print(f"CONTROL 3 -- permutation ({repeats} shuffles of the labels)")
    rng = random.Random(seed)
    counts = []
    for rep in range(repeats):
        qualifying = 0
        for task in evaluation.values():
            result = classify_task(task, shuffle_rng=rng)
            if result["corr"] is not None and result["baseline"] is None:
                qualifying += 1
        counts.append(qualifying)
        print(f"  shuffle {rep + 1}: CORR-ONLY = {qualifying}")
    mean = sum(counts) / len(counts)
    print(f"  chance CORR-ONLY = {mean:.1f} of {len(evaluation)}")
    return mean


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative", default=None,
                        help="task id to use as the adversarial negative control")
    parser.add_argument("--shortlist", action="store_true",
                        help="list training tasks eligible as negative controls")
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    training = load_all_tasks("training")

    if args.shortlist:
        for task_id, active in find_negative_candidates(training):
            print(f"{task_id}  {active} active correspondence atoms")
        return

    print("=" * 74)
    print("FALSIFICATION TEST -- the correspondence hypothesis")
    print(f"pre-registered threshold: CORR-ONLY >= {THRESHOLD} of 120")
    print("=" * 74)
    print()

    positive_ok = run_positive_control(training)
    print()

    negative_ok = None
    if args.negative:
        negative_ok = run_negative_control(training, args.negative)
        print()

    if args.skip_eval:
        return

    evaluation = load_all_tasks("evaluation")
    print(f"Scanning {len(evaluation)} evaluation tasks...")

    funnel = Counter()
    corr_only_tasks = []
    corr_tasks = []
    atom_use = Counter()

    for task_id, task in sorted(evaluation.items()):
        result = classify_task(task)
        if not result["applicable"]:
            funnel["not applicable (shape change / no split labels)"] += 1
            continue
        funnel["applicable"] += 1
        if result["baseline"] is not None:
            funnel["baseline-separable"] += 1
        if result["corr"] is not None:
            funnel["corr-separable"] += 1
            corr_tasks.append((task_id, result))
        if result["corr"] is not None and result["baseline"] is None:
            funnel["CORR-ONLY"] += 1
            corr_only_tasks.append((task_id, result))
            for name in result["corr"]:
                atom_use[name] += 1

    print()
    print("FUNNEL")
    for key in ("not applicable (shape change / no split labels)", "applicable",
                "baseline-separable", "corr-separable", "CORR-ONLY"):
        print(f"  {key:48s} {funnel[key]:4d}")

    new_corr = [t for t, r in corr_only_tasks if r["new_corr"]]
    print(f"  {'of those, needing a NEW correspondence atom':48s} "
          f"{len(new_corr):4d}")

    print()
    print("CORR-ONLY tasks")
    if not corr_only_tasks:
        print("  (none)")
    for task_id, result in corr_only_tasks:
        marker = "NEW " if result["new_corr"] else "    "
        print(f"  {marker}{task_id}  {' AND '.join(result['corr'])}"
              f"   [{result['strategy']}]")

    if atom_use:
        print()
        print("atoms doing the work")
        for name, count in atom_use.most_common():
            print(f"  {name:42s} {count}")

    print()
    chance = run_permutation_control(evaluation, repeats=args.permutations)

    print()
    print("=" * 74)
    observed = funnel["CORR-ONLY"]
    weak = funnel["corr-separable"]
    print(f"positive control : {'DETECTED' if positive_ok else 'MISSED'}")
    if negative_ok is not None:
        print(f"negative control : {'SILENT' if negative_ok else 'FIRED'}")
    print(f"CORR-ONLY        : {observed} of {len(evaluation)}   "
          f"(threshold {THRESHOLD}, chance {chance:.1f})")
    print(f"CORR-SEPARABLE   : {weak} of {len(evaluation)}   "
          f"(threshold {THRESHOLD})")
    if not positive_ok:
        print("VERDICT          : VOID -- instrument failed its own control")
    elif observed >= THRESHOLD:
        print("VERDICT          : SURVIVES")
    elif weak >= THRESHOLD:
        print("VERDICT          : FALSIFIED in the decision form, "
              "weak form survives")
    else:
        print("VERDICT          : FALSIFIED")
    print("=" * 74)


if __name__ == "__main__":
    main()
