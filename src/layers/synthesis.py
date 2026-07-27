import sys

sys.path.append(r"C:\arc_solver\src\layers")
sys.path.append(r"C:\arc_solver\src\primitives")

import numpy as np
from scipy import ndimage

from object_extractor import extract_objects, get_background_color
from property_encoder import encode_all_objects
from relationship_graph import build_relationship_graph

# ============================================================
# RULE SYNTHESIS ENGINE
#
# A different approach from verifier.py's generate_hypotheses().
#
# generate_hypotheses builds a list of COMPLETE rules with their
# parameters already baked in ("recolor the largest object to 4",
# "rotate 180") and hands every one of them to verify_all to filter.
# The rule that solves a task has to already be somewhere in that list.
#
# This module instead defines a small LANGUAGE of expressions with
# HOLES in them -- Recolor(<which objects?>, <what color?>) -- and
# SOLVES for the holes using the observed input/output change.
#
# The loop is counterexample-guided: induce from the first training
# pair, test on the next pair, and when a pair disproves the current
# rule, ADD that pair to the evidence set and re-induce. Each
# counterexample narrows what the language may say, rather than just
# rejecting a guess.
#
# STRUCTURE: the loop (synthesize) is operator-agnostic. An Operator
# supplies three things -- how to observe a pair, how to solve its
# holes, and how to describe a failure -- and the identical loop drives
# all of them. "Which objects?" is a hole EVERY operator has, so
# induce_selectors below is shared machinery, not per-operator code.
#
# NOT integrated into solver.py: this is a proof of concept under
# validation across operators, and is deliberately kept off the
# scoring path so nothing here can regress the existing sweep.
# ============================================================

MAX_DISTINCT_VALUES = 12   # cap on value-parameterized atoms, keeps the catalog bounded
MAX_CONJUNCTION     = 2    # how many atoms a selector may AND together

# Relational annotation builds an O(n^2) pairwise graph. Grids with very
# many objects (one training task has 209) would make that dominate
# induction, and tasks with that many objects are not the ones
# anchor-relative rules describe anyway, so they are skipped.
MAX_RELATIONAL_OBJECTS = 60

# Anchor-relative atoms are built and cached but kept OUT of the default
# catalog, because measuring them on all 1000 training tasks showed they
# do not pay for themselves: solved count was unchanged at 14, while
# Recolor precision fell from 75% to 64% as two extra tasks acquired
# contorted rules that fit training and failed the test, e.g.
#   rel:a_below_b@color3 -> 3 | is_sym_v AND rel:a_above_b@color6 -> 6 | ...
# What they DID change is where induction dies: across the three
# operators, tasks blocked at "no rule from pair 1" fell 51 -> 39 while
# "counterexample killed every candidate" rose 33 -> 43. So the extra
# vocabulary is expressive and genuinely used -- it just lets the engine
# fit the first pair in more ways without getting closer to the real
# rule. Read together, that says the binding constraint is no longer
# vocabulary but STRUCTURE (single-operator rules), which is the case
# for operator composition rather than more predicates.
# Flip to True to re-enable; the annotation machinery stays live so the
# experiment is one flag away.
USE_RELATIONAL_ATOMS = False

# Relations recorded from an object TO an anchor. Directional names read
# "object is <relation> the anchor".
ANCHOR_RELATIONS = (
    "touching",
    "touching_8",
    "same_centroid_row",
    "same_centroid_col",
    "a_above_b",
    "a_below_b",
    "a_left_of_b",
    "a_right_of_b",
    "a_inside_bbox_b",
    "same_color",
)

DIRECTIONS = {
    "up":    (-1, 0),
    "down":  (1, 0),
    "left":  (0, -1),
    "right": (0, 1),
}


# ============================================================
# THE LANGUAGE: selector atoms
#
# Each atom is a named predicate over (object, all objects in its grid).
# These are the VOCABULARY -- the words a rule can be written with.
# They are not rules: none of them says what to do, only which objects
# a rule would be talking about.
# ============================================================

def hole_count(obj):
    """
    Number of background regions fully enclosed by the object.

    Added because task 0a2355a6's rule is a clean function of it
    (1 hole -> 1, 2 -> 3, 3 -> 2) and nothing already in the catalog
    could separate those objects -- the engine was forced into a
    memorizing size lookup that then failed on the unseen test grid. A
    region touching the bounding-box edge is outside the shape, not a
    hole, so only strictly interior regions count.

    Cached on the object dict: induction evaluates atoms over every
    object many times while searching selectors.
    """
    if "_hole_count" in obj:
        return obj["_hole_count"]

    grid = obj["grid"]
    filled = (grid == obj["color"])
    labels, n = ndimage.label(~filled)

    holes = 0
    for i in range(1, n + 1):
        rows, cols = np.where(labels == i)
        if (rows.min() > 0 and cols.min() > 0
                and rows.max() < grid.shape[0] - 1
                and cols.max() < grid.shape[1] - 1):
            holes += 1

    obj["_hole_count"] = holes
    return holes


def _pick_anchors(objects):
    """
    Canonical anchor objects: each name must pick AT MOST ONE object, and
    must mean the same thing in every grid, so an atom built on it
    carries over to the test input.

    Colour anchors are only offered when exactly one object has that
    colour -- "the red one" is a usable landmark, "one of the four red
    ones" is not.
    """
    anchors = {}
    if not objects:
        return anchors

    by_size = sorted(objects, key=lambda o: (-o["pixel_count"], o["id"]))
    anchors["largest"]  = by_size[0]
    anchors["smallest"] = by_size[-1]

    color_counts = {}
    for o in objects:
        color_counts[o["color"]] = color_counts.get(o["color"], 0) + 1

    singletons = [o for o in objects if color_counts[o["color"]] == 1]
    if len(singletons) == 1:
        anchors["unique_color"] = singletons[0]

    for o in objects:
        if color_counts[o["color"]] == 1:
            anchors[f"color{o['color']}"] = o

    return anchors


def annotate_relational(objects, grid_shape):
    """
    Cache each object's relations to the grid's anchor objects.

    Why anchor-relative rather than existential: probing the tasks whose
    selector hole was unsolvable showed the distinguishing concept is
    almost never "touches SOMETHING" but "touches THE MARKER", "shares a
    column with THE LINE". Quantifying a relation existentially over all
    other objects throws away precisely the information the rule turns
    on.

    The pairwise graph is built ONCE per grid and the derived booleans
    are cached on the object dicts, because induction evaluates every
    atom over every object many times while searching for a selector.
    Rebuilding this graph inside the predicates instead would repeat the
    mistake that once made a single task take three hours.
    """
    for obj in objects:
        obj["_rel"] = {}

    if len(objects) < 2 or len(objects) > MAX_RELATIONAL_OBJECTS:
        return objects

    anchors = _pick_anchors(objects)
    if not anchors:
        return objects

    graph = build_relationship_graph(objects, grid_shape)
    pairs = graph["pairs"]

    for anchor_name, anchor in anchors.items():
        others = [o for o in objects if o["id"] != anchor["id"]]
        if not others:
            continue
        distances = {
            o["id"]: pairs[(o["id"], anchor["id"])]["centroid_distance"]
            for o in others
        }
        nearest = min(distances.values())

        for obj in objects:
            features = obj["_rel"]
            if obj["id"] == anchor["id"]:
                # An object is not "related to" itself; leaving these
                # False keeps the anchor out of its own selector.
                for relation in ANCHOR_RELATIONS:
                    features[f"{relation}@{anchor_name}"] = False
                features[f"is_nearest@{anchor_name}"] = False
                features[f"is_anchor@{anchor_name}"] = True
                continue

            relation_values = pairs[(obj["id"], anchor["id"])]
            for relation in ANCHOR_RELATIONS:
                features[f"{relation}@{anchor_name}"] = bool(relation_values[relation])
            features[f"is_nearest@{anchor_name}"] = (
                distances[obj["id"]] == nearest
            )
            features[f"is_anchor@{anchor_name}"] = False

    return objects


def build_atom_catalog(all_objects):
    """
    Build the atomic predicates available for this task, grounded in the
    property values that actually occur in it.

    Value-parameterized atoms (color==3, size==5) are only emitted for
    values genuinely present in the evidence, so the catalog describes
    this task rather than all of ARC.
    """
    atoms = [("all", lambda o, objs: True)]

    def add_bool(name, fn):
        atoms.append((name, fn))
        atoms.append((f"not_{name}", lambda o, objs, f=fn: not f(o, objs)))

    colors = sorted({o["color"] for o in all_objects})
    if len(colors) <= MAX_DISTINCT_VALUES:
        for c in colors:
            atoms.append((f"color=={c}", lambda o, objs, c=c: o["color"] == c))

    sizes = sorted({o["pixel_count"] for o in all_objects})
    if len(sizes) <= MAX_DISTINCT_VALUES:
        for s in sizes:
            atoms.append((f"size=={s}", lambda o, objs, s=s: o["pixel_count"] == s))

    heights = sorted({o["height"] for o in all_objects})
    if len(heights) <= MAX_DISTINCT_VALUES:
        for h in heights:
            atoms.append((f"height=={h}", lambda o, objs, h=h: o["height"] == h))

    widths = sorted({o["width"] for o in all_objects})
    if len(widths) <= MAX_DISTINCT_VALUES:
        for w in widths:
            atoms.append((f"width=={w}", lambda o, objs, w=w: o["width"] == w))

    # Relative-extreme atoms: computed against the object's own grid, so
    # "largest" means largest of what's present in that pair.
    add_bool("is_largest",
             lambda o, objs: o["pixel_count"] == max(x["pixel_count"] for x in objs))
    add_bool("is_smallest",
             lambda o, objs: o["pixel_count"] == min(x["pixel_count"] for x in objs))
    add_bool("is_tallest",
             lambda o, objs: o["height"] == max(x["height"] for x in objs))
    add_bool("is_widest",
             lambda o, objs: o["width"] == max(x["width"] for x in objs))

    # How many objects share this object's color / shape / size.
    add_bool("color_is_unique",
             lambda o, objs: sum(1 for x in objs if x["color"] == o["color"]) == 1)
    add_bool("shape_is_unique",
             lambda o, objs: sum(1 for x in objs
                                 if x["fingerprint"] == o["fingerprint"]) == 1)
    add_bool("size_is_unique",
             lambda o, objs: sum(1 for x in objs
                                 if x["pixel_count"] == o["pixel_count"]) == 1)

    for flag in ("touches_border", "touches_top", "touches_bottom",
                 "touches_left", "touches_right", "is_rectangle",
                 "is_single_pixel", "is_h_line", "is_v_line",
                 "is_sym_h", "is_sym_v", "is_sym_d"):
        add_bool(flag, lambda o, objs, f=flag: bool(o[f]))

    add_bool("is_square", lambda o, objs: o["height"] == o["width"])

    hole_counts = sorted({hole_count(o) for o in all_objects})
    if len(hole_counts) <= MAX_DISTINCT_VALUES:
        for k in hole_counts:
            atoms.append((f"holes=={k}", lambda o, objs, k=k: hole_count(o) == k))
    add_bool("has_holes", lambda o, objs: hole_count(o) > 0)

    # Anchor-relative atoms, from the cached annotations. A feature is
    # only offered if it appears in the evidence AND actually varies --
    # a relation that is False for every object separates nothing and
    # would just enlarge the conjunction search for no gain.
    relational_keys = set()
    if USE_RELATIONAL_ATOMS:
        for o in all_objects:
            relational_keys.update(o.get("_rel", {}).keys())

    for key in sorted(relational_keys):
        values = {bool(o.get("_rel", {}).get(key, False)) for o in all_objects}
        if len(values) < 2:
            continue
        add_bool(f"rel:{key}",
                 lambda o, objs, k=key: bool(o.get("_rel", {}).get(k, False)))

    return atoms


class Selector:
    """A conjunction of atoms -- the solved 'which objects?' hole."""

    def __init__(self, atoms):
        self.atoms = atoms  # list of (name, fn)

    def matches(self, obj, objs):
        return all(fn(obj, objs) for _, fn in self.atoms)

    def __repr__(self):
        return " AND ".join(name for name, _ in self.atoms)


def induce_selectors(observed):
    """
    Solve the 'which objects?' hole as a concept-learning problem.

    Objects the operator acted on are positive examples, objects it left
    alone are negatives, and we look for a predicate covering every
    positive and no negative ACROSS EVERY EVIDENCE PAIR at once. More
    evidence means strictly fewer selectors survive -- that is what
    makes adding a counterexample productive rather than merely a
    rejection.

    Shared by every operator: Recolor, Remove and Move all have this
    same hole, and none of them re-implements it.

    Ranked simplest-first (single atom before conjunction) so the least
    contrived concept that fits the evidence wins.
    """
    all_objects = [o for objs, _, _, _, _ in observed for o in objs]
    atoms = build_atom_catalog(all_objects)

    def covers_exactly(selector):
        for objs, positives, negatives, _, _ in observed:
            for obj, _ in positives:
                if not selector.matches(obj, objs):
                    return False
            for obj in negatives:
                if selector.matches(obj, objs):
                    return False
        return True

    good = [s for s in (Selector([a]) for a in atoms) if covers_exactly(s)]

    if not good and MAX_CONJUNCTION >= 2:
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                cand = Selector([atoms[i], atoms[j]])
                if covers_exactly(cand):
                    good.append(cand)

    return good


# ============================================================
# THE LANGUAGE: value expressions for each operator's second hole
#
# Every one of these is DERIVED from the observed change -- a lookup
# table or a form validated against evidence, never a swept guess.
# ============================================================

class ColorExpr:
    """Recolor's second hole: what color does a selected object become?"""

    def __init__(self, kind, table=None, value=None):
        self.kind  = kind
        self.table = table
        self.value = value

    def evaluate(self, obj, objs, grid, background):
        if self.kind == "literal":
            return self.value
        if self.kind == "from_color":
            return self.table.get(obj["color"])
        if self.kind == "from_size":
            return self.table.get(obj["pixel_count"])
        if self.kind == "from_shape_count":
            k = sum(1 for x in objs if x["fingerprint"] == obj["fingerprint"])
            return self.table.get(k)
        return None

    def __repr__(self):
        if self.kind == "literal":
            return f"color {self.value}"
        return f"{self.kind}{self.table}"


def induce_color_exprs(observations):
    """
    Solve the color hole from observed (obj, objs, grid, bg, new_color).

    Each candidate is a lookup table built directly from the evidence,
    returned only if it is actually a function -- if one key needs two
    different colors anywhere in the evidence, that form cannot explain
    the task and is dropped immediately.

    Returns (expr, cost) pairs. Cost drives ranking in Operator.induce:
    a table keyed on a raw measurement generalizes badly (an unseen size
    on the test grid simply isn't in the table), so tables are scored
    well above literals and grow more expensive the more entries they
    memorize.
    """
    candidates = []

    new_colors = {payload for _, _, _, _, payload in observations}
    if len(new_colors) == 1:
        candidates.append((ColorExpr("literal", value=next(iter(new_colors))), 0))

    def build_table(key_fn, kind):
        table = {}
        for obj, objs, _, _, payload in observations:
            key = key_fn(obj, objs)
            if key in table and table[key] != payload:
                return None
            table[key] = payload
        return ColorExpr(kind, table=table)

    for key_fn, kind in (
        (lambda o, objs: o["color"], "from_color"),
        (lambda o, objs: hole_count(o), "from_hole_count"),
        (lambda o, objs: sum(1 for x in objs if x["fingerprint"] == o["fingerprint"]),
         "from_shape_count"),
        (lambda o, objs: o["pixel_count"], "from_size"),
    ):
        expr = build_table(key_fn, kind)
        if expr is not None:
            candidates.append((expr, 100 + len(expr.table)))

    return candidates


class DecisionListExpr:
    """
    A list of (selector -> value) clauses, first match wins.

    This is the structural gap the overfit cases exposed. Tasks
    b230c067 and d2abd087 both have rules of the form "objects matching
    P become one value, everything else becomes another" -- and the
    predicates needed (shape_is_unique, size==6) were ALREADY in the
    catalog. What was missing was any way to say "different objects get
    different treatment", so the engine fell back on memorizing a size
    table and duly failed on the test grid. A clause list says it
    directly, and generalizes because each clause is a predicate rather
    than a remembered measurement.
    """

    def __init__(self, clauses):
        self.clauses = clauses

    def evaluate(self, obj, objs, grid, background):
        for selector, value in self.clauses:
            if selector.matches(obj, objs):
                return value
        return None

    def __repr__(self):
        return " | ".join(f"{sel} -> {val}" for sel, val in self.clauses)


def induce_decision_list(observed):
    """
    Solve a per-value selector: for each distinct observed value, find a
    predicate separating the objects that take that value from the
    objects that take any other value.

    This reuses induce_selectors unchanged -- it just relabels the
    evidence once per value. Objects the outer selector already excludes
    are irrelevant here, so only positives take part.
    """
    values = sorted({payload for _, positives, _, _, _ in observed
                     for _, payload in positives},
                    key=repr)
    if len(values) < 2:
        return None

    clauses = []
    for value in values:
        relabeled = []
        for objs, positives, negatives, background, grid in observed:
            same  = [(o, p) for o, p in positives if p == value]
            other = [o for o, p in positives if p != value]
            relabeled.append((objs, same, other, background, grid))

        selectors = induce_selectors(relabeled)
        if not selectors:
            return None
        clauses.append((selectors[0], value))

    # A clause list whose VALUES are geometric (pixel offsets) only
    # transfers if its KEYS are symbolic. Measured on the training set:
    # every Move decision list keyed on colour predicted the unseen test
    # correctly (342dd610, f45f5ca7), and every one keyed on a metric
    # property -- size, height, border contact -- failed, because rules
    # like "size==1 -> (-1,0) | size==2 -> (-2,0) | size==3 -> (-3,0)"
    # are memorizing an underlying relationship one object at a time,
    # which is the same failure mode decision lists were added to fix.
    # Colours come from a fixed palette and mean the same thing in the
    # test grid; a size or a row index does not.
    if all(isinstance(value, tuple) for _, value in clauses):
        for selector, _ in clauses:
            if not all(name.startswith("color==") for name, _ in selector.atoms):
                return None

    return DecisionListExpr(clauses)


class VectorExpr:
    """Move's second hole: by what offset does a selected object move?"""

    def __init__(self, kind, value=None, direction=None):
        self.kind      = kind
        self.value     = value
        self.direction = direction

    def evaluate(self, obj, objs, grid, background):
        if self.kind == "literal":
            return self.value
        if self.kind == "to_edge":
            return _delta_to_edge(obj, grid.shape, self.direction)
        if self.kind == "until_collision":
            return _delta_until_collision(obj, objs, grid.shape, self.direction,
                                          background)
        return None

    def __repr__(self):
        if self.kind == "literal":
            return f"offset {self.value}"
        return f"{self.kind}({self.direction})"


def _delta_to_edge(obj, grid_shape, direction):
    rows, cols = grid_shape
    if direction == "up":
        return (-obj["min_row"], 0)
    if direction == "down":
        return ((rows - 1) - obj["max_row"], 0)
    if direction == "left":
        return (0, -obj["min_col"])
    if direction == "right":
        return (0, (cols - 1) - obj["max_col"])
    return None


def _delta_until_collision(obj, objs, grid_shape, direction, background):
    """
    Slide until blocked by the grid edge or by another object's ORIGINAL
    position. Collision is measured against the starting scene, not a
    partially-settled one -- with no ordering rule, resolving objects
    against each other's final positions would make the result depend on
    the order objects happen to be listed in.
    """
    rows, cols = grid_shape
    dr, dc = DIRECTIONS[direction]
    own = set(obj["pixels"])
    blocked = {p for other in objs if other["id"] != obj["id"]
               for p in other["pixels"]}

    shift = 0
    while True:
        nxt = {(r + dr * (shift + 1), c + dc * (shift + 1)) for r, c in own}
        if any(r < 0 or r >= rows or c < 0 or c >= cols for r, c in nxt):
            break
        if nxt & blocked:
            break
        shift += 1
    return (dr * shift, dc * shift)


def induce_vector_exprs(observations):
    """
    Solve the movement hole. The literal offset is read straight off the
    evidence; the to_edge / until_collision forms are checked against
    every observed move and kept only if they reproduce all of them
    exactly. Either way the answer comes from the data -- nothing is
    proposed that the observations have not already confirmed.
    """
    candidates = []

    deltas = {payload for _, _, _, _, payload in observations}
    if len(deltas) == 1:
        candidates.append((VectorExpr("literal", value=next(iter(deltas))), 0))

    for kind in ("to_edge", "until_collision"):
        for direction in DIRECTIONS:
            expr = VectorExpr(kind, direction=direction)
            if all(expr.evaluate(obj, objs, grid, bg) == payload
                   for obj, objs, grid, bg, payload in observations):
                candidates.append((expr, 1))

    return candidates


# ============================================================
# PROGRAMS
# ============================================================

class RecolorProgram:
    operator_name = "Recolor"

    def __init__(self, selector, expr):
        self.selector = selector
        self.expr     = expr

    def apply(self, grid, background):
        objs = scene_objects(grid, background)
        result = grid.copy()
        for obj in objs:
            if not self.selector.matches(obj, objs):
                continue
            new_color = self.expr.evaluate(obj, objs, grid, background)
            if new_color is None:
                return None
            for (r, c) in obj["pixels"]:
                result[r, c] = new_color
        return result

    def __repr__(self):
        return f"Recolor(where {self.selector}, to {self.expr})"


class RemoveProgram:
    operator_name = "Remove"

    def __init__(self, selector, expr=None):
        self.selector = selector
        self.expr     = None   # Remove has no second hole

    def apply(self, grid, background):
        objs = scene_objects(grid, background)
        result = grid.copy()
        for obj in objs:
            if self.selector.matches(obj, objs):
                for (r, c) in obj["pixels"]:
                    result[r, c] = background
        return result

    def __repr__(self):
        return f"Remove(where {self.selector})"


class MoveProgram:
    operator_name = "Move"

    def __init__(self, selector, expr):
        self.selector = selector
        self.expr     = expr

    def apply(self, grid, background):
        objs = scene_objects(grid, background)
        rows, cols = grid.shape
        result = grid.copy()

        selected = [o for o in objs if self.selector.matches(o, objs)]
        for obj in selected:
            for (r, c) in obj["pixels"]:
                result[r, c] = background

        for obj in selected:
            delta = self.expr.evaluate(obj, objs, grid, background)
            if delta is None:
                return None
            dr, dc = delta
            for (r, c) in obj["pixels"]:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    return None
                result[nr, nc] = obj["color"]
        return result

    def __repr__(self):
        return f"Move(where {self.selector}, by {self.expr})"


# ============================================================
# OPERATORS: observe a pair, then solve that operator's holes
#
# Each observe_* returns (objects, positives, negatives, background,
# grid) or None when the pair is structurally outside what the operator
# can express -- there is no point inducing from a pair the operator
# could never explain.
# ============================================================

def scene_objects(grid, background):
    """
    Extract, encode and relationally annotate a grid's objects.

    Every place that builds a scene goes through here -- both induction
    and program application -- so an anchor-relative atom means the same
    thing when a rule is applied to the test grid as it did when the
    rule was induced.
    """
    objs = encode_all_objects(extract_objects(grid, background), grid.shape)
    return annotate_relational(objs, grid.shape)


def _grids_comparable(input_grid, output_grid):
    if input_grid.shape != output_grid.shape:
        return None
    background = get_background_color(input_grid)
    if get_background_color(output_grid) != background:
        return None
    return background


def observe_recolor(input_grid, output_grid):
    """
    Objects keep their exact pixels; some change color.

    Position matching (rather than shape-fingerprint matching) makes
    this exact: an object that kept its pixels but changed color is
    unambiguously "the same object, recolored", with no scoring
    heuristic in between.
    """
    background = _grids_comparable(input_grid, output_grid)
    if background is None:
        return None

    in_objs  = scene_objects(input_grid, background)
    out_objs = encode_all_objects(extract_objects(output_grid, background),
                                  output_grid.shape)
    if len(in_objs) != len(out_objs):
        return None

    out_by_pixels = {frozenset(o["pixels"]): o for o in out_objs}
    positives, negatives = [], []
    for obj in in_objs:
        match = out_by_pixels.get(frozenset(obj["pixels"]))
        if match is None:
            return None
        if match["color"] != obj["color"]:
            positives.append((obj, match["color"]))
        else:
            negatives.append(obj)

    return in_objs, positives, negatives, background, input_grid


def observe_remove(input_grid, output_grid):
    """
    Some objects vanish entirely; the rest are untouched. Nothing may
    appear or change, or this is not a Remove.
    """
    background = _grids_comparable(input_grid, output_grid)
    if background is None:
        return None

    in_objs = scene_objects(input_grid, background)
    out_objs = encode_all_objects(extract_objects(output_grid, background),
                                  output_grid.shape)

    in_by_pixels  = {frozenset(o["pixels"]): o for o in in_objs}
    out_by_pixels = {frozenset(o["pixels"]): o for o in out_objs}

    # Anything in the output that wasn't already in the input, unchanged,
    # means this pair involves more than removal.
    for key, obj in out_by_pixels.items():
        original = in_by_pixels.get(key)
        if original is None or original["color"] != obj["color"]:
            return None

    positives, negatives = [], []
    for obj in in_objs:
        key = frozenset(obj["pixels"])
        if key in out_by_pixels:
            negatives.append(obj)
        elif all(output_grid[r, c] == background for (r, c) in obj["pixels"]):
            positives.append((obj, None))
        else:
            return None

    return in_objs, positives, negatives, background, input_grid


def observe_move(input_grid, output_grid):
    """
    Objects keep their colour and shape but change position.

    Input and output objects are grouped by (colour, shape fingerprint)
    and matched greedily nearest-first inside each group, so identical
    twins are paired by proximity rather than by list order.
    """
    background = _grids_comparable(input_grid, output_grid)
    if background is None:
        return None

    in_objs  = scene_objects(input_grid, background)
    out_objs = encode_all_objects(extract_objects(output_grid, background),
                                  output_grid.shape)
    if len(in_objs) != len(out_objs) or not in_objs:
        return None

    def group_key(o):
        return (o["color"], o["fingerprint"])

    groups_in, groups_out = {}, {}
    for o in in_objs:
        groups_in.setdefault(group_key(o), []).append(o)
    for o in out_objs:
        groups_out.setdefault(group_key(o), []).append(o)

    if set(groups_in) != set(groups_out):
        return None

    positives, negatives = [], []
    for key, ins in groups_in.items():
        outs = list(groups_out[key])
        if len(ins) != len(outs):
            return None
        for obj in sorted(ins, key=lambda o: (o["min_row"], o["min_col"])):
            best = min(
                outs,
                key=lambda t: (abs(t["min_row"] - obj["min_row"])
                               + abs(t["min_col"] - obj["min_col"])),
            )
            outs.remove(best)
            delta = (best["min_row"] - obj["min_row"],
                     best["min_col"] - obj["min_col"])
            if delta == (0, 0):
                negatives.append(obj)
            else:
                positives.append((obj, delta))

    return in_objs, positives, negatives, background, input_grid


class Operator:
    """Bundles an operator's observer, hole-solvers and program type."""

    def __init__(self, name, observe, program_cls, induce_expr=None):
        self.name        = name
        self.observe     = observe
        self.program_cls = program_cls
        self.induce_expr = induce_expr

    def induce(self, evidence):
        """
        Solve every hole from the evidence and return consistent
        programs, cheapest first.

        Cost ranks how much the rule MEMORIZES versus how much it
        GENERALIZES: a literal value costs nothing, a decision list of
        predicates costs a little, and a lookup table keyed on a raw
        measurement costs a lot, because an unseen key on the test grid
        leaves it with no answer at all. Selector induction is the
        shared step; induce_expr is the operator's own second hole
        (Remove has none).
        """
        observed = []
        for input_grid, output_grid in evidence:
            obs = self.observe(input_grid, output_grid)
            if obs is None:
                return []
            observed.append(obs)

        if not any(positives for _, positives, _, _, _ in observed):
            return []

        selectors = induce_selectors(observed)
        if not selectors:
            return []

        def selector_cost(sel):
            return len(sel.atoms)

        if self.induce_expr is None:
            ranked = sorted(selectors, key=selector_cost)
            return [self.program_cls(sel) for sel in ranked]

        observations = [
            (obj, objs, grid, background, payload)
            for objs, positives, _, background, grid in observed
            for obj, payload in positives
        ]

        scored = []
        for expr, expr_cost in self.induce_expr(observations):
            for sel in selectors:
                scored.append((expr_cost + selector_cost(sel),
                               self.program_cls(sel, expr)))

        decision_list = induce_decision_list(observed)
        if decision_list is not None:
            cost = 10 + sum(len(sel.atoms) for sel, _ in decision_list.clauses)
            for sel in selectors:
                scored.append((cost + selector_cost(sel),
                               self.program_cls(sel, decision_list)))

        scored.sort(key=lambda pair: pair[0])
        return [program for _, program in scored]

    def diagnose(self, program, input_grid, output_grid):
        """
        Explain WHY a program failed, structurally rather than as a pixel
        count -- "it acted on the wrong objects" and "right objects,
        wrong value" are different failures needing different repairs.
        """
        obs = self.observe(input_grid, output_grid)
        if obs is None:
            return f"pair is not a pure {self.name.lower()}"

        objs, positives, negatives, background, grid = obs
        should = {obj["id"] for obj, _ in positives}
        would  = {obj["id"] for obj in objs if program.selector.matches(obj, objs)}

        if would != should:
            return (f"selector wrong: missed {sorted(should - would)}, "
                    f"wrongly included {sorted(would - should)}")

        if program.expr is None:
            return "selector matches but the result still differs"

        wrong = []
        for obj, expected in positives:
            got = program.expr.evaluate(obj, objs, grid, background)
            if got != expected:
                wrong.append(f"obj{obj['id']}: got {got}, expected {expected}")
        if wrong:
            return f"selector right, {self.name.lower()} value wrong: " + "; ".join(wrong)
        return "no structural difference found"


RECOLOR = Operator("Recolor", observe_recolor, RecolorProgram, induce_color_exprs)
REMOVE  = Operator("Remove",  observe_remove,  RemoveProgram)
MOVE    = Operator("Move",    observe_move,    MoveProgram,    induce_vector_exprs)

OPERATORS = [RECOLOR, REMOVE, MOVE]


# ============================================================
# THE LOOP: counterexample-guided refinement
#
# Unchanged from the single-operator proof of concept -- it now takes
# the operator as a parameter, but the control flow is identical.
# ============================================================

def synthesize(task, operator, verbose=False):
    """
    Induce a rule from training pair 1, then walk the remaining pairs.
    Every pair that disproves the current rule is ADDED to the evidence
    set and the rule is re-induced under that stronger constraint.

    Returns (program, trace) with program=None if no rule in the
    language explains every training pair.
    """
    pairs = [(p["input"], p["output"]) for p in task["train"]]
    trace = []

    evidence = [pairs[0]]
    programs = operator.induce(evidence)
    if not programs:
        trace.append(f"pair 1: could not induce any {operator.name} rule")
        return None, trace

    program = programs[0]
    trace.append(f"pair 1: induced {program}"
                 f"   ({len(programs)} candidates consistent with 1 pair)")

    for i in range(1, len(pairs)):
        input_grid, output_grid = pairs[i]
        predicted = program.apply(input_grid, get_background_color(input_grid))

        if predicted is not None and np.array_equal(predicted, output_grid):
            trace.append(f"pair {i+1}: rule holds, confidence up")
            continue

        trace.append(f"pair {i+1}: FAILED -- "
                     f"{operator.diagnose(program, input_grid, output_grid)}")

        evidence.append(pairs[i])
        programs = operator.induce(evidence)
        if not programs:
            trace.append(f"pair {i+1}: no {operator.name} rule fits "
                         f"{len(evidence)} pairs -- giving up")
            return None, trace

        program = programs[0]
        trace.append(f"pair {i+1}: re-induced from {len(evidence)} pairs "
                     f"-> {program}   ({len(programs)} candidates remain)")

    # Final check: the refined rule must explain every pair, including
    # ones that passed before refinement changed the rule underneath them.
    for i, (input_grid, output_grid) in enumerate(pairs):
        predicted = program.apply(input_grid, get_background_color(input_grid))
        if predicted is None or not np.array_equal(predicted, output_grid):
            trace.append(f"final check failed on pair {i+1}")
            return None, trace

    trace.append(f"FINAL RULE: {program}  (explains all {len(pairs)} pairs)")
    return program, trace


def solve_with_synthesis(task, operators=None, verbose=False):
    """
    Try each operator's induction loop in turn; first rule that explains
    every training pair wins. Returns (prediction, program, operator_name).
    """
    for operator in (operators or OPERATORS):
        try:
            program, trace = synthesize(task, operator)
        except Exception:
            continue
        if verbose:
            for line in trace:
                print(f"   [{operator.name}] {line}")
        if program is None:
            continue
        test_input = task["test"][0]["input"]
        prediction = program.apply(test_input, get_background_color(test_input))
        if prediction is not None:
            return prediction, program, operator.name
    return None, None, None


def compatible(task, operator):
    """True if this operator can structurally observe every training pair."""
    try:
        return all(
            operator.observe(p["input"], p["output"]) is not None
            for p in task["train"]
        )
    except Exception:
        return False
