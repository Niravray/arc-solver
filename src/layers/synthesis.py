import sys
import time

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


# Atoms that pin a specific measurement rather than describe structure.
# "size==4" is true of this grid's objects; "is_largest" is true of the
# same object in any grid.
METRIC_ATOM_PREFIXES = ("color==", "size==", "height==", "width==", "holes==")


def selector_cost(selector):
    """
    Rank selectors by how much they MEMORISE.

    Fewer atoms first, then fewer measurement-pinned atoms. Task
    72ca375d is the case that motivated this: three selectors explained
    every training pair identically --

        height==2 AND is_sym_v            -> wrong on the test grid
        color_is_unique AND is_sym_v      -> right
        not_is_single_pixel AND is_sym_v  -> right

    -- and with equal atom counts the tie fell to catalog order, which
    picked the metric one. "The symmetric object that is 2 rows tall"
    happens to be true of the training grids; "the symmetric object that
    isn't a single pixel" is the actual rule. Same principle already
    applied to Move's decision lists, now applied to selection, so every
    operator inherits it rather than just one.
    """
    metric = sum(1 for name, _ in selector.atoms
                 if name.startswith(METRIC_ATOM_PREFIXES))
    return (len(selector.atoms), metric)


def induce_selectors(observed, max_conjunction=None, deadline=None):
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

    max_conjunction controls how many atoms may be ANDed together and is
    what a search budget actually buys: widening from 2 to 3 turns a
    ~3k-candidate search into a ~90k one, so it is only worth paying for
    when the cheaper depth has already come up empty. deadline (a
    time.monotonic() value) abandons the widened search rather than
    letting one task run away with the budget.
    """
    if max_conjunction is None:
        max_conjunction = MAX_CONJUNCTION

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

    if not good and max_conjunction >= 2:
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                cand = Selector([atoms[i], atoms[j]])
                if covers_exactly(cand):
                    good.append(cand)

    if not good and max_conjunction >= 3:
        # Only reached when nothing simpler fit, and only under a budget.
        # The deadline is checked on the outer loop rather than per
        # candidate: the check itself is not free and one pass of the
        # inner loops is short.
        for i in range(len(atoms)):
            if deadline is not None and time.monotonic() > deadline:
                break
            for j in range(i + 1, len(atoms)):
                for k in range(j + 1, len(atoms)):
                    cand = Selector([atoms[i], atoms[j], atoms[k]])
                    if covers_exactly(cand):
                        good.append(cand)

    good.sort(key=selector_cost)
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

    # Decided up front, because it changes which selector to pick rather
    # than only whether to accept the result afterwards. Geometric values
    # need colour-keyed clauses (see below), and induce_selectors now
    # ranks colour== atoms LAST as measurement-pinned -- so taking its
    # first selector and checking afterwards would reject every geometric
    # list that a colour key could have explained.
    values_are_geometric = all(isinstance(value, tuple) for value in values)

    clauses = []
    for value in values:
        relabeled = []
        for objs, positives, negatives, background, grid in observed:
            same  = [(o, p) for o, p in positives if p == value]
            other = [o for o, p in positives if p != value]
            relabeled.append((objs, same, other, background, grid))

        selectors = induce_selectors(relabeled)
        if values_are_geometric:
            selectors = [s for s in selectors
                         if all(name.startswith("color==")
                                for name, _ in s.atoms)]
        if not selectors:
            return None
        clauses.append((selectors[0], value))

    # A clause list whose VALUES are geometric (pixel offsets) only
    # transfers if its KEYS are symbolic, which is why the colour-only
    # filter is applied while choosing selectors above. Measured on the
    # training set: every Move decision list keyed on colour predicted
    # the unseen test correctly (342dd610, f45f5ca7), and every one keyed
    # on a metric property -- size, height, border contact -- failed,
    # because rules like "size==1 -> (-1,0) | size==2 -> (-2,0)" memorize
    # an underlying relationship one object at a time, the same failure
    # mode decision lists were added to fix. Colours come from a fixed
    # palette and mean the same thing in the test grid; a size or a row
    # index does not.
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
    """
    Shared shape gate for the same-shape operators, returning the
    background to read both grids with.

    The INPUT's background is used for both rather than requiring the two
    to agree. get_background_color returns the most frequent colour, so
    recolouring a large region flips it even when the canvas itself never
    changed -- and demanding agreement rejected 109 training and 11
    evaluation tasks on that technicality alone, before any operator got
    to look at them. The background is a property of the scene the rule
    acts on, so the input's is the one that defines it.
    """
    if input_grid.shape != output_grid.shape:
        return None
    return get_background_color(input_grid)


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


def _match_survivors(in_objs, out_objs, output_grid, background,
                     ignore_color=False):
    """
    Decide, for each input object, whether it survived into the output in
    any form -- same place, or moved elsewhere.

    Shared by the tolerant observers below. The distinction that matters
    for composition is "gone" versus "still there but changed": an object
    that moved leaves its original cells empty and would otherwise look
    exactly like a removal.
    """
    out_by_pixels = {frozenset(o["pixels"]): o for o in out_objs}
    by_shape = {}
    for o in out_objs:
        key = o["fingerprint"] if ignore_color else (o["color"], o["fingerprint"])
        by_shape.setdefault(key, []).append(o)

    claimed = set()
    verdicts = []
    for obj in in_objs:
        same_place = out_by_pixels.get(frozenset(obj["pixels"]))
        if same_place is not None and id(same_place) not in claimed:
            claimed.add(id(same_place))
            verdicts.append((obj, "in_place", same_place))
            continue

        key = obj["fingerprint"] if ignore_color else (obj["color"], obj["fingerprint"])
        twins = [o for o in by_shape.get(key, []) if id(o) not in claimed]
        if twins:
            best = min(twins, key=lambda t: (abs(t["min_row"] - obj["min_row"])
                                             + abs(t["min_col"] - obj["min_col"])))
            claimed.add(id(best))
            verdicts.append((obj, "moved", best))
            continue

        if all(output_grid[r, c] == background for (r, c) in obj["pixels"]):
            verdicts.append((obj, "gone", None))
        else:
            verdicts.append((obj, "altered", None))

    return verdicts


def observe_remove_tolerant(input_grid, output_grid):
    """
    Label ONLY the removals, tolerating other changes in the same pair.

    The strict observers reject a pair the moment anything happens that
    their operator cannot explain, which is right for a single-operator
    rule but makes every mixed task invisible. A composed rule needs its
    first stage to explain its own part and leave the rest to the second
    stage, so this labels objects that genuinely vanished as positives
    and every survivor -- including ones that moved or were recoloured --
    as a negative that Remove must not touch.
    """
    background = _grids_comparable(input_grid, output_grid)
    if background is None:
        return None

    in_objs = scene_objects(input_grid, background)
    out_objs = encode_all_objects(extract_objects(output_grid, background),
                                  output_grid.shape)
    if not in_objs:
        return None

    positives, negatives = [], []
    for obj, verdict, _ in _match_survivors(in_objs, out_objs, output_grid,
                                            background):
        if verdict == "gone":
            positives.append((obj, None))
        else:
            negatives.append(obj)

    return in_objs, positives, negatives, background, input_grid


def observe_move_tolerant(input_grid, output_grid):
    """
    Label ONLY the movements, tolerating recolouring in the same pair.

    Objects are matched on shape alone, ignoring colour, so an object
    that moves AND changes colour is still recognised as the same object
    moving -- the recolour is then left for the second stage.
    """
    background = _grids_comparable(input_grid, output_grid)
    if background is None:
        return None

    in_objs = scene_objects(input_grid, background)
    out_objs = encode_all_objects(extract_objects(output_grid, background),
                                  output_grid.shape)
    if not in_objs:
        return None

    positives, negatives = [], []
    for obj, verdict, match in _match_survivors(in_objs, out_objs, output_grid,
                                                background, ignore_color=True):
        if verdict == "moved":
            positives.append((obj, (match["min_row"] - obj["min_row"],
                                    match["min_col"] - obj["min_col"])))
        else:
            negatives.append(obj)

    return in_objs, positives, negatives, background, input_grid


RAY_DIRECTIONS = {
    "up":         (-1, 0),
    "down":       (1, 0),
    "left":       (0, -1),
    "right":      (0, 1),
    "up_left":    (-1, -1),
    "up_right":   (-1, 1),
    "down_left":  (1, -1),
    "down_right": (1, 1),
}

# Direction sets worth trying, kept to a bounded, meaningful list rather
# than all 255 subsets of eight directions.
DIRECTION_SETS = [
    ("up", ("up",)),
    ("down", ("down",)),
    ("left", ("left",)),
    ("right", ("right",)),
    ("vertical", ("up", "down")),
    ("horizontal", ("left", "right")),
    ("orthogonal", ("up", "down", "left", "right")),
    ("diagonal", ("up_left", "up_right", "down_left", "down_right")),
    ("all8", tuple(RAY_DIRECTIONS)),
]

MAX_DRAW_OBJECTS = 40   # ray induction is per object per direction; keep it bounded


def cast_ray(grid, start, step, background):
    """
    Cells a ray covers leaving `start`, stopping before the grid edge or
    the first non-background cell.

    Stopping BEFORE existing content matters: Draw only fills background,
    so a ray that would run into an object ends against it rather than
    through it -- which is also what makes "until it hits something" fall
    out of the same code as "until the edge".
    """
    rows, cols = grid.shape
    dr, dc = step
    r, c = start[0] + dr, start[1] + dc
    cells = []
    while 0 <= r < rows and 0 <= c < cols and grid[r, c] == background:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def object_rays(grid, obj, direction_names, background):
    """Union of the rays every pixel of an object emits."""
    cells = set()
    for name in direction_names:
        step = RAY_DIRECTIONS[name]
        for pixel in obj["pixels"]:
            cells.update(cast_ray(grid, pixel, step, background))
    return cells


def object_rays_through(grid, obj, direction_names, background):
    """
    Like object_rays but continuing PAST obstacles to the grid edge,
    filling only the background cells on the way.

    A separate form rather than a flag because the two make genuinely
    different predictions once anything is in the way, and which one a
    task means is exactly what induction has to decide.
    """
    rows, cols = grid.shape
    cells = set()
    for name in direction_names:
        dr, dc = RAY_DIRECTIONS[name]
        for (pr, pc) in obj["pixels"]:
            r, c = pr + dr, pc + dc
            while 0 <= r < rows and 0 <= c < cols:
                if grid[r, c] == background:
                    cells.add((r, c))
                r += dr
                c += dc
    return cells


def object_halo(grid, obj, background, diagonal):
    """Background cells immediately around an object (outline / growth)."""
    rows, cols = grid.shape
    offsets = ([(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0)] if diagonal
               else [(-1, 0), (1, 0), (0, -1), (0, 1)])
    cells = set()
    for (r, c) in obj["pixels"]:
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == background:
                cells.add((nr, nc))
    return cells


def object_bbox_fill(grid, obj, background):
    """Background cells inside an object's bounding box."""
    return {(r, c)
            for r in range(obj["min_row"], obj["max_row"] + 1)
            for c in range(obj["min_col"], obj["max_col"] + 1)
            if grid[r, c] == background}


def object_bbox_outline(grid, obj, background):
    """Background cells forming a frame just outside an object's bbox."""
    rows, cols = grid.shape
    r0, r1 = obj["min_row"] - 1, obj["max_row"] + 1
    c0, c1 = obj["min_col"] - 1, obj["max_col"] + 1
    cells = set()
    for r in range(r0, r1 + 1):
        for c in (c0, c1):
            if 0 <= r < rows and 0 <= c < cols and grid[r, c] == background:
                cells.add((r, c))
    for c in range(c0, c1 + 1):
        for r in (r0, r1):
            if 0 <= r < rows and 0 <= c < cols and grid[r, c] == background:
                cells.add((r, c))
    return cells


def object_full_lines(grid, obj, background, mode):
    """Every background cell in the rows and/or columns an object occupies."""
    rows, cols = grid.shape
    cells = set()
    if mode in ("row", "both"):
        for r in {r for (r, _) in obj["pixels"]}:
            cells |= {(r, c) for c in range(cols) if grid[r, c] == background}
    if mode in ("col", "both"):
        for c in {c for (_, c) in obj["pixels"]}:
            cells |= {(r, c) for r in range(rows) if grid[r, c] == background}
    return cells


def _per_object_forms():
    """
    Registry of ways a single object can generate new cells.

    Measured over the 203 pure-addition training tasks: bbox_fill covers
    11, halo8 and bbox_outline 4 each, the line and ray variants 1-3
    each. No form dominates -- the bucket is a long tail -- so the point
    of a registry is breadth, and each entry stays cheap enough that
    breadth costs little.
    """
    forms = {}
    for set_name, directions in DIRECTION_SETS:
        forms[f"ray_{set_name}"] = (
            lambda g, o, b, d=directions: object_rays(g, o, d, b))
        forms[f"raythru_{set_name}"] = (
            lambda g, o, b, d=directions: object_rays_through(g, o, d, b))
    forms["halo4"]        = lambda g, o, b: object_halo(g, o, b, False)
    forms["halo8"]        = lambda g, o, b: object_halo(g, o, b, True)
    forms["bbox_fill"]    = object_bbox_fill
    forms["bbox_outline"] = object_bbox_outline
    for mode in ("row", "col", "both"):
        forms[f"full_{mode}"] = (
            lambda g, o, b, m=mode: object_full_lines(g, o, b, m))
    return forms


PER_OBJECT_DRAW_FORMS = _per_object_forms()


def global_mirror(grid, background, objects, kind):
    """
    Symmetry completion: a background cell is filled when its mirror
    image holds content. The colour comes from that mirror cell, so this
    form answers the colour hole itself.
    """
    rows, cols = grid.shape
    out = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != background:
                continue
            if kind == "h":
                sr, sc = r, cols - 1 - c
            elif kind == "v":
                sr, sc = rows - 1 - r, c
            else:
                sr, sc = rows - 1 - r, cols - 1 - c
            if grid[sr, sc] != background:
                out[(r, c)] = int(grid[sr, sc])
    return out


def global_connect_aligned(grid, background, objects):
    """Fill the gap between objects sharing an exact row or column band."""
    out = {}
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i], objects[j]
            if a["min_row"] == b["min_row"] and a["max_row"] == b["max_row"]:
                lo, hi = sorted((a, b), key=lambda o: o["min_col"])
                for r in range(a["min_row"], a["max_row"] + 1):
                    for c in range(lo["max_col"] + 1, hi["min_col"]):
                        if grid[r, c] == background:
                            out[(r, c)] = None
            if a["min_col"] == b["min_col"] and a["max_col"] == b["max_col"]:
                lo, hi = sorted((a, b), key=lambda o: o["min_row"])
                for c in range(a["min_col"], a["max_col"] + 1):
                    for r in range(lo["max_row"] + 1, hi["min_row"]):
                        if grid[r, c] == background:
                            out[(r, c)] = None
    return out


# Global forms produce the whole added set at once rather than per
# object, so they have no selector hole. They return {cell: colour},
# where a colour of None means the form does not determine it and the
# colour hole still has to be solved.
GLOBAL_DRAW_FORMS = {
    "mirror_h":        lambda g, b, objs: global_mirror(g, b, objs, "h"),
    "mirror_v":        lambda g, b, objs: global_mirror(g, b, objs, "v"),
    "mirror_180":      lambda g, b, objs: global_mirror(g, b, objs, "both"),
    "connect_aligned": global_connect_aligned,
}


class DrawProgram:
    """Selected objects each generate cells via one form."""

    operator_name = "Draw"

    def __init__(self, selector, form_name, color_mode, color_value=None):
        self.selector    = selector
        self.form_name   = form_name
        self.color_mode  = color_mode   # "literal" | "source"
        self.color_value = color_value
        self.expr        = None         # holes are named separately here

    def cells_for(self, grid, obj, background):
        return PER_OBJECT_DRAW_FORMS[self.form_name](grid, obj, background)

    def apply(self, grid, background):
        objs = scene_objects(grid, background)
        result = grid.copy()
        for obj in objs:
            if not self.selector.matches(obj, objs):
                continue
            color = (self.color_value if self.color_mode == "literal"
                     else obj["color"])
            for (r, c) in self.cells_for(grid, obj, background):
                result[r, c] = color
        return result

    def __repr__(self):
        color = (f"colour {self.color_value}" if self.color_mode == "literal"
                 else "the source object's colour")
        return f"Draw(from {self.selector}, {self.form_name}, in {color})"


class GlobalDrawProgram:
    """
    A form that generates the whole added set at once.

    Has no selector: symmetry completion and gap-filling are properties
    of the scene rather than of any one object, so there is nothing to
    choose between.
    """

    operator_name = "Draw"

    def __init__(self, form_name, color_value=None):
        self.form_name   = form_name
        self.color_value = color_value
        self.selector    = None
        self.expr        = None

    def apply(self, grid, background):
        objs = scene_objects(grid, background)
        result = grid.copy()
        for (r, c), color in GLOBAL_DRAW_FORMS[self.form_name](
                grid, background, objs).items():
            value = color if color is not None else self.color_value
            if value is None:
                return None
            result[r, c] = value
        return result

    def __repr__(self):
        color = ("" if self.color_value is None
                 else f", in colour {self.color_value}")
        return f"Draw({self.form_name}{color})"


def observe_draw(input_grid, output_grid):
    """
    A pure addition: existing content is untouched and only background
    cells are filled.

    Unlike the other observers this cannot label positives on its own --
    "which objects emit" is not visible from the change alone, it depends
    on which direction set is being considered. So this returns the raw
    scene and the cells that appeared, and DrawOperator.induce works out
    the emitters per candidate direction set.
    """
    background = _grids_comparable(input_grid, output_grid)
    if background is None:
        return None

    existing = input_grid != background
    if not np.array_equal(input_grid[existing], output_grid[existing]):
        return None

    added = (input_grid == background) & (output_grid != background)
    if not added.any():
        return None

    objs = scene_objects(input_grid, background)
    if not objs or len(objs) > MAX_DRAW_OBJECTS:
        return None

    return objs, added, background, input_grid, output_grid


class DrawOperator:
    """
    Creates content rather than transforming existing objects.

    The other three operators pick from objects that already exist, so
    their "which objects?" hole is answered by looking at what changed.
    Nothing that appears here corresponds to an input object at all, so
    the holes are different: which objects EMIT, in which directions, and
    in what colour. Measuring the 208 pure-addition training tasks showed
    120 of them place every new cell in the row or column of an existing
    object, which is why rays are the form built first.

    Emitters are still solved with the shared induce_selectors: for a
    given direction set, an object either accounts for part of the added
    cells or it does not, and that labelling is exactly the positive /
    negative split concept learning needs.
    """

    name = "Draw"
    induce_expr = None

    def observe(self, input_grid, output_grid):
        return observe_draw(input_grid, output_grid)

    def induce(self, evidence):
        observed = []
        for input_grid, output_grid in evidence:
            obs = self.observe(input_grid, output_grid)
            if obs is None:
                return []
            observed.append(obs)

        candidates = []
        candidates.extend(self._induce_per_object(observed))
        candidates.extend(self._induce_global(observed))
        candidates.sort(key=lambda pair: pair[0])
        return [program for _, program in candidates]

    def _induce_per_object(self, observed):
        candidates = []
        for form_name, form in PER_OBJECT_DRAW_FORMS.items():
            labelled = []
            usable = True

            for objs, added, background, grid, output_grid in observed:
                target = set(zip(*np.where(added)))
                positives, negatives = [], []
                covered = set()

                for obj in objs:
                    cells = form(grid, obj, background)
                    # An emitter's cells must land entirely inside the
                    # cells that actually appeared; anything straying onto
                    # an untouched cell disproves this object as a source.
                    if cells and cells <= target:
                        positives.append((obj, None))
                        covered |= cells
                    else:
                        negatives.append(obj)

                if not positives or covered != target:
                    usable = False
                    break
                labelled.append((objs, positives, negatives, background, grid))

            if not usable:
                continue

            selectors = induce_selectors(labelled)
            if not selectors:
                continue

            new_colors = set()
            source_ok = True
            for objs, added, background, grid, output_grid in observed:
                for (r, c) in zip(*np.where(added)):
                    new_colors.add(int(output_grid[r, c]))
                for obj in objs:
                    cells = form(grid, obj, background)
                    if cells and cells <= set(zip(*np.where(added))):
                        if any(output_grid[r, c] != obj["color"] for r, c in cells):
                            source_ok = False

            color_options = []
            if source_ok:
                color_options.append(("source", None))
            if len(new_colors) == 1:
                color_options.append(("literal", next(iter(new_colors))))

            for selector in selectors:
                for mode, value in color_options:
                    candidates.append(
                        (len(selector.atoms),
                         DrawProgram(selector, form_name, mode, value))
                    )
        return candidates

    def _induce_global(self, observed):
        candidates = []
        for form_name, form in GLOBAL_DRAW_FORMS.items():
            produced_colors = set()
            needs_color = False
            usable = True

            for objs, added, background, grid, output_grid in observed:
                target = set(zip(*np.where(added)))
                cells = form(grid, background, objs)
                if set(cells) != target:
                    usable = False
                    break
                for (r, c), color in cells.items():
                    if color is None:
                        needs_color = True
                        produced_colors.add(int(output_grid[r, c]))
                    elif color != output_grid[r, c]:
                        usable = False
                        break
                if not usable:
                    break

            if not usable:
                continue
            if needs_color and len(produced_colors) != 1:
                continue

            value = next(iter(produced_colors)) if needs_color else None
            # Global forms have no selector to justify, so they rank
            # ahead of per-object ones -- fewer holes solved from the
            # same evidence means less room to have fit it by accident.
            candidates.append((0, GlobalDrawProgram(form_name, value)))
        return candidates

    def diagnose(self, program, input_grid, output_grid):
        obs = self.observe(input_grid, output_grid)
        if obs is None:
            return "pair is not a pure addition"
        objs, added, background, grid, _ = obs
        target = set(zip(*np.where(added)))

        if isinstance(program, GlobalDrawProgram):
            drawn = set(GLOBAL_DRAW_FORMS[program.form_name](
                grid, background, objs))
        else:
            drawn = set()
            for obj in objs:
                if program.selector.matches(obj, objs):
                    drawn |= program.cells_for(grid, obj, background)

        if drawn == target:
            return "cells match; the colour must be wrong"
        return (f"cells wrong: {len(target - drawn)} expected cells never drawn, "
                f"{len(drawn - target)} drawn that should not be")


MAX_EXTRACT_OBJECTS = 80

# Two readings of "output the object": the raw window of the input at the
# object's bounding box, which carries anything else overlapping that
# box, versus the object alone on background. Induction picks whichever
# reproduces the training outputs rather than us deciding up front.
EXTRACT_FORMS = {
    "bbox_crop": lambda grid, obj: grid[obj["min_row"]:obj["max_row"] + 1,
                                        obj["min_col"]:obj["max_col"] + 1].copy(),
    "object_only": lambda grid, obj: obj["grid"].copy(),
}


class ExtractProgram:
    """
    The output IS one of the input's objects.

    First operator whose output has a different shape from its input, so
    it cannot go through _grids_comparable like the others.
    """

    operator_name = "Extract"

    def __init__(self, selector, form_name):
        self.selector  = selector
        self.form_name = form_name
        self.expr      = None

    def apply(self, grid, background):
        objs = scene_objects(grid, background)
        chosen = [o for o in objs if self.selector.matches(o, objs)]
        if not chosen:
            return None

        form = EXTRACT_FORMS[self.form_name]
        candidates = [form(grid, o) for o in chosen]
        # Several objects matching is only safe when they would all
        # produce the SAME grid -- identical twins, where any of them is
        # the right answer. If they disagree the selector has not
        # actually identified which object to output, and guessing one
        # would be answering a question the rule never settled.
        first = candidates[0]
        for other in candidates[1:]:
            if other.shape != first.shape or not np.array_equal(other, first):
                return None
        return first

    def __repr__(self):
        return f"Extract({self.selector}, as {self.form_name})"


def observe_extract(input_grid, output_grid):
    """
    Label which input objects reproduce the output when cropped.

    Deliberately does NOT call _grids_comparable: that gate returns None
    the moment the shapes differ, which is the whole point of this
    operator and the reason 316 unsolved training tasks had never been
    looked at by induction at all.
    """
    if input_grid.shape == output_grid.shape:
        return None   # same-shape changes belong to the other operators

    background = get_background_color(input_grid)
    objs = scene_objects(input_grid, background)
    if not objs or len(objs) > MAX_EXTRACT_OBJECTS:
        return None

    positives, negatives = [], []
    for obj in objs:
        produced = None
        for form_name, form in EXTRACT_FORMS.items():
            candidate = form(input_grid, obj)
            if (candidate.shape == output_grid.shape
                    and np.array_equal(candidate, output_grid)):
                produced = form_name
                break
        if produced:
            positives.append((obj, produced))
        else:
            negatives.append(obj)

    if not positives:
        return None
    return objs, positives, negatives, background, input_grid


class ExtractOperator:
    """
    Solves one hole: WHICH object becomes the output.

    There is no value hole -- the answer is the object itself -- so this
    is the purest reuse of induce_selectors in the engine. The form
    (bbox crop vs object alone) is fixed per candidate and checked
    across every pair, the same way DrawOperator handles its forms.
    """

    name = "Extract"
    induce_expr = None

    def observe(self, input_grid, output_grid):
        return observe_extract(input_grid, output_grid)

    def induce(self, evidence):
        observed = []
        for input_grid, output_grid in evidence:
            obs = self.observe(input_grid, output_grid)
            if obs is None:
                return []
            observed.append(obs)

        candidates = []
        for form_name in EXTRACT_FORMS:
            labelled = []
            usable = True
            for objs, positives, negatives, background, grid in observed:
                # Restrict to objects this particular form explains, so a
                # task solved by bbox_crop is not credited to object_only.
                same_form = [(o, f) for o, f in positives if f == form_name]
                if not same_form:
                    usable = False
                    break
                others = ([o for o, f in positives if f != form_name]
                          + list(negatives))
                labelled.append((objs, same_form, others, background, grid))

            if not usable:
                continue
            for selector in induce_selectors(labelled):
                candidates.append((len(selector.atoms),
                                   ExtractProgram(selector, form_name)))

        candidates.sort(key=lambda pair: pair[0])
        return [program for _, program in candidates]

    def diagnose(self, program, input_grid, output_grid):
        obs = self.observe(input_grid, output_grid)
        if obs is None:
            return "no input object reproduces this output when cropped"
        objs, positives, _, _, _ = obs
        should = {o["id"] for o, _ in positives}
        picked = {o["id"] for o in objs if program.selector.matches(o, objs)}
        if picked == should:
            return "right object picked; the crop form must be wrong"
        return (f"selector wrong: should pick {sorted(should)}, "
                f"picked {sorted(picked)}")


class ComposedProgram:
    """Two rules applied in sequence."""

    operator_name = "Composed"

    def __init__(self, first, second):
        self.first  = first
        self.second = second
        # Exposed so the shared diagnosis path can inspect the stage that
        # actually decides which objects get touched last.
        self.selector = second.selector
        self.expr     = second.expr

    def apply(self, grid, background):
        intermediate = self.first.apply(grid, background)
        if intermediate is None:
            return None
        return self.second.apply(intermediate, get_background_color(intermediate))

    def __repr__(self):
        return f"{self.first} THEN {self.second}"


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

# Tolerant variants, used only as the FIRST stage of a composition.
REMOVE_FIRST = Operator("Remove", observe_remove_tolerant, RemoveProgram)
MOVE_FIRST   = Operator("Move",   observe_move_tolerant,   MoveProgram,
                        induce_vector_exprs)

# How many stage-one rules to carry forward. Each one costs a full
# stage-two induction over the residual, so this is the knob that keeps
# composition from turning into the combinatorial sweep this engine
# exists to avoid.
MAX_FIRST_STAGE_CANDIDATES = 6


class ComposedOperator:
    """
    Two operators in sequence, presented as a single Operator.

    Deliberately shaped to satisfy the same interface as Operator, so
    synthesize() drives a composed rule with exactly the loop it uses for
    a single one -- induce, test, absorb the counterexample, re-induce.
    Composition is a richer hypothesis space, not a different algorithm.

    Stage one is induced from a TOLERANT observation, which labels only
    the part of the change that operator can explain. Stage two is then
    induced from the residual (stage one's output vs the real output)
    using the ordinary STRICT observer -- by that point the residual
    should be a pure single-operator change, and if it isn't, strict
    observation rejects it and this candidate dies cheaply.
    """

    def __init__(self, first, second):
        self.first  = first
        self.second = second
        self.name   = f"{first.name}->{second.name}"
        self.induce_expr = None

    def observe(self, input_grid, output_grid):
        return self.first.observe(input_grid, output_grid)

    def _residual(self, program, evidence):
        residual = []
        for input_grid, output_grid in evidence:
            intermediate = program.apply(input_grid,
                                         get_background_color(input_grid))
            if intermediate is None:
                return None
            residual.append((intermediate, output_grid))
        return residual

    def induce(self, evidence):
        first_stage = self.first.induce(evidence)
        if not first_stage:
            return []

        composed = []
        for first_program in first_stage[:MAX_FIRST_STAGE_CANDIDATES]:
            residual = self._residual(first_program, evidence)
            if residual is None:
                continue
            # Nothing left to explain means stage one did the whole job,
            # which the single-operator path already covers -- composing
            # a no-op second stage on top would only add a worse-ranked
            # duplicate of a rule we can already find.
            if all(np.array_equal(mid, out) for mid, out in residual):
                continue
            for second_program in self.second.induce(residual)[:2]:
                composed.append(ComposedProgram(first_program, second_program))

        return composed

    def diagnose(self, program, input_grid, output_grid):
        background = get_background_color(input_grid)
        intermediate = program.first.apply(input_grid, background)
        if intermediate is None:
            return f"stage 1 ({program.first}) could not be applied"
        if np.array_equal(intermediate, output_grid):
            return "stage 1 alone already matches; stage 2 overshot"
        return "stage 2: " + self.second.diagnose(
            program.second, intermediate, output_grid
        )


DRAW    = DrawOperator()
EXTRACT = ExtractOperator()

COMPOSED_OPERATORS = [
    ComposedOperator(REMOVE_FIRST, RECOLOR),
    ComposedOperator(REMOVE_FIRST, MOVE),
    ComposedOperator(MOVE_FIRST,   RECOLOR),
]

# Single operators are tried first: when both can explain a task, the
# simpler rule is the one more likely to be the real one.
OPERATORS = [RECOLOR, REMOVE, MOVE, DRAW, EXTRACT] + COMPOSED_OPERATORS


# ============================================================
# THE LOOP: counterexample-guided refinement
#
# Unchanged from the single-operator proof of concept -- it now takes
# the operator as a parameter, but the control flow is identical.
# ============================================================

# Search budget, in seconds, escalating on evidence: a task starts with
# the first allowance and earns the second once induction produces any
# candidate from pair 1, which is the point at which a rule is known to
# exist in the language.
#
# In practice this now functions as a RUNAWAY GUARD rather than a search
# lever, because induction turned out not to be time-bound: measured
# over 1000 failing induce() calls the median is 0.75ms and the maximum
# 325ms, so neither budget binds in normal operation. That is still
# worth having -- two separate incidents today saw a single task consume
# hours -- but the budget buys safety, not solutions.
#
# Raise both for a submission run, where a few hundred tasks against a
# long wall-clock makes a larger allowance affordable.
SEARCH_BUDGET_SECONDS  = 2.0
PROMOTED_BUDGET_SECONDS = 15.0

# How many of pair 1's ranked candidates to fall back through when the
# top-ranked one fails refinement.
#
# Set to 1 (no fallback) because measurement said it earns nothing:
# 8 restarts solved the same 56 tasks with the same 43 rules offered and
# the same 91% precision, for 52% more runtime (531s -> 807s).
#
# There is a structural reason, and it is worth keeping written down.
# _refine_from re-induces from ALL accumulated evidence at every
# counterexample, so the candidate set it is choosing from by the last
# pair depends only on the evidence, not on which candidate the walk
# started from. Every restart converges to the same place. If no rule in
# the language fits every pair, a different starting point cannot
# discover one -- the loop is evidence-driven, not path-dependent.
#
# The plumbing is left in place because it is nearly free at 1 and would
# start to matter if refinement ever became path-dependent, e.g. if
# re-induction were made incremental rather than recomputed from scratch.
MAX_CANDIDATE_RESTARTS = 1


def _refine_from(program, pairs, operator, trace, deadline):
    """
    Walk pairs 2..n from a given starting rule, absorbing each
    counterexample into the evidence and re-inducing.

    This is the original counterexample-guided loop, unchanged in
    behaviour -- lifted into its own function only so the caller can run
    it from more than one starting candidate.
    """
    evidence = [pairs[0]]

    for i in range(1, len(pairs)):
        if time.monotonic() > deadline:
            trace.append(f"pair {i+1}: budget exhausted")
            return None

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
                         f"{len(evidence)} pairs")
            return None

        program = programs[0]
        trace.append(f"pair {i+1}: re-induced from {len(evidence)} pairs "
                     f"-> {program}   ({len(programs)} candidates remain)")

    # Final check: the refined rule must explain every pair, including
    # ones that passed before refinement changed the rule underneath them.
    for i, (input_grid, output_grid) in enumerate(pairs):
        predicted = program.apply(input_grid, get_background_color(input_grid))
        if predicted is None or not np.array_equal(predicted, output_grid):
            trace.append(f"final check failed on pair {i+1}")
            return None

    return program


def synthesize(task, operator, verbose=False,
               budget=None, promoted_budget=None):
    """
    Induce a rule from training pair 1, then walk the remaining pairs.
    Every pair that disproves the current rule is ADDED to the evidence
    set and the rule is re-induced under that stronger constraint.

    Search is bounded by a budget that ESCALATES on evidence: a task
    starts with a small allowance, and earns the larger one the moment
    induction produces any candidate from pair 1, since that is the
    point at which a rule is known to exist in the language and further
    search has something to find.

    When the top-ranked candidate fails to survive refinement, the next
    candidate is tried rather than the whole operator being abandoned.
    Those alternatives already fit pair 1 and are already in the
    language, so this explores the existing space more thoroughly
    instead of enlarging it -- which matters, because measurement showed
    enlarging it (3-atom conjunctions) produced only overfitted rules.

    Returns (program, trace) with program=None if no rule in the
    language explains every training pair.
    """
    pairs = [(p["input"], p["output"]) for p in task["train"]]
    trace = []

    deadline = time.monotonic() + (budget or SEARCH_BUDGET_SECONDS)
    candidates = operator.induce([pairs[0]])
    if not candidates:
        trace.append(f"pair 1: could not induce any {operator.name} rule")
        return None, trace

    # Promotion: pair 1 yielded something, so this task has earned the
    # bigger allowance.
    deadline = time.monotonic() + (promoted_budget or PROMOTED_BUDGET_SECONDS)
    trace.append(f"pair 1: {len(candidates)} candidates consistent with 1 pair")

    for rank, candidate in enumerate(candidates[:MAX_CANDIDATE_RESTARTS]):
        if time.monotonic() > deadline:
            trace.append("budget exhausted before all candidates were tried")
            break

        if rank:
            trace.append(f"-- falling back to candidate {rank + 1}: {candidate}")
        else:
            trace.append(f"pair 1: induced {candidate}")

        program = _refine_from(candidate, pairs, operator, trace, deadline)
        if program is not None:
            trace.append(f"FINAL RULE: {program}  "
                         f"(explains all {len(pairs)} pairs)")
            return program, trace

    return None, trace


def synthesize_program(task, operators=None):
    """
    Induce a rule for a task without applying it to anything.

    Separate from solve_with_synthesis so a caller with several test
    inputs induces once and applies many times, rather than re-running
    the whole induction per test grid.

    Returns (program, operator_name), or (None, None) if no operator's
    language explains every training pair.
    """
    for operator in (operators or OPERATORS):
        try:
            program, _ = synthesize(task, operator)
        except Exception:
            continue
        if program is not None:
            return program, operator.name
    return None, None


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
