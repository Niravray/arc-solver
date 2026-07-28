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

import numpy as np
import sys
from typing import Any, TypedDict

from task_loader import load_all_tasks, get_task_ids
from object_extractor import (
    extract_objects,
    extract_objects_multi_strategy,
    get_background_color,
)
from property_encoder import encode_all_objects


class Relationship(TypedDict):
    """
    Typed contract for one directed edge from object A to object B.

    Directional fields describe B relative to A. Symmetric fields must
    have the same value on the reverse edge.
    """
    a_above_b: bool
    a_below_b: bool
    a_left_of_b: bool
    a_right_of_b: bool
    bboxes_overlap: bool
    touching: bool
    touching_diagonally: bool
    touching_8: bool
    contact_sides_a_to_b: tuple[str, ...]
    shared_edge_count: int
    a_inside_bbox_b: bool
    row_ranges_overlap: bool
    col_ranges_overlap: bool
    same_centroid_row: bool
    same_centroid_col: bool
    delta_row: float
    delta_col: float
    vertical_gap: int
    horizontal_gap: int
    centroid_distance: float
    manhattan_distance: float
    chebyshev_distance: float
    normalized_centroid_distance: float
    direction_a_to_b: str
    direction_8way_a_to_b: str
    same_color: bool
    different_color: bool
    a_larger_than_b: bool
    a_smaller_than_b: bool
    same_pixel_count: bool
    size_ratio: float
    a_bbox_larger_than_b: bool
    same_shape: bool
    same_shape_up_to_transform: bool
    shape_relation: str
    same_dimensions: bool


class RelationshipGraph(TypedDict):
    """Backward-compatible typed contract for a relationship graph."""
    objects: list[dict[str, Any]]
    pairs: dict[tuple[int, int], Relationship]
    color_counts: dict[int, int]
    size_ranking: list[int]
    grid_shape: tuple[int, int]


class RelationshipGraphInterpretations(TypedDict):
    """
    Alternative graph interpretations of the same grid.

    No strategy is declared correct at construction time. Downstream
    verification can retain or reject interpretations using all examples.
    """
    background: int
    graphs: dict[str, RelationshipGraph]
    skipped: dict[str, str]


class GraphValidationError(ValueError):
    """Raised when a relationship graph violates its contract."""


SYMMETRIC_RELATIONS = (
    "bboxes_overlap",
    "touching",
    "touching_diagonally",
    "touching_8",
    "shared_edge_count",
    "row_ranges_overlap",
    "col_ranges_overlap",
    "same_centroid_row",
    "same_centroid_col",
    "same_color",
    "different_color",
    "same_pixel_count",
    "same_shape",
    "same_shape_up_to_transform",
    "same_dimensions",
    "centroid_distance",
    "manhattan_distance",
    "chebyshev_distance",
    "normalized_centroid_distance",
    "vertical_gap",
    "horizontal_gap",
)

DIRECTION_INVERSES = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
    "diagonal": "diagonal",
}

DIRECTION_8WAY_INVERSES = {
    "same_position": "same_position",
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
    "up_left": "down_right",
    "up_right": "down_left",
    "down_left": "up_right",
    "down_right": "up_left",
}

# ============================================================
# RELATIONSHIP GRAPH
# Layer 4 of the ARC Solver Pipeline
#
# What this does:
# - Takes a list of encoded objects
# - For every pair of objects, computes their relationship
# - Returns a graph where:
#     nodes = objects
#     edges = relationships between object pairs
#
# Relationships computed:
#     spatial   : above, below, left_of, right_of, overlapping, touching, inside, outside
#     color     : same_color, different_color
#     size      : larger, smaller, equal_size
#     shape     : same_shape, different_shape
#     count     : how many objects of each color
#     distance  : pixel distance between centroids
# ============================================================


def build_relationship_graph(objects, grid_shape):
    """
    Build a complete relationship graph for a list of objects.

    objects    = list of encoded object dicts (from property_encoder)
    grid_shape = (rows, cols) of the full grid

    Returns a dict with:
        "objects"       : the original object list
        "pairs"         : dict of { (id_a, id_b): relationship_dict }
        "color_counts"  : dict of { color: count }
        "size_ranking"  : list of object ids sorted by pixel_count (largest first)
        "grid_shape"    : the grid shape
    """
    pairs = {}
    pixel_sets = {obj["id"]: set(obj["pixels"]) for obj in objects}
    shape_masks = {obj["id"]: _object_mask(obj) for obj in objects}

    for i in range(len(objects)):
        for j in range(len(objects)):
            if i == j:
                continue  # skip self-relationships

            obj_a = objects[i]
            obj_b = objects[j]

            rel = compute_relationship(
                obj_a,
                obj_b,
                grid_shape,
                pixel_sets=pixel_sets,
                shape_masks=shape_masks,
            )
            pairs[(obj_a["id"], obj_b["id"])] = rel

    # Count how many objects of each color exist
    color_counts = {}
    for obj in objects:
        c = obj["color"]
        color_counts[c] = color_counts.get(c, 0) + 1

    # Rank objects by size (largest first)
    # Rank by size, then by lowest object id. The explicit secondary key
    # keeps ties deterministic and consistent with spatial_color selectors.
    size_ranking = [
        obj["id"]
        for obj in sorted(objects, key=lambda obj: (-obj["pixel_count"], obj["id"]))
    ]

    graph = {
        "objects":      objects,
        "pairs":        pairs,
        "color_counts": color_counts,
        "size_ranking": size_ranking,
        "grid_shape":   grid_shape,
    }
    validate_relationship_graph(graph)
    return graph


def build_relationship_graph_interpretations(grid, max_objects=64):
    """
    Build one relationship graph for every available object interpretation.

    This is additive: build_relationship_graph() remains the single-graph
    API used by existing callers.
    """
    extracted = extract_objects_multi_strategy(grid)
    graphs = {}
    skipped = {}

    for strategy_name, objects in extracted.items():
        if len(objects) > max_objects:
            skipped[strategy_name] = (
                f"{len(objects)} objects exceeds limit {max_objects}"
            )
            continue
        encoded = encode_all_objects(objects, grid.shape)
        graphs[strategy_name] = build_relationship_graph(encoded, grid.shape)

    return {
        "background": get_background_color(grid),
        "graphs": graphs,
        "skipped": skipped,
    }


def compute_relationship(
    obj_a,
    obj_b,
    grid_shape,
    pixel_sets=None,
    shape_masks=None,
):
    """
    Compute the full relationship between two objects.

    Returns a dict describing how obj_a relates to obj_b.
    """
    rel = {}

    # ---- SPATIAL RELATIONSHIPS ----

    # Centroid-based direction
    # Signed vector from A to B. Positive rows point down and positive
    # columns point right in grid coordinates.
    dr = obj_b["centroid_row"] - obj_a["centroid_row"]
    dc = obj_b["centroid_col"] - obj_a["centroid_col"]

    rel["a_above_b"]   = obj_a["max_row"] < obj_b["min_row"]
    rel["a_below_b"]   = obj_a["min_row"] > obj_b["max_row"]
    rel["a_left_of_b"] = obj_a["max_col"] < obj_b["min_col"]
    rel["a_right_of_b"]= obj_a["min_col"] > obj_b["max_col"]

    # Bounding box overlap
    row_overlap = not (obj_a["max_row"] < obj_b["min_row"] or
                       obj_a["min_row"] > obj_b["max_row"])
    col_overlap = not (obj_a["max_col"] < obj_b["min_col"] or
                       obj_a["min_col"] > obj_b["max_col"])
    rel["row_ranges_overlap"] = row_overlap
    rel["col_ranges_overlap"] = col_overlap
    rel["bboxes_overlap"] = row_overlap and col_overlap

    # Pixel-level contact, preserving connectivity and contact geometry.
    pixels_b = (
        pixel_sets[obj_b["id"]]
        if pixel_sets is not None
        else None
    )
    contact = compute_contact(obj_a, obj_b, pixels_b=pixels_b)
    rel["touching"] = contact["touching"]
    rel["touching_diagonally"] = contact["touching_diagonally"]
    rel["touching_8"] = contact["touching_8"]
    rel["contact_sides_a_to_b"] = contact["contact_sides_a_to_b"]
    rel["shared_edge_count"] = contact["shared_edge_count"]

    # Containment: is obj_a fully inside obj_b's bounding box?
    rel["a_inside_bbox_b"] = (
        obj_a["min_row"] >= obj_b["min_row"] and
        obj_a["max_row"] <= obj_b["max_row"] and
        obj_a["min_col"] >= obj_b["min_col"] and
        obj_a["max_col"] <= obj_b["max_col"]
    )

    # ---- DISTANCE AND ALIGNMENT ----
    rel["same_centroid_row"] = (dr == 0)
    rel["same_centroid_col"] = (dc == 0)
    rel["delta_row"] = round(float(dr), 3)
    rel["delta_col"] = round(float(dc), 3)

    if obj_a["max_row"] < obj_b["min_row"]:
        vertical_gap = obj_b["min_row"] - obj_a["max_row"] - 1
    elif obj_b["max_row"] < obj_a["min_row"]:
        vertical_gap = obj_a["min_row"] - obj_b["max_row"] - 1
    else:
        vertical_gap = 0

    if obj_a["max_col"] < obj_b["min_col"]:
        horizontal_gap = obj_b["min_col"] - obj_a["max_col"] - 1
    elif obj_b["max_col"] < obj_a["min_col"]:
        horizontal_gap = obj_a["min_col"] - obj_b["max_col"] - 1
    else:
        horizontal_gap = 0

    rel["vertical_gap"] = int(vertical_gap)
    rel["horizontal_gap"] = int(horizontal_gap)

    centroid_distance = np.sqrt(dr**2 + dc**2)
    rel["centroid_distance"] = round(float(centroid_distance), 2)
    rel["manhattan_distance"] = round(float(abs(dr) + abs(dc)), 2)
    rel["chebyshev_distance"] = round(float(max(abs(dr), abs(dc))), 2)

    grid_rows, grid_cols = grid_shape
    max_grid_distance = np.sqrt(
        max(0, grid_rows - 1) ** 2 + max(0, grid_cols - 1) ** 2
    )
    rel["normalized_centroid_distance"] = (
        round(float(centroid_distance / max_grid_distance), 4)
        if max_grid_distance > 0
        else 0.0
    )

    # Direction from a to b (approximate)
    if abs(dr) > abs(dc):
        rel["direction_a_to_b"] = "down" if dr > 0 else "up"
    elif abs(dc) > abs(dr):
        rel["direction_a_to_b"] = "right" if dc > 0 else "left"
    else:
        rel["direction_a_to_b"] = "diagonal"
    rel["direction_8way_a_to_b"] = direction_8way(dr, dc)

    # ---- COLOR RELATIONSHIPS ----
    rel["same_color"]      = (obj_a["color"] == obj_b["color"])
    rel["different_color"] = (obj_a["color"] != obj_b["color"])

    # ---- SIZE RELATIONSHIPS ----
    pa = obj_a["pixel_count"]
    pb = obj_b["pixel_count"]

    rel["a_larger_than_b"]  = (pa > pb)
    rel["a_smaller_than_b"] = (pa < pb)
    rel["same_pixel_count"] = (pa == pb)
    rel["size_ratio"]       = round(pa / pb, 3) if pb > 0 else float("inf")

    # Bounding box size comparison
    area_a = obj_a["height"] * obj_a["width"]
    area_b = obj_b["height"] * obj_b["width"]
    rel["a_bbox_larger_than_b"] = (area_a > area_b)

    # Same dimensions (height x width)?
    rel["same_dimensions"] = (
        obj_a["height"] == obj_b["height"] and
        obj_a["width"]  == obj_b["width"]
    )
    rel["same_shape"] = (
        rel["same_dimensions"]
        and obj_a["fingerprint"] == obj_b["fingerprint"]
    )
    mask_a = shape_masks[obj_a["id"]] if shape_masks is not None else None
    mask_b = shape_masks[obj_b["id"]] if shape_masks is not None else None
    rel["shape_relation"] = get_shape_relation(
        obj_a,
        obj_b,
        mask_a=mask_a,
        mask_b=mask_b,
    )
    rel["same_shape_up_to_transform"] = (
        rel["shape_relation"] != "different"
    )

    return rel


def direction_8way(delta_row, delta_col):
    """Return the sign-based direction from object A to object B."""
    vertical = "up" if delta_row < 0 else "down" if delta_row > 0 else ""
    horizontal = (
        "left" if delta_col < 0 else "right" if delta_col > 0 else ""
    )
    if vertical and horizontal:
        return f"{vertical}_{horizontal}"
    return vertical or horizontal or "same_position"


def compute_contact(obj_a, obj_b, pixels_b=None):
    """Compute orthogonal and diagonal pixel contact details."""
    if pixels_b is None:
        pixels_b = set(obj_b["pixels"])
    sides = set()
    shared_edge_count = 0
    diagonal_contact = False

    orthogonal_steps = (
        (-1, 0, "up"),
        (1, 0, "down"),
        (0, -1, "left"),
        (0, 1, "right"),
    )
    diagonal_steps = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    for row, col in obj_a["pixels"]:
        for delta_row, delta_col, side in orthogonal_steps:
            if (row + delta_row, col + delta_col) in pixels_b:
                sides.add(side)
                shared_edge_count += 1
        for delta_row, delta_col in diagonal_steps:
            if (row + delta_row, col + delta_col) in pixels_b:
                diagonal_contact = True

    touching = shared_edge_count > 0
    return {
        "touching": touching,
        "touching_diagonally": diagonal_contact,
        "touching_8": touching or diagonal_contact,
        "contact_sides_a_to_b": tuple(sorted(sides)),
        "shared_edge_count": shared_edge_count,
    }


def _object_mask(obj):
    """Build a normalized binary mask without relying on obj['grid']."""
    mask = np.zeros((obj["height"], obj["width"]), dtype=np.int8)
    for row, col in obj["pixels"]:
        mask[row - obj["min_row"], col - obj["min_col"]] = 1
    return mask


def get_shape_relation(obj_a, obj_b, mask_a=None, mask_b=None):
    """Classify equality under exact, rotated, or reflected shape."""
    if mask_a is None:
        mask_a = _object_mask(obj_a)
    if mask_b is None:
        mask_b = _object_mask(obj_b)

    if np.array_equal(mask_a, mask_b):
        return "exact"

    rotations = [np.rot90(mask_a, turns) for turns in (1, 2, 3)]
    if any(np.array_equal(candidate, mask_b) for candidate in rotations):
        return "rotated"

    reflections = [
        np.fliplr(mask_a),
        np.flipud(mask_a),
        mask_a.T,
        np.flipud(mask_a.T),
    ]
    if any(np.array_equal(candidate, mask_b) for candidate in reflections):
        return "reflected"

    return "different"


def validate_relationship_graph(graph):
    """
    Validate graph structure and reciprocal relationship invariants.

    Returns the original graph so callers may validate inline. Raises
    GraphValidationError with a precise message on the first violation.
    """
    required_keys = {
        "objects", "pairs", "color_counts", "size_ranking", "grid_shape"
    }
    missing = required_keys - set(graph)
    if missing:
        raise GraphValidationError(
            f"Graph is missing required keys: {sorted(missing)}"
        )

    objects = graph["objects"]
    object_ids = [obj["id"] for obj in objects]
    if len(object_ids) != len(set(object_ids)):
        raise GraphValidationError("Object ids must be unique")

    rows, cols = graph["grid_shape"]
    if rows <= 0 or cols <= 0:
        raise GraphValidationError(
            f"Grid shape must be positive, got {graph['grid_shape']}"
        )

    expected_pairs = {
        (id_a, id_b)
        for id_a in object_ids
        for id_b in object_ids
        if id_a != id_b
    }
    actual_pairs = set(graph["pairs"])
    if actual_pairs != expected_pairs:
        missing_pairs = sorted(expected_pairs - actual_pairs)
        extra_pairs = sorted(actual_pairs - expected_pairs)
        raise GraphValidationError(
            f"Pair set mismatch; missing={missing_pairs}, extra={extra_pairs}"
        )

    expected_counts = {}
    for obj in objects:
        color = obj["color"]
        expected_counts[color] = expected_counts.get(color, 0) + 1
    if graph["color_counts"] != expected_counts:
        raise GraphValidationError(
            f"Color counts mismatch; expected {expected_counts}, "
            f"got {graph['color_counts']}"
        )

    object_by_id = {obj["id"]: obj for obj in objects}
    expected_ranking = [
        obj["id"]
        for obj in sorted(objects, key=lambda obj: (-obj["pixel_count"], obj["id"]))
    ]
    if graph["size_ranking"] != expected_ranking:
        raise GraphValidationError(
            f"Size ranking mismatch; expected {expected_ranking}, "
            f"got {graph['size_ranking']}"
        )

    for (id_a, id_b), rel in graph["pairs"].items():
        reverse = graph["pairs"][(id_b, id_a)]

        for key in SYMMETRIC_RELATIONS:
            if rel[key] != reverse[key]:
                raise GraphValidationError(
                    f"Relation '{key}' is inconsistent for {id_a}<->{id_b}"
                )

        inverse_direction = DIRECTION_INVERSES.get(rel["direction_a_to_b"])
        if inverse_direction is None:
            raise GraphValidationError(
                f"Unknown direction '{rel['direction_a_to_b']}' "
                f"for edge {id_a}->{id_b}"
            )
        if reverse["direction_a_to_b"] != inverse_direction:
            raise GraphValidationError(
                f"Direction mismatch for {id_a}<->{id_b}: "
                f"{rel['direction_a_to_b']} vs "
                f"{reverse['direction_a_to_b']}"
            )

        inverse_direction_8way = DIRECTION_8WAY_INVERSES.get(
            rel["direction_8way_a_to_b"]
        )
        if inverse_direction_8way is None:
            raise GraphValidationError(
                f"Unknown 8-way direction "
                f"'{rel['direction_8way_a_to_b']}' for edge {id_a}->{id_b}"
            )
        if reverse["direction_8way_a_to_b"] != inverse_direction_8way:
            raise GraphValidationError(
                f"8-way direction mismatch for {id_a}<->{id_b}: "
                f"{rel['direction_8way_a_to_b']} vs "
                f"{reverse['direction_8way_a_to_b']}"
            )

        if rel["delta_row"] != -reverse["delta_row"]:
            raise GraphValidationError(
                f"Row displacement mismatch for {id_a}<->{id_b}"
            )
        if rel["delta_col"] != -reverse["delta_col"]:
            raise GraphValidationError(
                f"Column displacement mismatch for {id_a}<->{id_b}"
            )

        inverse_checks = (
            ("a_above_b", "a_below_b"),
            ("a_below_b", "a_above_b"),
            ("a_left_of_b", "a_right_of_b"),
            ("a_right_of_b", "a_left_of_b"),
            ("a_larger_than_b", "a_smaller_than_b"),
            ("a_smaller_than_b", "a_larger_than_b"),
        )
        for forward_key, reverse_key in inverse_checks:
            if rel[forward_key] != reverse[reverse_key]:
                raise GraphValidationError(
                    f"Inverse mismatch: {id_a}->{id_b} '{forward_key}' "
                    f"does not match {id_b}->{id_a} '{reverse_key}'"
                )

        expected_bbox_relation = (
            object_by_id[id_a]["height"] * object_by_id[id_a]["width"]
            >
            object_by_id[id_b]["height"] * object_by_id[id_b]["width"]
        )
        if rel["a_bbox_larger_than_b"] != expected_bbox_relation:
            raise GraphValidationError(
                f"Bounding-box size relation is invalid for {id_a}->{id_b}"
            )

    return graph


def are_touching(obj_a, obj_b):
    """
    Check if two objects are touching (share an adjacent pixel edge).
    Two pixels touch if they are 1 step apart horizontally or vertically.
    """
    return compute_contact(obj_a, obj_b)["touching"]


def get_relationship(graph, id_a, id_b):
    """
    Get the relationship between two specific objects by their IDs.
    """
    return graph["pairs"].get((id_a, id_b), None)


def find_objects_by_color(graph, color):
    """
    Get all objects of a specific color from the graph.
    """
    return [obj for obj in graph["objects"] if obj["color"] == color]


def find_largest_object(graph):
    """
    Get the largest object (by pixel count) from the graph.
    """
    if not graph["size_ranking"]:
        return None
    largest_id = graph["size_ranking"][0]
    return next(o for o in graph["objects"] if o["id"] == largest_id)


def find_smallest_object(graph):
    """
    Get the smallest object (by pixel count) from the graph.
    """
    if not graph["size_ranking"]:
        return None
    return min(
        graph["objects"],
        key=lambda obj: (obj["pixel_count"], obj["id"])
    )


def print_graph_summary(graph):
    """
    Print a human-readable summary of the relationship graph.
    """
    objects = graph["objects"]
    pairs   = graph["pairs"]

    print(f"Graph: {len(objects)} objects, {len(pairs)} relationships")
    print(f"Color counts: {graph['color_counts']}")
    print(f"Size ranking (largest first): {graph['size_ranking']}")

    for (id_a, id_b), rel in pairs.items():
        obj_a = next(o for o in objects if o["id"] == id_a)
        obj_b = next(o for o in objects if o["id"] == id_b)

        spatial = []
        if rel["a_above_b"]:    spatial.append("above")
        if rel["a_below_b"]:    spatial.append("below")
        if rel["a_left_of_b"]:  spatial.append("left_of")
        if rel["a_right_of_b"]: spatial.append("right_of")
        if rel["touching"]:     spatial.append("touching")
        if not spatial:         spatial.append("overlapping_or_near")

        print(f"  obj{id_a}(color={obj_a['color']},px={obj_a['pixel_count']}) "
              f"-> obj{id_b}(color={obj_b['color']},px={obj_b['pixel_count']}): "
              f"{', '.join(spatial)} | "
              f"same_color={rel['same_color']} | "
              f"a_larger={rel['a_larger_than_b']} | "
              f"same_shape={rel['same_shape']} | "
              f"dist={rel['centroid_distance']}")


# ============================================================
# QUICK TEST
# py C:\arc_solver\src\layers\relationship_graph.py
# ============================================================

if __name__ == "__main__":
    print("Loading training tasks...")
    tasks = load_all_tasks("training")
    ids   = get_task_ids("training")
    print(f"Loaded {len(tasks)} tasks.\n")

    passed = 0
    failed = 0

    print("Testing relationship graph on first 5 tasks...\n")

    for i in range(5):
        task_id = ids[i]
        task    = tasks[task_id]

        try:
            for j, pair in enumerate(task["train"]):
                grid    = pair["input"]
                objects = extract_objects(grid)
                encoded = encode_all_objects(objects, grid.shape)
                graph   = build_relationship_graph(encoded, grid.shape)

                print(f"Task {i+1} ({task_id}) Pair {j+1}: "
                      f"{len(encoded)} objects, "
                      f"{len(graph['pairs'])} relationships, "
                      f"color_counts={graph['color_counts']}")

            passed += 1

        except Exception as e:
            failed += 1
            print(f"Task {i+1} FAILED: {e}")

    print(f"\nResults: {passed} passed, {failed} failed\n")

    # Detailed graph for the color indicator task (009d5c81)
    print("--- Detailed graph for task 009d5c81 ---")
    task    = tasks["009d5c81"]
    pair    = task["train"][0]
    grid    = pair["input"]
    objects = extract_objects(grid)
    encoded = encode_all_objects(objects, grid.shape)
    graph   = build_relationship_graph(encoded, grid.shape)

    print_graph_summary(graph)

    print("\nLargest object:")
    largest = find_largest_object(graph)
    if largest:
        print(f"  color={largest['color']} pixels={largest['pixel_count']}")

    print("\nSmallest object:")
    smallest = find_smallest_object(graph)
    if smallest:
        print(f"  color={smallest['color']} pixels={smallest['pixel_count']}")

    print("\nRelationship graph working correctly.")
