"""
Graph correspondence engine.

STATUS: built and unit-tested, NOT wired into the solver.

It compares relationship graphs between an input and output across
every shared extraction strategy and ranks the readings by how
completely their nodes correspond. Deliberately left unused rather
than deleted: the measured gap in the solver is not in how grids
are READ -- three additions of that kind (panels, raised guards,
region splitting) each made structure visible while solving
nothing -- so wiring another one in is the lowest-value option
available. It is kept because a correspondence score is the
natural input to the two-level rules the solver actually lacks,
where "which objects correspond" has to be answered before "what
was derived from them".
"""

from collections import Counter
from typing import Any, TypedDict

import numpy as np
from scipy.optimize import linear_sum_assignment

from relationship_graph import (
    RelationshipGraph,
    build_relationship_graph_interpretations,
    get_shape_relation,
)


class NodeChanges(TypedDict):
    moved: bool
    delta_row: int
    delta_col: int
    recolored: bool
    old_color: int
    new_color: int
    resized: bool
    old_pixel_count: int
    new_pixel_count: int
    shape_changed: bool
    shape_relation: str


class NodeMatch(TypedDict):
    input_id: int
    output_id: int
    score: float
    changes: NodeChanges


class RelationshipChange(TypedDict):
    input_pair: tuple[int, int]
    output_pair: tuple[int, int]
    changed_fields: dict[str, tuple[Any, Any]]


class GraphDifference(TypedDict):
    strategy: str
    node_matches: list[NodeMatch]
    removed_input_ids: list[int]
    added_output_ids: list[int]
    relationship_changes: list[RelationshipChange]
    preserved_relationships: int
    correspondence_score: float
    signature: str


# Relations that describe transferable structure rather than raw measurements.
STRUCTURAL_RELATION_KEYS = (
    "a_above_b",
    "a_below_b",
    "a_left_of_b",
    "a_right_of_b",
    "touching",
    "touching_diagonally",
    "a_inside_bbox_b",
    "same_color",
    "a_larger_than_b",
    "a_smaller_than_b",
    "same_pixel_count",
    "same_shape",
    "same_shape_up_to_transform",
    "same_centroid_row",
    "same_centroid_col",
)


def compute_graph_differences(input_grid, output_grid):
    """
    Compare every shared extraction strategy for an input/output pair.

    Interpretations remain separate. The returned list is ranked by how
    completely and confidently its nodes correspond.
    """
    input_interpretations = build_relationship_graph_interpretations(input_grid)
    output_interpretations = build_relationship_graph_interpretations(output_grid)
    differences = []

    shared_strategies = (
        set(input_interpretations["graphs"])
        & set(output_interpretations["graphs"])
    )
    for strategy in sorted(shared_strategies):
        differences.append(compare_relationship_graphs(
            input_interpretations["graphs"][strategy],
            output_interpretations["graphs"][strategy],
            strategy=strategy,
        ))

    differences.sort(
        key=lambda diff: (
            -diff["correspondence_score"],
            len(diff["removed_input_ids"]) + len(diff["added_output_ids"]),
            diff["strategy"],
        )
    )
    return differences


def compare_relationship_graphs(input_graph, output_graph, strategy="unknown"):
    """
    Find a globally optimal partial node correspondence, then compare edges.

    Dummy rows and columns allow nodes to remain unmatched. This supports
    additions and removals without forcing a bad real-to-real assignment.
    """
    input_objects = input_graph["objects"]
    output_objects = output_graph["objects"]
    node_matches = match_graph_nodes(input_graph, output_graph)

    matched_input_ids = {match["input_id"] for match in node_matches}
    matched_output_ids = {match["output_id"] for match in node_matches}
    removed = sorted(
        obj["id"] for obj in input_objects
        if obj["id"] not in matched_input_ids
    )
    added = sorted(
        obj["id"] for obj in output_objects
        if obj["id"] not in matched_output_ids
    )

    relationship_changes, preserved = compare_matched_relationships(
        input_graph,
        output_graph,
        node_matches,
    )

    raw_score = sum(match["score"] for match in node_matches)
    possible = max(len(input_objects), len(output_objects), 1) * 14.0
    correspondence_score = round(raw_score / possible, 4)

    return {
        "strategy": strategy,
        "node_matches": node_matches,
        "removed_input_ids": removed,
        "added_output_ids": added,
        "relationship_changes": relationship_changes,
        "preserved_relationships": preserved,
        "correspondence_score": correspondence_score,
        "signature": build_graph_diff_signature(
            node_matches,
            removed,
            added,
            relationship_changes,
        ),
    }


def match_graph_nodes(input_graph, output_graph, minimum_score=3.0):
    """Return a globally optimal partial correspondence between graph nodes."""
    input_objects = input_graph["objects"]
    output_objects = output_graph["objects"]
    input_count = len(input_objects)
    output_count = len(output_objects)

    if input_count == 0 or output_count == 0:
        return []

    # Real nodes plus one dummy slot for every node on the opposite side.
    matrix_size = input_count + output_count
    weights = np.zeros((matrix_size, matrix_size), dtype=float)

    for input_index, input_obj in enumerate(input_objects):
        for output_index, output_obj in enumerate(output_objects):
            weights[input_index, output_index] = node_match_score(
                input_obj,
                output_obj,
                input_graph,
                output_graph,
            )

    row_indices, col_indices = linear_sum_assignment(-weights)
    matches = []
    for row, col in zip(row_indices, col_indices):
        if row >= input_count or col >= output_count:
            continue
        score = float(weights[row, col])
        if score < minimum_score:
            continue

        input_obj = input_objects[row]
        output_obj = output_objects[col]
        matches.append({
            "input_id": input_obj["id"],
            "output_id": output_obj["id"],
            "score": round(score, 3),
            "changes": compute_node_changes(input_obj, output_obj),
        })

    matches.sort(key=lambda match: match["input_id"])
    return matches


def node_match_score(input_obj, output_obj, input_graph, output_graph):
    """Score node identity using appearance and relational context."""
    score = 0.0
    shape_relation = get_shape_relation(input_obj, output_obj)
    if shape_relation == "exact":
        score += 8.0
    elif shape_relation in {"rotated", "reflected"}:
        score += 6.0

    if input_obj["color"] == output_obj["color"]:
        score += 1.5

    larger_count = max(input_obj["pixel_count"], output_obj["pixel_count"])
    if larger_count:
        score += (
            min(input_obj["pixel_count"], output_obj["pixel_count"])
            / larger_count
        ) * 2.0

    if (
        input_obj["height"] == output_obj["height"]
        and input_obj["width"] == output_obj["width"]
    ):
        score += 1.0

    input_profile = node_relation_profile(input_graph, input_obj["id"])
    output_profile = node_relation_profile(output_graph, output_obj["id"])
    score += profile_similarity(input_profile, output_profile) * 1.5
    return score


def node_relation_profile(graph, object_id):
    """Build an ID-independent multiset of outgoing structural facts."""
    profile = Counter()
    for (source_id, _), relation in graph["pairs"].items():
        if source_id != object_id:
            continue
        for key in STRUCTURAL_RELATION_KEYS:
            if relation[key]:
                profile[key] += 1
    return profile


def profile_similarity(first, second):
    """Multiset Jaccard similarity in the range 0..1."""
    keys = set(first) | set(second)
    if not keys:
        return 1.0
    intersection = sum(min(first[key], second[key]) for key in keys)
    union = sum(max(first[key], second[key]) for key in keys)
    return intersection / union if union else 1.0


def compute_node_changes(input_obj, output_obj):
    delta_row = output_obj["min_row"] - input_obj["min_row"]
    delta_col = output_obj["min_col"] - input_obj["min_col"]
    shape_relation = get_shape_relation(input_obj, output_obj)
    return {
        "moved": delta_row != 0 or delta_col != 0,
        "delta_row": delta_row,
        "delta_col": delta_col,
        "recolored": input_obj["color"] != output_obj["color"],
        "old_color": input_obj["color"],
        "new_color": output_obj["color"],
        "resized": input_obj["pixel_count"] != output_obj["pixel_count"],
        "old_pixel_count": input_obj["pixel_count"],
        "new_pixel_count": output_obj["pixel_count"],
        "shape_changed": shape_relation != "exact",
        "shape_relation": shape_relation,
    }


def compare_matched_relationships(input_graph, output_graph, node_matches):
    """Compare structural edges induced by the matched node set."""
    output_id_by_input_id = {
        match["input_id"]: match["output_id"] for match in node_matches
    }
    matched_input_ids = sorted(output_id_by_input_id)
    changes = []
    preserved = 0

    for input_a in matched_input_ids:
        for input_b in matched_input_ids:
            if input_a == input_b:
                continue
            output_a = output_id_by_input_id[input_a]
            output_b = output_id_by_input_id[input_b]
            input_relation = input_graph["pairs"][(input_a, input_b)]
            output_relation = output_graph["pairs"][(output_a, output_b)]

            changed_fields = {}
            for key in STRUCTURAL_RELATION_KEYS:
                if input_relation[key] != output_relation[key]:
                    changed_fields[key] = (
                        input_relation[key],
                        output_relation[key],
                    )

            if changed_fields:
                changes.append({
                    "input_pair": (input_a, input_b),
                    "output_pair": (output_a, output_b),
                    "changed_fields": changed_fields,
                })
            else:
                preserved += 1

    return changes, preserved


def build_graph_diff_signature(matches, removed, added, relationship_changes):
    parts = []
    if removed:
        parts.append("NODES_REMOVED")
    if added:
        parts.append("NODES_ADDED")
    if any(match["changes"]["moved"] for match in matches):
        parts.append("NODES_MOVED")
    if any(match["changes"]["recolored"] for match in matches):
        parts.append("NODES_RECOLORED")
    if any(match["changes"]["resized"] for match in matches):
        parts.append("NODES_RESIZED")
    if any(match["changes"]["shape_changed"] for match in matches):
        parts.append("NODE_SHAPES_CHANGED")
    if relationship_changes:
        parts.append("RELATIONSHIPS_CHANGED")
    return "|".join(parts) if parts else "NO_GRAPH_CHANGE"
