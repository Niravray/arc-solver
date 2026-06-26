import numpy as np
import sys

sys.path.append(r"C:\arc_solver\src\layers")
from task_loader import load_all_tasks, get_task_ids
from object_extractor import extract_objects, get_background_color
from property_encoder import encode_all_objects

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

    for i in range(len(objects)):
        for j in range(len(objects)):
            if i == j:
                continue  # skip self-relationships

            obj_a = objects[i]
            obj_b = objects[j]

            rel = compute_relationship(obj_a, obj_b, grid_shape)
            pairs[(obj_a["id"], obj_b["id"])] = rel

    # Count how many objects of each color exist
    color_counts = {}
    for obj in objects:
        c = obj["color"]
        color_counts[c] = color_counts.get(c, 0) + 1

    # Rank objects by size (largest first)
    size_ranking = sorted(
        [obj["id"] for obj in objects],
        key=lambda oid: next(o["pixel_count"] for o in objects if o["id"] == oid),
        reverse=True
    )

    return {
        "objects":      objects,
        "pairs":        pairs,
        "color_counts": color_counts,
        "size_ranking": size_ranking,
        "grid_shape":   grid_shape,
    }


def compute_relationship(obj_a, obj_b, grid_shape):
    """
    Compute the full relationship between two objects.

    Returns a dict describing how obj_a relates to obj_b.
    """
    rel = {}

    # ---- SPATIAL RELATIONSHIPS ----

    # Centroid-based direction
    dr = obj_a["centroid_row"] - obj_b["centroid_row"]
    dc = obj_a["centroid_col"] - obj_b["centroid_col"]

    rel["a_above_b"]   = obj_a["max_row"] < obj_b["min_row"]
    rel["a_below_b"]   = obj_a["min_row"] > obj_b["max_row"]
    rel["a_left_of_b"] = obj_a["max_col"] < obj_b["min_col"]
    rel["a_right_of_b"]= obj_a["min_col"] > obj_b["max_col"]

    # Bounding box overlap
    row_overlap = not (obj_a["max_row"] < obj_b["min_row"] or
                       obj_a["min_row"] > obj_b["max_row"])
    col_overlap = not (obj_a["max_col"] < obj_b["min_col"] or
                       obj_a["min_col"] > obj_b["max_col"])
    rel["bboxes_overlap"] = row_overlap and col_overlap

    # Pixel-level touching (share an edge)
    rel["touching"] = are_touching(obj_a, obj_b)

    # Containment: is obj_a fully inside obj_b's bounding box?
    rel["a_inside_bbox_b"] = (
        obj_a["min_row"] >= obj_b["min_row"] and
        obj_a["max_row"] <= obj_b["max_row"] and
        obj_a["min_col"] >= obj_b["min_col"] and
        obj_a["max_col"] <= obj_b["max_col"]
    )

    # ---- DISTANCE ----
    centroid_distance = np.sqrt(dr**2 + dc**2)
    rel["centroid_distance"] = round(float(centroid_distance), 2)

    # Direction from a to b (approximate)
    if abs(dr) > abs(dc):
        rel["direction_a_to_b"] = "down" if dr > 0 else "up"
    elif abs(dc) > abs(dr):
        rel["direction_a_to_b"] = "right" if dc > 0 else "left"
    else:
        rel["direction_a_to_b"] = "diagonal"

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

    # ---- SHAPE RELATIONSHIPS ----
    rel["same_shape"] = (obj_a["fingerprint"] == obj_b["fingerprint"])

    # Same dimensions (height x width)?
    rel["same_dimensions"] = (
        obj_a["height"] == obj_b["height"] and
        obj_a["width"]  == obj_b["width"]
    )

    return rel


def are_touching(obj_a, obj_b):
    """
    Check if two objects are touching (share an adjacent pixel edge).
    Two pixels touch if they are 1 step apart horizontally or vertically.
    """
    pixels_a = set(obj_a["pixels"])

    for (r, c) in obj_b["pixels"]:
        # Check all 4 neighbors of each pixel in obj_b
        for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) in pixels_a:
                return True

    return False


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
    smallest_id = graph["size_ranking"][-1]
    return next(o for o in graph["objects"] if o["id"] == smallest_id)


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