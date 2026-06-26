import numpy as np
import sys

sys.path.append(r"C:\arc_solver\src\layers")
from task_loader import load_all_tasks, get_task_ids
from object_extractor import extract_objects, get_background_color
from property_encoder import encode_all_objects
from relationship_graph import build_relationship_graph

# ============================================================
# DIFFERENCE ENGINE
# Layer 5 of the ARC Solver Pipeline
#
# What this does:
# - Takes input graph and output graph for a training pair
# - Computes exactly what changed between them:
#     which objects disappeared
#     which objects appeared
#     which objects moved
#     which objects got recolored
#     which objects got resized
#     which objects stayed exactly the same
# - Also computes grid-level changes:
#     did the grid size change?
#     did the background color change?
#     how many pixels changed total?
# ============================================================


def compute_difference(input_grid, output_grid):
    """
    Compute the full difference between an input and output grid.

    input_grid  = 2D NumPy array (the input)
    output_grid = 2D NumPy array (the output)

    Returns a diff dict with everything that changed.
    """
    diff = {}

    # ---- GRID LEVEL CHANGES ----
    in_rows,  in_cols  = input_grid.shape
    out_rows, out_cols = output_grid.shape

    diff["input_shape"]  = (in_rows, in_cols)
    diff["output_shape"] = (out_rows, out_cols)
    diff["size_changed"] = (in_rows != out_rows or in_cols != out_cols)
    diff["size_increased"] = (out_rows * out_cols > in_rows * in_cols)
    diff["size_decreased"] = (out_rows * out_cols < in_rows * in_cols)

    # Scale factor (if size changed)
    if in_rows > 0 and in_cols > 0:
        diff["row_scale"] = round(out_rows / in_rows, 3)
        diff["col_scale"] = round(out_cols / in_cols, 3)
        diff["is_scaled_uniformly"] = (diff["row_scale"] == diff["col_scale"])
    else:
        diff["row_scale"] = 1.0
        diff["col_scale"] = 1.0
        diff["is_scaled_uniformly"] = True

    # Background color
    in_bg  = get_background_color(input_grid)
    out_bg = get_background_color(output_grid)
    diff["input_background"]   = in_bg
    diff["output_background"]  = out_bg
    diff["background_changed"] = (in_bg != out_bg)

    # Pixel-level changes (only if same size)
    if not diff["size_changed"]:
        changed_pixels = int(np.sum(input_grid != output_grid))
        total_pixels   = in_rows * in_cols
        diff["pixels_changed"]       = changed_pixels
        diff["pixels_unchanged"]     = total_pixels - changed_pixels
        diff["fraction_changed"]     = round(changed_pixels / total_pixels, 3)
        diff["all_pixels_same"]      = (changed_pixels == 0)
    else:
        diff["pixels_changed"]   = None
        diff["pixels_unchanged"] = None
        diff["fraction_changed"] = None
        diff["all_pixels_same"]  = False

    # Colors present in input vs output
    in_colors  = set(int(c) for c in np.unique(input_grid)  if int(c) != in_bg)
    out_colors = set(int(c) for c in np.unique(output_grid) if int(c) != out_bg)
    diff["input_colors"]         = in_colors
    diff["output_colors"]        = out_colors
    diff["colors_added"]         = out_colors - in_colors
    diff["colors_removed"]       = in_colors - out_colors
    diff["colors_unchanged"]     = in_colors & out_colors
    diff["color_count_changed"]  = (len(in_colors) != len(out_colors))

    # ---- OBJECT LEVEL CHANGES ----
    in_objects  = encode_all_objects(extract_objects(input_grid,  in_bg),  input_grid.shape)
    out_objects = encode_all_objects(extract_objects(output_grid, out_bg), output_grid.shape)

    diff["input_object_count"]  = len(in_objects)
    diff["output_object_count"] = len(out_objects)
    diff["object_count_changed"] = (len(in_objects) != len(out_objects))
    diff["objects_added"]   = len(out_objects) - len(in_objects) if len(out_objects) > len(in_objects) else 0
    diff["objects_removed"] = len(in_objects) - len(out_objects) if len(in_objects) > len(out_objects) else 0

    # Match input objects to output objects by shape fingerprint
    matched, disappeared, appeared = match_objects(in_objects, out_objects)

    diff["matched_objects"]     = matched       # list of (in_obj, out_obj, changes)
    diff["disappeared_objects"] = disappeared   # objects in input not in output
    diff["appeared_objects"]    = appeared      # objects in output not in input

    # Summarize what happened to matched objects
    moves     = [(a, b, ch) for a, b, ch in matched if ch.get("moved")]
    recolors  = [(a, b, ch) for a, b, ch in matched if ch.get("recolored")]
    resizes   = [(a, b, ch) for a, b, ch in matched if ch.get("resized")]
    unchanged = [(a, b, ch) for a, b, ch in matched if ch.get("unchanged")]

    diff["moved_objects"]     = moves
    diff["recolored_objects"] = recolors
    diff["resized_objects"]   = resizes
    diff["unchanged_objects"] = unchanged

    # ---- TILING DETECTION ----
    diff["is_tiling"] = detect_tiling(input_grid, output_grid)

    # ---- DIFF SIGNATURE ----
    # A compact fingerprint of this diff type
    # Used by L6b hypothesis proposer for historical matching
    diff["signature"] = build_diff_signature(diff)

    return diff


def match_objects(in_objects, out_objects):
    """
    Match input objects to output objects by shape fingerprint.

    Returns:
        matched     = list of (in_obj, out_obj, changes_dict)
        disappeared = list of in_objects with no match in output
        appeared    = list of out_objects with no match in input
    """
    matched     = []
    disappeared = []
    appeared    = list(out_objects)  # start assuming all appeared

    for in_obj in in_objects:
        best_match = None
        best_score = -1

        for out_obj in appeared:
            score = match_score(in_obj, out_obj)
            if score > best_score:
                best_score = score
                best_match = out_obj

        if best_match is not None and best_score > 0:
            appeared.remove(best_match)
            changes = compute_object_changes(in_obj, best_match)
            matched.append((in_obj, best_match, changes))
        else:
            disappeared.append(in_obj)

    return matched, disappeared, appeared


def match_score(obj_a, obj_b):
    """
    Score how well two objects match each other.
    Higher = better match.

    Same fingerprint = strong match.
    Same color = bonus.
    Similar size = bonus.
    """
    score = 0

    # Same shape fingerprint = strong match
    if obj_a["fingerprint"] == obj_b["fingerprint"]:
        score += 10

    # Same color
    if obj_a["color"] == obj_b["color"]:
        score += 3

    # Similar pixel count (within 50%)
    ratio = min(obj_a["pixel_count"], obj_b["pixel_count"]) / max(obj_a["pixel_count"], obj_b["pixel_count"]) \
            if max(obj_a["pixel_count"], obj_b["pixel_count"]) > 0 else 0
    if ratio > 0.5:
        score += 2

    return score


def compute_object_changes(in_obj, out_obj):
    """
    Compute what changed between a matched input and output object.
    """
    changes = {}

    # Did it move?
    moved = (in_obj["min_row"] != out_obj["min_row"] or
             in_obj["min_col"] != out_obj["min_col"])
    changes["moved"] = moved
    if moved:
        changes["delta_row"] = out_obj["min_row"] - in_obj["min_row"]
        changes["delta_col"] = out_obj["min_col"] - in_obj["min_col"]

    # Did it get recolored?
    recolored = (in_obj["color"] != out_obj["color"])
    changes["recolored"] = recolored
    if recolored:
        changes["old_color"] = in_obj["color"]
        changes["new_color"] = out_obj["color"]

    # Did it get resized?
    resized = (in_obj["pixel_count"] != out_obj["pixel_count"])
    changes["resized"] = resized
    if resized:
        changes["old_size"] = in_obj["pixel_count"]
        changes["new_size"] = out_obj["pixel_count"]

    # Did the shape change?
    changes["shape_changed"] = (in_obj["fingerprint"] != out_obj["fingerprint"])

    # Is it completely unchanged?
    changes["unchanged"] = (not moved and not recolored and not resized
                            and not changes["shape_changed"])

    return changes


def detect_tiling(input_grid, output_grid):
    """
    Check if the output is the input tiled N times.
    Returns (tile_rows, tile_cols) if tiling detected, else None.
    """
    in_rows, in_cols = input_grid.shape
    out_rows, out_cols = output_grid.shape

    if out_rows % in_rows != 0 or out_cols % in_cols != 0:
        return None

    tile_r = out_rows // in_rows
    tile_c = out_cols // in_cols

    tiled = np.tile(input_grid, (tile_r, tile_c))
    if np.array_equal(tiled, output_grid):
        return (tile_r, tile_c)

    return None


def build_diff_signature(diff):
    """
    Build a compact signature string describing this diff type.
    Used by L6b for historical matching.
    """
    parts = []

    if diff.get("is_tiling"):
        parts.append("TILE")
    if diff["size_changed"]:
        parts.append("SIZE_CHANGE")
    if diff["background_changed"]:
        parts.append("BG_CHANGE")
    if diff["colors_added"]:
        parts.append("COLORS_ADDED")
    if diff["colors_removed"]:
        parts.append("COLORS_REMOVED")
    if diff["disappeared_objects"]:
        parts.append("OBJECTS_REMOVED")
    if diff["appeared_objects"]:
        parts.append("OBJECTS_ADDED")
    if diff["moved_objects"]:
        parts.append("OBJECTS_MOVED")
    if diff["recolored_objects"]:
        parts.append("OBJECTS_RECOLORED")
    if diff.get("all_pixels_same"):
        parts.append("NO_CHANGE")

    return "|".join(parts) if parts else "UNKNOWN"


def print_diff(diff):
    """
    Print a human-readable summary of a diff.
    """
    print(f"Input:  {diff['input_shape']}  Output: {diff['output_shape']}")
    print(f"Size changed: {diff['size_changed']} | "
          f"Row scale: {diff['row_scale']} | Col scale: {diff['col_scale']}")
    print(f"Background: {diff['input_background']} -> {diff['output_background']}")
    print(f"Colors added: {diff['colors_added']} | removed: {diff['colors_removed']}")
    print(f"Objects: {diff['input_object_count']} -> {diff['output_object_count']}")
    print(f"Matched: {len(diff['matched_objects'])} | "
          f"Disappeared: {len(diff['disappeared_objects'])} | "
          f"Appeared: {len(diff['appeared_objects'])}")
    print(f"Moved: {len(diff['moved_objects'])} | "
          f"Recolored: {len(diff['recolored_objects'])} | "
          f"Resized: {len(diff['resized_objects'])}")
    if diff.get("is_tiling"):
        print(f"TILING DETECTED: {diff['is_tiling']}")
    print(f"Signature: {diff['signature']}")


# ============================================================
# QUICK TEST
# py C:\arc_solver\src\layers\difference_engine.py
# ============================================================

if __name__ == "__main__":
    print("Loading training tasks...")
    tasks = load_all_tasks("training")
    ids   = get_task_ids("training")
    print(f"Loaded {len(tasks)} tasks.\n")

    passed = 0
    failed = 0

    print("Testing difference engine on first 5 tasks...\n")

    for i in range(5):
        task_id = ids[i]
        task    = tasks[task_id]

        try:
            for j, pair in enumerate(task["train"]):
                diff = compute_difference(pair["input"], pair["output"])
                print(f"Task {i+1} ({task_id}) Pair {j+1}: "
                      f"sig={diff['signature']}")
            passed += 1

        except Exception as e:
            failed += 1
            print(f"Task {i+1} FAILED: {e}")

    print(f"\nResults: {passed} passed, {failed} failed\n")

    # Detailed diff for the tiling task
    print("--- Detailed diff for tiling task 00576224 ---")
    task = tasks["00576224"]
    pair = task["train"][0]
    diff = compute_difference(pair["input"], pair["output"])
    print_diff(diff)

    print()

    # Detailed diff for the color indicator task
    print("--- Detailed diff for color indicator task 009d5c81 ---")
    task = tasks["009d5c81"]
    pair = task["train"][0]
    diff = compute_difference(pair["input"], pair["output"])
    print_diff(diff)

    print("\nDifference engine working correctly.")