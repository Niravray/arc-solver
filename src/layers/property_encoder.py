import numpy as np
import sys

sys.path.append(r"C:\arc_solver\src\layers")
from task_loader import load_all_tasks, get_task_ids
from object_extractor import extract_objects, get_background_color

# ============================================================
# PROPERTY ENCODER
# Layer 3 of the ARC Solver Pipeline
#
# What this does:
# - Takes raw objects from the object extractor
# - Adds richer properties to each object:
#     symmetry (horizontal, vertical, diagonal)
#     shape fingerprint (normalized pattern)
#     border touching
#     is it a rectangle or irregular?
#     is it a single pixel?
#     is it a line (horizontal or vertical)?
#     mass density (how filled is the bounding box?)
# ============================================================


def encode_properties(obj, grid_shape):
    """
    Take a raw object dict from object_extractor
    and add rich properties to it.
    
    obj        = object dict from extract_objects()
    grid_shape = (rows, cols) of the full grid
    
    Returns the same object dict with new keys added.
    """
    grid = obj["grid"]  # small cropped grid for this object
    rows, cols = grid.shape
    grid_rows, grid_cols = grid_shape
    color = obj["color"]

    # ---- SYMMETRY ----

    # Horizontal symmetry: does the object look the same flipped top-bottom?
    is_sym_h = check_horizontal_symmetry(grid, color)

    # Vertical symmetry: does it look the same flipped left-right?
    is_sym_v = check_vertical_symmetry(grid, color)

    # Diagonal symmetry: does it look the same flipped along the diagonal?
    is_sym_d = check_diagonal_symmetry(grid, color)

    # ---- SHAPE TYPE ----

    pixel_count = obj["pixel_count"]
    height = obj["height"]
    width = obj["width"]
    bbox_area = height * width

    # Single pixel?
    is_single_pixel = (pixel_count == 1)

    # Horizontal line? (1 row, multiple cols, all filled)
    is_h_line = (height == 1 and width > 1 and pixel_count == width)

    # Vertical line? (multiple rows, 1 col, all filled)
    is_v_line = (width == 1 and height > 1 and pixel_count == height)

    # Rectangle? (bounding box completely filled)
    is_rectangle = (pixel_count == bbox_area)

    # Mass density: how much of the bounding box is filled?
    # 1.0 = perfect rectangle, lower = irregular shape
    mass_density = pixel_count / bbox_area if bbox_area > 0 else 0.0

    # ---- BORDER TOUCHING ----

    min_row = obj["min_row"]
    max_row = obj["max_row"]
    min_col = obj["min_col"]
    max_col = obj["max_col"]

    touches_top    = (min_row == 0)
    touches_bottom = (max_row == grid_rows - 1)
    touches_left   = (min_col == 0)
    touches_right  = (max_col == grid_cols - 1)
    touches_border = touches_top or touches_bottom or touches_left or touches_right

    # ---- POSITION CATEGORY ----
    # Where is this object in the grid? Useful for rules like "move to corner"
    center_row = grid_rows / 2
    center_col = grid_cols / 2
    centroid_row = obj["centroid_row"]
    centroid_col = obj["centroid_col"]

    position_v = "top"    if centroid_row < center_row else "bottom"
    position_h = "left"   if centroid_col < center_col else "right"
    position   = f"{position_v}_{position_h}"  # e.g. "top_left", "bottom_right"

    # ---- SHAPE FINGERPRINT ----
    # A normalized representation of the object's shape
    # Useful for comparing shapes across different tasks
    fingerprint = get_shape_fingerprint(grid, color)

    # ---- ADD ALL PROPERTIES TO THE OBJECT DICT ----
    obj["is_sym_h"]        = is_sym_h
    obj["is_sym_v"]        = is_sym_v
    obj["is_sym_d"]        = is_sym_d
    obj["is_single_pixel"] = is_single_pixel
    obj["is_h_line"]       = is_h_line
    obj["is_v_line"]       = is_v_line
    obj["is_rectangle"]    = is_rectangle
    obj["mass_density"]    = round(mass_density, 3)
    obj["touches_top"]     = touches_top
    obj["touches_bottom"]  = touches_bottom
    obj["touches_left"]    = touches_left
    obj["touches_right"]   = touches_right
    obj["touches_border"]  = touches_border
    obj["position"]        = position
    obj["fingerprint"]     = fingerprint

    return obj


def encode_all_objects(objects, grid_shape):
    """
    Encode properties for a list of objects.
    Returns the same list with properties added to each object.
    """
    return [encode_properties(obj, grid_shape) for obj in objects]


def check_horizontal_symmetry(grid, color):
    """
    Check if object is symmetric horizontally (top == bottom when flipped).
    """
    # Create binary mask (1 where object pixel exists)
    mask = (grid == color).astype(np.int32)
    flipped = np.flipud(mask)  # flip top-bottom
    return bool(np.array_equal(mask, flipped))


def check_vertical_symmetry(grid, color):
    """
    Check if object is symmetric vertically (left == right when flipped).
    """
    mask = (grid == color).astype(np.int32)
    flipped = np.fliplr(mask)  # flip left-right
    return bool(np.array_equal(mask, flipped))


def check_diagonal_symmetry(grid, color):
    """
    Check if object is symmetric along the main diagonal.
    Only possible for square objects.
    """
    rows, cols = grid.shape
    if rows != cols:
        return False  # can only be diagonally symmetric if square
    mask = (grid == color).astype(np.int32)
    transposed = mask.T
    return bool(np.array_equal(mask, transposed))


def get_shape_fingerprint(grid, color):
    """
    Create a normalized shape fingerprint for an object.
    
    The fingerprint is a tuple of 0s and 1s representing
    the object's shape normalized to a fixed size.
    
    Two objects with the same shape but different colors
    will have the same fingerprint.
    Two objects with the same shape but different sizes
    will have different fingerprints (size matters in ARC).
    """
    rows, cols = grid.shape
    # Create binary mask
    mask = (grid == color).astype(np.int32)
    # Flatten to a tuple for hashing/comparison
    return tuple(mask.flatten().tolist())


def same_shape(obj_a, obj_b):
    """
    Check if two objects have the same shape.
    Returns True if their fingerprints match.
    """
    return obj_a["fingerprint"] == obj_b["fingerprint"]


def same_size(obj_a, obj_b):
    """
    Check if two objects have the same pixel count.
    """
    return obj_a["pixel_count"] == obj_b["pixel_count"]


def same_color(obj_a, obj_b):
    """
    Check if two objects have the same color.
    """
    return obj_a["color"] == obj_b["color"]


def print_object_properties(obj):
    """
    Print all properties of an object. Useful for debugging.
    """
    print(f"Object {obj['id']}:")
    print(f"  color        = {obj['color']}")
    print(f"  pixel_count  = {obj['pixel_count']}")
    print(f"  size         = {obj['height']}x{obj['width']}")
    print(f"  position     = {obj['position']}")
    print(f"  bbox         = ({obj['min_row']},{obj['min_col']})-({obj['max_row']},{obj['max_col']})")
    print(f"  centroid     = ({obj['centroid_row']:.1f}, {obj['centroid_col']:.1f})")
    print(f"  mass_density = {obj['mass_density']}")
    print(f"  is_rectangle = {obj['is_rectangle']}")
    print(f"  is_h_line    = {obj['is_h_line']}")
    print(f"  is_v_line    = {obj['is_v_line']}")
    print(f"  is_single_px = {obj['is_single_pixel']}")
    print(f"  sym_h        = {obj['is_sym_h']}")
    print(f"  sym_v        = {obj['is_sym_v']}")
    print(f"  sym_d        = {obj['is_sym_d']}")
    print(f"  touches      = top:{obj['touches_top']} bottom:{obj['touches_bottom']} "
          f"left:{obj['touches_left']} right:{obj['touches_right']}")


# ============================================================
# QUICK TEST
# py C:\arc_solver\src\layers\property_encoder.py
# ============================================================

if __name__ == "__main__":
    print("Loading training tasks...")
    tasks = load_all_tasks("training")
    ids = get_task_ids("training")
    print(f"Loaded {len(tasks)} tasks.\n")

    passed = 0
    failed = 0

    print("Testing property encoder on first 5 tasks...\n")

    for i in range(5):
        task_id = ids[i]
        task = tasks[task_id]

        try:
            for j, pair in enumerate(task["train"]):
                grid = pair["input"]
                objects = extract_objects(grid)
                encoded = encode_all_objects(objects, grid.shape)

                print(f"Task {i+1} ({task_id}) Pair {j+1}: "
                      f"{len(encoded)} objects encoded")

            passed += 1

        except Exception as e:
            failed += 1
            print(f"Task {i+1} FAILED: {e}")

    print(f"\nResults: {passed} passed, {failed} failed\n")

    # Detailed breakdown of first task first pair
    print("--- Detailed property breakdown ---")
    first_task = tasks[ids[0]]
    first_grid = first_task["train"][0]["input"]
    print(f"Grid:\n{first_grid}\n")

    objects = extract_objects(first_grid)
    encoded = encode_all_objects(objects, first_grid.shape)

    for obj in encoded:
        print_object_properties(obj)
        print()

    print("Property encoder working correctly.")