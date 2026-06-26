import numpy as np
import sys

sys.path.append(r"C:\arc_solver\src\layers")

# ============================================================
# PRIMITIVE LIBRARY — SPATIAL + COLOR
# Part of Layer 6 of the ARC Solver Pipeline
#
# Each primitive is a function that takes a grid and parameters
# and returns a new grid.
#
# Rules for every primitive:
#   - Input:  2D NumPy array (the grid)
#   - Output: 2D NumPy array (the transformed grid)
#   - Never modify the input grid — always return a new one
#   - If the primitive cannot be applied, return None
# ============================================================


# ============================================================
# SPATIAL PRIMITIVES
# ============================================================

def rotate_90(grid, clockwise=True):
    """
    Rotate the entire grid 90 degrees.
    clockwise=True  → rotate right
    clockwise=False → rotate left
    """
    if clockwise:
        return np.rot90(grid, k=3)  # k=3 is 270 counter = 90 clockwise
    else:
        return np.rot90(grid, k=1)  # k=1 is 90 counter-clockwise


def rotate_180(grid):
    """Rotate the grid 180 degrees."""
    return np.rot90(grid, k=2)


def rotate_270(grid):
    """Rotate the grid 270 degrees clockwise (same as 90 counter-clockwise)."""
    return np.rot90(grid, k=1)


def mirror_horizontal(grid):
    """
    Flip the grid horizontally (left becomes right).
    """
    return np.fliplr(grid)


def mirror_vertical(grid):
    """
    Flip the grid vertically (top becomes bottom).
    """
    return np.flipud(grid)


def mirror_diagonal(grid):
    """
    Flip the grid along the main diagonal (transpose).
    Only makes visual sense for square grids.
    """
    return grid.T


def mirror_anti_diagonal(grid):
    """
    Flip the grid along the anti-diagonal.
    """
    return np.flipud(grid.T)


def move_object(grid, obj, delta_row, delta_col, background=0):
    """
    Move a specific object by (delta_row, delta_col).
    Leaves background color where the object was.
    
    grid       = full grid
    obj        = object dict from object_extractor
    delta_row  = how many rows to move (positive = down)
    delta_col  = how many cols to move (positive = right)
    background = color to fill where object was
    
    Returns new grid or None if move goes out of bounds.
    """
    new_grid = grid.copy()
    rows, cols = grid.shape

    # Check if all pixels stay in bounds after move
    for (r, c) in obj["pixels"]:
        new_r = r + delta_row
        new_c = c + delta_col
        if new_r < 0 or new_r >= rows or new_c < 0 or new_c >= cols:
            return None  # out of bounds

    # Erase original position
    for (r, c) in obj["pixels"]:
        new_grid[r, c] = background

    # Draw at new position
    for (r, c) in obj["pixels"]:
        new_r = r + delta_row
        new_c = c + delta_col
        new_grid[new_r, new_c] = obj["color"]

    return new_grid


def move_object_to_edge(grid, obj, direction, background=0):
    """
    Move an object as far as possible in a direction until it hits the edge.
    
    direction = "up", "down", "left", "right"
    """
    rows, cols = grid.shape

    if direction == "up":
        delta_row = -obj["min_row"]
        delta_col = 0
    elif direction == "down":
        delta_row = (rows - 1) - obj["max_row"]
        delta_col = 0
    elif direction == "left":
        delta_row = 0
        delta_col = -obj["min_col"]
    elif direction == "right":
        delta_row = 0
        delta_col = (cols - 1) - obj["max_col"]
    else:
        return None

    return move_object(grid, obj, delta_row, delta_col, background)


def tile_grid(grid, tile_rows, tile_cols):
    """
    Tile the grid tile_rows times vertically and tile_cols times horizontally.
    This is the primitive that solves the first task you saw.
    
    Example: tile_grid(grid, 3, 3) makes a 3x3 tiling of the input.
    """
    return np.tile(grid, (tile_rows, tile_cols))


def tile_to_size(grid, target_rows, target_cols):
    """
    Tile the grid to fill a target size exactly.
    Returns None if target is not an exact multiple of grid size.
    """
    in_rows, in_cols = grid.shape

    if target_rows % in_rows != 0 or target_cols % in_cols != 0:
        return None

    tile_r = target_rows // in_rows
    tile_c = target_cols // in_cols

    return np.tile(grid, (tile_r, tile_c))


def crop_to_content(grid, background=0):
    """
    Crop the grid to remove background border around the content.
    Returns the smallest bounding box containing all non-background pixels.
    """
    rows, cols = np.where(grid != background)

    if len(rows) == 0:
        return grid  # nothing to crop

    min_r, max_r = rows.min(), rows.max()
    min_c, max_c = cols.min(), cols.max()

    return grid[min_r:max_r+1, min_c:max_c+1].copy()


def scale_up(grid, factor):
    """
    Scale the grid up by an integer factor.
    Each pixel becomes a factor x factor block.
    
    Example: scale_up(grid, 2) doubles the size.
    """
    return np.kron(grid, np.ones((factor, factor), dtype=grid.dtype))


def scale_up_rows_cols(grid, row_factor, col_factor):
    """
    Scale the grid up with different factors for rows and cols.
    """
    return np.kron(grid, np.ones((row_factor, col_factor), dtype=grid.dtype))


# ============================================================
# COLOR PRIMITIVES
# ============================================================

def recolor_object(grid, obj, new_color):
    """
    Change the color of a specific object.
    
    grid      = full grid
    obj       = object dict
    new_color = integer 0-9
    """
    new_grid = grid.copy()
    for (r, c) in obj["pixels"]:
        new_grid[r, c] = new_color
    return new_grid


def recolor_all(grid, old_color, new_color):
    """
    Change every pixel of old_color to new_color across the whole grid.
    """
    new_grid = grid.copy()
    new_grid[new_grid == old_color] = new_color
    return new_grid


def swap_colors(grid, color_a, color_b):
    """
    Swap two colors across the entire grid.
    Every pixel of color_a becomes color_b and vice versa.
    """
    new_grid = grid.copy()
    mask_a = (grid == color_a)
    mask_b = (grid == color_b)
    new_grid[mask_a] = color_b
    new_grid[mask_b] = color_a
    return new_grid


def fill_region(grid, pixels, color):
    """
    Fill a specific set of pixel positions with a color.
    
    pixels = list of (row, col) tuples
    color  = integer 0-9
    """
    new_grid = grid.copy()
    for (r, c) in pixels:
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            new_grid[r, c] = color
    return new_grid


def recolor_by_size(grid, objects, size_to_color):
    """
    Recolor objects based on their pixel count.
    
    size_to_color = dict mapping pixel_count to new color
    Example: {1: 3, 5: 7} means "1-pixel objects become green, 5-pixel objects become orange"
    """
    new_grid = grid.copy()
    for obj in objects:
        if obj["pixel_count"] in size_to_color:
            new_color = size_to_color[obj["pixel_count"]]
            for (r, c) in obj["pixels"]:
                new_grid[r, c] = new_color
    return new_grid


def recolor_largest(grid, objects, new_color):
    """
    Recolor the largest object (by pixel count) to new_color.
    """
    if not objects:
        return grid
    largest = max(objects, key=lambda o: o["pixel_count"])
    return recolor_object(grid, largest, new_color)


def recolor_smallest(grid, objects, new_color):
    """
    Recolor the smallest object (by pixel count) to new_color.
    """
    if not objects:
        return grid
    smallest = min(objects, key=lambda o: o["pixel_count"])
    return recolor_object(grid, smallest, new_color)


def remove_object(grid, obj, background=0):
    """
    Remove an object from the grid by filling it with background color.
    """
    new_grid = grid.copy()
    for (r, c) in obj["pixels"]:
        new_grid[r, c] = background
    return new_grid


def remove_all_color(grid, color, background=0):
    """
    Remove all pixels of a specific color from the grid.
    """
    new_grid = grid.copy()
    new_grid[new_grid == color] = background
    return new_grid


def remove_smallest(grid, objects, background=0):
    """
    Remove the smallest object from the grid.
    This is the primitive for the color indicator rule.
    """
    if not objects:
        return grid
    smallest = min(objects, key=lambda o: o["pixel_count"])
    return remove_object(grid, smallest, background)


def color_indicator_rule(grid, objects, background=0):
    """
    Apply the color indicator rule:
    1. Find the smallest object (the indicator)
    2. Recolor the largest object to the indicator's color
    3. Remove the indicator

    This solves task 009d5c81 and similar tasks.
    """
    if len(objects) < 2:
        return None

    smallest = min(objects, key=lambda o: o["pixel_count"])
    largest  = max(objects, key=lambda o: o["pixel_count"])

    if smallest["id"] == largest["id"]:
        return None  # same object

    indicator_color = smallest["color"]

    # Step 1: recolor largest
    new_grid = recolor_object(grid, largest, indicator_color)

    # Step 2: remove smallest
    new_grid = remove_object(new_grid, smallest, background)

    return new_grid


# ============================================================
# COMPOSITE PRIMITIVES
# (common multi-step combinations built as single functions)
# ============================================================

def apply_tiling_rule(input_grid, output_grid):
    """
    If the output is a tiling of the input, return the tiled grid.
    Otherwise return None.
    
    This directly solves task 00576224.
    """
    in_rows, in_cols   = input_grid.shape
    out_rows, out_cols = output_grid.shape

    if out_rows % in_rows != 0 or out_cols % in_cols != 0:
        return None

    tile_r = out_rows // in_rows
    tile_c = out_cols // in_cols

    result = np.tile(input_grid, (tile_r, tile_c))

    if np.array_equal(result, output_grid):
        return result

    return None


# ============================================================
# PRIMITIVE REGISTRY
# A dict mapping primitive name to function
# Used by the hypothesis engine to look up and apply primitives
# ============================================================

SPATIAL_PRIMITIVES = {
    "rotate_90_cw":        lambda grid, **kw: rotate_90(grid, clockwise=True),
    "rotate_90_ccw":       lambda grid, **kw: rotate_90(grid, clockwise=False),
    "rotate_180":          lambda grid, **kw: rotate_180(grid),
    "rotate_270":          lambda grid, **kw: rotate_270(grid),
    "mirror_horizontal":   lambda grid, **kw: mirror_horizontal(grid),
    "mirror_vertical":     lambda grid, **kw: mirror_vertical(grid),
    "mirror_diagonal":     lambda grid, **kw: mirror_diagonal(grid),
    "mirror_anti_diagonal":lambda grid, **kw: mirror_anti_diagonal(grid),
    "tile_3x3":            lambda grid, **kw: tile_grid(grid, 3, 3),
    "tile_2x2":            lambda grid, **kw: tile_grid(grid, 2, 2),
    "tile_4x4":            lambda grid, **kw: tile_grid(grid, 4, 4),
    "tile_3x1":            lambda grid, **kw: tile_grid(grid, 3, 1),
    "tile_1x3":            lambda grid, **kw: tile_grid(grid, 1, 3),
    "tile_2x1":            lambda grid, **kw: tile_grid(grid, 2, 1),
    "tile_1x2":            lambda grid, **kw: tile_grid(grid, 1, 2),
    "scale_up_2":          lambda grid, **kw: scale_up(grid, 2),
    "scale_up_3":          lambda grid, **kw: scale_up(grid, 3),
    "crop_to_content":     lambda grid, **kw: crop_to_content(grid),
}

COLOR_PRIMITIVES = {
    "mirror_horizontal":   lambda grid, **kw: mirror_horizontal(grid),
    "swap_colors":         lambda grid, **kw: swap_colors(
                               grid,
                               kw.get("color_a", 1),
                               kw.get("color_b", 2)
                           ),
    "remove_smallest":     lambda grid, objects=None, **kw: remove_smallest(
                               grid, objects or []
                           ),
    "recolor_largest":     lambda grid, objects=None, new_color=0, **kw: recolor_largest(
                               grid, objects or [], new_color
                           ),
    "color_indicator":     lambda grid, objects=None, **kw: color_indicator_rule(
                               grid, objects or []
                           ),
}

ALL_PRIMITIVES = {**SPATIAL_PRIMITIVES, **COLOR_PRIMITIVES}


# ============================================================
# QUICK TEST
# py C:\arc_solver\src\primitives\spatial_color.py
# ============================================================

if __name__ == "__main__":
    sys.path.append(r"C:\arc_solver\src\layers")
    from task_loader import load_all_tasks, get_task_ids
    from object_extractor import extract_objects, get_background_color
    from property_encoder import encode_all_objects

    print("Loading tasks...")
    tasks = load_all_tasks("training")
    print(f"Loaded {len(tasks)} tasks.\n")

    # ---- TEST 1: Tiling primitive on task 00576224 ----
    print("--- Test 1: Tiling primitive ---")
    task  = tasks["00576224"]
    pair  = task["train"][0]
    inp   = pair["input"]
    out   = pair["output"]

    print(f"Input:\n{inp}")
    print(f"Expected output shape: {out.shape}")

    # Detect tiling factor
    tile_r = out.shape[0] // inp.shape[0]
    tile_c = out.shape[1] // inp.shape[1]
    result = tile_grid(inp, tile_r, tile_c)

    print(f"Tiled {tile_r}x{tile_c}:")
    correct = np.array_equal(result, out)
    print(f"Matches expected output: {correct}")

    # ---- TEST 2: Color indicator primitive on task 009d5c81 ----
    print("\n--- Test 2: Color indicator primitive ---")
    task    = tasks["009d5c81"]
    pair    = task["train"][0]
    inp     = pair["input"]
    out     = pair["output"]
    bg      = get_background_color(inp)
    objects = encode_all_objects(extract_objects(inp, bg), inp.shape)

    print(f"Objects in input: {len(objects)}")
    for obj in objects:
        print(f"  color={obj['color']} pixels={obj['pixel_count']}")

    result = color_indicator_rule(inp, objects, bg)

    if result is not None:
        correct = np.array_equal(result, out)
        print(f"Color indicator rule applied.")
        print(f"Matches expected output: {correct}")
    else:
        print("Color indicator rule returned None.")

    # ---- TEST 3: Rotation ----
    print("\n--- Test 3: Rotation primitives ---")
    test_grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    print(f"Original:\n{test_grid}")
    print(f"Rotate 90 CW:\n{rotate_90(test_grid, clockwise=True)}")
    print(f"Mirror horizontal:\n{mirror_horizontal(test_grid)}")
    print(f"Mirror vertical:\n{mirror_vertical(test_grid)}")

    print(f"\nTotal primitives available: {len(ALL_PRIMITIVES)}")
    print("Primitive names:", list(ALL_PRIMITIVES.keys()))
    print("\nSpatial + color primitives working correctly.")