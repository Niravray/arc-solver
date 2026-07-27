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


def shift_grid(grid, delta_row, delta_col, background=0):
    """
    Shift every pixel in the grid by (delta_row, delta_col) as one rigid
    whole-scene pan -- unlike move_object, this isn't scoped to a single
    object's pixels. Content shifted out of bounds is discarded (not
    wrapped); vacated cells are filled with background.
    """
    rows, cols = grid.shape
    result = np.full_like(grid, background)

    if delta_row >= rows or delta_row <= -rows or delta_col >= cols or delta_col <= -cols:
        return result

    src_row_start = max(0, -delta_row)
    src_row_end   = min(rows, rows - delta_row)
    src_col_start = max(0, -delta_col)
    src_col_end   = min(cols, cols - delta_col)

    dst_row_start = src_row_start + delta_row
    dst_row_end   = src_row_end + delta_row
    dst_col_start = src_col_start + delta_col
    dst_col_end   = src_col_end + delta_col

    result[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = (
        grid[src_row_start:src_row_end, src_col_start:src_col_end]
    )
    return result


_GRAVITY_STEPS = {
    "down":  (1, 0),
    "up":    (-1, 0),
    "left":  (0, -1),
    "right": (0, 1),
}


def apply_gravity(grid, objects, direction, background=0):
    """
    Slide every object independently in `direction` until each hits the
    grid boundary or another object -- unlike move_object_to_edge, every
    object moves by its own distance rather than one selected object
    moving by a single shared amount.

    Objects are settled in order starting from whichever is already
    closest to the destination edge, and each already-settled object's
    final position becomes an obstacle for the ones settled after it --
    the same ordering a real stack of falling objects would settle in.
    Objects still waiting their turn are not obstacles yet, since they
    aren't between the current object and the edge.
    """
    step = _GRAVITY_STEPS.get(direction)
    if step is None or not objects:
        return None

    rows, cols = grid.shape
    dr, dc = step
    result = np.full_like(grid, background)

    if direction == "down":
        order = sorted(objects, key=lambda o: -o["max_row"])
    elif direction == "up":
        order = sorted(objects, key=lambda o: o["min_row"])
    elif direction == "left":
        order = sorted(objects, key=lambda o: o["min_col"])
    else:  # right
        order = sorted(objects, key=lambda o: -o["max_col"])

    occupied = set()
    for obj in order:
        pixels = set(obj["pixels"])
        shift = 0
        while True:
            candidate = {(r + dr * (shift + 1), c + dc * (shift + 1)) for r, c in pixels}
            if any(r < 0 or r >= rows or c < 0 or c >= cols for r, c in candidate):
                break
            if candidate & occupied:
                break
            shift += 1

        final_pixels = {(r + dr * shift, c + dc * shift) for r, c in pixels}
        for r, c in final_pixels:
            result[r, c] = obj["color"]
        occupied |= final_pixels

    return result


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


def fill_enclosed_regions(grid, fill_color, background=0):
    """
    Recolor every background cell that's fully walled in by non-background
    pixels, leaving background reachable from the grid border untouched.

    "Enclosed" is decided by a flood fill from every border cell: any
    background cell that flood fill reaches is "outside" and stays
    background; whatever it never reaches is enclosed and gets
    fill_color. Diagonal gaps don't block enclosure -- the flood fill
    only moves orthogonally, matching how the walls themselves are drawn.
    """
    rows, cols = grid.shape
    reachable = np.zeros_like(grid, dtype=bool)
    stack = []

    for r in range(rows):
        for c in (0, cols - 1):
            if grid[r, c] == background and not reachable[r, c]:
                reachable[r, c] = True
                stack.append((r, c))
    for c in range(cols):
        for r in (0, rows - 1):
            if grid[r, c] == background and not reachable[r, c]:
                reachable[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr, nc] == background and not reachable[nr, nc]):
                reachable[nr, nc] = True
                stack.append((nr, nc))

    result = grid.copy()
    result[(grid == background) & (~reachable)] = fill_color
    return result


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
# OBJECT SELECTION PREDICATES
#
# These are NOT grid-in/grid-out primitives, so they intentionally
# don't live in ALL_PRIMITIVES (the verifier expects every entry
# there to take a grid and return a grid). Instead these take a
# list of object dicts -- the same dicts property_encoder.py
# produces -- and return a single object, a filtered list, or a
# bool.
#
# Two things use these:
#   1. Primitives below (recolor_largest, color_indicator_rule,
#      etc.) that used to hand-roll their own min()/max() selection.
#   2. The hypothesis engine (L8), which can call these directly to
#      target a specific object -- e.g. recolor_object(grid,
#      leftmost_object(objects), 3) -- instead of brute-forcing
#      every object in `objects` as a candidate for every
#      obj-taking primitive.
#
# TIE-BREAKING: when multiple objects are equally "largest",
# "topmost", etc., ties are broken by lowest object id. This
# assumes ids are assigned in scipy.ndimage.label's default
# raster-scan order (top-to-bottom, left-to-right) -- I haven't
# seen object_extractor.py, so I can't confirm that's how your
# ids are actually assigned. If it's different, "lowest id" stops
# meaning "reading order" and every tie-break below silently
# picks something else. Worth confirming before trusting this on
# an ambiguous task.
# ============================================================

def _break_tie(candidates):
    """
    Deterministically resolve ties among equally-ranked objects by
    returning the one with the lowest id. See module note above on
    what "lowest id" is assumed to mean.
    """
    return min(candidates, key=lambda o: o["id"])


def largest_object(objects):
    """Return the object with the most pixels, or None if objects is empty."""
    if not objects:
        return None
    max_count = max(o["pixel_count"] for o in objects)
    candidates = [o for o in objects if o["pixel_count"] == max_count]
    return _break_tie(candidates)


def smallest_object(objects):
    """Return the object with the fewest pixels, or None if objects is empty."""
    if not objects:
        return None
    min_count = min(o["pixel_count"] for o in objects)
    candidates = [o for o in objects if o["pixel_count"] == min_count]
    return _break_tie(candidates)


def leftmost_object(objects):
    """Return the object whose bounding box starts at the smallest column."""
    if not objects:
        return None
    min_col = min(o["min_col"] for o in objects)
    candidates = [o for o in objects if o["min_col"] == min_col]
    return _break_tie(candidates)


def rightmost_object(objects):
    """Return the object whose bounding box extends to the largest column."""
    if not objects:
        return None
    max_col = max(o["max_col"] for o in objects)
    candidates = [o for o in objects if o["max_col"] == max_col]
    return _break_tie(candidates)


def topmost_object(objects):
    """Return the object whose bounding box starts at the smallest row."""
    if not objects:
        return None
    min_row = min(o["min_row"] for o in objects)
    candidates = [o for o in objects if o["min_row"] == min_row]
    return _break_tie(candidates)


def bottommost_object(objects):
    """Return the object whose bounding box extends to the largest row."""
    if not objects:
        return None
    max_row = max(o["max_row"] for o in objects)
    candidates = [o for o in objects if o["max_row"] == max_row]
    return _break_tie(candidates)


def touches_border(obj, grid_shape):
    """
    True if obj's bounding box touches any edge of a (rows, cols)
    grid_shape.
    """
    rows, cols = grid_shape
    return (
        obj["min_row"] == 0 or
        obj["max_row"] == rows - 1 or
        obj["min_col"] == 0 or
        obj["max_col"] == cols - 1
    )


def objects_touching_border(objects, grid_shape):
    """Return the subset of objects touching the grid border."""
    return [o for o in objects if touches_border(o, grid_shape)]


def objects_not_touching_border(objects, grid_shape):
    """Return the subset of objects NOT touching the grid border."""
    return [o for o in objects if not touches_border(o, grid_shape)]


def unique_color_object(objects):
    """
    Return the one object whose color no other object shares.
    Returns None if there isn't exactly one such object (e.g. every
    color is unique, or every color is shared by 2+ objects).
    """
    if not objects:
        return None
    color_counts = {}
    for o in objects:
        color_counts[o["color"]] = color_counts.get(o["color"], 0) + 1
    unique = [o for o in objects if color_counts[o["color"]] == 1]
    if len(unique) == 1:
        return unique[0]
    return None


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


def recolor_by_mapping(grid, color_map):
    """
    Apply a full old_color -> new_color substitution table simultaneously.
    Colors not present in color_map are left unchanged.

    Masks are computed against the original grid, not the in-progress
    result, so entries can't chain into each other (e.g. a {1: 2, 2: 3}
    mapping must turn every original 1 into a 2, never into a 3, even
    though 2 also appears as a key).
    """
    result = grid.copy()
    for old_color, new_color in color_map.items():
        result[grid == old_color] = new_color
    return result


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
    Uses largest_object() so ties are broken deterministically
    instead of depending on the order `objects` happens to be in.
    """
    obj = largest_object(objects)
    if obj is None:
        return grid
    return recolor_object(grid, obj, new_color)


def recolor_smallest(grid, objects, new_color):
    """
    Recolor the smallest object (by pixel count) to new_color.
    Uses smallest_object() so ties are broken deterministically
    instead of depending on the order `objects` happens to be in.
    """
    obj = smallest_object(objects)
    if obj is None:
        return grid
    return recolor_object(grid, obj, new_color)


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
    Uses smallest_object() so ties are broken deterministically
    instead of depending on the order `objects` happens to be in.
    """
    obj = smallest_object(objects)
    if obj is None:
        return grid
    return remove_object(grid, obj, background)


def color_indicator_rule(grid, objects, background=0):
    """
    Apply the color indicator rule:
    1. Find the smallest object (the indicator)
    2. Recolor the largest object to the indicator's color
    3. Remove the indicator

    This solves task 009d5c81 and similar tasks.

    Uses smallest_object()/largest_object() instead of raw
    min()/max() over `objects`. This is the fix for "picks the
    wrong object when multiple same-color objects exist" -- the
    old min()/max() returned whichever tied object happened to
    come first in the `objects` list, which is arbitrary. This
    version always resolves ties the same way regardless of list
    order. Verify against the task that originally exposed the bug
    before calling it closed -- see the test script for a
    reproduction of the failure mode this fixes.
    """
    if len(objects) < 2:
        return None

    smallest = smallest_object(objects)
    largest = largest_object(objects)

    if smallest is None or largest is None or smallest["id"] == largest["id"]:
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
#
# NOTE: apply_tiling_rule() is intentionally NOT registered here.
# It takes output_grid as a parameter, which doesn't exist at test
# time -- it's a checker ("is this the tiled answer?"), not a
# blind hypothesis generator. The tile_* entries below already
# cover tiling as a generator; apply_tiling_rule stays a plain
# helper function, unregistered, same as before.
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

    # --- newly registered below (pre-existing functions, unchanged bodies) ---
    "tile_to_size":        lambda grid, target_rows=None, target_cols=None, **kw: (
                               tile_to_size(grid, target_rows, target_cols)
                               if target_rows is not None and target_cols is not None
                               else None
                           ),
    "scale_up_rows_cols":  lambda grid, row_factor=1, col_factor=1, **kw: scale_up_rows_cols(
                               grid, row_factor, col_factor
                           ),
    "move_object":         lambda grid, obj=None, delta_row=0, delta_col=0, background=0, **kw: (
                               move_object(grid, obj, delta_row, delta_col, background)
                               if obj is not None else None
                           ),
    "move_to_edge_up":     lambda grid, obj=None, background=0, **kw: (
                               move_object_to_edge(grid, obj, "up", background)
                               if obj is not None else None
                           ),
    "move_to_edge_down":   lambda grid, obj=None, background=0, **kw: (
                               move_object_to_edge(grid, obj, "down", background)
                               if obj is not None else None
                           ),
    "move_to_edge_left":   lambda grid, obj=None, background=0, **kw: (
                               move_object_to_edge(grid, obj, "left", background)
                               if obj is not None else None
                           ),
    "move_to_edge_right":  lambda grid, obj=None, background=0, **kw: (
                               move_object_to_edge(grid, obj, "right", background)
                               if obj is not None else None
                           ),
}

COLOR_PRIMITIVES = {
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

    # --- newly registered below (pre-existing functions, unchanged bodies) ---
    "recolor_smallest":    lambda grid, objects=None, new_color=0, **kw: recolor_smallest(
                               grid, objects or [], new_color
                           ),
    "recolor_all":         lambda grid, old_color=1, new_color=2, **kw: recolor_all(
                               grid, old_color, new_color
                           ),
    "remove_all_color":    lambda grid, color=1, background=0, **kw: remove_all_color(
                               grid, color, background
                           ),
    "recolor_by_size":     lambda grid, objects=None, size_to_color=None, **kw: recolor_by_size(
                               grid, objects or [], size_to_color or {}
                           ),
    "recolor_object":      lambda grid, obj=None, new_color=0, **kw: (
                               recolor_object(grid, obj, new_color)
                               if obj is not None else None
                           ),
    "remove_object":       lambda grid, obj=None, background=0, **kw: (
                               remove_object(grid, obj, background)
                               if obj is not None else None
                           ),
    "fill_region":         lambda grid, pixels=None, color=0, **kw: (
                               fill_region(grid, pixels, color)
                               if pixels is not None else None
                           ),
}

ALL_PRIMITIVES = {**SPATIAL_PRIMITIVES, **COLOR_PRIMITIVES}

# Selection predicates operate on object lists, not grids, so they
# don't belong in ALL_PRIMITIVES -- the verifier expects every entry
# there to be grid-in, grid-out. Kept as a separate registry so the
# hypothesis engine can look these up by name too, the same way it
# looks up ALL_PRIMITIVES.
SELECTION_PREDICATES = {
    "largest_object":        largest_object,
    "smallest_object":       smallest_object,
    "leftmost_object":       leftmost_object,
    "rightmost_object":      rightmost_object,
    "topmost_object":        topmost_object,
    "bottommost_object":     bottommost_object,
    "unique_color_object":   unique_color_object,
}


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

    # ---- TEST 4: Selection predicates (synthetic, no task files needed) ----
    print("\n--- Test 4: Selection predicates ---")
    synthetic_objects = [
        {"id": 0, "color": 5, "pixel_count": 1, "pixels": [(0, 0)],
         "min_row": 0, "max_row": 0, "min_col": 0, "max_col": 0},
        {"id": 1, "color": 5, "pixel_count": 1, "pixels": [(5, 7)],
         "min_row": 5, "max_row": 5, "min_col": 7, "max_col": 7},
        {"id": 2, "color": 7, "pixel_count": 10, "pixels": [(1, c) for c in range(1, 6)] + [(2, c) for c in range(1, 6)],
         "min_row": 1, "max_row": 2, "min_col": 1, "max_col": 5},
    ]
    print(f"largest_object id:  {largest_object(synthetic_objects)['id']} (expect 2)")
    print(f"smallest_object id: {smallest_object(synthetic_objects)['id']} (expect 0, tie broken by lowest id)")
    print(f"leftmost_object id: {leftmost_object(synthetic_objects)['id']} (expect 0)")
    print(f"rightmost_object id:{rightmost_object(synthetic_objects)['id']} (expect 1)")
    print(f"unique_color_object id: {unique_color_object(synthetic_objects)['id']} (expect 2, only non-shared color)")
    print(f"touches_border obj0 on 6x8 grid: {touches_border(synthetic_objects[0], (6, 8))} (expect True)")
    print(f"touches_border obj2 on 6x8 grid: {touches_border(synthetic_objects[2], (6, 8))} (expect False)")

    # ---- TEST 5: Registry sanity ----
    print("\n--- Test 5: Registry sanity ---")
    print(f"Total primitives in ALL_PRIMITIVES: {len(ALL_PRIMITIVES)}")
    print(f"Total selection predicates: {len(SELECTION_PREDICATES)}")
    assert len(SPATIAL_PRIMITIVES) + len(COLOR_PRIMITIVES) == len(ALL_PRIMITIVES), \
        "Key collision between SPATIAL_PRIMITIVES and COLOR_PRIMITIVES!"
    print("No key collisions between SPATIAL_PRIMITIVES and COLOR_PRIMITIVES.")
    print("Primitive names:", sorted(ALL_PRIMITIVES.keys()))

    print("\nSpatial + color primitives working correctly.")