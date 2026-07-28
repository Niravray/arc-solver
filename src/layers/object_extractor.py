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
from scipy import ndimage
import sys

from task_loader import load_all_tasks, get_task_ids

# ============================================================
# OBJECT EXTRACTOR
# Layer 2 of the ARC Solver Pipeline
#
# What this does:
# - Takes a grid of numbers (0-9)
# - Finds every distinct object (connected region of same color)
# - Returns a list of Object dicts with position, color, pixels
#
# An "object" = a group of pixels all the same color
# that are connected to each other (touching horizontally or vertically)
# ============================================================


def get_background_color(grid):
    """
    Detect the background color of a grid.
    The background is usually the most common color.
    
    Returns an integer 0-9.
    """
    # Count how many times each color appears
    values, counts = np.unique(grid, return_counts=True)
    
    # The most common color is the background
    background = values[np.argmax(counts)]
    return int(background)


def extract_objects(
    grid,
    background=None,
    connectivity=4,
    include_background=False,
):
    """
    Find all distinct objects in a grid.
    
    grid         = 2D NumPy array of integers 0-9
    background   = color to treat as background (ignored during extraction)
                   if None, auto-detected as most common color
    connectivity = 4 means objects connect up/down/left/right only
                   8 means objects also connect diagonally
    include_background = if True, extract components of the detected
                         background color as objects too
    
    Returns a list of object dicts. Each dict has:
        "id"         : unique integer ID for this object
        "color"      : integer 0-9
        "pixels"     : list of (row, col) tuples
        "pixel_count": number of pixels
        "min_row"    : top edge of bounding box
        "max_row"    : bottom edge of bounding box
        "min_col"    : left edge of bounding box
        "max_col"    : right edge of bounding box
        "height"     : bounding box height
        "width"      : bounding box width
        "centroid_row": center row (float)
        "centroid_col": center col (float)
        "grid"       : small NumPy array just containing this object
                       (cropped to bounding box, background filled with -1)
    """
    if background is None:
        background = get_background_color(grid)

    rows, cols = grid.shape
    objects = []
    obj_id = 0

    # Get all unique colors in the grid except background
    unique_colors = [
        int(c)
        for c in np.unique(grid)
        if include_background or int(c) != background
    ]

    for color in unique_colors:
        # Create a binary mask: 1 where this color exists, 0 elsewhere
        mask = (grid == color).astype(np.int32)

        # Use scipy to find connected components in this mask
        # connectivity=1 means 4-directional, connectivity=2 means 8-directional
        struct = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
        labeled, num_features = ndimage.label(mask, structure=struct)

        # Each labeled region is a separate object
        for label_id in range(1, num_features + 1):
            # Find all pixels belonging to this object
            pixel_coords = np.argwhere(labeled == label_id)
            pixels = [(int(r), int(c)) for r, c in pixel_coords]

            if len(pixels) == 0:
                continue

            # Compute bounding box
            pixel_array = np.array(pixels)
            min_row = int(pixel_array[:, 0].min())
            max_row = int(pixel_array[:, 0].max())
            min_col = int(pixel_array[:, 1].min())
            max_col = int(pixel_array[:, 1].max())

            height = max_row - min_row + 1
            width  = max_col - min_col + 1

            # Compute centroid (center of mass)
            centroid_row = float(pixel_array[:, 0].mean())
            centroid_col = float(pixel_array[:, 1].mean())

            # Build a small cropped grid just for this object
            # Fill with -1 (meaning "not part of this object")
            obj_grid = np.full((height, width), -1, dtype=np.int32)
            for (r, c) in pixels:
                obj_grid[r - min_row, c - min_col] = color

            objects.append({
                "id":          obj_id,
                "color":       color,
                "pixels":      pixels,
                "pixel_count": len(pixels),
                "min_row":     min_row,
                "max_row":     max_row,
                "min_col":     min_col,
                "max_col":     max_col,
                "height":      height,
                "width":       width,
                "centroid_row": centroid_row,
                "centroid_col": centroid_col,
                "grid":        obj_grid,
                "background":  background,
            })

            obj_id += 1

    return objects


def extract_objects_multi_strategy(grid):
    """
    Extract objects using multiple strategies in parallel.
    Returns a dict with results from each strategy.
    
    This handles ambiguous grids where it's unclear how to interpret objects.
    The hypothesis engine will try each interpretation separately.
    
    Strategies:
        "connected_4"  : standard 4-directional connectivity
        "connected_8"  : 8-directional connectivity (diagonals count)
        "by_color"     : treat each color as one single object (ignores connectivity)
        "no_background": same as connected_4 but keeps background color objects too
    """
    background = get_background_color(grid)

    strategies = {}

    # Strategy 1: standard 4-directional
    strategies["connected_4"] = extract_objects(grid, background=background, connectivity=4)

    # Strategy 2: 8-directional (diagonals connect)
    strategies["connected_8"] = extract_objects(grid, background=background, connectivity=8)

    # Strategy 3: each color is one object regardless of connectivity
    strategies["by_color"] = extract_by_color(grid, background=background)

    # Strategy 4: no background filtering - treat everything as objects
    strategies["no_background"] = extract_objects(
        grid,
        background=background,
        connectivity=4,
        include_background=True,
    )

    return strategies


def extract_by_color(grid, background=None):
    """
    Treat each unique color as a single object.
    Ignores connectivity — all pixels of the same color = one object.
    """
    if background is None:
        background = get_background_color(grid)

    objects = []
    obj_id = 0
    unique_colors = [int(c) for c in np.unique(grid) if int(c) != background]

    for color in unique_colors:
        pixel_coords = np.argwhere(grid == color)
        pixels = [(int(r), int(c)) for r, c in pixel_coords]

        if len(pixels) == 0:
            continue

        pixel_array = np.array(pixels)
        min_row = int(pixel_array[:, 0].min())
        max_row = int(pixel_array[:, 0].max())
        min_col = int(pixel_array[:, 1].min())
        max_col = int(pixel_array[:, 1].max())

        height = max_row - min_row + 1
        width  = max_col - min_col + 1

        centroid_row = float(pixel_array[:, 0].mean())
        centroid_col = float(pixel_array[:, 1].mean())

        obj_grid = np.full((height, width), -1, dtype=np.int32)
        for (r, c) in pixels:
            obj_grid[r - min_row, c - min_col] = color

        objects.append({
            "id":          obj_id,
            "color":       color,
            "pixels":      pixels,
            "pixel_count": len(pixels),
            "min_row":     min_row,
            "max_row":     max_row,
            "min_col":     min_col,
            "max_col":     max_col,
            "height":      height,
            "width":       width,
            "centroid_row": centroid_row,
            "centroid_col": centroid_col,
            "grid":        obj_grid,
            "background":  background,
        })

        obj_id += 1

    return objects


def print_objects(objects, grid_shape=None):
    """
    Print a human-readable summary of extracted objects.
    Useful for debugging.
    """
    print(f"Found {len(objects)} objects:")
    for obj in objects:
        print(f"  Object {obj['id']}: color={obj['color']} "
              f"pixels={obj['pixel_count']} "
              f"bbox=({obj['min_row']},{obj['min_col']})-({obj['max_row']},{obj['max_col']}) "
              f"size={obj['height']}x{obj['width']} "
              f"centroid=({obj['centroid_row']:.1f},{obj['centroid_col']:.1f})")


# ============================================================
# QUICK TEST
# py C:\arc_solver\src\layers\object_extractor.py
# ============================================================

if __name__ == "__main__":
    print("Loading training tasks...")
    tasks = load_all_tasks("training")
    ids = get_task_ids("training")

    print(f"Loaded {len(tasks)} tasks.")
    print("\nTesting object extractor on first 5 tasks...\n")

    passed = 0
    failed = 0

    for i in range(5):
        task_id = ids[i]
        task = tasks[task_id]

        print(f"Task {i+1}: {task_id}")

        try:
            for j, pair in enumerate(task["train"]):
                input_grid  = pair["input"]
                output_grid = pair["output"]

                bg = get_background_color(input_grid)
                input_objects  = extract_objects(input_grid)
                output_objects = extract_objects(output_grid)

                print(f"  Pair {j+1}: input {input_grid.shape} -> "
                      f"{len(input_objects)} objects | "
                      f"output {output_grid.shape} -> "
                      f"{len(output_objects)} objects | "
                      f"background color: {bg}")

            passed += 1
            print(f"  PASS")

        except Exception as e:
            failed += 1
            print(f"  FAIL: {e}")

        print()

    print(f"Results: {passed} passed, {failed} failed out of 5 tasks")

    # Show a detailed breakdown of one task
    print("\n--- Detailed breakdown of first task ---")
    first_task = tasks[ids[0]]
    first_input = first_task["train"][0]["input"]
    print(f"Grid:\n{first_input}")
    print()
    objects = extract_objects(first_input)
    print_objects(objects)
