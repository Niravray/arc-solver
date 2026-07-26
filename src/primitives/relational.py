import numpy as np


def move_until_touching(
    grid,
    moving_obj,
    target_obj,
    background=0,
    direction=None,
):
    """
    Move one object one cell at a time until it orthogonally touches target.

    Returns None if the chosen straight-line path exits the grid or collides
    with another object before contact.
    """
    if moving_obj is None or target_obj is None:
        return None
    if moving_obj["id"] == target_obj["id"]:
        return None

    moving_pixels = set(moving_obj["pixels"])
    target_pixels = set(target_obj["pixels"])
    if _sets_touch(moving_pixels, target_pixels):
        return grid.copy()

    step = _direction_step(direction)
    if step is None:
        row_delta = target_obj["centroid_row"] - moving_obj["centroid_row"]
        col_delta = target_obj["centroid_col"] - moving_obj["centroid_col"]
        if abs(row_delta) >= abs(col_delta) and row_delta != 0:
            step = (1 if row_delta > 0 else -1, 0)
        elif col_delta != 0:
            step = (0, 1 if col_delta > 0 else -1)
        else:
            return None

    rows, cols = grid.shape
    occupied = {
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if grid[row, col] != background and (row, col) not in moving_pixels
    }
    current = moving_pixels

    for _ in range(rows + cols):
        candidate = {
            (row + step[0], col + step[1])
            for row, col in current
        }
        if any(
            row < 0 or row >= rows or col < 0 or col >= cols
            for row, col in candidate
        ):
            return None
        if candidate & occupied:
            return None
        current = candidate
        if _sets_touch(current, target_pixels):
            result = grid.copy()
            for row, col in moving_pixels:
                result[row, col] = background
            for row, col in current:
                result[row, col] = moving_obj["color"]
            return result

    return None


def connect_aligned_objects(grid, obj_a, obj_b, color):
    """
    Draw a straight line through two objects with aligned centroids.

    Existing non-background pixels are preserved; the requested color fills
    the cells between the objects.
    """
    if obj_a is None or obj_b is None or obj_a["id"] == obj_b["id"]:
        return None

    result = grid.copy()
    if obj_a["centroid_row"] == obj_b["centroid_row"]:
        row = int(obj_a["centroid_row"])
        left, right = sorted(
            (obj_a, obj_b),
            key=lambda obj: obj["centroid_col"],
        )
        for col in range(left["max_col"] + 1, right["min_col"]):
            result[row, col] = color
        return result

    if obj_a["centroid_col"] == obj_b["centroid_col"]:
        col = int(obj_a["centroid_col"])
        top, bottom = sorted(
            (obj_a, obj_b),
            key=lambda obj: obj["centroid_row"],
        )
        for row in range(top["max_row"] + 1, bottom["min_row"]):
            result[row, col] = color
        return result

    return None


def apply_relational_primitive(
    operation,
    grid,
    graph,
    anchor_id,
    target_id,
    **parameters,
):
    """Bridge graph node bindings to relational grid transformations."""
    objects = {obj["id"]: obj for obj in graph["objects"]}
    anchor = objects.get(anchor_id)
    target = objects.get(target_id)
    if anchor is None or target is None:
        return None

    if operation == "move_until_touching":
        return move_until_touching(
            grid,
            anchor,
            target,
            background=parameters.get("background", 0),
            direction=parameters.get("direction"),
        )
    if operation == "connect_aligned":
        return connect_aligned_objects(
            grid,
            anchor,
            target,
            color=parameters.get("color", anchor["color"]),
        )
    raise ValueError(f"Unknown relational primitive: {operation}")


def _direction_step(direction):
    if direction is None:
        return None
    return {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }.get(direction)


def _sets_touch(first, second):
    for row, col in first:
        if any(
            neighbor in second
            for neighbor in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            )
        ):
            return True
    return False
