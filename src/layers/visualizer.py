import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import os
from pathlib import Path

# Add the project root to path so we can import task_loader
sys.path.append(str(Path(__file__).resolve().parent))

# ============================================================
# ARC VISUALIZER
# Layer 2 of the ARC Solver Pipeline (display tool)
#
# What this does:
# - Takes a grid of numbers (0-9) and draws it as colored squares
# - Shows training pairs side by side (input -> output)
# - Lets you visually inspect any task
# ============================================================

# The official ARC color palette
# Each number 0-9 maps to a specific color
ARC_COLORS = [
    "#000000",  # 0 = black
    "#0074D9",  # 1 = blue
    "#FF4136",  # 2 = red
    "#2ECC40",  # 3 = green
    "#FFDC00",  # 4 = yellow
    "#AAAAAA",  # 5 = grey
    "#F012BE",  # 6 = pink/magenta
    "#FF851B",  # 7 = orange
    "#7FDBFF",  # 8 = light blue
    "#870C25",  # 9 = dark red
]

# Build a matplotlib colormap from the ARC colors
ARC_CMAP = mcolors.ListedColormap(ARC_COLORS)


def draw_grid(ax, grid, title=""):
    """
    Draw a single ARC grid on a matplotlib axis.
    
    ax    = the matplotlib subplot to draw on
    grid  = a 2D NumPy array of integers 0-9
    title = text to show above the grid
    """
    rows, cols = grid.shape

    # Draw the colored grid
    ax.imshow(
        grid,
        cmap=ARC_CMAP,
        vmin=0,
        vmax=9,
        interpolation="nearest"
    )

    # Draw grid lines so each cell is clearly separated
    ax.set_xticks([x - 0.5 for x in range(cols + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(rows + 1)], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    # Remove axis tick labels (the numbers on the edges)
    ax.set_xticks([])
    ax.set_yticks([])

    # Set the title
    if title:
        ax.set_title(title, fontsize=9, pad=4)


def show_task(task, max_pairs=4):
    """
    Display a full ARC task — all training pairs plus the test input.
    
    task      = a task dict from task_loader (has "train" and "test" keys)
    max_pairs = maximum number of training pairs to show
    """
    train_pairs = task["train"][:max_pairs]
    test_inputs = task["test"]
    task_id = task.get("id", "unknown")

    # Total columns: 2 per training pair (input + output) + 1 per test input
    n_train = len(train_pairs)
    n_test = len(test_inputs)
    total_cols = (n_train * 2) + n_test

    fig, axes = plt.subplots(1, total_cols, figsize=(total_cols * 2.5, 3))
    fig.suptitle(f"Task: {task_id}", fontsize=11, fontweight="bold")

    # Handle case where there's only one subplot
    if total_cols == 1:
        axes = [axes]

    col = 0

    # Draw each training pair
    for i, pair in enumerate(train_pairs):
        draw_grid(axes[col], pair["input"],  title=f"Train {i+1} Input")
        col += 1
        draw_grid(axes[col], pair["output"], title=f"Train {i+1} Output")
        col += 1

    # Draw each test input
    for i, test in enumerate(test_inputs):
        draw_grid(axes[col], test["input"], title=f"Test Input {i+1}")
        col += 1

    plt.tight_layout()
    plt.show()


def show_grid(grid, title="Grid"):
    """
    Display a single grid on its own. Useful for debugging.
    
    grid  = a 2D NumPy array
    title = title for the window
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    draw_grid(ax, grid, title=title)
    plt.tight_layout()
    plt.show()


def show_pair(input_grid, output_grid, title=""):
    """
    Display an input and output grid side by side.
    Useful for checking if your solver produced the right answer.
    
    input_grid  = 2D NumPy array
    output_grid = 2D NumPy array
    """
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    draw_grid(axes[0], input_grid,  title="Input")
    draw_grid(axes[1], output_grid, title="Output")
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show()


def show_prediction(input_grid, predicted_grid, correct_grid=None, title=""):
    """
    Display input, your prediction, and optionally the correct answer.
    Use this to visually check if your solver is working.
    
    input_grid     = the test input
    predicted_grid = what your solver produced
    correct_grid   = the actual answer (optional, for training tasks)
    """
    n_cols = 3 if correct_grid is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 3, 3))

    draw_grid(axes[0], input_grid,     title="Input")
    draw_grid(axes[1], predicted_grid, title="Your Prediction")

    if correct_grid is not None:
        # Check if prediction is correct
        is_correct = np.array_equal(predicted_grid, correct_grid)
        correct_title = "Correct Answer ✓" if is_correct else "Correct Answer ✗"
        draw_grid(axes[2], correct_grid, title=correct_title)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.show()


# ============================================================
# QUICK TEST - run this file directly to visualize tasks
# In Command Prompt: py C:\arc_solver\src\layers\visualizer.py
# ============================================================

if __name__ == "__main__":
    from task_loader import load_all_tasks, get_task_ids

    print("Loading training tasks...")
    tasks = load_all_tasks("training")
    ids = get_task_ids("training")

    print(f"Loaded {len(tasks)} tasks.")
    print("Showing first 3 tasks visually...")
    print("Close each window to see the next one.")

    # Show the first 3 tasks
    for i in range(3):
        task_id = ids[i]
        task = tasks[task_id]
        print(f"\nShowing task {i+1}: {task_id}")
        print(f"  Training pairs: {len(task['train'])}")
        print(f"  Input shape:  {task['train'][0]['input'].shape}")
        print(f"  Output shape: {task['train'][0]['output'].shape}")
        show_task(task)

    print("\nVisualizer is working correctly.")