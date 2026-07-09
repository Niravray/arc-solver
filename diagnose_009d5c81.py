"""
Run this on your machine: py diagnose_009d5c81.py
Needs task_loader, object_extractor, property_encoder, difference_engine,
spatial_color, verifier all importable (same sys.path setup as your other
files, so just drop this in C:\\arc_solver\\ or add the same path appends).

What this does differently from the normal pipeline run: it isolates
color_indicator_rule against 009d5c81 specifically and prints the exact
fail_reason plus the actual objects extracted, instead of just the
aggregate "fallback" you get from analyze_failures. This is the fastest
way to find out for certain whether the fix is "remove all non-largest
objects" or something else entirely, rather than guessing from the
Objects: 3 -> 1 count alone.
"""
import sys
sys.path.append(r"C:\arc_solver\src\layers")
sys.path.append(r"C:\arc_solver\src\primitives")

import numpy as np
from task_loader import load_all_tasks
from object_extractor import extract_objects, get_background_color
from property_encoder import encode_all_objects
from spatial_color import color_indicator_rule
from verifier import Hypothesis, verify_hypothesis

tasks = load_all_tasks("training")
task = tasks["009d5c81"]

print(f"Task 009d5c81 has {len(task['train'])} training pairs.\n")

for i, pair in enumerate(task["train"]):
    inp = pair["input"]
    out = pair["output"]
    bg = get_background_color(inp)
    objects = encode_all_objects(extract_objects(inp, bg), inp.shape)

    print(f"--- Pair {i+1} ---")
    print(f"Input shape: {inp.shape} | Output shape: {out.shape}")
    print(f"Background color: {bg}")
    print(f"Objects in input ({len(objects)}):")
    for o in objects:
        print(f"  id={o['id']} color={o['color']} pixel_count={o['pixel_count']} "
              f"bbox=({o['min_row']},{o['min_col']})-({o['max_row']},{o['max_col']})")

    out_bg = get_background_color(out)
    out_objects = encode_all_objects(extract_objects(out, out_bg), out.shape)
    print(f"Objects in expected output ({len(out_objects)}):")
    for o in out_objects:
        print(f"  id={o['id']} color={o['color']} pixel_count={o['pixel_count']}")

    result = color_indicator_rule(inp, objects, bg)
    if result is None:
        print("color_indicator_rule returned None on this pair "
              "(guard clause tripped: fewer than 2 objects, or smallest==largest).")
    else:
        match = np.array_equal(result, out)
        print(f"color_indicator_rule output matches expected exactly: {match}")
        if not match and result.shape == out.shape:
            diff_mask = (result != out)
            n_diff = int(diff_mask.sum())
            print(f"  {n_diff} pixels differ. First few mismatches "
                  f"(row, col, got, expected):")
            rows, cols = np.where(diff_mask)
            for r, c in list(zip(rows, cols))[:10]:
                print(f"    ({r},{c}): got {result[r,c]}, expected {out[r,c]}")
        elif result.shape != out.shape:
            print(f"  Shape mismatch: got {result.shape}, expected {out.shape}")
    print()

# Now run it through the real Hypothesis/verify_hypothesis path so the
# fail_reason string matches exactly what the pipeline sees.
hyp = Hypothesis(
    name="color_indicator",
    apply_fn=lambda grid, objs, d: color_indicator_rule(grid, objs, d["input_background"]),
    confidence=0.9,
)
verified = verify_hypothesis(hyp, task)
print(f"verify_hypothesis result: {verified}")
print(f"fail_reason: {hyp.fail_reason}")