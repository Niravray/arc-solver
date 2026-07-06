import sys
sys.path.append(r"C:\arc_solver\src\layers")
sys.path.append(r"C:\arc_solver\src\primitives")
from task_loader import load_all_tasks
from object_extractor import extract_objects, get_background_color
from property_encoder import encode_all_objects
from spatial_color import color_indicator_rule
import numpy as np

tasks = load_all_tasks("training")
task = tasks["009d5c81"]
inp = task["train"][0]["input"]
out = task["train"][0]["output"]
bg = get_background_color(inp)
objects = encode_all_objects(extract_objects(inp, bg), inp.shape)

print("Objects:")
for o in objects:
    print(f"  id={o['id']} color={o['color']} pixels={o['pixel_count']}")

result = color_indicator_rule(inp, objects, bg)
print("Match:", np.array_equal(result, out))
print("Result:")
print(result)
print("Expected:")
print(out)