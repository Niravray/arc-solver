"""
Is there anything for library learning to eat?

Decompose every solved program into its structural parts and count what
recurs. Library learning promotes recurring structure into new named
primitives, so if nothing appears often enough, there is nothing to
promote and the corpus needs to grow first.

Counts three levels, because they mean different things:
  - ATOMS: already named; recurrence here is not an abstraction
  - SELECTORS: whole predicates, e.g. "not_is_largest AND is_sym_h"
  - SKELETONS: the operator shape with its holes blanked, e.g.
    "Recolor(?, literal) THEN Draw(?, halo8, source)" -- THIS is the
    level where a genuinely new primitive would come from
"""
import sys
import collections
import json

sys.path.append(r"C:\arc_solver\src\layers")
sys.path.append(r"C:\arc_solver\src\primitives")

from task_loader import load_all_tasks
from synthesis import synthesize_program

tasks = load_all_tasks("training")
records = json.load(open(r"C:\arc_solver\logs\full_evaluation_results.json"))
solved = [r["task_id"] for r in records if r["correct"]]

print(f"solved tasks in the corpus: {len(solved)}")

programs = {}
for tid in solved:
    try:
        program, name = synthesize_program(tasks[tid])
    except Exception:
        continue
    if program is not None:
        programs[tid] = (program, name)

print(f"of those, re-derived an induced program for: {len(programs)}")
print("(the rest were solved by the sweep, which has no program structure "
      "to mine)\n")


def selector_text(program):
    sel = getattr(program, "selector", None)
    return str(sel) if sel is not None else None


def atoms_of(program):
    sel = getattr(program, "selector", None)
    if sel is None or not hasattr(sel, "atoms"):
        return []
    return [name for name, _fn in sel.atoms]


def skeleton(program):
    """Operator shape with the holes blanked out."""
    parts = []
    node = program
    while True:
        first = getattr(node, "first", None)
        if first is None:
            break
        parts.append(_leaf_skeleton(first))
        node = getattr(node, "second", node)
        if not hasattr(node, "first"):
            break
    parts.append(_leaf_skeleton(node))
    return " THEN ".join(p for p in parts if p)


def _leaf_skeleton(program):
    name = getattr(program, "operator_name", type(program).__name__)
    expr = getattr(program, "expr", None)
    form = getattr(program, "form_name", None)
    if form:
        return f"{name}(?, {form})"
    if expr is not None:
        kind = getattr(expr, "kind", None) or type(expr).__name__
        return f"{name}(?, {kind})"
    return f"{name}(?)"


atom_counts = collections.Counter()
selector_counts = collections.Counter()
skeleton_counts = collections.Counter()

for tid, (program, _name) in programs.items():
    for a in atoms_of(program):
        atom_counts[a] += 1
    s = selector_text(program)
    if s:
        selector_counts[s] += 1
    skeleton_counts[skeleton(program)] += 1

def report(title, counter, note):
    print("=" * 62)
    print(f"{title}  ({note})")
    print("=" * 62)
    repeats = [(k, n) for k, n in counter.most_common() if n >= 2]
    if not repeats:
        print("  nothing occurs more than once")
    for k, n in repeats[:15]:
        print(f"  {n:3d}  {k}")
    singles = sum(1 for _k, n in counter.items() if n == 1)
    print(f"  ... plus {singles} that occur exactly once\n")

report("ATOMS", atom_counts, "already named -- recurrence is not new")
report("WHOLE SELECTORS", selector_counts, "reusable concepts")
report("SKELETONS", skeleton_counts,
       "operator shape with holes blanked -- where a NEW primitive would come from")

top = skeleton_counts.most_common(1)
print("=" * 62)
if top and top[0][1] >= 3:
    print(f"VERDICT: the most common skeleton occurs {top[0][1]} times "
          f"-- there is something to abstract.")
else:
    n = top[0][1] if top else 0
    print(f"VERDICT: the most common skeleton occurs only {n} time(s).")
    print("Nothing recurs often enough to promote. The corpus is too thin;")
    print("library learning has nothing to eat yet.")
