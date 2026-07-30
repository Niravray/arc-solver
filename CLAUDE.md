# ARC Solver — working context

A from-scratch solver for **ARC-AGI-2** (1000 training / 120 public eval /
240 test). No ML: symbolic rule induction plus a hypothesis sweep.
Python 3.14, NumPy, SciPy. Run everything with `py`, not `python`.

## Current numbers

```
split         tasks  solved   accuracy
training       1000      64      6.40%
evaluation      120       0      0.00%
```

Induction precision 90% (52 rules offered, 47 correct). 46 tests passing.
A clean full run is ~900s training + ~150s eval.

**The evaluation number is the one that counts** — Kaggle's test set comes
from that distribution, not training's. Training is the iteration target
because it's the only split giving signal; report both, always.

## Commands

```bash
py src/layers/solver.py --full-eval      # both splits + technical report
py src/layers/solver.py --training-only  # training only (legacy path)
py -m unittest discover -s tests         # 46 tests
py kaggle_notebook.py                    # build submission.json (240 tasks, ~76s)
```

## Architecture

Two independent solvers, merged rather than competing. They overlap on
**one task out of 64**, and ARC allows two attempts, so both stay:

- **`src/layers/synthesis.py`** (2582 lines) — the main engine. Solves 47.
  Induces a rule per task: defines a *language* of expressions with
  *holes*, then solves the holes from the observed change. Counterexample-
  guided — induce from pair 1, test on pair 2, absorb any pair that
  disproves the rule into the evidence and re-induce.
  Operators: Recolor, Remove, Move, Draw, Extract, plus 14 composed.
  Attempt 1 comes from here.
- **`src/layers/verifier.py`** — the older sweep. Solves 17. Generates a
  list of complete pre-parameterised rules and keeps whatever verifies.
  Attempt 2 comes from here.

Supporting layers: `task_loader` → `object_extractor` → `property_encoder`
→ `relationship_graph` → `difference_engine`, plus
`src/primitives/spatial_color.py` (the sweep's primitive library).

### Scene interpretations

"What counts as an object" is *searched over*, not assumed: `connected4`,
`connected8`, `by_color`, `no_background`, `panels`. Each is gated by a
cheap applicability check — trying all of them on every task once cost
12.6× runtime for nothing, because most grids read identically under
several. A winning interpretation is deliberately left **active** so the
rule is applied to the test grid under the same reading it was induced
under.

## Deliberately disabled, each with its measurement in-code

Do not flip these without re-measuring; a test pins them.

| Flag | Why |
|---|---|
| `USE_RELATIONAL_ATOMS = False` | 0 new solves, Recolor precision 75% → 64% |
| `USE_REGION_SPLIT = False` | fires on 180 tasks, produces 0 rules, 3.3× runtime |
| `MAX_CANDIDATE_RESTARTS = 1` | 0 gain; the loop is evidence-driven, so restarts converge |
| `MAX_CONJUNCTION = 2` | depth 3 found 1 selector in 43 tries, both programs wrong |

`src/layers/graph_difference.py` is built, tested, and intentionally not
wired in — see its docstring.

## Rules learned the hard way

1. **Measure the prize before building for it.** Bucket size has
   overestimated yield ~10× every single time. `COLORS_ADDED` was 106
   tasks → 2 solves. Extract: 230 → 15 form matches → 12 observable → 5.
2. **But weigh it against marginal cost.** The 16-form Draw probe came
   back a flat long tail with nothing above 11 and yielded +12, because
   each form was ~10 lines once the machinery existed.
3. **A measured ceiling expires when the system changes.** "Composition is
   worth 13 tasks" was measured with 3 operators; with 5 it was 68.
4. **A net-zero result can hide a regression.** 53 → 53 concealed
   Extract +2 / Move −2. Only the per-operator breakdown caught it.
5. **Restricting a language can beat extending it.** Forbidding geometric
   decision lists from metric keys took Move 29% → 100% precision.
6. **Check overlap before replacing anything.** Synthesis and the sweep
   overlap on 1 task; replacing either would have thrown away ~17.
7. **Environment before code on timing.** Misattributed machine
   contention/throttling to code three times; one run read 38012s vs 914s
   clean, a 42× swing on identical code.
8. **Does this let the solver SEE more, or DO more?** Only the second has
   ever paid. Panels, raised guards, and region splitting each made
   structure visible and solved ~nothing.

## Why evaluation is 0, precisely

Measured, not guessed:

- 102 of 120 tasks — **no operator can even observe** the change
- of the 18 it can, **all 18 fail to explain even one training pair**
- the 14 single-concept eval tasks: **0 solved**, so it isn't depth
- threw an hour of search at one task: **51 combinations died in 1 second**,
  all at observation. Compute is not the constraint
- 90% of eval tasks have repeated shapes (template reasoning); 37% need 3+
  concepts vs 23% training; grids are 3× the area with 2× the objects

The solver can only express *"for each object, do X where X depends on that
object's own properties."* Missing: two-level rules (derive something, then
use it), counting/arithmetic, template matching, multi-scale structure.
That's the research problem, and more operators of the current kind won't
close it.

**Do not trust my earlier "eval needs indirection" claim** — measurement
showed 0 of 120 have that shape, and the task it was based on
(`abc82100`) isn't a legend task at all. See the correction in the log.

## Known problems not yet fixed

- Exact-match verification only, so near-misses can't guide refinement
- 49/120 eval tasks have >1 test input (vs 7% training) and *all* must be
  right to count; the training number barely exercises that path
- Background is still a heuristic (most-frequent, plus a runner-up)
- Module-level mutable state: `_ACTIVE_INTERPRETATION`, `_in_region_split`
- `synthesis.py` should be 4-5 modules
- Eval is no longer a pristine check — a few of its tasks have been
  inspected while diagnosing

## Notes

- `logs/research_log.md` is the full narrative: every session, decision,
  and negative result. Read it before proposing a direction.
- Kaggle is a **code competition** — you submit a notebook that runs with
  no internet and writes `/kaggle/working/submission.json`. Paths are
  portable; `DATA_PATH` auto-detects `/kaggle/input/*`.
- `submission.json`, `logs/full_evaluation_results.json` and `__pycache__`
  are gitignored.
