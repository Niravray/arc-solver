# ARC Solver Research Log

## June 10, 2026

### What I built today
- Set up Python 3.14 environment on Windows
- Installed NumPy, SciPy, Matplotlib
- Created project folder structure at C:\arc_solver
- Downloaded all 6 ARC dataset files from Kaggle
- Built task_loader.py — reads all 1000 training tasks successfully
- First task observed: 2x2 input tiled to 6x6 output

### What I learned
- ARC grids are NumPy arrays of integers 0-9
- Each integer represents a color
- Tasks have 2-4 training pairs and 1 test input
- 1000 training tasks available

### Decisions made
- Using Python 3.14
- Project lives at C:\arc_solver
- Data at C:\arc_solver\data

# ARC Solver Research Log

## June 15, 2026

### What I built today

* Created visualizer.py for displaying ARC tasks graphically
* Added NumPy and Matplotlib imports for grid visualization
* Verified Python 3.14 installation and configuration
* Installed and verified NumPy 2.3.5 and Matplotlib packages
* Began setting up visualization pipeline for ARC task inspection
* Debugged VS Code environment and interpreter configuration
* Successfully confirmed package installation through terminal testing

### What I learned

* VS Code can use a different Python interpreter than the terminal
* Pylance warnings often indicate interpreter configuration issues rather than missing packages
* Git Bash and Windows paths can cause file execution problems
* Matplotlib uses pyplot and color mapping utilities to display ARC grids
* Visualization is essential for identifying transformation patterns in ARC tasks

### Problems encountered

* Pylance could not resolve NumPy and Matplotlib imports
* VS Code was not initially recognizing installed packages
* Script execution failed because of incorrect path handling in Git Bash
* Confusion between creating a virtual environment and selecting an interpreter

### How I solved them

* Verified Python version from terminal
* Confirmed NumPy and Matplotlib installation using pip
* Identified interpreter mismatch as the likely source of Pylance warnings
* Learned proper path handling for running Python scripts from Git Bash

### Decisions made

* Continue using Python 3.14 for the project
* Use NumPy arrays as the core representation of ARC grids
* Build visualization tools before implementing transformation solvers
* Prioritize understanding task patterns visually before coding solution strategies

### Current project status

* Dataset loaded successfully
* Task loader operational
* Development environment configured
* Visualization layer under construction
* Ready to begin systematic analysis of ARC task patterns

# ARC Solver Research Log

## June 16, 2026

### What I built today

* Initialized Git repository and pushed project to GitHub
* Built object_extractor.py as the next layer of the ARC solver pipeline
* Implemented background color detection
* Implemented connected-component object extraction
* Added object metadata generation:

  * position
  * dimensions
  * bounding boxes
  * centroids
  * object grids
* Added support for multiple object interpretation strategies
* Organized project into reusable pipeline components

### Object Extraction Strategies Implemented

1. connected_4

   * Standard horizontal/vertical connectivity

2. connected_8

   * Includes diagonal connections

3. by_color

   * Treats all pixels of the same color as one object

4. no_background

   * Extracts objects without removing background color

### What I learned

* ARC objects are not always uniquely defined
* Different tasks require different notions of "object"
* Connected-component analysis is a foundational ARC primitive
* Background color is usually the most frequent color but not always conceptually irrelevant
* Multiple competing interpretations must often be maintained simultaneously

### Research Insight

A major challenge in ARC is representation.

Before solving a task, the system must decide:

* What counts as an object?
* Which pixels belong together?
* Which color is background?
* Whether diagonally touching pixels form one object or multiple objects

Different answers can lead to completely different solutions.

### Decisions Made

* Use multiple object extraction strategies in parallel
* Preserve alternative interpretations rather than committing early
* Store rich object metadata for future reasoning
* Continue building solver primitives before attempting full task solving

### Current Solver Pipeline

Layer 1:

* Dataset loading ✔

Layer 2:

* Visualization ✔

Layer 3:

* Object extraction ✔

Next Layers:

* Object property analysis
* Transformation primitives
* Hypothesis generation
* Program search
* Solution verification

### Current Project Status

* Environment stable
* GitHub repository active
* Dataset operational
* Visualization operational
* Object extraction operational
* Ready to begin symbolic reasoning over extracted objects

## Task observations — June 2026

Task 00576224: tiling rule — input pattern tiled NxN times to fill output
Task 2: masking rule — output shape filled with input pattern
## Session — June 26, 2026

### What we did today
- Ran visualizer.py — first ARC tasks seen as colored grids
- Manually solved 3 tasks by eye:
  - Task 00576224: tiling rule — 2x2 pattern tiled to fill 6x6 output
  - Task 007bbfb7: pattern fill — shape filled with another object's pattern
  - Task 009d5c81: color indicator — small object recolors large object then disappears
- Ran object_extractor.py — 5/5 tasks passed
- Ran property_encoder.py — working
- Discussed DSL framing — our primitive library IS the domain specific language
- Discussed convolution/line buffer techniques — applicable in software for pattern detection
- Decision: build baseline first with 20-30 primitives, get a real Kaggle score, then iterate

### Key insight today
The same rule type (color indicator) appeared in multiple tasks. This means every primitive we build solves multiple tasks at once. Coverage compounds.

### Pipeline status today
- L1 Task Loader ✅
- L2 Object Extractor ✅
- L3 Property Encoder ✅
- L4 Relationship Graph — building next

## Session — July 26, 2026

### What we did today
- Ran the full 1000-task training evaluation for the first time — baseline was 12/1000 (1.2%) accuracy, but the run took 3.6 hours
- Diagnosed why: the `resize_then_recolor` composed hypothesis generator was crossing every tile/scale/crop stage against every selector and color (up to ~1330 hypotheses on one task) — this was 99.8% of total runtime and had never solved a single task. Fixed by only generating resize stages whose output shape could actually match the observed pair (read straight off the diff, same trick as everywhere else). 3.6h → 6min, accuracy unchanged (12/1000), since everything pruned was guaranteed to fail the shape check anyway
- Found that the relational subsystem from an earlier session (`graph_difference.py`, `relational_selectors.py`, `relational.py`) was fully built and unit-tested but never wired into `generate_hypotheses()` — wired in `move_until_touching` (leftmost/rightmost-style selectors), then a second tier using the graph-based selectors (nearest/farthest neighbor, aligned-object connection). 12 → 13/1000
- Broke it immediately after: the tier-2 code rebuilt the relationship graph from scratch inside every hypothesis's apply_fn with no caching. One training task (`0a1d4ef5`) has 209 objects, so that O(n²) graph build was getting redone ~300 times for that task alone — caused a 3+ hour hang that I had to kill mid-run. Fixed with a per-task graph cache keyed by grid identity, same discipline already used for objects/diff elsewhere. Task dropped from "would take hours" to 11s. Lesson: any new hypothesis family needs to be checked for "is something expensive getting recomputed per-hypothesis instead of per-pair" before it ships, not after
- Clustered the ~987 unsolved training tasks by diff signature to find where to point new primitives next, instead of guessing. Biggest clean bucket: movement-only signatures (168 tasks). Drilled further — 12 were a uniform whole-grid shift, 107 were multiple objects each moving independently (gravity-shaped)
- Added `shift_grid` (reads the pan vector straight off the diff) — turned out none of the 12 candidate tasks had a truly constant vector across all their pairs, but it picked up an unrelated task (`25ff71a9`) anyway. 13 → 14/1000
- Added `apply_gravity` (every object slides independently until it hits the boundary or another object, settling closest-to-edge first) — solved 2 real gravity tasks (`1e0a9b12`, `5ffb2104`). 14 → 16/1000. Also learned the diff engine's shape-based object matching can mislabel a "pattern got stamped into a different tile" as "an object moved," which is worth remembering next time a movement-signature cluster looks bigger than it turns out to be
- Started on a genuinely different technique: instead of sweeping a bounded set of guessed parameters, infer the parameter directly from training pixels (counterexample-guided / derive-then-verify, the FlashFill idea). First case: `_infer_color_mapping` reads a full N-color substitution table straight off a pair's pixels (plus a "tune again" version built from all pairs at once). Solved `0d3d703e`, which needed 8 colors swapped simultaneously — something the old single-swap sweep could never reach at any reasonable search budget. 16 → 17/1000
- Cleaned up the repo: deleted dead `format_submission()` (had two known format bugs, fully superseded), deleted the one-off `debug.py`/`diagnose_009d5c81.py` diagnostic scripts, added `.gitignore` for `__pycache__`/`.pyc`/generated eval results, untracked the `.pyc` files that had been committed by accident
- Committed and pushed everything to GitHub

### Key insight today
Two different failure modes showed up and they need different fixes: (1) coverage gaps — no hypothesis in the library can even express the right rule, fixed by adding primitives — and (2) cost blowups — the right hypothesis exists but something is getting recomputed combinatorially, fixed by caching/pruning. Both perf incidents today (resize_then_recolor, uncached graph rebuild) were the same root cause: something correctly scoped to run "once per training pair" instead ran "once per hypothesis," and hypothesis counts scale fast. Worth a checklist item on every future primitive: what's the actual cost per apply_fn call, and is anything in it shared across hypotheses that should be cached once per pair.

Also: reading a parameter directly off the training pixels (derive) beats sweeping a bounded guess list (search) whenever the parameter space is too large to bound cheaply — the 8-color mapping task is proof, not just theory. Worth extending this "derive, don't sweep" approach to other primitives (deltas, tile factors already partly do this) before writing more brute-force selector sweeps.

### Pipeline status today
- L1–L7 ✅ (unchanged)
- L8 Solver — full 1000-task eval running in ~7 minutes (was 3.6 hours)
- Relational subsystem (graph_difference, relational_selectors, relational primitives) — wired into the solver, contributing real solves
- Accuracy: 12/1000 (1.2%) → 17/1000 (1.7%) today
- Next: extend the "derive, don't sweep" approach to more primitive families; keep clustering remaining failures to find the next bucket worth targeting

## Session — July 27, 2026

### What we did today

Two separate tracks: finished off the old sweep, then built a genuinely different solver alongside it.

**Track 1 — the existing sweep (`verifier.py`), 17 → 19/1000**
- Clustered the unsolved failures again. The biggest single signature is `COLORS_ADDED|OBJECTS_ADDED` at 106 tasks, but inspecting actual examples showed it's a grab-bag of at least three unrelated rules — flood fill, sequence extrapolation, per-colour stamping — not one pattern. Lower yield per primitive than the movement bucket was
- Added `fill_enclosed_regions` (flood fill from the border; anything unreached is walled in and gets recoloured). Validated standalone across the unsolved set BEFORE wiring it in: predicted exactly 2 tasks, delivered exactly those 2 (`00d62c1b`, `a5313dff`). 17 → 19/1000

**Track 2 — the rule synthesis engine (`synthesis.py`, new, ~1200 lines)**

The change of approach: stop picking rules from a prebuilt list, and start *inducing* a rule per task. The predefined part is a LANGUAGE of expressions with holes — `Recolor(<which objects?>, <what colour?>)` — and the holes get solved from the observed change. The loop is counterexample-guided: induce from pair 1, test on pair 2, and when a pair disproves the rule, add it to the evidence and re-induce under that stronger constraint.

- Built it first for Recolor only. Selector induction is the core: objects that changed are positive examples, objects left alone are negatives, solve for a predicate covering all positives and no negatives. 6 tasks, 6/6 correct on the unseen test
- Generalised the loop to be operator-agnostic and added **Remove** — the key test, because Remove has no second hole, so it reuses `induce_selectors` with ZERO operator-specific code. Worked immediately, which proved the selector inducer is genuinely shared machinery and not recolor code in disguise
- Added **Move** — a new hole type (offsets, not colours). Loop, diagnosis and selector induction all worked unchanged
- Added **two-stage composition** as an `Operator` satisfying the same interface, so `synthesize()` drives a composed rule with literally the same loop. Stage 1 uses a *tolerant* observer (labels only the part it explains); stage 2 induces from the residual with the ordinary strict observer
- Added **Draw** — the first operator that CREATES content, so "which objects?" becomes "which objects EMIT?". Rays cast from objects, coloured by the source object or a fixed colour

**Final: 22 rules found, 19 correct, 86% precision — 18 of them tasks the sweep can't solve. Combined potential 37/1000 (3.7%).**

### Vocabulary expansions — every one forced by a measured failure, not a hunch
- `hole_count`, after `0a2355a6` turned out to be a clean function of it (1 hole→1, 2→3, 3→2) with nothing in the catalog able to separate those objects — the engine was falling back on a memorising size table that then failed on test
- **decision lists** (selector → value clauses), after `b230c067`/`d2abd087` showed the predicates needed already existed but the language had no way to say "these objects get one value, those another"
- a **symbolic-key restriction** on geometric clause lists, after 10 Move rules memorised per-object offsets like `size==1 → (-1,0) | size==2 → (-2,0)`. This one REMOVED expressiveness and took Move from 29% → 100% precision with zero loss of correct answers

### What didn't work (worth recording)
- **Anchor-relative relational predicates.** Probed first and the probe was informative — the concept needed isn't "touches SOMETHING" but "touches THE MARKER", so I built proper anchor-relative atoms (relations to largest / smallest / uniquely-coloured / each single-instance colour). Result: **zero new solves, Recolor precision 75% → 64%**. Left in the code but disabled behind `USE_RELATIONAL_ATOMS = False` with the measurement written into the comment
- The failure profile did shift though — across operators, tasks blocked at "cannot express a rule at all" fell 51 → 39 while "counterexample killed every candidate" rose 33 → 43. Twelve tasks stopped being blocked at induction and NONE became correct answers. That's the signal that vocabulary had stopped being the binding constraint
- **Composition of the three existing operators.** Measured the ceiling before building: only 13 of 1000 tasks are combinations of Recolor/Remove/Move at all. Delivered +2 at 29% precision. Kept enabled only because single operators are tried first, so it fires purely additively where nothing else answered
- **Draw's recall.** 208 pure-addition tasks available, delivered 2. The failure-stage measurement is unambiguous: of 203 reaching induction, **190 fail because the ray model never reproduces the added cells**, 7 at the selector hole, 6 get through. The machinery is sound; the single generative rule is too narrow

### Key insight today

**Measure the size of the prize before building for it.** This ran three times and was right three times. Composition looked like the obvious next step and measurement capped it at 13 tasks. The `COLORS_ADDED` bucket looked like the biggest opportunity at 106 tasks and turned out to be three unrelated problems. Draw looked like a 208-task win and the ray form only reaches a fraction. The one time I skipped a proper pre-measurement — relational predicates, where I probed but built anyway after the probe came back negative — was the one expansion that cost precision for nothing.

**Restricting a language can beat extending it.** The single most valuable change all day was taking expressiveness AWAY: forbidding geometric decision lists from keying on metric properties took Move from 29% to 100% precision and lost no correct answers. More expressive means fitting the training pairs in more ways, and most of those ways are wrong. Every expansion needs its precision measured, not just its recall.

**"Draw" is a family, not an operator.** Rays that stop before the obstacle vs on it, emission from every pixel vs the leading edge, stamping, symmetry completion — each is a separate generative form needing its own induction. The 208-task bucket is real but it's not one operator's worth of work.

**Induction generalises better than search when it commits.** The synthesis engine sits at 86% precision — when it finds a rule explaining every training pair, it's usually the real rule. That's the property that makes it worth more than its raw 19 tasks, and it's the opposite of what the sweep does, where a verified hypothesis is frequently a coincidence.

### Pipeline status today
- Sweep (`solver.py`, on the scoring path): **19/1000 (1.9%)**, full eval ~5 min
- Synthesis engine (`synthesis.py`, deliberately NOT imported by solver.py): **19 correct at 86% precision**, 18 unique to it, validated across 5 operator configurations
- **Combined potential: 37/1000 (3.7%)** — 3× where the day before started
- Tests 17/17, zero exceptions across 1000 tasks
- Next: more Draw forms (the 190 blocked tasks are the largest single opportunity on the board); then decide on integrating synthesis into `solver.py` as a first-pass path with the sweep as fallback