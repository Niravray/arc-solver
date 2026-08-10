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

- Draw v1 only reached 2 tasks, so measured *why* rather than guessing: of 203 pure-addition tasks reaching induction, **190 failed because the ray model never reproduced the added cells**, 7 at the selector hole, 6 got through. The machinery was fine; the single generative form was the bottleneck
- So measured 16 candidate forms against those 203 tasks before writing any of them. Result was a flat long tail — `bbox_fill` 11, `connect_aligned`/`halo8`/`bbox_outline` 4 each, line and ray variants 1–3 each, 32 total. No dominant form
- Built the registry anyway, in two kinds: **per-object forms** that keep the selector hole (rays, rays-through-obstacles, halo4/8, bbox_fill, bbox_outline, full row/col/both) and **global forms** with no selector at all (mirror_h/v/180, connect_aligned) — symmetry completion is a property of the scene, not of any one object, so there's nothing to choose between. Global forms rank ahead of per-object ones: fewer holes solved from the same evidence means less room to have fit it by accident. Mirror forms also answer the colour hole themselves, from the mirrored cell. **Draw 2 → 14 correct at 93% precision**
- **Wired synthesis into `solver.py`** alongside the sweep — merged, not replaced. **19 → 49/1000 (4.90%)**, runtime unchanged at 5.7 min, zero exceptions

- **Ran the evaluation split for the first time — 0/120.** Every design decision all day had been made by looking at training results, so this was the honest check on whether any of it transfers
- Verified it wasn't a harness bug (task ids match solution keys, grids parse, zero exceptions), then measured *how* it failed, which matters far more than the number: **synthesis induced 0 rules and the sweep verified 0 hypotheses.** Not one candidate on any of the 120 tasks
- Made both splits the default reporting path, with attribution behind the number rather than just accuracy

**Final: training 49/1000 (4.90%) — 30 by induction alone, 18 by the sweep alone, 1 by both. Evaluation 0/120 (0.00%).**

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

**"Draw" is a family, not an operator** — and acting on that was the biggest single win of the day. Rays stopping before vs on the obstacle, emission from every pixel vs the leading edge, halos, bbox fills, symmetry completion: each is a separate generative form. Turning Draw's one hardcoded form into a registry took it from 2 to 14 correct.

**Measure the prize, but weigh it against the MARGINAL cost.** This refines the earlier rule and is the correction I'd most want to remember. The 16-form probe came back as a flat long tail with nothing above 11, and on the earlier heuristic ("is this bucket big enough?") that reads as *skip it*. Building it anyway converted 32 cell-level matches into 12 extra correct answers — because once the shared machinery exists, each new form is about ten lines. A long tail is worth harvesting exactly when the per-item cost has already been paid. The right question isn't "how big is the biggest bucket" but "how much does one more entry cost me".

**Induction generalises better than search when it commits.** The synthesis engine sits at 89% precision — when it finds a rule explaining every training pair, it's usually the real rule. That's the opposite of the sweep, where a hypothesis that verifies on the training pairs is frequently a coincidence.

**Two weak solvers that fail differently beat one better solver.** Final attribution: 30 tasks solved by induction alone, 18 by the sweep alone, **1 by both**. They overlap on one task in forty-nine. Replacing the sweep with synthesis would have scored 31; keeping only the sweep, 19; running both across ARC's two attempt slots, 49. The lesson is to check the OVERLAP before deciding whether a new approach replaces an old one — near-zero overlap means merge, and the weaker system keeps earning its place.

**The number we'd been optimising was measured on the easy half of the benchmark.** This is the most important thing learned today and it recontextualises everything above. 49/1000 on training, 0/120 on evaluation. The dataset is ARC-AGI-2 (1000 train / 120 public eval / 240 test), whose evaluation set is deliberately built so that patterns learned from training do NOT carry over — it exists to defeat systems that pattern-match instead of reason. Eval grids have 3× the cells (median 324 vs 100) and 2× the objects (11 vs 5), and the tasks need composed and indirect rules: one I looked at uses a legend row to say which colour to apply elsewhere (indirection — one part of the grid tells you how to read another), another redraws a 6×6 input at two nested scales into 16×16 (fractal self-similarity). Neither language can express either. The Kaggle test set of 240 is drawn from the hard distribution, not from training, so **the realistic submission score right now is 0%**.

**"Zero rules proposed" and "zero rules correct" look identical in an accuracy column and mean opposite things.** This is the diagnostic that made the eval result interpretable rather than just demoralising. Had we proposed rules that turned out wrong, the problem would be overfitting — my design choices memorising training quirks — and the fix would be to undo today's work. We proposed *nothing*, which means it's a pure coverage failure: the framework is sound, the languages are simply silent on tasks of that complexity. Same headline number, opposite diagnosis, opposite remedy. That's why the new report always prints "rules offered" next to "correct", and why attribution is derived from which ATTEMPT slot matched rather than from the method string — the method names both sources whenever both offered an answer, so it can tell you who spoke but not who was right.

**Measure on the held-out split from the start, not once at the end.** The per-task induction can't overfit — it only ever sees that task's own training pairs — but the *designer* can, and I did have that risk: which Draw forms to build, the symbolic-key restriction, the cost ranking, `hole_count` were all chosen by looking at training numbers. That turned out not to be the failure mode here, but I only know that because we finally ran the other split. A whole day of gains was reported against a measurement that couldn't have revealed this.

### Decision taken
Keep iterating against training, because it's the only split giving usable signal (eval offers no gradient to climb from 0), but **report both numbers on every run** so a training gain is never again mistaken for a real one. When choosing what to build next, bias toward what eval actually needs — indirection, multi-scale structure, many-object scenes — over more variants of what training rewards.

### Pipeline status today
- **`solver.py` now runs BOTH solvers and reports BOTH splits** (`--full-eval`; `--training-only` keeps the old behaviour)

```
split         tasks  solved   accuracy  errors   runtime
training       1000      49      4.90%       0      347s
evaluation      120       0      0.00%       0      179s
```

- training attribution: 31 synthesis, 18 sweep, induction precision 89% (35 rules offered)
- evaluation attribution: nothing — no rule produced on any task
- eval is *slower* per task (1.49s vs 0.35s) while finding nothing, which is consistent with the coverage read: bigger grids, more objects, more work done, still nothing expressible
  - attempt 1 = the induced rule, attempt 2 = the sweep's best, deduplicated so a shared answer doesn't waste the second slot
  - each source independently exception-wrapped; old fallback preserved when neither answers
- Synthesis engine: 35 rules found, 31 correct, 89% precision, across Recolor / Remove / Move / Draw / two composition pairs
- Sweep: 19 correct, unchanged and still pulling its weight on 18 tasks nothing else reaches
- Kaggle submission path verified — list-per-task_id with attempt_1/attempt_2, rule derived once per task and applied to every test input
- Tests 17/17
- Accuracy across the two days: 12/1000 → **49/1000** on training, runtime 3.6 hours → under 6 minutes
- Next, in order:
  1. **Size-changing operators.** Every synthesis observer routes through `_grids_comparable`, which returns None the moment input and output shapes differ — so **316 of the 951 unsolved training tasks are structurally invisible to induction**, never even looked at. Of those, 230 have an output always smaller than the input, and the most common ARC pattern there is "the output IS one of the input's objects", which reuses selector induction directly. `Extract(selector)` is the obvious first one
  2. The 171 pure-addition tasks still unmatched by any of the 16 Draw forms
  3. Revisit `MAX_RELATIONAL_OBJECTS = 60` and `MAX_DRAW_OBJECTS = 40` — those perf guards were tuned on training-scale grids and will exclude eval-scale scenes, though they aren't the main blocker
- Standing caveat to carry forward: everything above is training-split progress. Eval is 0 and getting it off 0 is a research problem, not a few more primitives
## Session — July 28, 2026

### What we did today

Training 49/1000 → **56/1000 (5.60%)**. Evaluation still 0/120. Four changes shipped, three experiments rejected on measurement.

**Extract operator — the output IS one of the input's objects (+5, later +7)**
- Motivated by a structural blind spot: every synthesis observer routed through `_grids_comparable`, which returns None the moment shapes differ, so **316 of 951 unsolved tasks had never been looked at by induction at all**
- Before building, measured the actual opportunity rather than trusting the bucket: of 230 shrinking tasks, only **15** have an output that literally equals a cropped object. The other 179 aren't windows of the input at all
- Extract has NO value hole — the answer is the object itself — so it's the purest reuse of `induce_selectors` in the engine. Two crop forms (raw bbox window vs object isolated on background), chosen by induction
- 5 correct at 83% precision, rising to 7 after the ranking fix below

**Selector ranking: structure over measurement (+2)**
- `72ca375d` found three selectors that explained every training pair identically: `height==2 AND is_sym_v` (wrong on test), `color_is_unique AND is_sym_v` (right), `not_is_single_pixel AND is_sym_v` (right). With equal atom counts the tie fell to catalog order and picked the metric one
- `induce_selectors` now sorts by (atom count, metric-atom count). Same principle already proven on Move's decision lists, now applied to selection, so all five operators inherit it

**Background gate relaxed (+1)**
- `_grids_comparable` required input and output to share a background colour. But `get_background_color` returns the most FREQUENT colour, so recolouring a large region flips it even when the canvas never changed — rejecting 109 training and 11 evaluation tasks on a technicality before any operator looked at them
- Now uses the input's background for both. Precision 90% → 91%

**Escalating search budget (kept as a safety net, not a search lever)**
- Implemented as specified: 2s per task, promoted to 15s once induction yields any candidate from pair 1 — the point at which a rule is known to exist in the language
- But measurement showed induction is **not time-bound**: over 1000 failing `induce()` calls, median 0.75ms, max 325ms. Neither budget ever binds
- Retained as a runaway guard, since two separate incidents have seen one task consume hours. It buys safety, not solutions

**Composition expanded from 3 operator pairs to 12 (+3)**
- The 13-task composition ceiling measured yesterday was **stale** — it predated Draw and Extract. Re-measuring with the current operator set puts the reachable multi-concept space at **68 tasks**, a 5× increase, almost entirely because APPEAR became expressible once Draw existed (APPEAR+MOVE+REMOVE 14, APPEAR+RECOLOR 13, APPEAR+REMOVE 12, APPEAR+MOVE 9)
- Needed tolerant observers for Recolor and Draw: a first stage must label only *its* part of the change and leave the rest to stage two, whereas the strict observers reject a pair the moment anything else happens — precisely the situation composition exists for
- First-stage candidates went 2 → 4, so ordered pairs went 3 → 12. Measured on the unsolved set first: 7 rules found, 3 correct, 78s — cheap enough to justify wiring. New solves are `Recolor→Draw` (×2) and `Draw→Remove`

**Kaggle submission dry run — pipeline validated**
- 240/240 task ids matching Kaggle's own `sample_submission.json`, **0 malformed entries**, 17 multi-test-input tasks with correct entry counts, 58s build, 0.38 MB
- The 17 multi-test-input tasks are the meaningful check — that is exactly the case the deleted `format_submission()` got silently wrong, emitting one prediction where several were required

**Portability — the code now runs somewhere other than this laptop**
- Absolute Windows paths were baked into **10 files**: eight `sys.path.append` calls plus `DATA_PATH` and the `save_submission`/`save_evaluation_results` defaults. Kaggle runs on Linux under `/kaggle`, so none of them resolve there and the code would not have imported, let alone run
- `sys.path` now derives from `__file__`; `DATA_PATH` resolves via `ARC_DATA_DIR` → any `/kaggle/input/*` containing `arc-agi*` files → local `data/`; `save_submission` targets `/kaggle/working/submission.json` when that exists
- Added `kaggle_notebook.py`. ARC Prize is a **code** competition: the submission is a notebook Kaggle runs itself with no internet, which must write `submission.json` during the run, and the private test set is swapped in at scoring time — so nothing may assume the contents of the test file currently on disk. The entry point also backfills well-formed placeholders for any missing task id, since Kaggle rejects an incomplete file outright rather than scoring zero on the gap
- Verified no behaviour change: 59/1000 and 0/120 identical before and after

### What didn't work (measured and rejected)

- **Depth-3 conjunctions.** Found a selector in 1 of 43 depth-2 failures, and both programs it produced were wrong on test — `color==1 AND width==3 AND not_shape_is_unique`, three atoms, two of them metric. Widening the language again bought only new ways to overfit
- **Candidate restarts.** Falling back through pair 1's ranked candidates when the top one fails refinement: zero tasks gained, same 56 solved, same 43 offered, same 91% precision
- **Ten candidate selector atoms** probed against the seven Extract selector failures (`shape_is_most_common`, `is_densest`, `most_holes`, `n_colors_gt1`, …). **None separated the target in any of them**, so none were added. Inspection also showed the failure set was contaminated: `d9fac9be` is an observer false positive, where a 1×1 output is trivially reproduced by eight different objects

### Key insight today

**The refinement loop is evidence-driven, not path-dependent — and that rules out a whole category of future ideas.** `_refine_from` re-induces from ALL accumulated evidence at every counterexample, so the candidate set it chooses from by the final pair depends only on the evidence, not on which candidate the walk started from. Every restart converges to the same place. This is why candidate restarts gained nothing and structurally could not have. The consequence is general: **the only levers on induction are richer evidence or a better language, never better search over the language we already have.** That is worth remembering before building any future search machinery.

**A net-zero result can be hiding a regression.** The selector ranking change first landed as 53 → 53, which reads as harmless. The per-operator breakdown showed Extract +2 and Move −2 — and Move's two hadn't become wrong, they had stopped producing any rule at all. Two independently-correct rules were fighting: ranking demotes `color==` as measurement-pinned, while geometric decision lists REQUIRE colour-keyed clauses, and that requirement was implemented as a check AFTER taking `selectors[0]`. Once ranking pushed colour atoms last, every geometric list was rejected. Fixed by filtering during selection rather than validating afterwards, so the constraint lives where the choice is made. Without the per-operator breakdown this would have shipped silently.

**I misread machine contention as a code problem twice.** A run went 1h12m and I called it "the same signature as the earlier hang" — it was three of my own leftover Python processes competing for CPU. Later I quoted "+52% runtime" as the measured cost of candidate restarts when eval timings for identical work swung 131s → 317s → 2223s across runs. Correctness numbers were rock stable throughout (56 solved, 43 offered, 91% precision, every run). Lesson: **before attributing a timing change to code, check what else is running** — and never quote a precise cost figure from a contended measurement.

**Bucket size keeps overestimating yield, now with a second filter identified.** Extract: 230 shrinking tasks → 15 exact form matches → 12 observable → 5 solved. Previously the gap was form-match; this time a second filter appeared behind it, the selector hole. Estimating from the widest number available has been wrong every single time.

**A measured ceiling expires when the thing it measured changes.** "Composition is worth 13 tasks" was measured with three operators and then treated as settled — while adding Draw and Extract quietly invalidated it. Re-measuring put the real figure at 68 and the expansion delivered +3. On yesterday's number I would have skipped composition entirely. The habit to keep: when the system gains a capability, the *old measurements* are now suspect, not just the old code. Several other "settled" numbers in this log are probably stale for the same reason and worth re-running before being trusted again.

### Pipeline status today

```
split         tasks  solved   accuracy  errors   runtime
training       1000      59      5.90%       0      378s
evaluation      120       0      0.00%       0       60s
```

- training attribution: 42 synthesis, 17 sweep, induction precision 91% (46 rules offered)
- synthesis operators: Draw 15, Recolor 10, Extract 7, Move 4, Recolor→Draw 2, Remove 1, Remove→Recolor 1, Draw→Remove 1
- Tests 17/17, zero exceptions on either split
- **Submission pipeline verified end to end** — `kaggle_notebook.py` produces a valid 240-task submission in 76s against a 12-hour limit, and the code is no longer machine-bound

**Evaluation diagnosed properly for the first time.** 18 of 120 tasks ARE structurally observable (Draw 12, Move 4, Recolor 2) — and all 18 die at the same point: *no rule from pair 1*. Not a counterexample killing candidates, not a ranking problem — a single example cannot be explained. The other 102 no operator can even observe: 50 are same-shape but need more than one operator, 39 change shape without being an Extract, 11 change background colour, 2 exceed the object guards.

- Next: the eval failure is now precisely located, and it is a language problem at the single-pair level. Getting off 0 needs concepts we do not have — indirection (a legend saying how to read the rest of the grid), multi-scale/fractal structure, counting and aggregation. Not more variants of what training rewards

> **CORRECTION (July 30).** The "indirection" claim above is wrong and was disproven by measurement the next day: **0 of 120 evaluation tasks** have the indirection shape (a value that varies across pairs and is readable from elsewhere in the scene). Worse, the task I built the claim on — `abc82100`, which I kept calling "the legend task" — is not a legend task at all. I read pair 1 (`12888` looking like a colour key) and generalised; across its four pairs row 0 is variously cleared, filled from empty, and colour-swapped. Extrapolating from one example, while writing log entries warning against extrapolating from one example.

## Session — July 29-30, 2026

### What we did today

Training 59/1000 → **64/1000 (6.40%)**. Evaluation still 0/120. One audit, five fixes, three capabilities that didn't pay, and two of my own misdiagnoses.

**Audited the pipeline for flaws and fixed five (+5 tasks)**
1. **The solver committed to one definition of "object"** — the big one. `scene_objects` always used 4-connectivity. Measured: 324 unsolved tasks are observable under it, and **10 more become observable only under 8-connectivity**, silently locked out. The June 16 entry decided to "preserve alternative interpretations rather than committing early" and the synthesis engine then committed early anyway. Interpretation is now something the solver **searches over**, verification picking the reading exactly as it picks every other hole. Earned 3 tasks (`Extract@connected8`) + 1 (`Extract@panels`)
2. **Background was an unvalidated guess** — most-frequent colour, which is wrong when a large object dominates a small canvas, and then everything downstream is wrong with nothing raised. Added a runner-up candidate
3. **No panel structure** — 21/120 eval tasks are separator-divided grids where the separator read as just another object. Panels are now an interpretation giving one pseudo-object per panel, so every operator applies unchanged. 24 of 29 separator-bearing eval tasks extract panels. Fixing #1 first is what made this cheap — panels became a list entry, not a subsystem
4. **Guards tuned to the wrong split** — 60/40/80 → 120/90/150
5. **Attempt 2 was spent on the sweep** even when induction had a runner-up that also explained every pair

**Applicability gating (6.2× speedup, zero cost).** Applying every interpretation to every task cost 12.6× runtime (389s → 4902s) for those five tasks, because almost every grid has no separators and reads identically under both connectivities — so the retries re-derived identical objects and re-ran identical failed inductions. Gating each on one cheap extraction or separator scan: **4902s → 794s, all five tasks kept.**

**Region splitting — built, measured, disabled.** Isolates an auxiliary edge strip, solves the remainder, reassembles. Motivated by a real signal: stripping `abc82100`'s top row moved it from "no operator can observe this" to "observable by six composed operators". Result: fires on **180** unsolved training tasks and **27** eval tasks, produces **zero** rules, costs 3.3× runtime. The reason is in the modes — **159 of the 180 are "preserve"**, where the strip is *identical* in input and output, so cropping it away is a no-op by construction and the remainder is the same rule at the same difficulty. Only "clear" splits can simplify: 21 training, 1 eval

**Threw an hour of compute at one task — answered in 1 second.** 51 combinations (3 interpretations × 17 operators, 3-atom conjunctions, raised per-task budgets, restarts on) against `abc82100`. **All 51 died at "nothing from pair 1" in one second.** Not the search giving up — the search never *starting*: every observer fails to describe the pair, so there are no holes to solve and a budget has nothing to buy. An hour, a day, a week: identical. This cleanly rules out the entire "more compute" direction

### Key insight today

**Three additions made structure VISIBLE without making it SOLVABLE.** Panels, the raised guards, and region splitting — each cost real runtime, each gained ~nothing. Set against what *did* work all week (Draw's form registry +12, Extract +7, composition +3, gravity, shift), the pattern is unambiguous: **the gap is not in how the solver reads grids, it is in what it can do once it has read them.** Every future idea should be checked against this before building — "does this let the solver *see* more, or *do* more?" Only the second has ever paid.

**Compute is not the constraint, and now that is measured rather than assumed.** The one-hour experiment is the cleanest negative result of the project: failures happen at *observation*, before any search begins. That retires the whole family of "search harder / longer / wider" proposals, including the escalating budget and candidate restarts from yesterday.

**I mischaracterised a task from one example and built several turns on it.** `abc82100` is not a legend task — I read pair 1, saw `12888`, called it a colour key, and told the user eval needs indirection. Its four pairs do three different things to row 0. The indirection measurement (0/120) had already contradicted me and I kept the label anyway. The correction is written into the July 28 entry above. **When a measurement contradicts an earlier claim, go back and revise the claim, not just the next decision.**

**I misattributed runtime to code for the third time.** An overnight run reported 38012s for training; a clean re-run of the identical code reported **914s** — a 42× swing. The tell was available immediately: *evaluation in that same run took 266s*, and a genuine regression would have slowed both. Environment first, code second.

### Pipeline status today

```
split         tasks  solved   accuracy  errors   runtime
training       1000      64      6.40%       0      914s
evaluation      120       0      0.00%       0      150s
```

- attribution: 47 synthesis, 17 sweep, induction precision 90% (52 rules offered)
- operators: Draw 16, Recolor 10, Extract 7, Move 4, Extract@connected8 3, Recolor→Draw 2, Extract@panels 1, Remove 1, Remove→Recolor 1
- Tests 17/17, zero exceptions on either split
- **Flags now off by default, each with its measurement in-code:** `USE_RELATIONAL_ATOMS`, `USE_REGION_SPLIT`, `MAX_CANDIDATE_RESTARTS=1`, `MAX_CONJUNCTION=2`
- Next: on the evidence above, the only direction with a track record is new **operators and forms** — things that let the solver *do* more. Anything that only lets it *see* more should be measured before it is built, not after

## Session — July 30, 2026 (continued)

### What we did

Training 64/1000 → **72/1000 (7.20%)**, precision 90% → **92%**. Evaluation still 0/120.

- **Added tests for `synthesis.py`, which had none** — 17 → 46 tests. The file is ~2700 lines and produces 58 of the 72 solves, and nothing imported it; every change all week was validated only by the accuracy number not dropping, which had already been shown to hide a regression. Tests pin the structural-over-metric ranking, the colour-key restriction on geometric decision lists, that observers *reject* pairs they cannot explain, the loop's core invariant (any returned rule reproduces every training output), and the four measured-off flags
- **Writing the tests found a real bug**, now fixed: `find_separator_lines` treated every uniform non-background line as a divider, but a solid-coloured panel's own columns are uniform too — so a grid of solid blocks had every column flagged, no segment survived, and no panel was produced. The panels interpretation silently did nothing on exactly the layouts it was built for. Fixed by picking the colour whose lines actually *partition* the grid. Detection went from contradictory (29 detected / 24 extracted) to consistent (24/24 eval, 151/151 training)
- **Used the two extraction strategies built on June 16 and never wired in** — `by_color` and `no_background`, unused for six weeks. **+8 tasks**, the biggest single gain since the Draw form registry
- Three-stage composition added in the same batch: **contributed 0**
- Answered two open questions: attempt-2 alternates **[measured]** 29 solved tasks had alternates, **0 were ever saved by one**; the guard raise earns exactly **1** task
- Wrote `CLAUDE.md` (working context for a fresh session) and `docs/RESEARCH_AGENDA.md` (24 flaws across six tiers, with what to research for each)

### Key insight today

**I wrote a rule into `CLAUDE.md` and disproved it within the hour.** The rule was *"does this let the solver SEE more or DO more — only the second has ever paid"*, generalised from panels, raised guards and region splitting all gaining ~nothing. Then `no_background` and `by_color` — pure changes to how grids are read — added 8 tasks in one run.

The corrected rule is narrower and fits all the evidence: **regrouping the same pixels into different objects pays (14 tasks: `no_background` 9, `connected8` 3, `by_color` 2); making extra structure visible without changing the grouping does not (panels 1, region split 0, guards 1).** The distinction is whether the *object set itself* changes, not whether the solver "sees more".

Worth noting how the wrong rule formed: three consecutive negative results, all in one narrow direction, generalised into a law. Three data points in the same corner of the space is not a law — and I had just spent two days warning about exactly this failure mode in the task-level measurements.

**Two capabilities sat built, tested and unused for six weeks.** `by_color` and `no_background` were written on June 16, with a log entry that day explicitly deciding to "use multiple object extraction strategies in parallel". They were then never called. That is now the second time this project has found significant value in code it had already written and forgotten (the relational subsystem on July 26 was the first, +1 task). **Before building anything new, check what is already in the repo and unwired.**

### Pipeline status

```
split         tasks  solved   accuracy  errors
training       1000      72      7.20%       0
evaluation      120       0      0.00%       0
```

- attribution: 58 synthesis, 14 sweep, induction precision 92% (63 rules offered)
- interpretations now searched: `connected4`, `connected8`, `by_color`, `no_background`, `panels` — **the second-largest source of solves in the project**
- tests 46/46; submission pipeline verified; code portable to Kaggle
- Next: see `docs/RESEARCH_AGENDA.md`. The cheapest high-value item is a held-out split *within* training — all 1000 tasks are currently used both to design and to score, so 7.20% is optimistic even for unseen tasks of the same difficulty

### Later — the held-out split

Implemented the top item from `docs/RESEARCH_AGENDA.md`: a **750 dev / 250 holdout** split within training, deterministic (ids sorted, then shuffled with a fixed seed) so the partition cannot drift between runs or machines. `--full-eval` now reports `training-dev`, `training-holdout`, `training-all` and `evaluation`.

```
training-dev       750 tasks   60 solved   8.00%   precision  91%
training-holdout   250 tasks   12 solved   4.80%   precision 100%
training-all      1000 tasks   72 solved   7.20%   precision  92%
```

**The 3.2-point gap is NOT significant.** If the true rate were 7.2% everywhere, a 250-task holdout gives 18 ± 4.1 solves; 12 observed is z = −1.47, p = 0.14, Fisher exact on dev vs holdout p = 0.12. Reporting that gap as evidence of overfitting would have been wrong, and it was tempting — it points exactly the way the hypothesis predicted.

**The resolution finding matters more than the number.** At a ~7% solve rate, one standard deviation on a 250-task holdout is ~4 solves, so **a change must be worth roughly 8+ tasks on holdout to clear noise.** Almost every win in this project has been 1–8 tasks *across all 1000*. So the holdout as sized is a guard against large fitting effects, not a measure of incremental progress — and treating it as the latter would produce confident nonsense in both directions. Making it able to adjudicate a typical change means enlarging it, which costs dev signal; that tradeoff is unresolved.

Two caveats written into the code rather than left in a commit message:
1. The split was created **after** the solver was already tuned against all 1000 tasks, so this first holdout number is itself contaminated. It only starts meaning something for decisions taken from here on.
2. The resolution limit above.

**Process note:** the first attempt re-ran the full evaluation and was still going after 70 minutes (machine throttling again, ~7s/task against 0.9s clean). Killed it once I realised the split is a **partition of records, not a computation** — the previous run's results were already saved, so the answer came from `logs/full_evaluation_results.json` in seconds. Worth remembering before spending an hour on a reporting change.

### Where this leaves the project

```
training       1000 tasks    72 solved   7.20%
evaluation      120 tasks     0 solved   0.00%
```

48 tests. Submission pipeline verified end to end. Code portable to Kaggle. Everything committed and pushed.

`CLAUDE.md` carries the working context for a fresh session; `docs/RESEARCH_AGENDA.md` carries 24 flaws across six tiers with research directions for each. The cheapest remaining high-value items are parallelism (1000 independent tasks currently run serially on one core, blocked by module-level mutable state) and, on the research side, template/motif matching — 90% of evaluation tasks contain repeated shapes and the solver has no concept of one.

## Session — August 5, 2026

### What we did

Tested the one hypothesis `docs/VOCABULARY_PROPOSAL.md` Part 5 had explicitly flagged as never tested: whether the objects that change are decided by **correspondence** — shape twin, twin-up-to-rotation, containment, scaled copy — rather than by any property computable from a single object alone. This is exactly what `graph_difference.py` computes and has sat unwired since it was built.

Wrote `scripts/falsify_correspondence.py` as a proper falsification test, pre-registered before the evaluation set was touched: **CORR-ONLY** (a task a correspondence atom separates that no baseline atom also separates, at conjunction depth ≤2 — the solver's real `MAX_CONJUNCTION`) ≥ 8 of 120 to survive.

**Result: 1 of 120, and that one hit is not distinguishable from chance.** A permutation control — shuffle the changed/unchanged labels five times, keep the correspondence atoms active — produced CORR-ONLY = 1 on one of five shuffles, identical to the real count. The weak form (28/120 tasks where a correspondence atom separates at all, regardless of whether baseline also can) numerically clears the same threshold, but was never the decision criterion: most of it coincides with what 51 baseline-separable tasks already explain.

**Both controls came from reading tasks by hand, not from what the instrument happened to pass — and the positive control took two tries.** The first candidate, training task `776ffc46`, read on a 30-second glance like a clean correspondence case: loose shapes matching a template's shape get recoloured. Reading it fully to build the control disproved that: the real rule needs picking which of two frames is LARGER, reading THAT frame's content, then matching loose shapes against it — a three-step derive-then-use chain, not a single relation. Caught before the evaluation set was touched, not after. Replaced with `7d1f7ee8`: two framed objects have their nested content simplified, a same-general-shape decoy outside any frame does not change. Hand-verified against all 3 training pairs; no baseline atom separates this task at all, `corr:is_contained` alone does. The negative control, `810b9b61`, has a correspondence atom that is genuinely active and correlated, and the instrument correctly stayed silent because a real baseline separator (`has_holes`) also explains it.

Recorded as Extension D in a new Part 8 of `docs/VOCABULARY_PROPOSAL.md`.

### Key insight

Three vocabulary extensions have now been tested this way, each on a properly validated instrument, and the pattern is no longer a coincidence:

| device | training control | evaluation yield |
|---|---|---|
| cell-neighbourhood predicates (A) | 4 / 7 | 0 / 32 |
| symmetry & period completion (B v2) | hard case detected | 1 / 120 |
| cross-object correspondence (D) | hard case detected | 1 / 120, at chance |

`graph_difference.py`'s docstring calls a correspondence score "the natural input to the two-level rules the solver actually lacks" — and the discarded `776ffc46` control is direct evidence that gap is real. But this result says correspondence AS A SELECTOR, on its own, is not where the missing value is. What would move the needle is the derive-then-use chain itself — read something, then act on what was read — which none of A, B, C, or D attempt to build, and which is a materially bigger thing to construct than any atom catalog.

### Where this leaves the project

```
training       1000 tasks    72 solved   7.20%
evaluation      120 tasks     0 solved   0.00%
```

No solver code changed this session — pure measurement. `scripts/falsify_correspondence.py` is reusable for any future correspondence-shaped hypothesis. Four vocabulary extensions proposed, three tested, each surviving its own control and each landing at 0–1 evaluation tasks. The honest reading of `docs/VOCABULARY_PROPOSAL.md` Part 7 is looking more right with every extension added to it: no single-concept vocabulary addition is going to move evaluation off zero, because the benchmark is built to defeat exactly that strategy.

## Session — August 5, 2026 (continued)

### What we did

Went to start `docs/RESEARCH_AGENDA.md` item #17 (held-out split) as the next piece of work. It was already done — implemented and logged on July 30, and I said otherwise in conversation before checking. Reading `task_loader.py` and `solver.py` directly instead of trusting my own summary of the docs found:

- **#17 (held-out split): done 2026-07-30.** `get_dev_holdout_split` exists, `--full-eval` already reports `training-dev` / `training-holdout` / `training-all` separately. Exactly as `logs/research_log.md`'s own July 30 entry two sessions up describes — I had read that entry and then told the user it wasn't started anyway.
- **#21 (parallelism): also already done**, dated 2026-08-05, landed inside commit `236b5bf` — whose message was entirely about the library-learning probe. `run_full_evaluation_parallel` spreads all 1000 (or 120) tasks over a `multiprocessing.Pool`, `evaluate_split` uses it automatically whenever `PARALLEL_WORKERS = cpu_count - 1 > 1`. It doesn't fix #22 (module-level mutable state) — it sidesteps it, since separate OS processes each get an independent copy of the globals, so the thread-safety problem this project flagged never actually comes up. #22 stays open as a code-quality item, no longer a blocker.

**Both capabilities sat implemented and undocumented as "done" for up to six days.** This is the third time this project has found real, already-built value sitting unadvertised (`by_color`/`no_background` on June 16, the relational subsystem on July 26, now this) — except this time the cost wasn't unused code, it was *my own incorrect summary to the user*, corrected only because I checked the code instead of trusting a doc I'd already read once. Fixed `docs/RESEARCH_AGENDA.md` in place (both items marked DONE with dates, "Suggested order" updated) rather than leaving the drift for the next session to rediscover.

**Ran `--full-eval` fresh anyway**, since confirming the split and the parallel path actually work end-to-end is exactly what "do the fix" should produce even when the fix pre-exists. Result: 0 errors on both splits, accuracy unchanged — `training-dev` 60/750 (8.00%), `training-holdout` 12/250 (4.80%), `training-all` 72/1000 (7.20%), `evaluation` 0/120 (0.00%), induction precision 92% (training-all) / 91% (dev) / 100% (holdout). Tests: 48/48 (was 46 — also stale in `CLAUDE.md`, fixed).

**The runtime from that run should not be trusted, and is logged rather than acted on.** Training took 5673s and evaluation took 6026s, against the documented clean baseline of ~900s / ~150s — 6.3x and 40x respectively. Both splits slowed down, which is the signature of machine contention (rule 7) rather than a code regression (which should hit one path, not both) — but a genuine contention story predicts *similar* multiplicative slowdown on both, and 6.3x vs 40x is not similar. Windows multiprocessing uses `spawn`, not `fork`, which re-imports every module (numpy, scipy, the whole solver) per worker process rather than sharing memory — a real, structural cost that plain environment noise doesn't produce, and evaluation's larger grids (**[measured elsewhere]** 3x the area, 2x the objects of training) would make that per-task import-and-pickle overhead worse specifically on the split that already looked disproportionate. Three candidate causes, one noisy run, no controlled comparison — recorded as open rather than diagnosed. **Do not write this run's wall-clock into `CLAUDE.md` as a new baseline.** If runtime work happens next, the first step is a controlled A/B on the same task subset with `PARALLEL_WORKERS=1` forced, on an otherwise idle machine.

### Key insight

**I repeated, in miniature, the exact failure `docs/VOCABULARY_PROPOSAL.md` and this log have both already named.** Told the user "none of these are started" about a list where two items were not only started but finished, one of them by me two sessions ago. The fix both times was the same: stop trusting a summary — including my own — and check the artifact (code, in this case; grids, in the correspondence test two sessions ago) directly before acting on it or reporting it.

### Where this leaves the project

```
training-dev       750 tasks   60 solved   8.00%   precision  91%
training-holdout   250 tasks   12 solved   4.80%   precision 100%
training-all      1000 tasks   72 solved   7.20%   precision  92%
evaluation         120 tasks    0 solved   0.00%
```

Both cheap methodological items from `docs/RESEARCH_AGENDA.md`'s "suggested order" are now done and documented as such. Next up per that list: **#3, template matching** — 90% of evaluation tasks contain repeated shapes and the solver has no concept of one. A runtime investigation (the paragraph above) is a plausible detour before that if the 6-40x swing turns out to matter for iteration speed, but it hasn't been scoped yet.

## Session — August 5, 2026 (continued, solver audit)

### What we did

Asked to check selector induction, value induction, all-pair verification and the evaluation pipeline for correctness and report back — a straight code audit, not a measurement or a build. Read `induce_selectors`, `ColorExpr`/`VectorExpr`/`DecisionListExpr` and their induction functions, Draw's and Extract's form-based induction, the counterexample loop (`_refine_from`/`synthesize` in `synthesis.py`) and the sweep's own all-pairs check (`verify_hypothesis`/`verify_all` in `verifier.py`), and the whole scoring path in `solver.py`.

**Found one real bug.** `_evaluate_one`, `run_full_evaluation` and `solve_all_tasks` each independently scored a task's predictions — which `solve_task` derives from `test[0]` only, by design — against **every** entry in `solutions[task_id]`, not just the one for `test[0]`. `solutions[task_id]` is one grid per test INPUT (confirmed structurally: training task `12997ef3` has two test inputs with differently-shaped solutions, `(12,3)` and `(3,12)`), not a set of acceptable alternate answers for a single input. A task with multiple test inputs could therefore be scored "correct" by a `test[0]`-derived guess coincidentally matching a *different* test input's answer, without the solver ever looking at that input.

**Measured the actual damage before calling it urgent.** Ran all 69 multi-test-input training tasks directly and compared: **0 currently benefit from the leniency**, so `72/1000` was never inflated by it. But 41% of evaluation tasks have multiple test inputs against 7% of training (already-measured, `docs/RESEARCH_AGENDA.md` #19), so the exposure sits almost entirely on the split that counts, and the all-zero/copy-of-input fallback prediction is a plausible coincidental match against an unrelated same-shaped answer as the solver improves.

**Fixed and deduped in the same change**, since the bug was the identical ~10-line loop copy-pasted three times, not three separate mistakes. Added `score_predictions(predictions, correct_outputs)` in `solver.py`, checking only `correct_outputs[0]`, and replaced all three call sites with it. 48/48 tests still pass (none of them had ever exercised multi-test-input scoring — confirmed by grep before writing the fix, which is also why this shipped unnoticed). Did not re-run the full evaluation to confirm the numbers are unchanged, because the argument is airtight without it: the new check only *removes* acceptances relative to the old one, training's removed acceptances were already measured at zero, and evaluation was already `0/120` under the strictly more permissive old logic, so it cannot have gone down. Smoke-tested the new code path directly on 5 multi-test-input tasks to confirm it runs clean end to end.

### Key insight

Two other things surfaced while reading, neither worth fixing on their own: a `finally: pass` in `synthesize_program` that does nothing — the real "only reset the active interpretation on failure" logic lives entirely in the control flow (return-inside-try vs. falling through), not in that block, so it's a landmine for whoever next assumes the finally clause is where cleanup happens; and `compatible()` in `synthesis.py` is defined but called from nowhere in the repo.

The bug itself is a good example of the class of thing a single accuracy number cannot catch: it can only ever move the count in the generous direction, so a solver that's currently bad at everything (0/120 eval) will never show it — it would first appear as an unexplained jump the day the solver starts actually getting eval tasks right, which is exactly the moment a false signal would be most damaging to trust.

### Where this leaves the project

Fix is committed; `docs/RESEARCH_AGENDA.md` doesn't need a new entry since this was never on it — the review that found it wasn't chasing a documented flaw, it was reading code fresh. Worth remembering next time a session starts "what's the highest-value thing to do" purely from the agenda: an undirected read of core code found something the agenda-driven measurement work of the last week did not.

## Session — August 6, 2026: hand-reading the eval set for two-level rules, and where it led

### What we did

Went looking for the one hypothesis nothing had tested yet: the richer form of "two-level rules" `776ffc46` needed when it got dropped from the Extension D control — select a source by a derived property, read its content, apply that content elsewhere. Pre-registered a threshold (12 of 120, higher than Extensions B/D's 8, because this needs a new hole type plus an induction procedure rather than one more atom) and hand-read all 120 evaluation tasks cold, no automated detector, the same method that found `135a2760`/`28a6681f`/`dd6b8c4b`. Landed at **21 MATCH, 13 MAYBE** — the first result all week to clear its threshold.

**Then verified the 5 highest-confidence calls with exact computed diffs instead of eyeballing grids, and it did not hold up.** `62593bfd` turned out to be four objects each doing a plain vertical Move to a new row — same shape, same colour, already-expressible, no content read at all. `b10624e5` turned out to be self-referential: each coloured block's own size determines two new same-sized blocks placed next to it, needing no separate source object. `16b78196` held up but was messier than described (several loose motifs merge into one composite, not a single shape-to-notch match). `7c66cb00` held up in spirit but the real mechanism was smaller and different: recolour to match your own containing region's colour. `e87109e9` stayed genuinely unresolved. **2 of 5 clearest calls collapsed outright.** Extrapolating that attrition rate across all 21 puts the honest count at roughly 8-13 — right on the pre-registered threshold, not comfortably above it. The raw 21 was a real overconfidence, most visible on exactly the ambiguous middle cases, which fits: three losses in a row create real pressure to finally find something that survives, and that pressure shows up on judgment calls, not the clear ones.

**Built the one piece that survived verification anyway, to get a real number instead of another hand-estimate.** Added `from_container_color` to `synthesis.py`: a new `ColorExpr` kind reading the colour of the smallest object whose bounding box contains the selected one, structural (cost 1, no stored table) rather than memorised. Verified end-to-end on a constructed case first — two nested-ring/inner-pixel pairs where the inner pixel's own colour and size are identical across both pairs but the ring's colour differs, so `literal` and `from_color` are both mathematically impossible and only the container read can explain both. The counterexample loop behaved exactly as designed: induces the cheaper `literal(5)` from pair 1, pair 2 disproves it, re-induces onto `from_container_color`, final check passes. Added 3 tests (offer-when-explains, absent-without-a-container, the full loop case) — 51/51 passing.

**Measured yield on the actual benchmark: exactly zero.** `7c66cb00` itself still doesn't solve — checked empirically before writing any code, not just predicted: no single scene interpretation gives one clean container per section across all four of its sections at once (`by_color` merges unrelated same-coloured regions across sections; `no_background` still splits each border into two disconnected strips). Full `--full-eval`: training-all 72/1000 (unchanged), evaluation 0/120 (unchanged), identical attribution and precision to the pre-change baseline in every field. A correctly-built, honestly-verified, zero-regression capability, measured with the same rigor as everything else this week, added nothing.

### Key insight

This closes the loop on the whole week rather than opening a new one. Four extensions now proposed, all four tested with real controls or real measurement, all four landing at 0-1 tasks or an outright zero: cells (0/32), symmetry (1/120), correspondence (1/120, at chance), and now two-level rules — the one candidate that survived its own falsification test still measured to nothing once actually built. The pattern is no longer "unlucky four times" — `VOCABULARY_PROPOSAL.md` Part 7 already named it: the benchmark is constructed so that no single concept covers many tasks. This session adds a sharper version of that finding: it isn't only that no single concept covers many tasks, it's that a correctly-verified, individually-real concept still covers **none**, because what's missing is either something structurally larger than any one addition, or a long tail of small, mutually unrelated capabilities of which this specific one wasn't a member for this particular 120-task set.

Asked directly whether this architecture could reach 90%: no, and not as a fixable engineering gap. Two separate ceilings — ARC-AGI-2 is built to resist brute force and no system, symbolic or LLM-ensemble, is within reach of 90% on it; and this architecture specifically is a no-ML system whose entire language is "for each object, do X based on that object's own properties," which this week's evidence (three falsifications plus a correctly-built capability measuring to zero) shows cannot close a gap made of dozens of unrelated missing concepts through language extension alone.

### Where this leaves the project

```
training-dev       750 tasks   60 solved   8.00%   precision  91%
training-holdout   250 tasks   12 solved   4.80%   precision 100%
training-all      1000 tasks   72 solved   7.20%   precision  92%
evaluation         120 tasks    0 solved   0.00%
```

`from_container_color` stays in the language — it's cheap, structural, tested, and cost nothing in precision or runtime; there's no reason to gate it the way the genuinely harmful flags in `CLAUDE.md`'s table are gated, it simply isn't the piece that was missing. `docs/VOCABULARY_PROPOSAL.md` is not updated with this session's finding yet; worth doing before the next vocabulary-shaped idea gets proposed, so the pattern is visible in the document built to track exactly this pattern.

## Session — August 10, 2026: cross-operator cost ranking, another verified zero

### What we did

Asked directly whether this architecture could reach 90% (no — see the closing note two sessions up) and then to ideate on genuinely different reasoning strategies rather than another vocabulary extension. Landed on version-space algebra: instead of picking one hypothesis and testing it against the next pair, maintain the set of every hypothesis consistent with the evidence and narrow it as pairs arrive.

Worked out what that means concretely for this codebase before writing anything, since "version-space algebra" as a textbook idea needed translating into this solver's actual structure. Two pieces of it turned out to already exist: `induce_selectors` already enumerates every depth-≤2 selector and filters to what's consistent with *all* evidence at once, and `_refine_from`'s counterexample loop already re-induces from the full accumulated evidence at every step rather than patching one hypothesis — both are why "more restarts" and "deeper conjunctions" measured to zero earlier this project: there was never a discarded alternative sitting around waiting to be found. The one place true version-space thinking was absent: `synthesize_program` tried each operator (Recolor, Move, Draw, Extract, every composed pair) in a fixed list order and returned on the *first* one that produced any complete rule, with no comparison against what a different operator might have found more cheaply.

**Built it.** Threaded a comparable `induce_cost` through every one of the four `induce()` implementations (`Operator`, `DrawOperator`, `ExtractOperator`, `ComposedOperator` — the last of which had no internal ranking at all before this and now sums its two stages' costs, which also fixed it never sorting its own composed candidates). `synthesize_program` now collects every operator's result within an interpretation and returns the cheapest, only falling through to a later interpretation if nothing succeeded at all — preserving the existing "default interpretation costs nothing extra" design while fixing first-match-wins within whichever interpretation actually gets used. 51/51 tests still pass.

**Measured: 72/1000 training, 0/120 evaluation — byte-identical attribution and precision to before the change.** Skipped the cheap pre-check I'd proposed (does any currently-solved task actually have a cheaper alternative sitting unused behind the one that won?) at explicit instruction, and the full run confirmed what that check would have shown for free: no task in this evidence had two operators competing for the same rule, so first-match-wins and cheapest-wins were never actually in disagreement here. The skipped measurement wasn't wasted effort avoided, it was signal deferred to a slower, more expensive channel.

**Runtime note, not a claim.** This run — genuinely more work per task than any run all week, since nothing short-circuits on first success anymore — finished in 2236s training / 314s eval, the fastest full run this entire project has logged, well under the 5673-9542s range recent runs have shown. Consistent with (not proof of) the standing read that the multi-hour swings this project keeps hitting are machine contention, not code: more real work, less wall-clock, on the same machine, same week.

### Key insight

The change is real, tested, and correct — a rule no longer wins an operator competition just by being first in a list — and it is the second capability this week (after `from_container_color`) that is individually verified, individually justified, and individually worth exactly nothing on this benchmark. That is itself information: it says the operators this solver already has rarely genuinely compete for the same task. Most tasks that get solved have exactly one operator that can explain them at all — which lines up with `induce_selectors`' `all` being the winning selector in most solves (the July 30 log entry) and with `#15b`'s finding that selection does little work in most successful solves. The engine is not leaving cheap wins on the table by choosing badly between competing rules; when it fails, it fails because nothing in the language reaches the task at all, which is the same conclusion five different tests have now converged on from five different angles.

### Where this leaves the project

```
training-dev       750 tasks   60 solved   8.00%   precision  91%
training-holdout   250 tasks   12 solved   4.80%   precision 100%
training-all      1000 tasks   72 solved   7.20%   precision  92%
evaluation         120 tasks    0 solved   0.00%
```

Cross-operator ranking stays in the language for the same reason `from_container_color` did — correct, cheap, harms nothing — but it is not the lever. Two reasoning-algorithm ideas were raised this session; version-space algebra (this one) is spent. The other, treating "describe the structure, then describe the deviation" as the primary induction strategy rather than an add-on descriptor, has not been tried and remains open.
