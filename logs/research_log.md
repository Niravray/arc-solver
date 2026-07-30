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
