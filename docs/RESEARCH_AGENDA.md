# Flaws, bottlenecks and research directions

Every item below is a real limitation of the solver as of 2026-07-30
(64/1000 training, 0/120 evaluation). Measured claims are marked
**[measured]**; the rest is analysis and should be treated as a hypothesis
to test, not a fact.

Ordered by how much a fix would plausibly matter, not by difficulty.

---

## Tier 1 — the things actually blocking evaluation

### 1. The rule language is single-level

Every rule the solver can express has one shape:

> for each object, do X to it, where X depends on **that object's own
> properties**

There is no way to say "derive something from part of the grid, then use
it elsewhere". No intermediate value is ever computed and then consumed.

**[measured]** 102/120 evaluation tasks cannot even be *observed* by any
operator; of the 18 that can, all 18 fail to explain a single training
pair; the 14 single-concept evaluation tasks yield 0 solves — so it is not
a depth problem, the individual transformations are inexpressible.

**Research:** inductive program synthesis over a DSL with higher-order
constructs; version-space algebra (Lau/Gulwani); sketch-based synthesis
where a hole may be filled by an *expression over the scene* rather than a
constant. The minimal concrete step is "parameter lifting": let a value
expression read from a key region instead of being a literal. Note
**[measured]** only 22 training and **0** evaluation tasks have the simple
form of this, so scope it as a training-side win unless a richer version
is designed.

### 2. No counting or arithmetic

The solver cannot represent a number derived from the scene. "Output a
grid whose size is the object count", "recolour by how many neighbours a
cell has", "the answer is the shape that appears three times" are all
unreachable.

**Research:** quantitative extensions to the DSL; how DreamCoder-style
systems bootstrap arithmetic primitives; whether counting belongs as a
value expression (`CountOf(selector)`) or a separate operator class.
Cheap first probe: how many unsolved tasks have an output dimension equal
to some countable property of the input?

### 3. No template or motif matching

**[measured]** 90% of evaluation tasks contain repeated shapes. The
solver has no concept of "this motif appears again over there" — shapes
are compared only by exact fingerprint equality inside a single grid.

**Research:** perceptual grouping / Gestalt principles as inductive bias;
second-order (pattern) unification; approximate shape matching under
transformation groups. This is the single largest structural signal in the
evaluation data and the solver is blind to it.

### 4. No multi-scale or recursive structure

Tiling and scaling are flat, one-shot operations. A task that redraws its
input at two nested scales (`eee78d87`, 6×6 → 16×16) is unreachable in
principle, not just in practice.

**Research:** recursive program synthesis; L-systems; self-similarity
detection as a preprocessing step that proposes a recursion depth.

---

## Tier 2 — verification and search

### 5. Verification is exact-match only

A rule reproducing 99% of the output scores identically to one
reproducing 0%. So a near-miss carries no signal and cannot guide repair,
which is why the loop can only accept or discard.

**Research:** partial-credit objectives (per-pixel, per-object, or
edit-distance); how to use them *without* admitting rules that fit
loosely — the current all-or-nothing check is exactly what gives the
engine its 90% precision, so any softening needs to preserve that.

### 6. No program repair

When a rule fails on pair 2, the whole rule is thrown away and re-induced
from scratch. Nothing tries to *patch* it.

**Research:** counterexample-guided inductive repair; treating a failing
rule's residual as a sub-problem to solve and compose (the engine already
does this for composition, but only across operator boundaries, never
within a rule).

### 7. The loop cannot benefit from more search

**[measured]** Restarts gained 0; depth-3 conjunctions found 1 selector
in 43 tries and both programs were wrong; an hour of widened search on one
task died in **1 second** with all 51 combinations failing at
*observation*. Because `_refine_from` re-induces from all accumulated
evidence at every counterexample, every starting point converges to the
same candidate set.

**Consequence, worth internalising:** the only levers are *richer
evidence* or *a better language* — never better search over the language
we have. This retires a whole family of tempting ideas.

### 8. Overfitting is fought with hand-tuned heuristics

`selector_cost`, expression costs, the symbolic-key restriction on
geometric decision lists — all hand-designed after observing specific
failures. They work **[measured]** (the Move restriction took precision
29% → 100%) but they are not principled and will not generalise to new
operators.

**Research:** minimum description length as the ranking criterion;
Bayesian program induction with a prior over programs (Lake et al. 2015);
whether a single principled complexity measure reproduces the hand-tuned
orderings.

### 9. No probabilistic ranking

A rule either fits the evidence or does not. There is no posterior, no
confidence, no way to say two rules are equally plausible and hedge.

**Research:** probabilistic program induction; using a posterior to pick
the *two* Kaggle attempts to maximise the chance one is right —
**[measured]** attempt 2 currently adds nothing (29 solved tasks had
alternates, 0 were saved by one).

---

## Tier 3 — representation

### 10. Object extraction is a fixed enumeration

Five interpretations are searched (`connected4/8`, `by_color`,
`no_background`, `panels`). The real space is much larger and includes
**nested** objects — a shape made of shapes — which nothing here can
represent. Objects are flat and non-overlapping by construction.

**[measured]** `connected8` alone earned 3 tasks, so this axis does pay.

**Research:** hierarchical/part-whole object models; overlapping
segmentations; grouping by texture, symmetry or repetition rather than
adjacency.

### 11. Background is an unvalidated heuristic

Most-frequent colour, with a runner-up candidate added. It is still a
guess, and when wrong every object, diff and rule downstream is wrong
with nothing raised.

**Research:** infer background jointly with objects as a latent variable
chosen by whichever reading makes the task explainable — the same
"search over interpretations" trick already used for connectivity.

### 12. No cross-pair object correspondence

Objects are matched input↔output *within* a pair. Nothing reasons about
how objects correspond *across* training pairs, which is where the real
regularity often lives. `src/layers/graph_difference.py` was built and
tested for exactly this and is still unwired.

**Research:** graph matching / structural correspondence; whether a
correspondence score can rank scene interpretations instead of the
current fixed order.

### 13. No relational or arrangement-level geometry

Objects have bounding boxes and centroids. There is no way to say "the
third object from the left", "the one in the middle row of the
arrangement", or to use a polar/relative frame. Anchor-relative
predicates were tried and **[measured]** gained 0 while costing 11 points
of precision — but that was one narrow implementation, not proof the
whole idea fails.

### 14. Verification shares extraction with induction

If object extraction misreads a grid, verification misreads it the same
way and cannot catch the error. The two are not independent, so a
systematic representation error is invisible.

---

## Tier 4 — learning

### 15. No cross-task learning whatsoever

Every task is solved from scratch. Nothing accumulates. A concept
discovered on task 300 is unavailable on task 301.

**Research:** DreamCoder's wake/sleep library learning — discover reusable
abstractions from solved tasks and add them to the DSL; this is the single
biggest architectural difference between this solver and the strongest
published program-synthesis approaches.

### 16. No learned search guidance

Search order comes from hand-written cost functions. Nothing learns which
operators or interpretations are likely for a given task's appearance.

**Research:** neural-guided synthesis; a cheap classifier over task
features predicting which operator to try first (would mostly buy speed,
given #7 says search order is not what limits correctness).

---

## Tier 5 — methodology (cheap to fix, distorts everything above)

### 17. There is no held-out split *within* training

All 1000 training tasks are used both to design the solver and to score
it. Every form, atom, cost heuristic and threshold was chosen by looking
at those tasks. **So 6.40% is optimistic even as a measure of
"generalises to unseen tasks of this difficulty."**

**Fix:** split training into dev/holdout, tune only on dev, report both.
This is the cheapest high-value methodological fix available and would
make every future number trustworthy.

### 18. Evaluation is no longer pristine

Several evaluation tasks have been inspected while diagnosing the 0
(`abc82100`, `eee78d87`, plus a whole-split mechanism classification). Not
yet designed against, but the independence is weakened.

### 19. Multi-test-input tasks are barely exercised

**[measured]** 49/120 evaluation tasks have more than one test input
(41%, versus 7% on training) and **all** of them must be correct for the
task to count. Local scoring uses `test[0]` only, so this path is nearly
untested.

### 20. Bucket size has overestimated yield ~10× every time

Not a code flaw but a persistent estimation bias worth naming:
`COLORS_ADDED` 106 tasks → 2 solves; Extract 230 → 15 form matches → 12
observable → 5 solves. Always measure the *form match*, then the *hole
solvability*, before committing.

---

## Tier 6 — engineering

### 21. No parallelism

**The single biggest cheap win.** 1000 tasks are completely independent
and run serially on one core. A ~900s cycle could plausibly be ~150s.
Blocked by #22.

### 22. Module-level mutable state

`_ACTIVE_INTERPRETATION` and `_in_region_split` are globals, and
`synthesize_program` deliberately *leaks* the winning interpretation so
`apply` reads the grid the same way. Correct but fragile, and it makes the
solver non-thread-safe, which blocks #21.

**Fix:** thread the interpretation through explicitly, or bind it into the
returned program object.

### 23. `synthesis.py` is 2582 lines

Should be 4-5 modules (language/atoms, operators, composition, loop,
interpretations). No functional gain, but it is now the main file and
hard to navigate.

### 24. Timing on this machine is unreliable

**[measured]** identical code has reported 914s and 38012s — a 42× swing
from contention and overnight throttling. Any performance claim needs a
clean, idle re-run before it is believed.

---

## Suggested order if optimising for learning

1. **#17** (held-out split) — makes every subsequent number honest
2. **#21/#22** (parallelism) — 6× faster iteration compounds into everything
3. **#3** (template matching) — largest measured structural signal in eval
4. **#15** (library learning) — the biggest architectural gap versus
   published work
5. **#1** (two-level rules) — the clearest expressiveness gap

## Suggested order if optimising for score on training

1. **#10** (richer object readings — this axis has already paid)
2. **#2** (counting; probe first)
3. **#6** (repair, so near-misses are not wasted)
