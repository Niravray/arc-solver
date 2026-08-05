# Vocabulary proposal: cells, structure, and description length

Written 2026-07-30 against the evaluation diagnosis in this document's
first section. **Proposal only — nothing here is implemented.**

The constraint this respects: *build a richer language in which
transformations can be inferred from evidence, not a bigger set of
primitives searched combinatorially.* Every item below is an **inducible
hole**, not an enumerated operation.

---

## Part 1 — The diagnosis this is built on

### Evaluation funnel (120 tasks)

```
TOTAL                                   120
  observable by a STRICT operator        24
  observable only by TOLERANT composed   57
  not observable at all                  39
  candidate rule induced                  0   <-- everything dies here
  correct                                 0
  exceptions                              0
```

**Pipeline bugs ruled out**: task ids match solution keys, every task has
train and test, solution counts match test counts, the scorer credits
ground truth 30/30, and zero exceptions occur.

### Where the 39 unobservable tasks fail

| count | family |
|---|---|
| 31 | geometric construction: output smaller, not an object crop |
| 5 | geometric construction: output size varies |
| 3 | geometric construction: output larger (build/tile/scale) |

One family: **the solver can modify an input, never construct an
output.**

### Where our best-covered operator fails

Strict `Draw` observes 18 evaluation tasks and induces **0**:

| count | stage |
|---|---|
| **17** | **no FORM reproduces the added cells** |
| 1 | form matched, selector/colour hole failed |

**94% of Draw's failures are the generative form, not the holes.**

### What the small edits look like

Of 15 evaluation tasks changing ≤20 cells: densities of 0.02–0.10 (cells
scattered across a wide span, not a blob), touching 3–20 cells that sit
*on* existing objects while adding others in empty space, in scenes with
10–24 objects.

**These have no whole-object story.** Our entire vocabulary assumes
"identify objects, act on them as units."

### The corroborating oddity

`all` is the induced selector in 24 of ~40 solved programs. Selector
induction — which the architecture treats as its shared core — does no
work in most successful solves. **The work is in the form.** And forms
are exactly where evaluation breaks.

---

## Part 2 — What strong systems represent that we do not

Mined for *abstractions*, not primitives.

### From ARC-NCA (neural cellular automata)

A transformation is a **local update rule applied at every cell**, where
a cell's new state is a function of its neighbourhood. The unit of
description is the cell-in-context, not the object.

We have no cell-level subject at all. Every operator we own takes whole
objects.

### From MADIL (MDL-based synthesis)

Two things:

1. **Structure as a compression unit.** A periodic region becomes *one*
   representational unit — a period plus a tile — rather than a
   collection of redundant cell descriptions. Symmetry likewise: describe
   a region once plus a symmetry operator.
2. **Description length as the single selection criterion.** Competing
   descriptions (periodic? symmetric? irregular?) are ranked by total
   encoding cost = program length + residual length.

We have neither periodicity nor symmetry as scene descriptors, and our
ranking is a pile of hand-tuned costs.

### From the joint-compression argument

Describing only *output given input* leaves the input uncompressed.
Regularities that live in the input's own structure are never named.

We only ever describe the change, never the scene.

### The abstraction all three share

**Describe the structure, then describe the deviation from it.** None of
them enumerate transformations; they build a compact description of what
is regular and treat the transformation as what breaks or completes that
regularity.

---

## Part 3 — Three proposed extensions

### Extension A — the CELL as an inducible subject

**WHAT concept is missing?** A cell selected by its *neighbourhood
context* rather than by object membership.

**WHICH failed tasks require it?** The 15 scattered small-edit tasks
directly. More importantly, the 17 Draw form-failures: `halo8` is
literally "cells 8-adjacent to non-background", `bbox_fill` is "cells
inside an object's bounding box". **We hand-wrote 20 Draw forms because
we could not induce cell selection.** If cells become inducible subjects,
those forms stop being enumerated and start being *learned*.

**HOW observed?** Identically to objects, one level down: cells whose
colour changed are positives, cells that did not are negatives. Every
grid gives thousands of labelled examples — far more evidence per task
than the handful of objects we currently learn from.

**HOW are holes induced?** `induce_selectors` **unchanged**, over a cell
atom catalog instead of an object one:
`neighbour_count(colour c) == k`, `is_adjacent_to(colour c)`,
`in_bbox_of_some_object`, `on_row_with(colour c)`, `distance_to_nearest(c)
== d`, `is_background`, `has_n_nonbg_neighbours`.

**HOW does it generalize?** Neighbourhood predicates are *structural*, not
positional — "cells with exactly two red neighbours" transfers to a grid
of any size with any layout. This is the same structural-beats-metric
principle already measured as worth +2 tasks.

**HOW much search does it add?** None combinatorially — it is one more
subject type for the existing loop. Cost is per-cell evaluation of the
atom catalog: ~900 cells × ~40 atoms worst case, comparable to today's
relational annotation. Guarded the same way.

**HOW is memorization penalized?** The existing cost model applies
directly: a predicate keyed on absolute position (`row==3`) is metric and
must lose to a structural one (`has_2_red_neighbours`), exactly as
`height==2` loses to `not_is_single_pixel` today.

**WHAT would falsify it?** Take the 15 small-edit tasks plus the 17 Draw
form-failures. For each, label changed vs unchanged cells and ask whether
*any* predicate over a fixed neighbourhood catalog separates them on
every training pair. **If fewer than ~5 of 32 admit a separating cell
predicate, the abstraction is wrong and should not be built.**

### RESULT: FALSIFIED (2026-07-30). Do not build this.

```
target tasks              32
separable by a cell rule   0     (pre-registered threshold: 5)
```

**Zero of 32.** Run before any implementation, against a threshold fixed
in advance.

**The instrument was validated first.** On seven training tasks we
already solve with a known cell-level rule, the same catalog separated
**4 of 7** — and it recovered exactly the predicates this proposal
claimed the Draw forms secretly are:

```
4258a5f9  SEPARABLE  adj8_colour5==True                 (was: halo8)
dc1df850  SEPARABLE  adj8_colour2==True                 (was: halo8)
3aa6fb7a  SEPARABLE  is_background AND in_object_bbox   (was: bbox_fill)
6d75e8bb  SEPARABLE  is_background AND in_object_bbox   (was: bbox_fill)
```

So the *"hand-written forms are really uninduced cell predicates"* claim
was correct — and still did not matter, because the evaluation tasks are
not decided by cell context at all.

The three control misses are the informative part. `00d62c1b` and
`a5313dff` need "not reachable from the grid border", which **no 3×3
neighbourhood can express**: enclosure is a connectivity property, and
connectivity is global.

**Conclusion, and it generalises past this extension:** the problem was
never that we describe the wrong *unit*. Cells are not a better unit than
objects here — both are LOCAL descriptions. Whether a cell or an object
changes in these tasks is decided by NON-LOCAL, relational structure:
which object it belongs to, whether that object is the largest, whether
it is enclosed, how it corresponds to something else in the scene.

This retires the whole cellular-automata / local-rule family for these
tasks, and it shifts weight toward descriptors that are inherently global
(Extension B) and toward cross-object correspondence — which is exactly
what the unwired `graph_difference.py` computes.

---

### Extension B — SCENE DESCRIPTORS: periodicity and symmetry

**WHAT concept is missing?** A description of what is *regular* about a
grid, independent of any change.

**WHICH failed tasks require it?** 90% of evaluation tasks contain
repeated shapes (measured). The pure-erasure and pure-addition small
edits look like **breaking or repairing a regularity**. Several of the 39
construction tasks are plausibly "output the tile" or "output the
repaired grid".

**HOW observed?** Computed, never searched: row/column period by
autocorrelation; symmetry group by testing h/v/180/diagonal in O(cells);
the set of cells that *violate* each regularity.

**HOW are holes induced?** Two new expression forms, both derived:
- `CompleteSymmetry(<which axis?>)` — axis solved by which symmetry the
  output satisfies and the input does not
- `CompletePeriod(<which period?>)` — period read off the input directly

The violating cells *are* the positives. No enumeration.

**HOW does it generalize?** A period or an axis is a property of the
grid, recomputed per test input. Nothing task-specific is memorized.

**HOW much search does it add?** O(cells) per descriptor, computed once
per grid and cached like the relational graph.

**HOW is memorization penalized?** These are the cheapest possible
descriptions — one integer or one axis name. Under any description-length
ranking they beat lookup tables automatically, which is the correct
outcome.

**WHAT would falsify it?** Count evaluation tasks where (a) some symmetry
holds in the output but not the input, or (b) the output is the input's
period-completion. **If under ~8 tasks qualify, the yield does not justify
the machinery.**

### RESULT: **RETRACTED — the test was wrong, not the extension.**

The original run reported 0/120 qualifying against a threshold of 8, and
this document said "FALSIFIED". **That verdict is withdrawn.** The
detector measured WHOLE-GRID periodicity and WHOLE-GRID symmetry. Reading
task `135a2760` by hand shows the regularity is *regional*:

```
3333333333333   <- outer frame, colour 3
3222222222223   <- inner frame, colour 2
3213131333123   <- periodic interior, ONE defect
3222222222223
3333333333333
```

Interior `[1,3,1,3,1,3,3,3,1]` under period 2 has canon `{0:1, 1:3}` and
exactly one defect — at the precise cell the output changes. The whole-row
and whole-grid tests both fail here because the *frame cells* break any
regularity measured across the full extent.

The control passed 3/3 only because those three training tasks happen to
be **globally** symmetric, which is exactly the case the instrument could
see. A control that only covers the easy case cannot validate an
instrument against the hard one.

### RESULT (superseded heading kept for history)

```
output gains a symmetry the input lacks (every pair)  :  1
changed cells ARE the symmetry violations             :  0
output is the input's period-completion               :  0
QUALIFYING                                            :  0 / 120
```

Threshold was 8, fixed before running. **Zero.**

**Control passed 3/3** on training tasks solved by mirror forms —
`496994bd` (gains mirror_ud and rot180), `f25ffba3` (mirror_ud),
`e729b7be` (rot180), all with the changed cells matching the symmetry
violations exactly. The detector works; evaluation simply has none of
this.

---

## Part 5 — The pattern across both falsifications

Three times now, a representational device has been validated on
training tasks and produced **zero** on evaluation:

| device | training control | evaluation |
|---|---|---|
| cell-neighbourhood predicates | 4 / 7 | **0 / 32** |
| symmetry & period completion | 3 / 3 | **0 / 120** |
| the object operators themselves | 72 solves | **0 / 120** |

Every instrument works where it was designed and finds nothing where it
was not. That is a stronger and more uncomfortable signal than either
individual result.

**The hidden assumption both extensions shared:** that the
transformation is a function of the input grid's *geometry* — local
neighbourhood in A, global symmetry in B — so that computing the right
geometric property would reveal which cells or objects change. Evaluation
tasks appear not to work that way.

**What has never been tested:** whether the deciding structure is
*correspondence* — this thing relates to that thing — rather than any
property computable from one grid alone. `src/layers/graph_difference.py`
computes exactly that and has been built, tested and unwired for five
days.

**Methodological note.** Both extensions were hypotheses generated from
intuitions formed on training tasks, then tested against evaluation. Both
failed the same way. Before proposing a third, the honest step is to stop
generating candidates from training intuition and read a sample of
evaluation tasks directly — describing what each actually requires, with
no hypothesis in hand. Structural taxonomy has been done; semantic
reading has not.


---

### Extension C — DESCRIPTION LENGTH as the single ranking principle

**WHAT concept is missing?** A principled cost. We have `selector_cost`,
expression costs of 0/1/10/100, `MAX_CONJUNCTION`, and a bespoke
symbolic-key restriction on geometric decision lists — each added
reactively after a specific failure.

**WHICH failed tasks require it?** None directly. This is **precision
insurance**, and it matters precisely because Extensions A and B widen the
language, and every widening this project has attempted cost precision
until a restriction was hand-added afterwards.

**HOW induced?** Not induced — it replaces the ranking function. Cost of
a program ≈ bits to encode its structure + bits to encode its parameters.
A literal is a few bits; a 10-entry lookup table is ~10× that; a 3-atom
conjunction more than a 1-atom one. The existing orderings should fall out
rather than being stipulated.

**WHAT would falsify it?** Re-rank today's 72 solutions by description
length. **If the MDL ordering disagrees with the hand-tuned ordering on
the cases where the hand-tuning was measured to matter** — `72ca375d`
(structural beat metric), the Move decision lists (colour keys beat size
keys) — then MDL is not capturing what those restrictions capture, and
the hand-tuned costs should stay.

---

## Part 4 — Ordering, and what is deliberately excluded

**Order:** falsification experiment for **A** first. It is a measurement,
not a build, and A is the only extension that addresses both the largest
failure family (Draw's form gap, 17 tasks) and the scattered-edit tasks
(15) with a single mechanism — while *removing* hand-written forms rather
than adding primitives.

Then **B** if its count clears. Then **C** before either ships, since it
is the guard on the precision they put at risk.

**Explicitly NOT proposed:**

- more transformation primitives to enumerate
- deeper composition (measured: three-stage contributed 0)
- larger beam or restarts (measured: 0 gain; the loop converges because
  it re-induces from all evidence)
- deeper conjunctions (measured: 1 selector in 43 tries, both wrong)
- copying arc-dsl's 181 primitives into a search space
- library learning (measured: best composed motif recurs twice — nothing
  to compress until the corpus roughly triples)

**The through-line:** every one of the above adds *search over the
existing representation*. The measurements say the representation is the
constraint — failures happen at observation and at form, before search
begins. Extensions A and B add **new things to say**; C keeps the language
honest while they are added.


---

## Part 6 — Extension B, retested properly

The retracted test measured whole-grid regularity. The corrected one
**segments first, then measures inside each region**: peel uniform border
rings, split on uniform lines, then look for per-row and per-column
periodicity with a bounded number of defects. All computed, no search.

**The control was designed to be able to fail.** The previous control
contained only globally-symmetric tasks — exactly what the broken
instrument could already see — so it passed 3/3 while the instrument was
useless. This one includes `135a2760`, the framed regional case verified
by hand, as a known-positive the detector *must* find.

```
CONTROL  135a2760 (regional, framed, known-positive):  DETECTED
QUALIFYING:  1 / 120        (pre-registered threshold: 6)
VERDICT:     FALSIFIED, on a validated instrument
```

Regional periodicity is real, and it is worth **one** evaluation task.

---

## Part 7 — What three falsifications add up to

| extension | instrument validated? | evaluation yield |
|---|---|---|
| A — cell-neighbourhood predicates | yes (4/7 control) | **0 / 32** |
| B v1 — whole-grid regularity | **no** — retracted | (void) |
| B v2 — regional regularity | yes (hard control detected) | **1 / 120** |

Two extensions, properly tested, worth a combined **one task**.

**The meta-finding is the useful part.** Reading the tasks directly
showed `135a2760` is periodic repair, `28a6681f` is staircase
regularisation, `dd6b8c4b` is noise removal inside framed regions — three
tasks, three unrelated concepts. That is ARC-AGI-2's evaluation set
working as designed: it is *constructed* so that no single concept covers
many tasks.

**So the strategy of "find the high-leverage vocabulary extension" may
have no winner.** Every candidate tested lands at 0–1 tasks, and the
reason is structural rather than a failure of imagination: the benchmark
is built to defeat exactly that strategy. A vocabulary that covers 6+
evaluation tasks would have to be *general* in a way none of these
geometric descriptors are.

**Methodological lesson, worth more than the extensions:** a control must
include a case the instrument is expected to find HARD. The B v1 control
passed 3/3 and certified a broken detector, because every control case
was one the detector was built for. Cheap thresholds fixed in advance
caught two bad ideas; only a properly adversarial control caught the bad
*instrument*.

---

## Part 8 — Extension D: correspondence, tested

Part 5 named the question no one had tested: is the deciding structure
*correspondence* — this object relates to that one — rather than any
property computable from a single grid? `src/layers/graph_difference.py`
has sat built and unwired on exactly this bet for five days. This is that
test: `scripts/falsify_correspondence.py`.

**What was measured.** For every evaluation task, label each input object
CHANGED or UNCHANGED. Ask whether a depth ≤2 conjunction — the solver's
real `MAX_CONJUNCTION` — separates the two sets on every training pair
simultaneously, under any of the solver's five scene interpretations.
Two atom pools: the solver's existing catalog (BASELINE), and eight new
relational atoms computed pairwise against every other object in the same
scene — exact shape twin, twin up to rotation/reflection (dihedral,
excluding the exact match), scaled-integer copy, bounding-box containment
in either direction, and majority-shape membership. The decision-relevant
count is **CORR-ONLY**: tasks a correspondence atom separates that no
baseline atom does — coincidence with what the solver can already say
doesn't count as new capability.

**Pre-registered before the evaluation set was touched:** CORR-ONLY ≥ 8 ⇒
survives (same order of magnitude as A's and B's thresholds, discounted
for the 3–10× attrition every "observable" count in this project has
shown between separability and an actual solve).

**The positive control took two tries, and the first miss is the useful
part.** The original choice, training task `776ffc46`, was picked from a
30-second read: a framed template holds a shape, loose shapes matching it
get recoloured. Reading it fully to build the control disproved that read
— the true rule needs choosing *which of two frames is bigger*, reading
*its* content, and matching loose shapes against *that* — a three-step
derive-then-use chain, not a single correspondence relation. Using it
would have blamed a miss on correspondence when the actual gap is the
"two-level rules" line already in `CLAUDE.md`. Replaced with `7d1f7ee8`,
hand-verified against all 3 training pairs: two framed objects have their
nested content simplified, a third same-general-shape object outside any
frame is a decoy and is left alone. No baseline atom separates this task
at all; `corr:is_contained` alone does, exactly.

**The negative control also came from reading the task, not from what the
detector could already pass.** `810b9b61` recolours objects that enclose a
background hole (`has_holes` — a single-object property) and, under one
scene interpretation, `corr:contains_object` happens to be active and
correlated. The instrument correctly stays silent on CORR-ONLY because a
baseline separator (`has_holes`) is also found — coincidence, not
capability.

```
CONTROL 1 (positive, 7d1f7ee8)   DETECTED   corr:is_contained
CONTROL 2 (negative, 810b9b61)   SILENT     baseline (has_holes) also separates
```

**Evaluation result:**

```
applicable                                    79 / 120
baseline-separable                            51
corr-separable (weak form)                    28
CORR-ONLY (decision form)                      1     3dc255db, corr:is_contained
permutation control, chance CORR-ONLY        0.2     (5 shuffles: 0,1,0,0,0)
```

### RESULT: FALSIFIED in the decision form. Threshold was 8; result is 1.

The weak form (CORR-SEPARABLE, 28) numerically clears the same threshold,
but **the permutation control removes it as evidence.** Shuffling the
CHANGED/UNCHANGED labels while keeping the correspondence atoms active
produced CORR-ONLY = 1 on one of five shuffles — identical to the real
count. The single real hit cannot be distinguished from noise. And
CORR-SEPARABLE was never the decision criterion (stated in the
pre-registration before this run): a correspondence atom coinciding with
what 51 baseline-separable tasks can already say is not new capability,
it's the `shape_is_unique` situation from Extension A's own instrument
repeating itself — a real relation, doing no marginal work.

**This is the same shape as A and B, a third time, with a fully-tested
instrument this time — both controls came from reading tasks by hand, and
the first attempt at the positive control was wrong and was caught before
the evaluation set was touched, not after.**

| device | training control | evaluation yield |
|---|---|---|
| cell-neighbourhood predicates (A) | 4 / 7 | **0 / 32** |
| symmetry & period completion (B v2) | hard case detected | **1 / 120** |
| cross-object correspondence (D) | hard case detected | **1 / 120 (indistinguishable from chance)** |

**What this closes.** `graph_difference.py`'s docstring calls a
correspondence score "the natural input to the two-level rules the solver
actually lacks." That description survives — `776ffc46` is direct
evidence a two-level chain is what some evaluation tasks need — but this
result says correspondence AS A SELECTOR, on its own, is not where the
missing value is. The gap Part 5 pointed at was real; it just doesn't
cash out as "add relational atoms to selector induction." It cashes out
as the derive-then-use chain itself, which no extension proposed here —
A, B, C, or D — attempts to build.
