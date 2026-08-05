# How the solver works

A technical description of the system as of 2026-07-30
(72/1000 training, 0/120 evaluation).

---

## 1. The problem

An ARC task gives you 2–5 **training pairs** (input grid → output grid)
and one or more **test inputs**. You must produce the test output. Grids
are 1–30 cells per side, each cell an integer 0–9 (a colour). Scoring is
exact match — a single wrong cell fails the task. ARC allows **two
attempts** per test input.

The rule is different in every task. Nothing transfers between tasks.

---

## 2. Two solvers, merged

`solve_task()` in `src/layers/solver.py` asks two independent systems for
an answer and puts them in the two attempt slots.

```
                    ┌─────────────────────────┐
   task ──────────► │  synthesis.py           │──► rule ──► attempt 1
            │       │  INDUCES a rule         │
            │       └─────────────────────────┘
            │       ┌─────────────────────────┐
            └─────► │  verifier.py            │──► rule ──► attempt 2
                    │  SWEEPS prebuilt rules  │
                    └─────────────────────────┘
```

They are kept separate because they fail in different places and
**overlap on one task out of 72** — replacing either would throw away
almost everything the other contributes.

| | solves | how |
|---|---|---|
| `synthesis.py` | 58 | induces a rule per task from the observed change |
| `verifier.py` | 14 | enumerates complete prebuilt rules, keeps whatever verifies |

Synthesis takes attempt 1 because its precision is higher: when it
commits to a rule it is right ~92% of the time, whereas a sweep
hypothesis that fits the training pairs frequently does not generalise.

---

## 3. Supporting layers

Both solvers sit on the same perception stack.

```
task_loader        JSON → numpy int32 grids
    ↓
object_extractor   grid → list of objects (connected components)
    ↓
property_encoder   object → size, bbox, centroid, symmetry, fingerprint…
    ↓
relationship_graph pairwise relations between objects (touching, above…)
    ↓
difference_engine  input vs output → what changed, as a signature
```

An **object** is a dict: pixels, colour, bounding box, centroid, height,
width, symmetry flags, a shape fingerprint, and a cropped grid of just
that object.

---

## 4. The synthesis engine

This is the main solver. The idea: rather than searching a list of
complete rules, define a **language of expressions with holes** and
**solve the holes from the evidence**.

```
Recolor( <which objects?> , <what colour?> )
           ↑                    ↑
        hole 1               hole 2
```

### 4.1 Scene interpretations

Before anything else, the solver must decide what counts as an object.
This is **searched over**, not assumed:

| interpretation | grouping rule |
|---|---|
| `connected4` | orthogonally adjacent same-colour pixels |
| `connected8` | as above, plus diagonals |
| `by_color` | every pixel of a colour is one object |
| `no_background` | background regions become objects too |
| `panels` | separator-divided grid → one object per panel |

Each is gated by a cheap applicability check — `panels` is skipped unless
separator lines exist, `connected8` unless diagonal adjacency actually
merges something. Without those gates, trying all of them cost **12.6×
runtime for nothing**, because most grids read identically under several.

The winning interpretation is left **active** so the rule is applied to
the test grid under the same reading it was induced under. This axis is
the second-largest source of solves in the project (14 tasks).

### 4.2 Operators

Five base operators, each defining three things: how to **observe** a
pair, how to **solve** its holes, and how to **describe a failure**.

| operator | positives are | second hole |
|---|---|---|
| `Recolor` | objects that kept their pixels but changed colour | what colour |
| `Remove` | objects that vanished | *none* |
| `Move` | objects that changed position | what offset |
| `Draw` | *cells that appeared* (not objects) | which form, what colour |
| `Extract` | the object that became the whole output | which crop form |

An observer **rejects** a pair it cannot describe. `observe_remove`
returns None if anything also moved or was recoloured. This strictness is
what keeps rules honest — and it is why composition needs separate
*tolerant* observers that label only their own part of the change.

### 4.3 Solving hole 1 — which objects?

`induce_selectors` is shared by every operator and treats this as
**concept learning**:

- objects the operator acted on → **positive** examples
- objects it left alone → **negative** examples
- find a predicate covering every positive and no negative, **across
  every evidence pair at once**

Predicates come from an **atom catalog** built from the values actually
present in the task: `color==3`, `size==5`, `is_largest`, `is_sym_v`,
`touches_border`, `holes==2`, `shape_is_unique`, and so on. Single atoms
first, then 2-atom conjunctions if none suffice.

Results are ranked by `selector_cost` = (atom count, **metric-atom
count**). A selector pinned to a measurement loses to a structural one
that fits equally well — `is_sym_v AND not_is_single_pixel` beats
`is_sym_v AND height==2`, because the second is true of the training
grids while the first is the rule.

> **Measured caveat:** `all` is the selector in 24 of ~40 induced
> programs. In most solves there is no selection problem at all.

### 4.4 Solving hole 2 — what value?

Every value form is **derived from the evidence**, never swept:

- **literal** — every observed value is the same
- **lookup table** — keyed on the object's colour, size, hole count or
  shape count. Built directly from observations, rejected if one key ever
  needs two values
- **decision list** — clauses of `selector → value`, first match wins
- **`to_edge` / `until_collision`** (Move only) — checked against every
  observed offset, kept only if they reproduce all of them

Ranking is by how much the form **memorises**: a literal costs 0, a
lookup table costs 100 + its size. A table keyed on a raw measurement has
no answer for an unseen key on the test grid, so it is a last resort.

One restriction earned its place empirically: a decision list whose
*values* are geometric (pixel offsets) is only allowed if its *keys* are
colours. Offsets keyed on size — `size==1 → (-1,0) | size==2 → (-2,0)` —
are memorisation, and forbidding them took Move from **29% to 100%
precision**.

### 4.5 The loop

Counterexample-guided, in `synthesize()`:

```
evidence ← [pair 1]
candidates ← induce(evidence)          # solve the holes
program ← candidates[0]

for each remaining pair:
    if program reproduces this pair:   continue
    diagnose the failure structurally  # "wrong objects" vs "wrong value"
    evidence ← evidence + [this pair]  # absorb the counterexample
    program ← induce(evidence)[0]      # re-induce, now more constrained

final check: program must reproduce EVERY pair
```

The final check matters — refinement can change the rule under pairs that
already passed. **Any rule the engine returns reproduces every training
output**, which is the invariant the 92% precision rests on.

> **Measured property:** because re-induction uses *all* accumulated
> evidence, every starting candidate converges to the same set. So
> restarts and deeper search gain nothing — the only levers are richer
> evidence or a better language.

### 4.6 Composition

14 composed operators, as ordered pairs of a *tolerant* first stage and a
*strict* second stage. Stage one labels only the part of the change it
explains; stage two is induced from the **residual** (stage one's output
vs the real output). If the residual is not a clean single-operator
change, strict observation rejects it and the candidate dies cheaply.

### 4.7 Budget

A per-task deadline (2s, rising to 15s once a candidate exists) exists
purely as a **runaway guard**. Measured over 1000 failing inductions, the
median is 0.75ms and the maximum 325ms, so it never binds in normal
operation — but two incidents have seen a single task consume hours.

---

## 5. The sweep

`verifier.py` is the original approach: build a list of **complete,
already-parameterised** rules from the diff signature — `rotate_180`,
`recolor_all_6_to_2`, `gravity_down`, `tile_3x3` — then apply each to
every training pair and keep those that match exactly.

It is weaker and less precise, but it reaches 14 tasks nothing else does,
mostly whole-grid geometric transformations that the object-centric
synthesis engine never considers.

---

## 6. Merging the two attempts

```
attempt 1 ← induced rule
attempt 2 ← an induced ALTERNATE if one exists, else the sweep's best
```

Deduplicated, so if both solvers land on the same grid the second slot
goes to a distinct runner-up rather than a repeat. Each source is
exception-wrapped independently; if both fail, the fallback is the test
input unchanged plus a blank grid.

> **Measured:** attempt 2 currently adds nothing — 29 solved tasks had
> alternates and 0 were saved by one.

---

## 7. Worked example

Task `810b9b61`, solved by `Recolor(where holes==1, to colour 3)`.

1. **Interpretation** — `connected4` applies; used.
2. **Observe pair 1** — objects keep their pixels; some changed colour.
   Positives: the objects that became 3. Negatives: the rest.
3. **Hole 1** — the atom catalog includes `holes==1` because some objects
   have exactly one enclosed background region. It covers every positive
   and no negative.
4. **Hole 2** — every positive became colour 3 → `literal(3)`, cost 0.
5. **Test on pair 2** — holds.
6. **Test on pair 3** — holds.
7. **Final check** — reproduces all three pairs. Returned.
8. **Apply to test input** under `connected4`, the same reading it was
   induced under.

---

## 8. What it can and cannot express

**Can:** *for each object, do X to it, where X depends on that object's
own properties* — with X being recolour, remove, move, draw, or extract,
optionally chained two or three deep, under five different readings of
what an object is.

**Cannot:**

- **Two-level rules** — derive a value from one part of the grid and use
  it on another
- **Counting or arithmetic** — no number is ever computed from the scene
- **Template matching** — no concept of "this shape appears again over
  there", despite 90% of evaluation tasks containing repeated shapes
- **Multi-scale/recursive structure** — tiling and scaling are flat
- **Grid construction** — output is always a modified input, never built
- **Conditionals at program level** — decision lists cover values only

This is why evaluation is 0: **102 of 120 tasks cannot even be
*observed*** by any operator, and of the 18 that can, all 18 fail to
explain even one training pair. It is a representation gap, not a search
gap — an hour of widened search on one task died in 1 second, with all 51
combinations failing at observation before any search began.

---

## 9. Measurement

`py src/layers/solver.py --full-eval` reports four rows —
`training-dev`, `training-holdout`, `training-all`, `evaluation` — with
per-solver attribution, induction precision, and a per-operator
breakdown.

Attribution comes from **which attempt slot matched**, not from the
method string: the string names both sources whenever both offered an
answer, so it says who spoke, not who was right.

"Rules offered vs correct" is printed beside accuracy because *0 rules
offered* and *0 rules correct* look identical in an accuracy column and
mean opposite things — a coverage gap versus overfitting, which call for
opposite work.

See `docs/RESEARCH_AGENDA.md` for the 24 known flaws and `CLAUDE.md` for
the short working context.
