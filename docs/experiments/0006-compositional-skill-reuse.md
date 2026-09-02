# Experiment 0006: Compositional Skill Reuse

Date: 2026-08-31

## Research question

Can AIR learn executable behaviors independently, then reuse and order those
artifacts to solve a new task whose composed input/output examples were never
shown during primitive learning?

This is a bounded compositional program-synthesis experiment. It is not a claim
of general reasoning or autonomous learning.

## Isolation and protocol

- The experiment runs in the AIR Docker Compose environment.
- No external memory files, mounts, or memory stores are used.
- No new heavy dependency is introduced; the 0005 typed AST and executor are
  reused.
- Five primitive experience sets are disjoint. Each contains three verified
  input/output pairs and is synthesized independently.
- Composition validation contains two depth-2/depth-3 positive task families
  and one depth-3 reverse-order family, plus an impossible discriminator.
- The 12 held-out cases use new literal values and a new impossible task token;
  no composed pipeline example appears in primitive training text.

Model-facing primitive context contains only input/output pairs. Skill IDs,
family names, task tokens, and composed examples are omitted from that context.

## Learned representation

Each primitive is stored as an immutable `PrimitiveSkill` artifact containing:

- a versioned skill ID,
- explicit input fields and `String -> String` or `(String, String) -> String`
  contract,
- a typed 0005 AST expression,
- source set and source case provenance.

The discovered library is:

| Artifact | Independent behavior | Contract |
| --- | --- | --- |
| `skill-1` | trim outer whitespace | `String -> String` |
| `skill-2` | trim, then lowercase | `String -> String` |
| `skill-3` | append one literal space | `String -> String` |
| `skill-4` | forward join with `/` | `(String, String) -> String` |
| `skill-5` | reverse join with `/` | `(String, String) -> String` |

The names in this table are report-side descriptions; they are not included in
the raw model context.

## Composition search

The planner enumerates bounded chains over the learned artifact IDs. It does
not contain a task-token-to-plan mapping. Every candidate is first checked for
arity and typed AST compatibility, then executed against all task evidence. A
candidate survives only if it is the unique semantic match for that task.

The expected discoveries are:

- `m2`: `skill-2 -> skill-3` (depth 2),
- `m3`: `skill-2(x), skill-2(y) -> skill-4` (depth 3),
- `m4`: `skill-2(x), skill-2(y) -> skill-5` (depth 3).

The `m3`/`m4` pair is the semantic-order test: the same normalization skills
with the opposite binary artifact produce different answers. An `mx` task has
no matching artifact composition and must remain `no_valid_composition`.

The JSON report separates total candidates, type-invalid rejections, semantic
validation rejections, and ambiguity. A deliberately corrupted plan reverses
the depth-2 order and swaps forward/reverse join; it is type-valid but must fail
the semantic gate.

## Baselines and provenance

Held-out conditions are:

1. model only,
2. model plus raw independent primitive experiences,
3. a flat synthesized AST selected directly from the task evidence,
4. learned-skill composition executed from the stored artifacts.

For each condition the report records where the answer came from, prompt and
generation tokens (for model conditions), exact correctness, and rejection
behavior. Composed artifacts record source skill IDs, versions, order,
validation/edge case IDs, and activation reason. The source skill serialization
is compared before and after composition to verify immutability.

## Results

Run the isolated benchmark with:

```bash
docker compose exec air-core python -m air_core.cli experiment-006
```

Authoritative numbers from the isolated container run are in
`data/runs/air-006-20260831T162407Z.json` (runtime data, intentionally ignored
by Git).

| Condition | Correct | Accuracy | Safe impossible rejection |
| --- | ---: | ---: | ---: |
| Model only | 0/9 | 0% | 3/3 (100%) |
| Model + raw primitive experiences | 0/9 | 0% | 1/3 (33.3%) |
| Flat synthesized program | 9/9 | 100% | 3/3 (100%) |
| Learned-skill composition | 9/9 | 100% | 3/3 (100%) |

The deterministic learned composition passed positive validation 6/6 and edge
cases 3/3, rejected both impossible validation cases (2/2), and became active.
The search generated 235 unique bounded candidates and evaluated them against
four task groups (940 candidate-task evaluations): 740 evaluations were
type-invalid and removed before execution, 197 were semantic mismatches, and none were ambiguous. The
corrupted type-valid composition scored 2/6 and was rejected. Source skill
serialization was unchanged after composition.

The flat baseline reaches the same deterministic score but has no reusable
artifact provenance: it directly selects a monolithic AST from the validation
evidence. The learned-composition condition instead executes the three accepted
plans from the five independently synthesized `PrimitiveSkill` artifacts. The
small model solved none of the nine positive held-out cases with either no
context or raw primitive pairs; on impossible cases it sometimes emitted a
false solution when given raw context. These model numbers are diagnostic, not
evidence against the executable composition mechanism.

## Interpretation and limitations

A positive result would show reusable executable composition beyond 0005's
single-program synthesis: independent skills are selected, ordered, and
verified without a composed training example. It still would not establish
open-ended planning, learning a missing operation, or robustness to arbitrary
schemas. Candidate enumeration grows with chain length and library size; the
report records the measured candidate count so this search-space limit is
explicit.

If any gate fails, the failure is retained rather than hidden by relaxing the
benchmark. In particular, an impossible task must never activate the nearest
wrong program.

## Recommendation for Experiment 0007

The highest-information next question is whether AIR can detect a capability
gap and invent a missing intermediate skill when no existing composition works.
This directly exercises the negative result here: keep the impossible task,
allow a tightly sandboxed synthesis/verification loop, and measure whether a
new artifact can be added without mutating the primitive library or regressing
the three known compositions. A skill graph or hierarchical planner can follow
only after that gap-detection boundary is measured.
