# Experiment 0007: Capability-Gap Detection and Missing-Skill Synthesis

Date: 2026-08-31

## Research question

When the existing AIR skill library cannot solve a new task, can AIR identify
that capability gap, learn the missing intermediate behavior from independent
verified experiences, and then reuse it without mutating older skills?

This is still a bounded typed-program experiment. It is not yet open-ended
autonomous learning or general coding ability.

## Protocol and isolation

- The 0006 five-skill library is loaded as the immutable starting point.
- New gap tasks require replacing literal spaces with `_`, a behavior absent
  from the starting library, followed by existing normalization, append, and
  forward/reverse join behaviors.
- Four independent training pairs describe only the missing behavior. They do
  not contain any `g1`, `g2`, `g3`, or composed final outputs.
- Three missing-skill validation pairs and three edge pairs are separate from
  training. Gap-task validation and the 12-case held-out set use new values.
- An opaque impossible task is present before and after learning; it must remain
  `no_valid_composition`.
- The run stays in the existing Docker/llama.cpp/SQLite environment. No
  external memory file or mount is used and no heavy dependency is added.

The 0005 typed DSL gains one backward-compatible operation:
`REPLACE(String, String, String) -> String`. It is available to the bounded
synthesizer, but no `REPLACE` skill exists in the initial 0006 library.

## Capability-gap phase

Before learning, AIR searches the existing five artifacts over bounded chains
up to three unary steps or two steps per binary branch. Positive `g1`, `g2`, and
`g3` evidence must produce no valid plan; the detector labels those groups
`gap_detected`. The opaque `gx` group has no positive evidence and is reported
as `safe_unknown`, not as a request to invent a solution.

## Missing-skill synthesis

The missing-skill consolidator searches typed AST candidates including direct
replacement, normalization variants, and an intentionally invalid integer
candidate. It accepts exactly one expression that explains all four independent
training pairs:

```text
REPLACE(FIELD("input"), LITERAL(" "), LITERAL("_"))
```

The candidate must pass its own validation and edge gates before it is appended
to the library as immutable version-1 `skill-6`.

## Composition after learning

The extended artifact search must discover these plans without a task-token
mapping:

- `g1`: `skill-2 -> skill-6 -> skill-3` (depth 3),
- `g2`: `skill-2(x) -> skill-6(x)` and
  `skill-2(y) -> skill-6(y) -> skill-4` (depth 5),
- `g3`: the same normalized/replaced branches followed by `skill-5` (depth 5).

The order test is meaningful: applying replacement before normalization turns
outer spaces into underscores and fails the validation examples. A corrupted
candidate reverses `g1` and swaps the forward/reverse join; it remains
type-valid but must be rejected semantically.

## Baselines and provenance

The held-out report compares:

1. model only,
2. model with only the old 0006 primitive pairs,
3. model with old plus missing-skill raw pairs,
4. a flat monolithic `REPLACE` AST selected from gap validation,
5. the old library before missing-skill learning,
6. the extended learned-skill composition.

Each condition records whether raw context or an artifact was used and whether
a new monolithic/composed program was produced. The new skill and composed
artifacts retain source cases, versions, order, validation evidence, and an
activation reason. A before/after snapshot verifies that all five 0006 source
skills are unchanged.

## Results

Run the benchmark with:

```bash
docker compose exec air-core python -m air_core.cli experiment-007
```

The authoritative runtime report is
`data/runs/air-007-20260831T164125Z.json` (runtime data, intentionally ignored
by Git).

| Condition | Positive held-out | Impossible rejection |
| --- | ---: | ---: |
| Model only | 0/9 | 3/3 (100%) |
| Model + prior raw primitives | 0/9 | 3/3 (100%) |
| Model + all raw experiences | 0/9 | 3/3 (100%) |
| Flat synthesized program | 9/9 | 3/3 (100%) |
| Existing library before learning | 0/9 | 3/3 (100%) |
| Learned missing skill + composition | 9/9 | 3/3 (100%) |

The strong-success gate passed: all three positive groups were diagnosed as
gaps before learning; `skill-6` scored 4/4 on training, 3/3 on its independent
validation, and 3/3 on edge cases; post-learning composition scored 3/3;
the impossible task scored 2/2 safe rejection; the corrupted composition scored
1/3 and was rejected; and all five source skills were unchanged.

Before learning, 4,960 unique plans (19,840 task evaluations) were searched;
18,332 evaluations were type-invalid and 1,508 were semantic rejects. After
adding `skill-6`, the extended search covered 11,352 unique plans (45,408 task
evaluations), with 41,544 type-invalid and 3,861 semantic rejects. Ambiguity
was zero in both phases. The accepted plans were `g1 = skill-2 -> skill-6 ->
skill-3`, and normalized/replaced branches followed by `skill-4` or `skill-5`
for `g2`/`g3`.

The model did not solve positive held-out cases even with all independent raw
pairs; the deterministic flat AST and learned composition both solved all nine
positive cases. The flat result is a control, not evidence of artifact reuse:
it selects a monolithic AST from validation, while the final condition executes
the newly synthesized `skill-6` together with the five existing immutable
artifacts and records their provenance.

## Interpretation and next boundary

A positive result would establish a useful step beyond 0006: AIR can convert a
verified failure into a new reusable artifact and retry composition. It still
would not show that AIR can choose arbitrary learning experiments, use external
documentation, write Python, or scale a skill graph.

If the gate fails, the failure must remain visible. The next experiment should
then target the specific failing phase rather than relaxing validation. If it
passes, 0008 should move the same gap-detection loop into a sandboxed Python/API
task with explicit tests and documentation lookup.
