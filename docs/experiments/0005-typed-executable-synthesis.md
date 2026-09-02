# Experiment 0005: Typed Executable Skill Synthesis

Date: 2026-08-31

## Hypothesis

Replacing prose skill compression with a typed, executable program should retain
numeric types, string normalization, operand order, and exact separators better
than the 0004-style prose skill.

This is a bounded program-synthesis experiment, not a claim of general learning.

## Architecture

The task schema remains `recipe, x, y -> value, label`. Four recipe values select
integer sum, integer difference, normalized forward join, and normalized reverse
join. Family/procedure names are absent from model-facing examples. The 0005
held-out partition is newly generated and is not the 0004 held-out set.

The pipeline is:

```text
verified experiences
  -> group by observed recipe discriminator
  -> enumerate typed AST candidates
  -> type-check and execute on training examples
  -> reject ambiguity/conflict
  -> run validation and independent edge tests
  -> activate executable programs
```

The executor is deterministic. The LLM is used only for the model, raw, and
compact-prose comparison conditions; it does not reinterpret an accepted AST.

## Typed DSL

The minimum types are `String` and `Int`. The implemented operations and
contracts are:

- `FIELD(name) -> String`
- `LITERAL(String|Int) -> String|Int`
- `TRIM(String) -> String`
- `LOWER(String) -> String`
- `PARSE_INT(String) -> Int`
- `ADD(Int, Int) -> Int`
- `SUB(Int, Int) -> Int`
- `CONCAT(String, String) -> String`

Both static contracts and runtime contracts are enforced. Numeric outputs remain
JSON integers; strings are compared with exact case, whitespace, ordering, and
separator semantics.

## Synthesis process

The consolidator groups only by the ordinary `recipe` value and searches a
bounded set of typed expression trees, including intentionally invalid
string/integer combinations. For every recipe it records candidate search count,
type-invalid candidates, and ambiguity. A recipe is accepted only if exactly one
typed program explains every training pair. Conflicting evidence or multiple
equally valid programs raises a synthesis error. No semantic recipe-to-operation
mapping is hard-coded; operation choice is selected by exact evidence matching.

## Validation design

There are 8 training, 8 validation, 8 independently authored edge/adversarial,
and 12 new held-out cases. Edge cases cover zero, negative values, equal and
opposite-sign subtraction, leading/trailing whitespace, mixed case, strings with
spaces, and forward/reverse operand order. The candidate must reach 90% on
validation and 100% on edge tests before activation.

A second candidate is syntactically and type-correct but deliberately corrupted:
ADD/SUB swaps, reversed subtraction operands, reverse join operands, and missing
TRIM/LOWER. It is run through the same semantic gate.

## Benchmark

Held-out conditions are:

1. model only
2. model + raw experiences
3. 0004-style compact prose skill
4. deterministic typed executable skill

The report also records raw/prose/executable character and byte sizes and their
compression ratios. Runtime JSON files remain under `data/runs/` and are ignored
by Git.

## Results

The benchmark ran in the isolated AIR Docker environment with no external memory
mount and no new dependency. The consolidator discovered four recipe groups and
searched 224 program candidates: 48 were rejected as type-invalid and no
ambiguity remained after the independent whitespace training example.

The synthesized programs passed validation **8/8 (100%)** and independently
authored edge tests **8/8 (100%)**, so the executable skill became active. A
syntactically and type-correct corrupted candidate passed **0/8 (0%)** and was
rejected.

On the new 12-case held-out set:

| Condition | Correct | Accuracy |
| --- | ---: | ---: |
| Model only | 0/12 | 0% |
| Model + raw experiences | 12/12 | 100% |
| Compact prose skill | 4/12 | 33.3% |
| Typed executable skill | 12/12 | 100% |

The typed representation therefore stayed within the primary target (at most
one case behind raw; in this run it matched raw) and was clearly better than the
0004-style compact prose condition. This is evidence that deterministic typed
execution preserves the tested semantics; it is not evidence of general
reasoning or autonomous learning.

Size measurements were 969 raw-experience characters, 436 compact-prose
characters, and 1,046 serialized executable-program characters. The AST is
larger than this tiny prose sample because it stores explicit operation nodes and
types; its benefit here is executable semantics, not byte compression.

Container report: `data/runs/air-005-20260831T153743Z.json` (runtime data,
intentionally ignored by Git).

## Failures and limitations

The DSL is deliberately bounded and cannot represent arbitrary Python, loops,
external calls, or learned latent concepts. A deterministic executor can make an
accepted program reliable, but it cannot make an incorrectly synthesized program
correct. The raw baseline may still outperform a compact program if the program
language or synthesis search omits a needed semantic detail.

## Decision for Experiment 0006

The primary 0005 gate passed: typed executable skill 12/12 held-out, at least
11/12 target, clearly above compact prose, and corrupted candidate rejected.
Proceed to skill composition, while carrying the typed AST, deterministic
executor, edge tests, and semantic rejection gate forward. 0006 must test
composing independently learned programs on a new task, not simply add more
recipes to this same four-operation benchmark.
