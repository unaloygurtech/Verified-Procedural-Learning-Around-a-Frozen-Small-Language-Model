# Experiment 0004: breaking schema-only routing

Date: 2026-08-31

## Purpose

Experiment 0003 let the router identify a family from its field schema. This
experiment removes that shortcut. Four procedures all receive exactly
`recipe`, `x`, and `y`, and all return exactly `value` and `label`.

The procedures are sum, difference, forward join, and reverse join. Their
family labels are test-harness metadata only. The model-facing raw experience
prompt contains input/output pairs and the ordinary `recipe` value, but no
family name.

## Consolidation and safety

The consolidator groups examples by the content discriminator (`recipe`) and
searches a bounded program language: integer parse/add/subtract, trim/lowercase,
forward/reverse concatenation, and a literal separator. It must find exactly one
program per recipe. A recipe with conflicting evidence is rejected as ambiguous
instead of being activated.

Every candidate is run on eight disjoint validation cases and must reach 80%
before the twelve held-out cases. The benchmark includes model-only, raw,
manual-skill, automatic-skill, and deliberately corrupted-candidate conditions.

No external memory files or mounts are available to either container. The result
is not a claim of general learning; it is a test that AIR can move from
schema-based dispatch toward content-based program selection while preserving
verification and safe rejection.

## Result

The consolidator discovered all four recipe rules from the same shared schema.
The positive candidate scored 7/8 (87.5%) on validation and became active. A
deliberately corrupted candidate scored 3/8 (37.5%) and was rejected.

On the 12 held-out cases:

| Condition | Correct | Accuracy |
| --- | ---: | ---: |
| Model only | 0/12 | 0% |
| Model + raw experiences | 12/12 | 100% |
| Manual rule + router | 6/12 | 50% |
| Auto-discovered rule + router | 7/12 | 58.3% |

This is not yet a positive skill-improvement result. Raw examples retain enough
surface information for this small model to solve the task, while the compact
generated rule loses robustness on negative integer output, exact whitespace,
and reverse string normalization. The useful result is diagnostic: content
discriminator discovery and rejection work, but skill compression must preserve
type/format semantics before it can replace raw experience context.

Container report: `data/runs/air-004-20260831T152437Z.json` (runtime data,
intentionally ignored by Git).

## Decision

Proceed to 0005 only with this failure mode explicit. Program synthesis should
emit typed intermediate values and executable tests, not only a short prose
rule. The next benchmark must compare synthesized programs against both raw
examples and the current compact skill, with exact string and numeric typing
checks.
