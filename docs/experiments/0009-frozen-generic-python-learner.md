# Experiment 0009: Frozen Generic Python Learner

Date: 2026-08-31

## Research question

Can the same learning loop acquire executable skills in several previously
unseen Python/API task families without a family-specific prompt patch?

The four families are deliberately small but use different standard-library
surfaces:

```text
json          -> canonical compact JSON
datetime      -> DD-MM-YYYY to ISO date
pathlib       -> lowercase POSIX path suffix
collections   -> character frequency summary
```

This is a transfer and protocol experiment, not a claim of arbitrary Python
competence.

## Frozen protocol

Every family receives only its contract, a short API note, and four public
examples. The learner uses one immutable generic template, one three-attempt
repair budget, the same static checker shape, the same subprocess sandbox, and
the same hidden validation/edge gates. The template is recorded by version and
SHA-256 so later runs can be compared without silently changing the prompt.

The final protocol values are:

- prompt version: `air-009-generic-learner-v1`
- prompt SHA-256: `d6b6c4da5226e826343de0bdf3864ba00a95f34957443c6f0c3c8f04b72c2833`
- family-specific prompt patches: false
- repeats: 2
- maximum proposals per family run: 3
- candidate execution: `python -I`, sanitized environment, temporary cwd,
  two-second timeout, family-specific standard-library allowlist

Family API notes are data supplied to the same template; they are not separate
hand-written learner prompts. Public examples are never reused as hidden or
held-out examples.

## Measurements

For each family and repeat the report records gap detection, proposal and
repair count, rejected and unsafe proposals, public/hidden/edge gates,
activation, held-out reuse, wrong activation, cross-family help, and source
artifact immutability. A prior 0008 query skill is placed in the starting
library as a cross-family control. Its original 12/12 held-out score is checked
before and after the experiment for regression.

The static checker rejects dangerous imports/names and unsupported calls before
execution. A rejected proposal is not necessarily unsafe: syntax and shape
errors are counted separately from explicit sandbox-safety violations.

## Results

Authoritative report:
`data/runs/air-009-20260831T172012Z.json`

| Family | Active runs | Public/hidden/edge | Learned held-out on active runs |
| --- | ---: | ---: | ---: |
| json-canonical | 1/2 | 1/1, 1/1, 1/1 | 8/8 |
| datetime-date | 0/2 | 0/2, 0/2, 0/2 | n/a |
| pathlib-suffix | 1/2 | 1/1, 1/1, 1/1 | 8/8 |
| collections-count | 0/2 | 0/2, 0/2, 0/2 | n/a |

Across eight family runs, gap detection was 8/8 and activation was 2/8 (25%).
The two activated artifacts each achieved 8/8 on new held-out literals. There
were 20 proposals, 12 repair steps, and 18 rejected proposals; no proposal in
the final two-repeat run used an explicitly unsafe import/name. Hidden
validation failures after a public pass: 0. Wrong pre-learning activation: 0;
previous 0008 skill helped a new family: 0. All base artifacts remained
unchanged, and the prior 0008 regression stayed 12/12 before and after.

The direct baselines show why reuse matters but are not interchangeable with
the learner: across all 64 family-held-out evaluations, model-only scored
32/64 and model-plus-docs 24/64, while inactive before-gap execution scored
0/64. Learned artifact execution scored 16/16 on the two active runs (the
overall 16/64 includes the six correctly inactive runs).

## Interpretation

This is a substantially stronger result than one hand-authored Python task:
the same frozen generic learner acquired two of four API-family skills, with
perfect hidden/edge gates whenever it activated, while refusing to activate on
the other six runs. It also shows the current bottleneck clearly. The small
SmolLM3-3B model often emitted a lambda or one-line `def` shape that violated
the executable artifact contract; its repair loop repeated those candidates
within the fixed budget. The failure is recorded rather than patched with a
family-specific recipe.

The next experiment should keep this prompt hash fixed and improve neither the
task nor the result after seeing failures. Useful additions are independent
seeds/runs, a fifth family, explicit capability-gap calibration, and measured
retrieval/interference at a larger skill library. Only after that should AIR
attempt documentation search or multi-step tool use.
