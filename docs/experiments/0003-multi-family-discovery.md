# Experiment 0003: multi-family discovery and routing

Date: 2026-08-31

## Purpose

Experiment 0002 used one narrow arithmetic task. This experiment tests whether
AIR can discover and route across four different task families while keeping
family labels out of all model-facing examples.

## Families and DSL

The bounded DSL includes field-schema routing, string trim/lowercase/space
replacement, field concatenation, integer add/multiply, finite enum maps,
coupon adjustments, and a boolean conditional. The families are:

1. signal normalization (`code`, `signal`, `tag`, `value`)
2. ticket triage (`title`, `priority`, `state`)
3. inventory pricing (`sku`, `qty`, `unit_price`, `coupon`)
4. message routing (`body`, `channel`, `urgent`)

Each family has 4 training, 2 validation, and 3 held-out cases. The raw
experience prompt contains only input/output pairs; family labels are retained
only in the test harness for scoring and partition checks.

## Protocol

The automatic consolidator first clusters training pairs by observed input and
output schema. It then compiles one rule set per discovered cluster. A candidate
is active only when it reaches 80% on all eight validation cases. The held-out
comparison measures model-only, model+raw experiences, manual multi-skill, and
automatic multi-skill conditions.

No external memory files or mounts are available to either container.

## Success criteria

- 4 balanced schemas discovered from examples with no family labels
- candidate validation accuracy at least 80%
- invalid or ambiguous candidates remain rejected
- automatic skill materially beats model-only and raw-experience baselines on
  held-out cases
- results are stored as a JSON report under `data/runs/`

The benchmark is intentionally bounded and does not claim general learning. A
positive result supports expanding the DSL and adding new task families; a
negative result tells us whether routing, rule extraction, or model execution is
the limiting component.

## Result

The automatic candidate discovered four schemas and passed all 8 validation
cases (100%), so it was activated. On all 12 held-out cases:

| Condition | Correct | Accuracy |
| --- | ---: | ---: |
| Model only | 0/12 | 0% |
| Model + raw experiences | 1/12 | 8.3% |
| Manual skill + router | 7/12 | 58.3% |
| Automatically discovered skill + router | 9/12 | 75% |

Automatic-skill family breakdown: signal normalization 3/3, message routing
3/3, inventory pricing 2/3, ticket triage 1/3. The gain is therefore real but
not complete. Remaining errors are concentrated in applying unseen combinations
of ticket state/priority and one inventory coupon arithmetic case, not in family
schema discovery. The next iteration should target compositional enum handling
and add a separate negative-candidate rejection measurement.

Container report: `data/runs/air-003-20260831T150859Z.json` (runtime data,
intentionally ignored by Git).
