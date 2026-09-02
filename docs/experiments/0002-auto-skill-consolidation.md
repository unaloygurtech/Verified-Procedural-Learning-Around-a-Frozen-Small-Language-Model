# Experiment 0002: automatic skill consolidation

Date: 2026-08-31

## Question

Can AIR turn verified raw experiences into a compact skill automatically, reject
bad candidates before use, and improve a small local model on unseen cases?

## Isolation and method

- The experiment ran only inside the AIR Docker Compose environment.
- No external memory files or mounts were used.
- Training, validation, and held-out cases are disjoint.
- A candidate must score at least 80% on five validation cases before it can see
  the eight held-out cases.
- The runtime model is SmolLM3-3B GGUF Q4 through llama.cpp on CPU.

## Attempts

1. One-shot LLM consolidation inferred parts of the mapping but scored 0/5 on
   validation. It was rejected.
2. Decomposed LLM consolidation, including a reasoning-mode attempt, either
   produced an invalid rule or failed to finish valid JSON. It was rejected.
3. A deterministic symbolic hypothesis search tested a deliberately small rule
   language against verified training examples. It found one unique rule set,
   compiled it into the same explicit format as the working manual skill, and
   scored 5/5 on validation. Only then was it activated.

## Held-out result

| Condition | Correct | Accuracy |
| --- | ---: | ---: |
| Model only | 0/8 | 0% |
| Model + raw experiences | 1/8 | 12.5% |
| Model + manual verified skill | 7/8 | 87.5% |
| Model + auto-generated verified skill | 8/8 | 100% |

Container report: `data/runs/neralis-auto-20260831T143808Z.json` (runtime data,
intentionally ignored by Git).

## Interpretation

This is a positive result for a narrow, falsifiable mechanism: a small model can
execute a compact skill much better than it can infer the same rule from a long
raw history, and a constrained tool can automatically distill that skill when a
separate validation gate controls activation.

It is not evidence of general autonomous learning. The successful consolidator
searches a hand-bounded DSL (key templates, simple integer operations, and
constant labels). The next useful test is to broaden the rule language and add
multiple task families while preserving disjoint data and rejection behavior.

## Decision

- Keep SmolLM3-3B as the local skill executor for now.
- Do not use it as the sole skill consolidator.
- Use tool-assisted hypothesis search plus validation as the initial automatic
  consolidation path.
- Continue only through measured task families; replace the model or approach if
  verified skills stop producing repeatable held-out gains.
