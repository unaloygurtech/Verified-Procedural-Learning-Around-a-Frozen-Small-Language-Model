# Experiment 0001: Neralis-3 manual skill compression

Date: 2026-08-31 (Europe/Istanbul)

## Question

Can a fixed small model solve held-out tasks more accurately and with less
prompt context when verified raw experiences are compressed into a short skill?

## Runtime

- Model: `ggml-org/SmolLM3-3B-GGUF:Q4_K_M`
- Upstream: `HuggingFaceTB/SmolLM3-3B`
- Runtime: `llama.cpp`, CPU baseline
- Context: 4096 tokens
- Temperature: 0
- Seed: 42

## Method

Neralis-3 is a synthetic normalization domain with disjoint partitions:

- 12 training experiences
- 5 skill-validation cases
- 8 held-out cases

The three held-out conditions used the same model and tasks:

1. Model only, without rules or history.
2. Model with all 12 verified raw input/output experiences.
3. Model with a manually distilled skill that had first passed the separate
   validation gate.

The verifier compared parsed JSON values exactly. Markdown JSON fences were
tolerated so formatting did not hide semantic correctness.

## Results

| Condition | Correct | Accuracy | Prompt tokens | Generated tokens | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model | 0/8 | 0.0% | 907 | 285 | 2.877 |
| Model + raw experiences | 1/8 | 12.5% | 5,835 | 211 | 2.682 |
| Model + verified skill | 7/8 | 87.5% | 2,003 | 246 | 2.779 |

Skill validation was 5/5, so the candidate crossed the 80% activation gate.
Compared with raw experiences, the skill used 65.7% fewer prompt tokens and
improved held-out accuracy by 75 percentage points.

The one skill failure produced the correct score and label but inserted the
signal name into the output key.

## Invalid pilot

An earlier pilot used character reversal and multiple character-count rules.
SmolLM3 failed even with the full skill. That pilot was rejected as a learning
measurement because character-level reversal disproportionately tests tokenizer
and small-model manipulation limits. It remains recorded in the local SQLite
ledger but is not included in the result table.

## Interpretation

This result supports continuing the AIR investigation. It does not yet prove
the main hypothesis because:

- the skill was manually distilled;
- the synthetic domain is intentionally simple;
- only one model, quantization, and deterministic seed were tested;
- repeated prompt prefixes may benefit from llama.cpp prompt caching, so wall
  time is not yet a reliable comparison;
- no automatic skill proposal or independent gatekeeper exists yet.

## Next falsification step

Generate candidate skills from raw experiences, validate them without access to
held-out cases, and repeat across multiple synthetic domains and seeds. Compare
the generated skill against both the manual upper bound and the raw-experience
baseline.

