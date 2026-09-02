# Experiment 0018 — Verified Candidate Search and Program Induction

Experiment 0017 left direct Python and canonical semantic-IR generation at 0/8 for SmolLM3-3B. Experiment 0018 preserves that direct docs→IR arm as a baseline and measures a key distinction:

```text
Does the model fail to understand the semantics,
or can it understand them but not produce an executable representation?
```

This is not a new memory, storage, or planning layer. The model no longer writes the program; it selects among candidates generated inside the frozen typed search space.

## Frozen protocol

- `ggml-org/SmolLM3-3B-GGUF:Q4_K_M`, llama.cpp, CPU, context 4096;
- weights frozen; no model swap, Qwen, 7B/8B/14B model, LoRA, or fine-tuning;
- retrieval, canonical state, sandbox, verifier, provenance, and hidden/edge gates from Experiments 0013–0017 unchanged;
- 12 independent opaque families: eight from Experiment 0017 and four new families from seed 1203;
- four public, three hidden, three edge, and eight held-out examples per family;
- the candidate generator does not see the family ID, ground-truth rule, or hidden/edge examples; the grammar is frozen before the benchmark;
- the direct-generation prompt is reused exactly from Experiment 0017; ranking and constraint prompt SHA-256 values are recorded before the first call;
- checkpoint output performs no silent retry; a timeout is stored as a result.

The candidate language is typed `AIR-SEMANTIC-IR` v1 from Experiment 0017:

```text
INPUT, INT, CALL, REVERSE, ROTATE, CONCAT, RETURN
```

For `CALL`, the generator scans the complete frozen `air_synth_012` opaque-operation namespace independently of the model. This is not the addition of a family-specific primitive. The result therefore cannot be interpreted as arbitrary Python or general program synthesis. The measured scope is candidate coverage and selection within a known safe typed primitive namespace.

Search budget:

| Limit | Frozen value |
|---|---:|
| Maximum expression depth | 3 |
| Maximum raw candidates | 5,000 |
| Maximum public survivors | 32 |
| Candidate-size curve | 2, 3, 5, 8 |
| Random seed | 18018 |

Search uses typed pruning, canonical AST hashes, and public-output behavior signatures. Public-equivalent candidates are not deleted completely; the canonical candidate and alternatives are retained as metadata. If multiple candidates survive hidden and edge validation, pure search safely rejects with `ambiguous_program`.

## Arms

| Arm | Model task |
|---|---|
| A — direct generation | Experiment 0017 docs→IR baseline; the model generates IR |
| B — pure search | No model; typed search + public + hidden + edge validation |
| C — search + SmolLM | Selects only a `candidate_id` from the public-survivor list |
| D — random | Fixed-seed random selection from the same candidate list |
| E — oracle | Selects the ground-truth-equivalent candidate as a diagnostic upper bound |
| F — docs/no-docs | Documentation and no-documentation ranking on the same candidate set |
| G — constraint search | The model produces only a small descriptor; search narrows using that descriptor |

Arm E is not a deployable model. It measures the upper bound available when candidate generation is correct but ranking/ambiguity remains.

## Frozen runtime result

Raw report:
`data/runs/air-018-20260901T184555Z.json`.

The run used **130 model calls**, **60,508 input tokens**, **2,608 output tokens**, and **0 timeouts**. Call distribution:

| Arm | Calls | Input | Output |
|---|---:|---:|---:|
| A direct generation | 12 | 4,159 | 689 |
| B pure search | 0 | 0 | 0 |
| C search + SmolLM | 53 | 28,049 | 691 |
| D random | 0 | 0 | 0 |
| E oracle | 0 | 0 | 0 |
| F docs/no-docs | 53 | 25,092 | 689 |
| G constraint | 12 | 3,208 | 539 |

Overall model latency was p50 **2.11 s** and p95 **6.16 s**; total acquisition latency was approximately **335.86 s**.

## Required comparison

| Measurement | A direct | B pure search | C SmolLM | D random | E oracle | F docs/no-docs | G constraint |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACTIVE | 0/12 | 0/12 | 12/12 | 12/12 | 12/12 | 12/12 | 0/12 |
| Exact candidate selection | 0/12 | — | 9/12 | 3/12 | 12/12 | docs 9/12; no-docs 8/12 | — |
| Active held-out artifacts | 0 | 0 | 12 | 12 | 12 | 12 | 0 |
| Mean held-out accuracy | — | — | 100% | 100% | 100% | 100% | — |
| Model calls | 12 | 0 | 53 | 0 | 0 | 53 | 12 |

ACTIVE, hidden/edge, and held-out results are calculated using the verifier gate; exact candidate selection is calculated using the oracle IR hash. Most candidates are syntactic alternatives with identical behavior on public, hidden, edge, and held-out examples. Exact identity and behavioral activation are therefore reported separately.

## Candidate-coverage funnel

| Stage | Result |
|---|---:|
| Correct candidate generated | **12/12 (100%)** |
| Correct candidate survived public validation | **12/12 (100%)** |
| Mean raw candidates | **1,554/family** |
| Mean type-invalid candidates rejected | **0** (the generator produces only type-valid ASTs) |
| Mean AST duplicates rejected | **555** |
| Mean behavior-equivalent alternatives | **807** |
| Mean public survivors | **5** |
| Oracle candidate available rate | **100%** |
| Public-overfit candidates | **0** |
| Search budget exhausted | **0/12** |

Correct-candidate coverage is complete within this bounded primitive namespace. This does not show that the generator invented Python or a new semantic primitive; it shows that it found the correct opaque callable and generic wrappers within the search scope.

## Candidate-size ranking curve

Each row uses a deterministic prefix of the same public-survivor evidence. “Correct available” indicates whether a ground-truth-equivalent candidate is present in that prefix.

| Candidate count | Families | Correct available | SmolLM exact | No-doc exact |
|---:|---:|---:|---:|---:|
| 2 | 12 | 11/12 | 10/12 (83.3%) | 11/12 (91.7%) |
| 3 | 12 | 12/12 | 12/12 (100%) | 12/12 (100%) |
| 5 | 12 | 12/12 | 10/12 (83.3%) | 9/12 (75.0%) |
| 8 | 12 | 12/12 | 9/12 (75.0%) | 8/12 (66.7%) |

On the main candidate set, exact ranking was **9/12** for SmolLM and **3/12** for fixed random, a descriptive lift of **+0.50**. This single 12-family frozen run is not sufficient for an inferential-significance claim. The documentation/no-documentation difference was **9−8 = +1/12 (+8.3 percentage points)** on the main set, and its direction was mixed across candidate sizes; this is not strong evidence of documentation-semantic lift.

## Pure search and ambiguity

Pure search generated the correct candidate but produced **0/12 ACTIVE**. In every family, more than one candidate survived hidden and edge validation; the strict ambiguity policy did not select among them. Therefore:

- candidate-coverage failure: none;
- public-pruning failure: none;
- hidden/edge validation failure: none;
- pure model-free deployment: **safe reject**, not automatic acquisition;
- the oracle selected 12/12 on the same candidate sets and confirmed the generator/verifier upper bound;
- SmolLM candidate-ID selection resolved ambiguity operationally and activated 12/12 artifacts, although exact identity was 9/12.

This gives a strict **no** to the question “does the system learn when the model is completely removed?” The model-free arm finds and verifies the correct program, but it does not activate it safely when equivalent alternatives require a tie-break.

## Constraint arm

The model was asked for a descriptor rather than a program:

```json
{"uses_call": true, "uses_reverse": false, "uses_rotate": false,
 "uses_concat": false, "max_depth": 3}
```

Actual descriptor accuracy was **0/12**, and correct-program retention was **0/12**. Incorrect constraints removed the correct candidate from the search space; the arm was recorded as `constraint_prediction_failure` and did not activate.

## Failure taxonomy

```text
direct_generation_failure     12
ambiguous_program              12
random_selection_failure        9
constraint_prediction_failure  12
candidate_coverage_failure       0
public_pruning_failure           0
public_overfit_candidate         0
hidden_validation_failure        0
edge_failure                     0
compilation_failure              0
safety_rejection                 0
heldout_failure                  0
timeout                          0
invalid_candidate_id              0
search_budget_exhausted           0
```

Search output is not the primary failure bucket in Experiment 0018; direct canonical program generation remains the failure identified in Experiment 0017. The new boundary in Experiment 0018 is syntactic/equivalent-candidate ambiguity after public evidence.

## Direct answers

1. **Did the system learn a novel capability through candidate search and verification without the model?**  
   **Strictly, no; PARTIAL.** The correct candidate was found 12/12 and oracle verification passed 12/12, but the strict pure-search ambiguity gate activated 0/12. The model-free path is currently “verified but not selected.”

2. **Did SmolLM ranking add a meaningful contribution over pure search?**  
   **Yes, for bounded selection.** Pure search safely rejected 0/12, while SmolLM activated 12/12 artifacts; exact oracle identity was 9/12.

3. **Was SmolLM meaningfully better than the random baseline?**  
   **Descriptively, yes:** exact selection was 9/12 versus 3/12 for random, a **+50 percentage-point lift**. Twelve families in one frozen run are not enough for inferential significance.

4. **Did documentation improve ranking?**  
   **Not strongly.** The main set was 9/12 with correct documentation versus 8/12 without; the +8.3-point difference was small and the candidate-size curve was mixed. This is only a weak/unstable signal of novel documentation understanding.

5. **Could the model recognize novel semantics but fail to produce an executable representation?**  
   **Yes, this is the most compatible interpretation, but it is not complete proof.** Direct IR generation was 0/12, candidate coverage was 12/12, and ranking was 50 percentage points above random. Because candidates came from an opaque primitive namespace, this cannot establish general semantic cognition.

6. **Did the candidate generator cover the correct programs sufficiently?**  
   **Yes for this bounded grammar and 12 families: 12/12.** This does not generalize to Python or invention of new primitives.

7. **Is the main limit the search engine, output generation, or semantic capacity?**  
   **Primarily output generation/output following.** Generator coverage and oracle verification were 12/12; the secondary limit is public-equivalent ambiguity. Ranking is positive for semantic capacity, but the weak documentation lift means the question is not fully resolved.

8. **Is SmolLM3-3B sufficient as a reasoning/learning core for this system?**  
   **For a limited bounded core, yes; for a general acquisition core, no.** It can select a candidate ID, but it cannot directly produce a new executable capability and the constraint-descriptor arm was 0/12.

9. **Is a larger-model comparison scientifically justified?**  
   **Yes.** Generator and oracle validation succeeded completely; the unresolved boundary is candidate discrimination and documentation use under the 3B model. A larger model should be tested on the same frozen candidate sets and ranking protocol. Presenting candidate search as model success would not be meaningful.

10. **Does the 3B-plus-system hypothesis continue?**  
    **Yes for the bounded architecture; PARTIAL for the general vision.** Search + SmolLM ranking + verifier achieved 12/12 ACTIVE and 100% held-out accuracy. Pure model-free acquisition achieved 0/12, so “the model is unnecessary” does not follow; neither arbitrary program induction nor Level 3 can be claimed.

## Engine decision

```text
OUTPUT_GENERATION_LIMIT
```

The basis is: the correct candidate was generated 12/12 and oracle-verified 12/12, but direct model IR generation remained 0/12. The SmolLM ranking signal is positive, so the 3B model has non-zero semantic-discrimination ability. However, behavior-equivalent candidates and the weak documentation lift do not prove that the model generally understands new semantics.

## Safety and regression

- manual activation: none;
- hidden + edge gate: mandatory;
- unsupported opcode: rejected;
- invalid candidate ID: rejected outside the candidate set;
- public-overfit candidate: not activated;
- wrong activation: 0;
- canonical learned state, sandbox, and verifier unchanged;
- existing-skill regression: none;
- Level 3: not declared.

## References and reproduction

- [candidate generator](../../src/air_core/program_search.py)
- [Experiment 0018 runtime](../../src/air_core/exp018.py)
- [Experiment 0017 semantic IR](../../src/air_core/semantic_ir.py)

```bash
docker compose exec air-core python -m air_core.cli experiment-018
docker compose exec air-core python -m air_core.cli experiment-018 --resume /workspace/data/runs/air-018-<timestamp>.json
```

Runtime JSON is excluded from Git. Source, tests, and this report are committed.
