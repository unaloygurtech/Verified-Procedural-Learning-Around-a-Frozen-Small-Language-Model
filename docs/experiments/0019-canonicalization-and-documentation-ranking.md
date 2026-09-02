# Experiment 0019 — Behavioral Canonicalization and Documentation-Grounded Ranking

Experiment 0019 separately measured the two questions left open by Experiment 0018:

1. Can different programs with identical behavior on public, hidden, and edge sets be reduced to one artifact safely and deterministically?
2. Can SmolLM3-3B select a candidate using normative documentation semantics rather than public examples and candidate ordering?

The experiment added no new memory, storage, planning, model, or primitive layer.

## Frozen conditions

- `ggml-org/SmolLM3-3B-GGUF:Q4_K_M`, llama.cpp, CPU, context 4096;
- weights frozen; no model swap, Qwen, 7B/8B/14B model, LoRA, or fine-tuning;
- the Experiment 0018 candidate generator, `AIR-SEMANTIC-IR` v1, depth 3, 5,000 raw-candidate budget, and 32-public-survivor limit unchanged;
- search saw public examples only; hidden and edge examples were used only by the activation/equivalence gate;
- no family-specific prompt, answer-specific candidate, manual tie-break, silent retry, or post-result seed/prompt change;
- ranking-prompt versions and SHA-256 values frozen before the first model call.

Canonical selection order:

```text
minimum semantic depth
→ minimum operation count
→ minimum serialized IR bytes
→ minimum estimated execution cost
→ lexical canonical IR
```

Raw frozen runtime:
`data/runs/air-019-20260901T193544Z.json`.

## Part A — Behavioral canonicalization

Part A reused the same 12 families and candidate search from Experiment 0018.

| Measurement | Result |
|---|---:|
| Families | 12 |
| Correct candidate coverage | **12/12** |
| Mean public survivors | 5 |
| Ambiguous families before canonicalization | **12/12** |
| Ambiguous families after canonicalization | **0/12** |
| Combined behavior classes | 1/family |
| Canonical ACTIVE | **12/12** |
| Behaviorally correct activation | **100%** |
| Exact oracle IR | 9/12 |
| Canonical-choice stability | **12/12** |
| Held-out accuracy | **100%** |
| False merge | **0/6** |
| Model calls | **0** |
| Mean search latency | 107.7 ms/family |
| Mean canonicalization latency | 3.60 ms/family |
| Mean artifact size | 167 B |

Although the exact oracle hash matched only 9/12, every selected artifact behaved correctly on hidden, edge, and held-out examples. The primary metric is therefore verified behavioral correctness rather than exact syntax.

### False-equivalence controls

All six negative pairs looked identical on the public set. The canonicalizer retained all six as separate classes using later gate witnesses:

| Control | Distinguishing witness | False merge |
|---|---|---:|
| Same public, different hidden | hidden | 0 |
| Same public + hidden, different edge | edge | 0 |
| Same on short input, different on long input | hidden | 0 |
| Same type, different ordering | hidden | 0 |
| Same callable, different wrapper | hidden | 0 |
| Same shape, different rotate constant | hidden | 0 |

This does not mean canonicalization proves arbitrary semantic equivalence. Equality is established only under the frozen public+hidden+edge witness set; a new input domain could break it. The 100% held-out result provides additional support for these 12 families, not a mathematical proof of program equivalence.

## Part B — Documentation-grounded ranking

Part B used 20 deterministic dataset variants. These are **not 20 new primitives**; they are 20 new public/hidden/edge/held-out splits within the 15 opaque-operation namespace frozen in Experiment 0018. The first 12 targets come from the existing target set, and at least eight come from independent data seeds 1901–1920.

Public examples were intentionally made non-discriminative. After search, each family retained three structurally minimal candidates. Separate diagnostic-probe outputs for the candidates were shown to the model; these probes were not hidden, edge, or held-out data. The normative rule that distinguished the correct candidate was present in the documentation.

For each accuracy value, the parenthetical value is the Wilson 95% confidence interval.

| Condition | Correct selection |
|---|---:|
| No documentation | **7/20, 35%** (18.1–56.7%) |
| Correct documentation | **6/20, 30%** (14.5–51.9%) |
| Wrong-but-plausible documentation | **6/20, 30%** (14.5–51.9%) |
| Correct documentation + distractors | **8/20, 40%** (21.9–61.3%) |
| Behavior-descriptor representation + correct documentation | **7/20, 35%** (18.1–56.7%) |
| Fixed random | **8/20, 40%** (21.9–61.3%) |
| Oracle | **20/20, 100%** (83.9–100%) |

Main differences:

```text
correct-documentation lift vs no documentation = -5 percentage points
correct documentation vs wrong documentation =  0 percentage points
wrong-documentation harm                       = +5 percentage points
```

Correct documentation did not outperform either the no-documentation or random baseline. Although the confidence intervals are wide, the point-estimate direction is not positive.

### Counterfactual documentation

In eight pairs, the candidate set and public evidence were held constant. Document A declared target A normative; Document B declared alternative target B normative.

```text
Correctly follow both Document A and Document B: 0/8
Wilson 95% CI: 0–32.4%
```

In the wrong-documentation arm, the alternative behavior was selected 11/20 times. However, because there was no predictable A→B flip paired with correct-documentation selection in the same family, this is not strong evidence of semantic conditioning.

### Candidate order and size

| Measurement | Result |
|---|---:|
| Behavioral stability between two permutations | 11/20, 55% |
| First-candidate selection | 55% |
| Last-candidate selection | 10% |
| Correct candidate available at size 2 | 17/20 |
| Size-2 correct-documentation accuracy | 9/20, 45%; 9/17, 52.9% when available |
| Correct candidate available at size 3 | 20/20 |
| Size-3 correct-documentation accuracy | 6/20, 30% |

The selection result did not improve as the candidate count increased. The 55% first-position rate and only 55% order stability provide an additional negative signal that ranking was substantially influenced by pattern or position.

Correct-documentation results by kind also showed no consistent semantic pattern:

```text
object 3/7
mixed  1/7
runs   2/6
```

## Part C — Combined acquisition

Three deployable arms were compared on the same 20 families:

| Arm | ACTIVE | Held-out | Model calls |
|---|---:|---:|---:|
| Pure search + canonicalization | **20/20** | **100%** | **0** |
| Search + SmolLM correct-documentation ranking | 6/20 | 100% for active artifacts | 20 |
| Hybrid | **20/20** | **100%** | **0** |

The hybrid first reduced candidates to behavior classes that passed hidden and edge gates. One verified class remained per family, so no model call was required.

| Efficiency metric | Result |
|---|---:|
| Model avoidance | **20/20, 100%** |
| Model-call savings versus ranked arm | **20** |
| Input-token savings | **12,030** |
| Output-token savings | **350** |
| Hybrid total latency | 2.63 s |
| Hybrid latency/acquired skill | 131.5 ms |
| Ranked total latency | 182.68 s |
| Active learned state / artifact bytes per query | 167 B |
| Wrong activation | **0** |

`correct_acquisition_per_model_call` is not reported as a numeric ratio for the hybrid because its denominator is zero. The runtime labels it `model_free`.

## Model cost

All diagnostic arms in Part B:

| Measurement | Result |
|---|---:|
| Model calls | 148 |
| Input tokens | 85,389 |
| Output tokens | 2,579 |
| Timeouts | 0 |
| Model latency p50 | 7.92 s |
| Model latency p95 | 11.35 s |
| Mean model latency | 7.76 s |
| Total model-call latency | 1,148.91 s |

Only 20 of these 148 calls belonged to the deployable ranked arm. The rest were diagnostic controls for no-documentation, wrong-documentation, distractor, order, representation, candidate-size, and counterfactual conditions.

## Failure taxonomy

```text
candidate_coverage_failure   0
public_ambiguity            32
hidden_ambiguity            12
edge_ambiguity              12
false_equivalence_failure    0
canonicalization_failure     0
ranking_failure             14
documentation_misuse        14
counterfactual_doc_failure   8
position_bias_failure        9
wrong_activation             0
heldout_failure              0
timeout                      0
safe_unknown                 0
invalid_candidate_id         0
```

`hidden_ambiguity` and `edge_ambiguity` count multiple verified syntaxes before Part A canonicalization; distinct behavioral ambiguity after canonicalization was 0/12.

## Decisions

### Documentation semantic-use decision

```text
WEAK_OR_NO_DOC_SEMANTIC_USE
```

In this frozen test, SmolLM3-3B did not reliably use novel documentation semantics for candidate discrimination. Correct documentation was 5 points worse than no documentation, tied wrong documentation, and 10 points worse than random. Counterfactual following was 0/8 and order stability was 55%. Much of the Experiment 0018 ranking result appears to come from candidate, example, and position patterns.

### Bounded-system decision

```text
YES — for the bounded architecture
NO — as a reliable documentation-semantic ranking core
```

These statements are compatible. Bounded acquisition worked 20/20 through search + verifier + canonicalization without requiring 3B ranking. The current 3B model may remain a UI, task-routing, or bounded-scoring assistant when useful, but this experiment does not support the claim that it is the primary learning/reasoning core for truly understanding new documentation.

### Interpretation cases

- **CASE 1 — Pure Search + Canonicalization Strong:** passed.
- **CASE 3 — Documentation Still No Lift:** passed.
- **CASE 5 — Hybrid Strongest:** matched pure search in accuracy; it is the production preference because of model avoidance and safe fallback behavior.
- CASE 2 and CASE 4 were not supported.

## Direct answers

1. **Did canonicalization solve the Experiment 0018 pure-search ambiguity?**  
   **Yes.** All 12/12 families were ambiguous before canonicalization; 12/12 became canonical ACTIVE with 0/12 final behavioral ambiguity.

2. **Could the system safely activate behaviorally equivalent skills without a model?**  
   **Yes within this bounded witness set:** 12/12, 100% held-out, false merge 0/6, wrong activation 0. This is not a general program-equivalence proof.

3. **Did SmolLM really use documentation semantics?**  
   **Not reliably.** The decision is `WEAK_OR_NO_DOC_SEMANTIC_USE`.

4. **Did wrong or counterfactual documentation predictably change decisions?**  
   **No.** Joint counterfactual following was 0/8.

5. **Was ranking semantic or more pattern/position-driven?**  
   **It appears pattern/position-driven.** Correct documentation was 30%, random was 40%, order stability was 55%, and first-position selection was 55%.

6. **Did the hybrid reduce model use without reducing accuracy?**  
   **Yes.** It achieved 20/20 ACTIVE, 100% held-out accuracy, 100% model avoidance, and saved 20 calls plus 12,030 input tokens relative to the deployable ranked arm.

7. **Is the current 3B model sufficient for the bounded learning core?**  
   **Architecturally yes, because the critical acquisition path became model-free.** There is no evidence that it is sufficient as a documentation-reasoning core.

8. **Is the acquisition path robust enough to return to sequential accumulation/planning?**  
   **Within the bounded scope, yes:** Part A 12/12, Part C 20/20, 100% held-out, wrong activation 0, and false merge 0. This still does not declare Level 3.

9. **What should the next experiment be?**  
   **End-to-end sequential accumulation/planning.** It should measure whether task decomposition can be selected in a scoped, verifiable, model-independent way over more varied and previously unplanned task families. A larger-model comparison is valuable for semantic-ranking research, but bounded acquisition no longer depends on it and it is not the primary next step.

## Safety, regression, and tests

- wrong activation: 0;
- false canonical merge: 0/6;
- hidden leakage: none;
- invalid candidate ID: rejected;
- grammar, generator, sandbox, and verifier unchanged;
- no silent retry;
- Docker full suite: **138 tests, all passed**.

## Reproduction

```bash
docker compose exec air-core python -m air_core.cli experiment-019
docker compose exec air-core python -m air_core.cli experiment-019 --resume /workspace/data/runs/air-019-<timestamp>.json
```

Runtime JSON is excluded from Git; source, tests, and this report are committed.
