# Experiment 0020 — Persistent Skill Accumulation, Reuse, Transfer, and Efficiency

## Result

Experiment 0020 tested whether verified procedures remained usable after the process was closed and restarted, without changing the model-free typed-search and behavioral-canonicalization path from Experiment 0019. All 32 distinct procedures became ACTIVE. Base skills were written to canonical JSON learned state without storing acquisition examples or expected answers; after a cold restart, top-1 retrieval was 32/32 and held-out accuracy was 100%.

This is not a general AGI or benchmark-equivalent model claim. It is a persistent procedural-memory and reuse result within a bounded semantic-IR space.

## Frozen protocol

- Model/runtime: `ggml-org/SmolLM3-3B-GGUF:Q4_K_M`, llama.cpp, CPU, context 4096.
- Weights, search grammar, verifier, sandbox, retrieval, and canonicalization unchanged; no LoRA, fine-tuning, or model swap.
- 32 skills: call, reverse-call, rotate-call, nested-call, and concat procedures.
- Four public, three hidden, three edge, and eight held-out examples were kept separate for every skill.
- Because of hardware cost, the final clean transfer set was reduced to 48 tasks and frozen after acquisition: 18 direct, 12 near, 12 two-skill, and 6 three-skill tasks.
- Candidate-search budget: maximum depth 3, maximum 5,000 raw candidates, maximum 32 public survivors. Hidden and edge examples were not used during search.
- Final vanilla-prompt SHA-256:
  `9f0508c0af370ea0a537601e2d14acbbba2969d3bebbe02877c037dd26f82344`.

## Acquisition and persistence

| Measurement | Result |
|---|---:|
| Attempted base skills | 32 |
| ACTIVE base skills | 32/32 (100%) |
| Wrong activation | 0 |
| Duplicate requests | 6/6 correctly reused |
| Duplicate avoidance | 100% |
| Conflict requests | 4/4 kept as separate conflict facets |
| Persisted basic skills before restart | 36 (32 + 4 controlled conflicts) |
| Loaded skills after restart | 39 (including 3 compiled compositions) |
| Zero-relearning reuse | 100% |
| Restart held-out accuracy | 100% |
| Canonical artifact bytes | 7,625 |

Controlled conflicts were kept as behaviorally distinct procedures under separate metadata facets within the same kind; they did not shadow the primary exact-retrieval path. All six duplicate requests reused an existing artifact without generating a new executable.

## Library growth and retrieval

At the 4, 8, 16, 24, and 32-skill checkpoints, existing-skill held-out retention and top-1 retrieval remained 100%. `catastrophic_external_interference = 0`.

In the controlled distractor fixture, the target skill remained top-1 with 32, 100, 1,000, 10,000, and 100,000 metadata entries, with one candidate inspected per query. This fixture measures metadata/index scale; it does not create 100k heavy executable Python objects. Canonical artifacts were not copied per facet.

## Transfer and composition

| Transfer | Tasks | Accuracy |
|---|---:|---:|
| Direct | 18 | 100% |
| Near / surface-varied | 12 | 100% |
| Two-skill composition | 12 | 100% |
| Three-skill composition | 6 | 100% |

The first controlled composition use took approximately 0.00018–0.00022 seconds with bounded scoped search. Three verified compositions were persisted as canonical artifacts; on the second use, direct retrieval reduced search calls to zero. The measured reuse speedup was approximately 10.2×–15.4×. This small in-process measurement contains latency noise and should be treated as directional.

## Final arms

| Metric | Vanilla 3B | Empty control* | 3B + learned state | Executor-only |
|---|---:|---:|---:|---:|
| Overall accuracy | 0/48 (0%) | 0/48 (0%) | 48/48 (100%) | 48/48 (100%) |
| Direct | 0% | 0% | 100% | 100% |
| Near | 0% | 0% | 100% | 100% |
| Two-skill | 0% | 0% | 100% | 100% |
| Three-skill | 0% | 0% | 100% | 100% |
| Model calls | 48 | 0 incremental | 0 | 0 |
| Input tokens | 4,923 | paired | 0 | 0 |
| Output tokens | 597 | paired | 0 | 0 |
| Latency p50 | 1.165 s | paired | 25 µs | 0 |
| Latency p95 | 1.568 s | paired | 67 µs | 0 |
| Wrong reuse | 0 | 0 | 0 | 0 |

* The Empty control reuses the same frozen vanilla-prompt answers as a paired control instead of making another 48 model calls. This saves compute but is not an independent second model sample. Vanilla calls: 48; timeouts: 0. The model could not answer the opaque procedures directly. The learned-state arm made no acquisition or search calls.

## Decision

- **ACQUISITION: PASS** — 32/32 ACTIVE, hidden/edge gates, and held-out passage.
- **ACCUMULATION: PASS** — state survived across processes.
- **PERSISTENCE: PASS** — cold-restart load with verified integrity hashes.
- **REUSE: PASS** — zero-relearning top-1 reuse at 100%.
- **TRANSFER: PASS** — direct and near transfer at 100%.
- **COMPOSITION: PASS (bounded)** — two- and three-skill composition at 100%.
- **EFFICIENCY: PASS** — the learned hot path was substantially faster without model or search calls.
- **Benchmark readiness: ALMOST_READY** — a strong persistent-procedural-memory signal, but broader independent and natural task families are still required.

This experiment supports the conclusion that the reusable procedural library is not an answer cache: learned state contains only canonical IR and metadata; expected answers for the final 48 tasks were not written into state. However, because the tasks were generated in the same frozen opaque semantic-IR space, transfer generalization remains bounded.

## Highest-information next step

A `composition/planning` experiment should measure whether task decomposition can be selected in a model-independent, scoped, and verifiable way for learned skills, using the same persistent store over more varied and previously unplanned task families.

Runtime report (excluded from Git):
`data/runs/air-020-20260901T204551Z.json`.
