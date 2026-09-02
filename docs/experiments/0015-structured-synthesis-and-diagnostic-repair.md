# Experiment 0015 — Structured Synthesis, Failure Diagnosis, and Reliable Skill Acquisition

Experiment 0015 isolates the acquisition bottleneck observed in Experiment 0014. Retrieval, canonical storage, facet/fingerprint indexes, composition, model, and sandbox boundaries remain unchanged. The only change is the protocol used to generate new Python procedures and repair failed candidates.

This is not a general continual-learning claim; it is a bounded acquisition measurement over five independent deterministic opaque-API families.

## Frozen operating conditions

- `ggml-org/SmolLM3-3B-GGUF:Q4_K_M`, llama.cpp, CPU, context 4096.
- Model weights, model selection, retrieval, storage, and verifier remain unchanged.
- Five families with disjoint `4 discovery / 3 hidden / 3 edge / 8 held-out` examples.
- The fixed seed and opaque `air_synth_012` package from Experiment 0012 are used.
- Before the first real model call, SHA-256 values for the contract, structured-synthesis, and diagnostic-repair prompts are computed and written to the runtime JSON.
- A timeout is a result; there are no silent retries. The runtime checkpoint is stored under `data/runs/air-015-<timestamp>.json` and is excluded from Git.

## Three acquisition arms

### Arm A — Full-program baseline

The generic learner from Experiments 0014/0009 generates the complete `transform` program. It receives one initial proposal and at most three generic repair attempts.

### Arm B — Structured synthesis

The system deterministically generates the function name, signature, permitted import/API surface, wrapper, and return contract. The model returns only the semantic function body. The structured skeleton therefore encodes the solution rule; opaque API semantics still have to be inferred from documentation and public examples. This arm measures only the first proposal.

### Arm C — Structured synthesis plus diagnostic repair

This arm is identical to Arm B, except that a failed candidate receives a targeted repair prompt based on verifier evidence rather than a generic “try again” request. Syntax, import, signature, type, runtime, output, hidden, and edge failures are classified separately. The total repair budget is fixed at `3` before the benchmark begins.

For every candidate, the source SHA-256 and normalized-AST hash are recorded. A repeated candidate is marked `duplicate_candidate`; its model-call cost is included, but it is not counted as a new hypothesis.

## Activation gate and measurements

```text
retrieval → contract → candidate → static safety → public
          → hidden → edge → ACTIVE / REJECTED
```

An artifact becomes ACTIVE only after correct retrieval, gap detection, and every validation gate pass. Each arm reports the acquisition funnel, `activation_given_correct_retrieval`, wrong activation, public/hidden/edge transitions, total/unique/duplicate candidates, repair success rate, failure taxonomy, model calls/tokens/latency, and cost per success.

For an active artifact, eight held-out examples compare the executable artifact with SmolLM and the relevant documentation. The artifact path uses zero model calls and reports `bytes/query`; this is interpreted as external procedural-state reuse, not learning in the model weights.

## Reproduction

```bash
docker compose exec air-core python -m air_core.cli experiment-015
```

For a quick smoke or unit-style run, the held-out count can be limited:

```bash
docker compose exec air-core python -m air_core.cli experiment-015 --heldout-limit 2
```

To resume from a completed family checkpoint:

```bash
docker compose exec air-core python -m air_core.cli experiment-015 --resume /workspace/data/runs/air-015-<timestamp>.json
```

## Pre-registered interpretation boundaries

Experiment 0015 answers only these questions: does structured synthesis reduce acquisition failure compared with full-program generation; does diagnostic repair outperform generic retry; and does any gain extend across multiple novel families? If the result is insufficient, the report must say “no.” A Level 3 or general continual-learning claim cannot be inferred automatically. The decomposition/planning axis from Experiment 0014 remains unresolved. Experiment 0016 should be selected only after inspecting the Experiment 0015 failure taxonomy.

## Frozen runtime result (2026-09-01)

Runtime report: `data/runs/air-015-20260901T120406Z.json`.

The model/runtime conditions were unchanged: SmolLM3 3B Q4_K_M, llama.cpp, CPU, context 4096. A total of **30 model calls**, **13,148 input tokens**, **2,079 output tokens**, and **0 timeouts** were observed; latency was p50 **11.84 s** and p95 **18.76 s**.

| Measurement | Arm A — full program | Arm B — structured | Arm C — diagnostic |
|---|---:|---:|---:|
| Correct retrieval (5 families) | 5/5 | 5/5 | 5/5 |
| Contract extraction | shared 0/5 | shared 0/5 | shared 0/5 |
| Activation | 0/5 | 0/5 | 0/5 |
| `activation_given_correct_retrieval` | 0% | 0% | 0% |
| Initial structurally valid candidate | 0/5 | 0/5 | 0/5* |
| Hidden/edge → ACTIVE transition | 0/5 | 0/5 | 0/5* |
| Wrong activation | 0 | 0 | 0 |
| Repair attempts / successful repairs | 15 / 0 | 0 / 0 | 0 / 0 |
| Repair success rate | 0% | N/A | N/A |
| Total / unique / duplicate candidates | 20 / 8 / 10 | 0 / 0 / 0 | 0 / 0 / 0 |
| Held-out artifact reuse | N/A | N/A | N/A |

* Because contract extraction failed, Arms B/C were safely blocked before candidate generation. This does not indicate an error in the hidden/edge verifier; no candidate reached the ACTIVE stage.

Retrieval was correct for every family and the previous learned library was unchanged. Safety controls rejected unsafe and semantically wrong candidates; regression, provenance, and artifact-immutability checks passed. However, none of the three arms produced an ACTIVE skill, so there was no artifact available for held-out reuse.

### Failure diagnosis and decision

Failure-taxonomy counts were: `contract_extraction_failure=15` (5 families × 3 arms), `static_safety_rejection=3`, `syntax_failure=1`, `repair_failure=1`; all other activation-failure classes were 0. The largest and earliest bottleneck was **contract extraction**. The model selected the opaque operation name as a callable, returned empty permitted-import lists, and produced no invariants; the frozen parser correctly rejected these contracts. Arm A also generated the same or duplicate candidate in 10 of 20 full-program proposals, and no repair succeeded.

Experiment 0015 therefore reports **failed acquisition**:

- structured synthesis outperformed the baseline: **no**;
- diagnostic repair provided additional benefit: **no** (the contract gate was not reached);
- the gain extended across multiple novel families: **no**;
- hidden/edge-validated ACTIVE skills: **0**;
- held-out reuse: **not applicable**;
- wrong activation and regression of existing skills: **0 / no regression**.

### Experiment 0016 gate result

**End-to-end capability accumulation was not implemented in Experiment 0016.** The required conditions—multiple ACTIVE novel families, hidden-plus-edge passage, structured/diagnostic superiority, and held-out reuse—were not met. The highest-information next study should first isolate **contract-acquisition reliability**: measure correct extraction of contract fields with the same frozen model and documentation before moving to end-to-end planning or composition.
