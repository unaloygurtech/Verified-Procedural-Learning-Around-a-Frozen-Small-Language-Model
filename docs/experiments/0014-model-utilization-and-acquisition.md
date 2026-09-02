# Experiment 0014 — Frozen Model Utilization, Decomposition, Ranking, and Acquisition

0014 connects the real SmolLM3 runtime to the bounded retrieval and scoping
system measured by 0013.  It does not introduce a new index, storage format,
model, or context-budget optimizer.

## Frozen protocol

- Model/runtime: `ggml-org/SmolLM3-3B-GGUF:Q4_K_M`, llama.cpp, CPU, context 4096.
- Model parameters are not updated and no model swap is permitted.
- 0013’s canonical skill store, multi-facet indexes, fingerprints, deduplication,
  and scoped composition remain the source of candidates.
- The 0009 generic learner and 0011 retrieval prompt are reused unchanged.
- Ranking, direct-answer, decomposition, and composition-ranking templates are
  frozen before the first benchmark call.  Their versions and SHA-256 hashes are
  written to the runtime JSON.
- Runtime output is a checkpointed JSON file under `data/runs/`; it is ignored
  by Git.  A timeout is a result and is not silently retried.

## Blocks

### A — bounded candidate ranking

Controlled distractors include wrong capability, same type with another
operation, near-match input type, deprecated skill, and the correct skill.
Deterministic top-1 retrieval is compared with model ranking over top-3 and
top-5 candidates.  A missing correct candidate is `retrieval_failure`; a wrong
selection when it is present is `ranking_failure`.  Ambiguous candidates permit
safe abstention (`skill_id: null`).

### B — real context compression

A 100-document pool contains one normative document and controlled distractors.
The same task is evaluated with the full pool, retrieved documents, a relevant
snippet, and a compact fingerprint/contract.  No hard or soft token budget is
applied.  Compression is useful only when the semantic verifier still passes.

### C — model decomposition and scoped composition

The model emits an ordered capability list for two-stage, three-stage, and
missing-capability tasks.  Only the selected scope is opened for each subtask.
The report compares global brute-force theory, deterministic typed scoping,
model decomposition with scoped retrieval, and model decomposition plus
scoped ranking.  Wrong decomposition, wrong ranking, and wrong composition are
separate failure categories.

### D — novel capability acquisition

Three deterministic opaque API families are generated with disjoint discovery,
hidden-validation, edge, and held-out partitions.  The pipeline is:

```text
task → gap → 0013 retrieval/prefilter → frozen 0009 learner
     → static safety → sandbox → public → hidden → edge → activation
```

The main metric is `activation_given_correct_retrieval`.  Unsafe, malformed,
and semantically wrong artifacts remain rejected.  Acquisition cannot mutate
the previous library.

### E — learned-state reuse

An activated artifact is compared with SmolLM solving from context on held-out
inputs.  Artifact reuse reports executable calls, bytes read/query, and zero
model calls.  This is external procedural-state reuse, not model-weight
learning.

## Failure taxonomy

The runtime report uses: `retrieval_failure`, `decomposition_failure`,
`ranking_failure`, `composition_failure`, `gap_detection_failure`,
`synthesis_failure`, `repair_failure`, `public_validation_failure`,
`hidden_validation_failure`, `edge_failure`, `safety_rejection`, `timeout`,
and `safe_unknown`.

## Required interpretation

The result is not a single pass/fail claim and one activated family cannot be
called continual learning.  Interpret ranking, context quality, decomposition,
composition, acquisition, and external-state reuse independently.  The next
experiment must be chosen from the largest measured failure bucket, not in
advance.

Run it inside the existing container:

```bash
docker compose exec air-core python -m air_core.cli experiment-014 --heldout-limit 2
```

## Measured run (2026-09-01, frozen protocol)

The first real SmolLM3 run completed with **42 model calls**, **18,987 input
tokens**, **2,294 output tokens**, and **0 timeouts** (latency p50 4.65 s,
p95 39.88 s). The JSON checkpoint is
`data/runs/air-014-20260901T000653Z.json` and remains ignored by Git.

| Block | Measured result | Interpretation |
|---|---:|---|
| A — deterministic top-1 retrieval | 66.7% | Includes intentional ambiguous/miss controls; one target was deliberately absent. |
| A — model ranking top-3/top-5 | 100% / 100% when the target was retrieved | Retrieval and ranking are separate; the ambiguous case was a ranking failure because SmolLM did not abstain. |
| B — context accuracy | 4/4 conditions passed | Full pool: 2,478 tokens; top-5: 231 (90.7% reduction); snippet: 130 (94.8%); fingerprint/contract: 140 (94.4%). |
| C — decomposition | 1/3 correct | Two decomposition failures; the valid three-stage composition completed after scoped ranking. |
| C — search reduction | 5,896,800 global theoretical candidates vs 405 scoped candidates | Mechanical scoping works, but model planning is currently the bottleneck. |
| D — documentation retrieval | 3/3 correct | Correct retrieval did not guarantee a usable learned artifact. |
| D — activation given correct retrieval | 0/3 (0%) | All three families reached three bounded repair attempts and ended as `repair_failure`. |
| D — safety controls | 3 unsafe and 3 semantic-wrong proposals rejected | No unsafe activation; the previous library stayed immutable. |
| E — artifact reuse | Not exercised | No new skill activated, so there was no valid artifact to reuse. |

The strongest positive result is that natural retrieval compression preserved the
real model’s task accuracy in this controlled token task. The largest measured
bottleneck is **repair/synthesis after correct documentation retrieval**, not
storage or context size. This is a bounded result, not evidence that AIR has
solved continual learning or learned Python generally.
