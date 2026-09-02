# Experiment 0013 — Hierarchical Memory, Capability Fingerprints, Deduplication, and Scoped Composition

## Scope

This study preserves the same `SmolLM3-3B-GGUF:Q4_K_M` runtime and 4096-token context setting. Its purpose is not to change the model or claim general intelligence, but to decompose the learned-state retrieval, context, and composition bottlenecks observed in Experiment 0012 into measurable system controls. The core benchmark makes no model calls. This prevents prompt or ranking behavior from obscuring index performance; a frozen model-ranking arm can be added separately over a bounded candidate set in Experiment 0014.

## Architectural decision

The system uses a single-ownership model:

    canonical skill store (skill_id -> immutable artifact)
            + domain/family/type/operation/trust/facet indexes
            + deterministic capability fingerprint index

A skill may appear under multiple facets, but the executable artifact is physically stored only once. Indexes contain only `skill_id` references. Retrieval can use intersection (all known facets) or union (partial task clues). Providing three facets in a query does not require all five facets to be known exactly; scalar verification remains the final gate.

## Frozen measurement protocol

- Synthetic libraries of 100, 1,000, 10,000, and 100,000 metadata-plus-artifact records.
- At 1,000,000 records, an explicit working-set guard activates; the process is not forced into memory exhaustion and the point is reported as `skipped_safe_resource_guard`.
- Retrieval comparison: naive linear scan, the Experiment 0012 exact-key baseline, and a hierarchical/fingerprint-aware multi-index.
- Each query measures top-1/top-3/top-5, candidate count, bytes read, active learned state, RAM proxy, and p50/p95 latency. Core benchmark model calls and model input tokens are zero.
- The fingerprint is deterministic canonical JSON containing input/output type and shape, operation, pure/deterministic flags, side effects, trust, cost, and version.
- Deduplication merges artifacts only after structural bucketing, behavioral tests, and the equivalence gate. Syntax differences do not prevent semantic comparison; edge-case differences prevent merging.
- Composition is measured with two-stage, three-stage, and missing-capability tasks. Global brute force, type-filtered/hierarchical scope, and bounded stage-subagent scope all pass through the same contract verifier.
- The context arm does not use a hard token budget. It compares a full 100-document pool, top-K documents, a relevant snippet, and fingerprint/artifact-only conditions using token counts and the downstream semantic gate.

## Measurement summary

The runtime report is written to `data/runs/air-013-<timestamp>.json` and is excluded from Git. The main expected/measured signals are:

| Field | Result |
|---|---|
| Maximum real library | 100,000; the 1M point was skipped by the safe resource guard |
| Hierarchical retrieval accuracy | Target found at every 100/1k/10k/100k point; top-1 and top-5 recall 1 |
| Candidates/query | 1 for full-facet plus unique lexical queries; 33,333 at 100k for a single-facet query |
| Naive p50 | 111.788 ms at 100k; 100,000 candidates scanned and approximately 38.1 MB read |
| Experiment 0012 exact-index p50 | 73.8 µs at 100k; still the fastest exact-lookup baseline |
| Hierarchical p50 | 2.069 ms at 100k; one candidate and approximately 822 B of hot state read |
| 100k index/RAM | 18.84 MB logical index; 202.8 MB deterministic Python working-set proxy; 600 kB canonical artifact |
| Active state/context | The full library was not sent to the model; core model calls/tokens: 0 |
| Fingerprint | Approximately 132 B/skill; approximately 35% of metadata at 10k; approximately 12% candidate pruning with no accuracy regression |
| Multiple facets | Three-facet intersection produced one candidate at 100k; a single facet produced 33,333; artifact duplication was 0 B; union expanded to 40,000 and produced top-5 ambiguity |
| Deduplication | 6 → 4 representatives (33.3% reduction), 87 B saved, precision/recall 1, false merge 0 |
| Composition | At 180 skills, depth-3 global brute force: 5,832,000 combinations; agent scope: 1 candidate; reduction >99.999%; accuracy preserved |
| Missing-capability safety | Missing-capability composition was safely rejected |
| Context compression | 1,844 tokens over 100 documents → 15-token relevant snippet (99.2% reduction); downstream semantic correctness preserved; hard budget OFF |

Latency, RAM proxy, and byte measurements depend on hardware and interpreter behavior; the exact p50/p95 values in the report are authoritative. The Experiment 0012 SQLite baseline is included as a logical exact-key control and remains faster for exact lookup than the hierarchical index. The hierarchical index is not intended to beat exact-lookup latency; its contribution is candidate pruning and single-artifact ownership for partial or ambiguous multi-facet queries. The union query is intentionally broad: without ranking, it cannot guarantee the target appears in the top five, and it is a negative ambiguity/latency signal in this experiment.

## Negative results and limitations

1. One million executable Python objects were not run because of the safe resource limit. This is not a claim of 1M-scale success.
2. Because core Experiment 0013 makes no model calls, real SmolLM3 output accuracy, acquisition success, and model-ranking latency were not measured here. The context arm is a retrieval-semantic control; it does not by itself invalidate the 100-document model-timeout finding from Experiment 0012.
3. Fingerprints add metadata overhead and do not automatically reduce latency for every query under these conditions. Their gains are in candidate pruning, bytes read, and context size.
4. The composition result uses a deterministic typed fixture. The frozen Experiment 0014 protocol is required to measure real model decomposition errors.
5. The RAM value is a deterministic Python-object working-set proxy, not process RSS.

## Interpretation and next experiment

Experiment 0013 makes storage measurable as a “single copy plus multiple retrieval paths” system: while naive scanning remains linear as the library grows, facet/contract intersections keep the inspected candidate set small. Semantic-gate deduplication reduces state without false merges, and scoped composition dramatically reduces combinatorial search.

The largest remaining uncertainty is not retrieval, but whether the model can perform correct decomposition/ranking and acquire a new capability in the narrowed search space. The highest-information next step for Experiment 0014 is therefore to run one generic frozen SmolLM3 ranking-plus-acquisition protocol over 3–5 new task families on the same canonical/faceted index, measuring model calls, input tokens, wrong activation, timeouts, and held-out regression together with the Experiment 0013 telemetry.
