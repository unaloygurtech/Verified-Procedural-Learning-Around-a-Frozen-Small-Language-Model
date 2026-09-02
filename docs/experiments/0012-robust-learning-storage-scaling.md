# Experiment 0012 — Robust Learning, Storage, and Scaling

Experiment 0012 is a bounded systems experiment, not a claim of general
autonomous learning. It keeps `SmolLM3-3B-GGUF-Q4`, llama.cpp, the 4096-token
context, and the frozen 0009 learner unchanged.

Authoritative runtime report (ignored by Git):

`data/runs/air-012-20260831T212714Z.json`

The long run used SQLite checkpoints after a model call timed out. Completed
retrieval/proposal responses were reused verbatim. A runtime timeout is stored
as a failed attempt and is not retried for the same family. No prompt, task,
gate, document, seed, or expected result was changed after observing outcomes.

## Block A — Multi-API robustness

Five opaque operations were generated for each of three deterministic seeds:
string shards, integer-list processing, JSON-object processing, mixed
string/numeric processing, and run-length encoding. Each API/seed pair has
disjoint 4 public, 3 validation, 3 edge, and 8 held-out cases. The same 15
API/seed pairs were tested with 10, 50, and 100-document pools (45 runs).

The learner prompt remained `air-009-generic-learner-v1`, SHA-256
`d6b6c4da5226e826343de0bdf3864ba00a95f34957443c6f0c3c8f04b72c2833`.
The retriever reused `air-011-document-retriever-v1`, SHA-256
`38feee0a77605938e629c71ec477f46f5154589072c183488ce040617722b17e`.

| Documents | Correct retrieval | Related wrong | Hallucinated ID | Activated | Wrong activation | Active held-out | Proposals / repairs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 15/15 (100%) | 0 | 0 | 1/15 | 0 | 8/8 | 45 / 30 |
| 50 | 6/15 (40%) | 1 | 6 | 1/15 | 0 | 8/8 | 45 / 30 |
| 100 | 0/15 (0%) | 2 | 1 | 0/15 | 0 | none | 25 / 10 |

Overall retrieval was 21/45; activation was 2/45, or 2/21 after a correct
retrieval. Only one of the 15 unique API/seed pairs activated in at least one
repeat (twice under seed 1202). Both activated artifacts solved every held-out
case, 16/16 total. No wrong retrieval became a wrong activation. Unsafe imports
and semantic identity candidates were rejected, prior 0008 regression remained
1.0, and source/base artifacts remained immutable.

This is a negative robustness result. At 10 documents retrieval was perfect,
but the generic learner usually tried to reimplement the documented semantics
instead of emitting the minimal opaque API wrapper. Those programs then used
calls or attributes outside the allowlist and exhausted the frozen repair
budget. Acquisition, not only retrieval, is a bottleneck.

### Context sustainability and output quality

Retrieval pools are sent to the model and therefore consume active context.
The skill library/index itself is not sent to the model.

| Documents | Successful calls | Mean prompt tokens | Max prompt tokens | Max of 4096 context | Mean successful retrieval | Timeouts | Retrieval quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 15/15 | 484 | 497 | 12.1% | 7.50 s | 0 | 100% |
| 50 | 15/15 | 1,549 | 1,569 | 38.3% | 24.94 s | 0 | 40% |
| 100 | 7/15 | 2,881 | 2,908 | 71.0% | 73.41 s | 8 | 0% |

The 100-document mean is calculated only over calls with token telemetry;
timed-out calls reported no token count. Therefore `0 tokens` on those rows
does **not** mean zero context use. At 100 documents, context and CPU latency
became operationally unsustainable before reaching the 4096 hard limit. More
context also reduced retrieval output quality rather than improving it.

## Block B — Learned-state storage engine v0

Storage is separated into:

- HOT: identity, typed contract, category, trust/status, version, tiny
  retrieval descriptor, and executable pointer/bytes.
- WARM: lexical terms, usage/success statistics, relations, ranking metadata,
  and operation family.
- COLD: raw experiences, documentation, full provenance, validation history,
  edge tests, and readable source.

COLD is not loaded during a normal query. For the binary representation the
measured HOT/WARM/COLD sizes were 475/157/1,616 bytes; active HOT+WARM was 632
bytes.

AIR IR v1 has six typed opcodes: `PARSE_TOKEN`, `LOOKUP_INT`, `PARITY_INT`,
`MUL_INT`, `TO_STR`, and `RETURN`. It is deterministic and model-independent,
but deliberately not general Python. Unsupported Python remains readable
Python. Derived artifacts preserve source ID/version/hash, compiler/IR version,
equivalence test IDs, activation, timestamp, and source immutability.

| Representation | Bytes | Load p50 | Execute p50 | Equivalence | vs Python bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Readable Python | 272 | 140.8 µs | 1.1 µs | 18/18 | 1.00× |
| JSON typed AST | 770 | 123.5 µs | 34.9 µs | 18/18 | 2.83× |
| Compact JSON AIR IR | 294 | 99.5 µs | 30.2 µs | 18/18 | 1.08× |
| Packed binary AIR IR | 192 | 110.1 µs | 33.7 µs | 18/18 | 0.71× |

Storage is not semantically broken: every representation passed all 18 cases
and deterministic round-trip tests. Malformed binary, unknown opcode, wrong
version, type-invalid operands, and a structurally valid semantic-wrong IR were
all rejected.

However, smaller did not mean faster. Packed binary saved 29.4% versus source
and 75.1% versus JSON AST, but IR execution was about 30 times slower than the
trusted in-process Python function for this tiny rule. Compact JSON IR was
larger than Python. Metadata/index bytes (about 440 bytes per skill) also exceed
the 192-byte binary artifact, limiting the system-level value of code-byte
compression. AIR IR currently earns its place through explicit safety and
portability, not speed.

## Block C — Skill-library scaling

Definitions:

- SLS is all persistent HOT+WARM+COLD learned-state bytes, excluding SQLite
  page overhead.
- ALS is only returned HOT+WARM metadata and executable bytes loaded for a
  query.
- Bytes read/query is a logical serialized-byte measurement plus a B-tree page
  traversal proxy, not physical disk telemetry.

| Skills | SLS | SQLite index | ALS/query | Indexed bytes/query | Indexed p50 | Naive p50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 5,115 B | 12,288 B | 408 B | 4,504 B | 91.7 µs | 2.6 µs |
| 100 | 51,555 B | 20,480 B | 415 B | 4,511 B | 71.7 µs | 8.6 µs |
| 1,000 | 515,679 B | 155,648 B | 415 B | 8,607 B | 78.6 µs | 132.7 µs |
| 10,000 | 5,156,974 B | 1,527,808 B | 415 B | 8,607 B | 52.1 µs | 820.9 µs |
| 100,000 | 51,569,967 B | 15,192,064 B | 415 B | 12,703 B | 46.6 µs | 9,816.8 µs |

The correct 100,000th artifact was retrieved. SLS grew about 10× at each scale
step while ALS stayed approximately 415 bytes; the 100k SLS/ALS ratio was
124,265×. Indexed retrieval is overhead at 10 and 100 skills, crosses over near
1,000, and is about 210× faster than the naive scan at 100k. The complete
library and COLD state were never loaded into model context.

Composition remains a separate combinatorial problem. At 100k skills, depth-2
brute force is 10 billion candidates and depth-3 is 1 quadrillion. Type
filtering leaves 2.5 billion and 62.5 trillion respectively. The synthetic
category distribution reduced both to zero because no within-category
contracts composed; this is a dataset-specific control, not evidence that
category filtering solves general composition.

## Block D — Efficiency control

The generated `Talven-12` rule has SHA-256
`d78770fc1aa2c580258e4bd96bbeb03f1df0a68fa462d7eb5fa5d9511031f5f6`.
The acquisition call timed out and no Python artifact activated. All eight raw
context calls also hit the 180-second runtime timeout. Consequently Python,
JSON AST, compact IR, and binary IR conditions had no learned source to run.
This block is **inconclusive**, not evidence that all representations have zero
quality or zero context cost. It is retained as the observed runtime failure.

Block B proves representation equivalence for a canonical verified source;
Block D does not prove end-to-end model-acquired source → IR conversion under
this run's degraded model runtime.

## Answers and 0013 recommendation

The system differs from a normal text memory layer by storing gated executable state,
typed contracts, provenance, trust, and deterministic verifiers. Readable
source remains in COLD; compact executable state can be HOT. The index/storage
path scales well enough to 100k, while model-facing documentation retrieval and
the frozen learner are the bottlenecks. Disk/index is not currently the main
problem; acquisition and composition are.

Python plus SQLite is sufficient for current performance. AIR IR is justified
only where fail-closed execution and model-independent portability matter; it
should not replace all Python artifacts merely to save bytes. External
executable state is a meaningful procedural system-state change, but it is not
a model-parameter update.

0013 should perform the proposed brain-transplant test with a fixed set of
already validated Python/IR artifacts, the same SQLite index, and the same
verifiers. Swap only the model, use zero retraining, and compare skill reuse,
wrong activation, regression, model calls, and portability failures. This is
the direct falsification test for model-independent learned state. It should
not claim success from IR execution alone: model-independent acquisition and
selection remain untested, and 0012 shows they are the fragile parts.
