# Experiment 0011 — Documentation Retrieval and Learning Efficiency

Experiment 0011 contains two independent blocks. The complete run was
executed in the AIR Docker container with
`SmolLM3-3B-GGUF-Q4 via llama.cpp` on 2026-08-31.

Runtime report (ignored by Git):

`data/runs/air-011-20260831T181537Z.json`

## Block A — Documentation retrieval

The target is the four-operation opaque `zorvik_010` package introduced by
Experiment 0010. Its operation semantics are hidden from the model in a small
pool of eight records: four normative manuals and four distractors (including
an unrelated note and a historical wrong draft). The retrieval prompt is
frozen as `air-011-document-retriever-v1` with SHA-256
`38feee0a77605938e629c71ec477f46f5154589072c183488ce040617722b17e`.
The learner is exactly the frozen 0009 generic protocol, SHA-256
`d6b6c4da5226e826343de0bdf3864ba00a95f34957443c6f0c3c8f04b72c2833`; no
family-specific learner patch is used.

Results across four families:

| measure | result |
| --- | ---: |
| correct documentation retrieval | 3/4 (75%) |
| related/wrong retrievals | 1 |
| capability gaps detected | 4/4 |
| skills activated after all gates | 3/4 |
| activated-skill held-out reuse | 3/3 families at 8/8 (100%) |
| wrong activation | 0 |
| hidden-validation failures among accepted skills | 0 |
| base-skill immutability | true for every family |
| prior 0008 regression | 1.0 accuracy |

The three correctly retrieved manuals led to executable skills that passed
public discovery, hidden validation, and edge gates, then solved all held-out
cases without another model call. The `vum` query selected hallucinated
`manual-vum` rather than the expected `manual-saffron`; its candidate failed
the allowlist/public gate and was not activated. An unsafe `os` import and a
semantic identity candidate were both rejected. This is evidence for the
full retrieval-to-reuse chain on 3/4 families, not evidence that retrieval or
arbitrary Python learning is solved.

## Block B — Novel rule learning and efficiency

At run time a deterministic `Neralis-11` rule system was generated. Its rule
specification hash is
`c8f08e5b3daafd299e92d34d17afe6412034e2d1f406235d75e1f7c3edff2895` and its
documentation hash is
`ef0c97d7d6b3d70250e9c214626c56428531318b18ec6deb8ed1e387e6a56ab2`.
Training/discovery, validation, edge, and held-out inputs are disjoint.

| condition | accuracy | mean latency | p50 / p95 latency | input tokens | output tokens | model calls | attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model only (zero knowledge) | 0/8 (0%) | 970.0 ms | 722.9 / 2245.1 ms | 640 | 135 | 8 | 8 |
| model + full rule document | 1/8 (12.5%) | 2030.8 ms | 2043.1 / 2124.1 ms | 1392 | 89 | 8 | 8 |
| AIR learned artifact | 8/8 (100%) | 18.4 ms | 18.4 / 20.3 ms | 0 | 0 | 0 | 8 |

The generic learner detected a gap, produced one deterministic Python
artifact, and passed discovery (4/4), validation (3/3), and edge (3/3)
without repair. The artifact was then reused on all eight disjoint held-out
inputs. Its 53-token code footprint is the active artifact context; execution
itself used no model prompt. Thus the artifact had a large active-cost
advantage over both model baselines in this run, while remaining external
state and not a model-parameter update.

The unknown token `Z7` was rejected, the plausible semantic-wrong candidate
failed validation, and the unsafe `os` candidate failed the static gate. The
model-only and raw-context scores are low enough that contamination/guessing
was not a positive explanation here; this control is still only one model and
one generated rule family.

## Interpretation and next question

0011 demonstrates a bounded documentation-retrieval → executable-skill →
held-out-reuse path, with a real retrieval failure preserved rather than
repaired after looking at the result. It also demonstrates that a validated
external artifact can reduce repeated active computation on a newly generated
rule. It does **not** yet show that AIR invents missing primitives, performs
arbitrary program synthesis, or scales beyond this narrow sandbox.

The highest-information 0012 question is whether the same frozen retriever and
learner protocol generalize to several newly generated APIs with opaque
operation names, larger/noisier documentation pools, and independent seeds,
while retaining safe rejection and old-skill regression guarantees. That
separates a repeatable learning mechanism from a one-run/document-layout
effect.
