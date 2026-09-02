# Experiment 0010: Novel Synthetic API Learning

Date: 2026-08-31

## Question

Can AIR acquire an executable procedure for an API created during the
experiment, whose exact namespace and semantics could not have been in the
model's pretraining data?

The repository creates the local package `zorvik_010` with four opaque
operations: `kel`, `nam`, `tesh`, and `vum`. Their deterministic semantics are
not exposed to the model except through the family documentation and public
tests. The package is imported inside the candidate sandbox; it is not an
existing internet package.

## Frozen protocol

The 0009 generic learner is reused unchanged at the template level:

- prompt version: `air-009-generic-learner-v1`
- prompt SHA-256: `d6b6c4da5226e826343de0bdf3864ba00a95f34957443c6f0c3c8f04b72c2833`
- one generic template; no family-specific prompt patches
- maximum three proposals per primitive
- four discovery, three hidden validation, three edge, and eight held-out
  cases per primitive

Only the family data changes: operation signature, short behavior description,
and examples. The learning protocol itself is frozen. The model/runtime stays
`SmolLM3-3B-GGUF-Q4 via llama.cpp` in the existing Docker Compose isolation.

## Package and sandbox

`zorvik_010` is available only through the sandbox import path. Candidate code
may import that package and the one documented operation for its family; other
imports, dangerous names, indirect calls, and unsupported attributes are
rejected by the AST gate. Execution uses `python -I`, a sanitized environment,
temporary cwd, and a two-second timeout. The package source hash is recorded
before and after the run.

The semantic oracle calls the package directly in the harness. It is not put in
the model-facing prompt. A deliberately identity candidate is type/runtime
valid but semantically wrong; an `os` candidate is explicitly unsafe.

## Zero-knowledge contamination control

Before documentation or examples are supplied, the model is asked to answer
public inputs under two controls: no API semantics at all, and API names/signatures
only. In the full run, model-only scored 3/16 and names-only 0/16. This is not a
meaningful semantic signal (threshold 50%), so the synthetic API was retained;
the report marks `contamination_flag: false`.

## Results

Authoritative full report:
`data/runs/air-010-20260831T175020Z.json`

| Primitive | Active | Discovery / validation / edge | Learned held-out | Docs-only / docs+raw |
| --- | ---: | ---: | ---: | ---: |
| `kel` | no | 0/4 · 0/3 · 0/3 | 0/8 | 3/8 · 2/8 |
| `nam` | yes | 4/4 · 3/3 · 3/3 | 8/8 | 3/8 · 2/8 |
| `tesh` | yes | 4/4 · 3/3 · 3/3 | 8/8 | 2/8 · 4/8 |
| `vum` | yes | 4/4 · 3/3 · 3/3 | 8/8 | 1/8 · 1/8 |

AIR detected a gap for all four primitives (4/4) and activated 3/4 skills.
The three activated artifacts are thin executable wrappers around the newly
created operations, carry documentation/public/validation/edge provenance, and
reused successfully on new literals. The `kel` learner exhausted its repair
budget on malformed/missing JSON code and remained inactive; this failure was
not patched away.

The pre-existing library scored 0/32 on the four family held-outs. The direct
documentation/raw-example baselines did not reproduce the learned artifact:
they reached at most 4/8 for a family, while each active artifact reached 8/8.
An unsafe `os` proposal was rejected statically, and the semantic identity
candidate scored 0/3 on hidden validation. No wrong pre-learning activation or
cross-family help occurred. The synthetic package hash was unchanged and all
old artifacts remained immutable.

## Composition

The composition search is behavior-based over the independently learned
artifacts; no mapping is hard-coded and no composed example appears in a
primitive prompt. The intended hidden task is `tesh(kel(value))`. Because
`kel` failed to activate in the full run, there was no valid pair and the
composition result is explicitly `no_valid_composition`. This is a negative
result, not a forced success.

## Interpretation

The strongest supported claim is narrower than “AIR learned Python”:

> With zero-knowledge controls near chance, AIR used experiment-time
> documentation and public evidence to create and gate three reusable
> executable wrappers for a previously nonexistent API namespace.

This is evidence for bounded test-time acquisition, not proof that the model
learned arbitrary Python or that contamination is mathematically impossible.
The package itself is simple, the API is supplied by documentation, and one of
four primitive learners failed. The next high-information experiment is to
keep the prompt hash fixed while adding documentation retrieval: require AIR
to locate the relevant synthetic API note from a small noisy document set, then
repeat the same hidden/held-out and composition gates.
