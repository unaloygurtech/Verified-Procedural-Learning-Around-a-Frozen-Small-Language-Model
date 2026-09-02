# Experiment 0008: Python/API Capability Gap

Date: 2026-08-31

## Research question

Can AIR carry the 0007 capability-gap loop into a real programming language?
The target is a small but nontrivial standard-library procedure, not a
question-answering test:

```text
query string -> parse query pairs -> sort key/value -> canonical encode
```

The learned artifact is Python source executed in a sandbox. This is the first
AIR experiment in which the search space is Python programs rather than only
our typed DSL.

## Isolation and task

- The run uses the existing AIR Docker Compose environment and SmolLM3-3B
  runtime. No external memory files or mounts are available.
- Candidate code may use only `urllib.parse.parse_qsl` and `urlencode`, plus
  list sorting and ordinary expressions.
- A pre-existing unrelated identity artifact is tested first and fails the
  public discovery cases, so AIR must report `gap_detected`.
- The model sees a short API document and four public discovery tests. The
  independent validation, edge, and 12 held-out queries are not supplied as
  training context.
- No external package or network request is required.

The reference evaluator uses `parse_qsl(query, keep_blank_values=True)`, sorts
pairs by `(key, value)`, then calls `urlencode(pairs, doseq=True)`. This oracle
is in the harness, not in the model-facing prompt.

## Learning loop

```text
existing Python library
  -> public tests fail
  -> gap_detected
  -> model proposal using limited docs
  -> static safety check
  -> sandbox execution on public tests
  -> repair prompt when public tests fail (maximum 3 attempts)
  -> hidden validation + edge gate
  -> immutable Python skill artifact
  -> held-out reuse
```

The candidate must define exactly `transform(query: str) -> str`. The static
checker rejects non-`urllib.parse` imports, dangerous names/calls, indirect
calls, and top-level executable statements. Execution uses `python -I`, a
sanitized environment, a temporary working directory, and a two-second timeout.
This is defense-in-depth inside an already isolated container, not a claim of a
kernel-level sandbox.

## Safety and semantic gates

The code allowlist deliberately permits a type/runtime-valid corrupted
procedure that parses and encodes without sorting. It should run successfully
but fail semantic validation. A candidate that imports `os` is rejected before
execution. A model proposal that fails public tests is repaired; a proposal
that passes public tests but fails hidden validation is not silently activated.

The accepted artifact carries a version, input/output contract, source case IDs,
and the exact Python code. The unrelated starting artifact is snapshotted to
verify immutability.

## Baselines

Held-out conditions are:

1. model only: direct canonical-query answer with no API document,
2. model plus limited API document: direct answer without a stored artifact,
3. existing Python library before learning: safe failure/no active skill,
4. learned Python skill: accepted source executed in the sandbox.

The report records prompt/generation tokens for model conditions and whether
documentation or artifact reuse was involved. A flat code baseline is not used
as the primary comparison here because the model-generated Python artifact is
itself the object under test; the corrupted source is the semantic control.

## Results

Run the benchmark with:

```bash
docker compose exec air-core python -m air_core.cli experiment-008
```

The authoritative runtime report is written under `data/runs/` and is ignored
by Git. The table is filled from the final isolated container run.

| Condition | Positive held-out | Safe/valid behavior |
| --- | ---: | --- |
| Model only | 0/12 (0%) | direct generation did not solve the contract |
| Model + limited docs | 2/12 (16.7%) | documentation helped, but did not make a reusable procedure |
| Existing library before learning | 0/12 (0%) | `gap_detected`; the unrelated identity artifact was safely inactive |
| Learned Python skill | 12/12 (100%) | accepted artifact reused in the sandbox on all new literals |

Final report: `data/runs/air-008-20260831T165636Z.json`.

The learning gates were all positive: discovery 4/4, hidden validation 4/4,
edge cases 3/3, and the deliberately unsorted corrupted candidate 0/4. The
starting artifact remained byte-for-byte unchanged. A first full run exposed a
real repair weakness (the model omitted imports and repeated the candidate);
the prompt was tightened to require a self-contained program and the rerun
then passed on its first proposal. This stochastic observation is retained as
part of the experiment record rather than treated as a guaranteed property.

The strong-success gate requires: gap detection before learning; a statically
safe candidate; public discovery tests passed after at most three attempts;
hidden validation and edge tests passed; corrupted code rejected semantically;
and the pre-existing artifact unchanged.

## Interpretation and next step

This run shows that AIR can turn a verified code-generation loop into a
reusable executable procedure outside its hand-authored DSL for one bounded
standard-library task. It does not establish arbitrary Python competence,
external documentation search, networked tool use, or long-term skill scaling.

If the model cannot produce a safe passing candidate, preserve that failure and
separate whether the bottleneck is API comprehension, repair, static allowlist,
or execution. If it passes, the next experiment should add a second API task
family and measure transfer without increasing the prompt with task-specific
recipes.
