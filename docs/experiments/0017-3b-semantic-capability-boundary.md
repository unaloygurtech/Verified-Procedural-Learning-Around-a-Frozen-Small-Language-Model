# Experiment 0017 — 3B Semantic Capability Boundary

Experiment 0017 was designed from the result of Experiment 0016. Although retrieval was 8/8 and structural metadata accuracy in Arms B/C/D was 100%, complete contracts remained 0/8 and oracle downstream semantic-body success remained 0/8. Experiment 0017 therefore adds no new memory, planning, or agent layer; it measures where SmolLM3-3B fails under the narrowest executable semantic representation.

## Frozen conditions

- `ggml-org/SmolLM3-3B-GGUF:Q4_K_M`, llama.cpp, CPU, context 4096;
- model weights frozen; no model swap, 7B/14B/Qwen model, LoRA, or fine-tuning;
- retrieval, canonical learned state, facet/fingerprint indexes, sandbox, verifier, provenance, and earlier safety gates frozen;
- eight opaque families: five from seed 1201 and three from independent seed 1202;
- four public/discovery, three hidden, three edge, and eight held-out examples per family;
- no family-specific prompt, hidden leakage, answer-specific opcode, silent retry, or relaxed validation.

The existing Experiment 0012 AIR-IR was not force-expanded to this task class. Instead, a frozen `AIR-SEMANTIC-IR` v1 was defined for Experiment 0017:

```text
INPUT, INT, CALL, REVERSE, ROTATE, CONCAT, RETURN
```

The IR is typed, schema-validated, deterministic, and fail-closed. Unknown opcodes, invalid operands, incorrect APIs, and `RETURN` outside the root are rejected. The system deterministically generates the module, callable, wrapper, imports, serialization, sandbox, and provenance code; the model generates only the semantic representation.

The minimum semantic-plan policy was frozen before the benchmark:

```text
required: operation, arguments
optional: ordering, return
```

The `preconditions`, `postconditions`, `semantic_invariants`, `special_cases`, and `failure_behavior` fields from Experiment 0016 are measured as unnecessarily demanding for this task class; validation is not relaxed.

## Diagnostic ladder

| Arm | Model input | Model output |
|---|---|---|
| A — oracle Python body | Correct oracle contract | Semantic Python body |
| B — oracle → IR | Correct oracle contract | Minimal typed IR |
| C — docs → IR | Retrieved docs, public examples, deterministic structural metadata | Minimal typed IR |
| D — docs → plan → IR | Retrieved docs and public examples | Minimum semantic plan, then IR |
| E — oracle plan → IR | Correct minimum semantic plan | Minimal typed IR |
| F — candidate selection | Correct oracle plan + five frozen candidates | `candidate_id` |
| G — oracle compiler | No model; ground-truth IR | Deterministic executable artifact |

The version and SHA-256 values of Arms A–F prompts were written to the runtime JSON before the first real call. Arm G is an infrastructure upper-bound control only.

## Frozen runtime result (2026-09-01)

Raw runtime report:
`data/runs/air-017-20260901T163634Z.json`.

The run used **56 model calls**, **17,026 input tokens**, **2,782 output tokens**, and **0 timeouts**. Overall model latency was p50 **3.73 s** and p95 **9.54 s**.

| Measurement | A Python body | B oracle → IR | C docs → IR | D docs → plan → IR | E oracle plan → IR | F selection | G compiler |
|---|---:|---:|---:|---:|---:|---:|---:|
| Valid output | 8/8 | 7/8 | 8/8 | 0/8 plan | 8/8 | 8/8 | 8/8 |
| Valid IR | — | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 8/8 |
| Semantic correctness | — | 0/8 | 0/8 | — | 0/8 | 3/8 | 8/8 |
| Public pass | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 3/8 | 8/8 |
| Hidden pass | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 3/8 | 8/8 |
| Edge pass | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 3/8 | 8/8 |
| ACTIVE | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 3/8 | 8/8 |
| `activation_given_correct_retrieval` | 0% | 0% | 0% | 0% | 0% | 37.5% | 100% |

Retrieval remained **8/8 correct** for every arm. In Arm F, the correct candidate was selected from five frozen typed candidates **3/8** times. Those three artifacts achieved **3/3 family and 100% held-out accuracy**; the artifact path used **0 model calls** and was measured as external procedural-state reuse. Wrong activation was **0** and there was no regression in the canonical library.

The oracle compiler passed schema, type, compilation, public, hidden, edge, and held-out gates for every family. The IR executor/compiler infrastructure therefore works; model-capacity evaluation was not blocked by an infrastructure failure.

## Capacity diagnosis matrix

| Stage | Result |
|---|---:|
| Oracle Python body | 0/8 ACTIVE; eight bodies were produced, but most failed at static/safety or public validation |
| Oracle → minimal IR | 0/8 valid IR |
| Docs → minimal IR | 0/8 valid IR |
| Docs → semantic plan | 0/8 valid plans |
| Semantic plan → IR | 0/8 (no plan was produced) |
| Oracle plan → IR | 0/8 valid IR |
| Candidate selection | 3/8 correct (**37.5%**) |
| Oracle compiler | 8/8 (**100%**) |

Model-generated IR-like outputs usually deviated from the canonical schema: the format/version changed, Python-like text was returned instead of expressions, wrapper metadata was moved into the IR, or real example inputs were placed in plan arguments. These were recorded as `ir_schema_failure`; no later repair or normalization was applied to obtain the correct result.

## Failure taxonomy

```text
ir_schema_failure            23
ir_generation_failure         9
semantic_plan_failure         8
python_body_failure           6
semantic_induction_failure    5
candidate_selection_failure   5
public_validation_failure     6
safety_rejection              1
retrieval_failure             0
hidden_validation_failure     0
edge_failure                  0
duplicate_candidate           0
timeout                       0
heldout_failure               0
```

The largest and decisive bucket was **IR schema/output-following failure**. Semantic candidate selection produced a limited recognition signal at 3/8, above chance in this bounded setting; however, the model could not convert even the correct oracle plan into canonical IR. The docs→IR and docs→plan arms produced no usable artifact for novel semantic induction.

## Minimum-contract result

A minimum plan carrying only `operation + arguments` was a sufficient deterministic rule. Nevertheless, docs→plan remained **0/8**. This shows that the broad contract schema from Experiment 0016 was not the only or decisive obstacle: it may have increased the previous failure, but even the narrowest plan and IR interface exceeded the model’s output-following boundary.

## Direct answers

1. **Was Python/full-program generation the main problem?**  
   **Partly, but not by itself.** The Python-body arm failed under syntax and safety constraints; oracle→IR also remained 0/8.

2. **Was the large contract schema an unnecessary obstacle?**  
   **Partly, but it is not a sufficient explanation.** Structural contract extraction was a bottleneck in Experiment 0016; the minimal IR in Experiment 0017 also could not be produced canonically.

3. **Did minimal semantic IR improve acquisition?**  
   **Not for direct generation.** A partial benefit appeared in candidate selection, but docs→IR and oracle-plan→IR were both 0/8.

4. **Can the model recognize correct semantic behavior among candidates?**  
   **Partly: 3/8 (37.5%).** The three correct selections passed hidden, edge, and held-out tests.

5. **Can the model convert an oracle semantic plan into executable IR?**  
   **No, 0/8 under this frozen interface.**

6. **Can the model derive a new semantic rule from documentation?**  
   **No under the strict executable gate: 0/8.** Neither docs→plan nor docs→IR produced a usable result.

7. **Where is the SmolLM3-3B failure boundary?**  
   **Not in the deterministic compiler; it is in the model’s canonical structured-output and IR-following stage.** Limited recognition exists, but reliable representation generation does not.

8. **Is the 3B plus system hypothesis still technically plausible?**  
   **Partially.** The oracle compiler achieved 100% and candidate selection 37.5%, so a search-plus-rank-plus-verify architecture remains technically plausible; the direct model-synthesis assumption was not supported.

9. **Is reaching a robust experimental system with the current 3B model reasonable?**  
   **Not with the current direct-acquisition design.** A narrow candidate-search/retrieval assistant role remains reasonable; the evidence is insufficient for direct, reliable acquisition of reusable capabilities from documentation.

This experiment does **not** declare Level 3. External procedural-state reuse was demonstrated by the three successful selections in Arm F, but no claim of multi-stage planning, general continual learning, or AGI can be inferred.

## Safety, regression, and tests

- wrong activation: 0;
- false canonical merge: 0/6;
- hidden leakage: none;
- invalid candidate ID: rejected;
- grammar, generator, sandbox, and verifier unchanged;
- no silent retry;
- Docker full suite: **138 tests, all passed**.

## Reproduction and resume

```bash
docker compose exec air-core python -m air_core.cli experiment-017
docker compose exec air-core python -m air_core.cli experiment-017 --resume /workspace/data/runs/air-017-<timestamp>.json
```

Runtime JSON checkpoints are excluded from Git; source, tests, and this report are committed.
