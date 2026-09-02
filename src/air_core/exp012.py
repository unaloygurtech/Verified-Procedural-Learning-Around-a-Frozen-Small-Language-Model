"""Experiment 0012: robust acquisition, compact learned state, and scaling."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import random
import statistics
import tempfile
import time
from typing import Any, Callable

import air_synth_012

from .air_ir import (
    AIRProgram,
    COMPILER_VERSION,
    IR_VERSION,
    IRExecutionError,
    IRValidationError,
    Instruction,
    build_rule_program,
    compile_python_rule_subset,
    deserialize_binary_ir,
    deserialize_compact_ir,
    deserialize_json_ast,
    execute_program,
    semantic_equivalence,
    serialize_binary_ir,
    serialize_compact_ir,
    serialize_json_ast,
    slot,
    source_sha256,
    validate_program,
)
from .exp008 import HELD_OUT_008, run_python_gate
from .exp009 import (
    LEARNING_PROMPT_HASH_009,
    LEARNING_PROMPT_VERSION_009,
    FamilyCase009,
    PythonFamily009,
    PythonGate009,
    PythonSkillArtifact009,
    RepairAttempt009,
    SandboxResult009,
    StaticCheck009,
    _extract_code,
    generic_learning_prompt,
    run_python_gate_009,
    run_python_in_sandbox_009,
    static_check_python_009,
)
from .exp010 import BASE_PYTHON_LIBRARY_010, MODEL_IDENTITY_010
from .exp011 import RETRIEVAL_PROMPT_HASH_011, RETRIEVAL_PROMPT_TEMPLATE_011, RETRIEVAL_PROMPT_VERSION_011
from .learned_state import (
    ColdState,
    DerivedArtifactProvenance,
    HotState,
    LayeredSkillState,
    SQLiteSkillIndex,
    WarmState,
    benchmark_naive_retrieval,
    canonical_json_bytes,
    composition_candidate_counts,
    deep_size,
    generate_skill_records,
    query_for_record,
)
from .model_client import LlamaCppClient, ModelUnavailable
from .neralis import parse_response
from .store import ExperimentStore


POOL_SIZES_012 = (10, 50, 100)
ROBUSTNESS_SEEDS_012 = air_synth_012.SEEDS_012
SYNTHETIC_MODULE_012 = "air_synth_012"
SYNTHETIC_IMPORT_ROOT_012 = str(Path(__file__).resolve().parents[1])
SCALE_POINTS_012 = (10, 100, 1_000, 10_000, 100_000)
STORAGE_BENCHMARK_REPEATS_012 = 300


def _percentiles_us(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    if not ordered:
        return {"mean_us": 0.0, "p50_us": 0.0, "p95_us": 0.0}
    p95_index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "mean_us": statistics.mean(ordered) * 1_000_000,
        "p50_us": statistics.median(ordered) * 1_000_000,
        "p95_us": ordered[p95_index] * 1_000_000,
    }


def _measure(call: Callable[[], Any], repeats: int = STORAGE_BENCHMARK_REPEATS_012) -> tuple[dict[str, float], Any]:
    samples: list[float] = []
    result: Any = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - started)
    return _percentiles_us(samples), result


def _case_tuple(seed: int, kind: str, split: str, values: tuple[str, ...]) -> tuple[FamilyCase009, ...]:
    return tuple(
        FamilyCase009(
            f"air012-{seed}-{kind}-{split}-{index:02d}",
            value,
            air_synth_012.apply_operation(seed, kind, value),
            split,
        )
        for index, value in enumerate(values, 1)
    )


def _generated_literals(seed: int, kind: str) -> tuple[str, ...]:
    rng = random.Random(seed * 101 + air_synth_012.FAMILY_KINDS_012.index(kind))
    delimiter = ("|", "~", "^")[air_synth_012.SEEDS_012.index(seed)]
    values: list[str] = []
    while len(values) < 18:
        if kind == "shards":
            parts = ["".join(rng.choice("abcdefxyz") for _ in range(rng.randint(1, 5))) for _ in range(rng.randint(2, 4))]
            value = delimiter.join(parts)
        elif kind == "numbers":
            value = ",".join(str(rng.randint(-15, 35)) for _ in range(rng.randint(3, 6)))
        elif kind == "object":
            keys = rng.sample(tuple("abcdefg"), rng.randint(2, 5))
            payload = {key: rng.randint(0, 30) if index % 2 == 0 else "".join(rng.choice("pqrs") for _ in range(3)) for index, key in enumerate(keys)}
            value = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        elif kind == "mixed":
            text = "".join(rng.choice("abcdefghjkmn") for _ in range(rng.randint(4, 8)))
            value = f"{text}#{rng.randint(-8, 24)}"
        else:
            chunks = [rng.choice("abcd") * rng.randint(1, 4) for _ in range(rng.randint(3, 6))]
            value = "".join(chunks)
        if value not in values:
            values.append(value)
    return tuple(values)


def _operation_docs(seed: int, kind: str, operation: str) -> str:
    variant = air_synth_012.SEEDS_012.index(seed)
    details = {
        "shards": f"Split on {(('|', '~', '^')[variant])!r}, reverse shard order, reverse and uppercase every shard, then join with the same delimiter.",
        "numbers": f"Parse comma-separated integers, map each n to n * {(3, 5, 7)[variant]} + {(1, 2, 4)[variant]}, deduplicate, sort descending, and join with semicolons.",
        "object": f"Parse a JSON object, order keys {'descending' if variant % 2 else 'ascending'}, and join key=value fields with vertical bars.",
        "mixed": f"Split at the final #, rotate the text left by {variant + 1}, reverse and uppercase it, and append ':' plus n * {variant + 2} + {variant + 3}.",
        "runs": f"Run-length encode consecutive characters as character+count and join groups with {(('/', ':', '.')[variant])!r}.",
    }[kind]
    return (
        f"Normative manual for the current API. Use `from {SYNTHETIC_MODULE_012} import {operation}`. "
        f"Signature: `{operation}(value: str) -> str`. {details}"
    )


def make_robustness_families_012(seed: int) -> tuple[PythonFamily009, ...]:
    names = air_synth_012.operation_names(seed)
    families: list[PythonFamily009] = []
    for kind in air_synth_012.FAMILY_KINDS_012:
        operation = names[kind]
        values = _generated_literals(seed, kind)
        families.append(
            PythonFamily009(
                family_id=f"air012-{seed}-{kind}-{operation}",
                title=f"Opaque operation {operation}",
                api_docs=_operation_docs(seed, kind, operation),
                contract="Define transform(value: str) -> str for the requested opaque operation. The selected normative manual is the only API authority.",
                allowed_imports=frozenset({SYNTHETIC_MODULE_012}),
                allowed_import_members=frozenset({operation}),
                allowed_call_names=frozenset({operation}),
                allowed_attrs=frozenset({operation}),
                discovery=_case_tuple(seed, kind, "discovery", values[:4]),
                validation=_case_tuple(seed, kind, "validation", values[4:7]),
                edge=_case_tuple(seed, kind, "edge", values[7:10]),
                heldout=_case_tuple(seed, kind, "heldout", values[10:18]),
                sandbox_import_root=SYNTHETIC_IMPORT_ROOT_012,
            )
        )
    return tuple(families)


def _doc_id(seed: int, family_id: str, label: str) -> str:
    digest = hashlib.sha256(f"{seed}:{family_id}:{label}".encode()).hexdigest()[:12]
    return f"record-{digest}"


def make_document_pool_012(family: PythonFamily009, pool_size: int, seed: int) -> tuple[tuple[dict[str, str], ...], str]:
    if pool_size not in POOL_SIZES_012:
        raise ValueError("unsupported document pool size")
    operation = family.family_id.rsplit("-", 1)[-1]
    correct_id = _doc_id(seed, family.family_id, "normative")
    records = [
        {"doc_id": correct_id, "kind": "correct", "text": family.api_docs},
        {
            "doc_id": _doc_id(seed, family.family_id, "historical"),
            "kind": "related_wrong",
            "text": f"Historical non-normative draft for {operation}. It used legacy_012.{operation}_old and must not be used for current API code.",
        },
        {
            "doc_id": _doc_id(seed, family.family_id, "terminology-a"),
            "kind": "terminology_distractor",
            "text": f"Current deployment note mentions {SYNTHETIC_MODULE_012} and {operation}, but contains no callable signature or operation semantics.",
        },
        {
            "doc_id": _doc_id(seed, family.family_id, "terminology-b"),
            "kind": "terminology_distractor",
            "text": f"Compatibility checklist for opaque operation {operation}; this is operational metadata, not the normative programming manual.",
        },
    ]
    topics = ("gardening", "network timing", "invoice layout", "music catalog", "weather archive", "image metadata", "hardware inventory")
    for index in range(pool_size - len(records)):
        records.append(
            {
                "doc_id": _doc_id(seed, family.family_id, f"unrelated-{index}"),
                "kind": "unrelated",
                "text": f"Unrelated reference {index:03d} about {topics[index % len(topics)]}; no synthetic API definition is present.",
            }
        )
    random.Random(seed * 1000 + pool_size + len(family.family_id)).shuffle(records)
    return tuple(records), correct_id


def _retrieval_prompt_012(family: PythonFamily009, pool: tuple[dict[str, str], ...]) -> str:
    documents = "\n\n".join(f"[{record['doc_id']}]\n{record['text']}" for record in pool)
    operation = family.family_id.rsplit("-", 1)[-1]
    return RETRIEVAL_PROMPT_TEMPLATE_011.format(operation=operation, api_name=SYNTHETIC_MODULE_012, documents=documents)


def _retrieve_012(client: LlamaCppClient, store: ExperimentStore, family: PythonFamily009, pool: tuple[dict[str, str], ...], correct_id: str, model_seed: int) -> dict[str, Any]:
    kind = f"air-012:retrieval:{family.family_id}:{len(pool)}"
    with store.connect() as connection:
        cached = connection.execute("SELECT metadata_json FROM runs WHERE kind = ? ORDER BY id DESC LIMIT 1", (kind,)).fetchone()
    if cached:
        result = json.loads(cached["metadata_json"])
        result["cached_resume"] = True
        return result
    prompt = _retrieval_prompt_012(family, pool)
    runtime_error: str | None = None
    try:
        completion = client.chat_json(prompt, max_tokens=96, seed=model_seed)
        parsed = parse_response(completion.text)
        selected = parsed.get("doc_id") if isinstance(parsed, dict) and isinstance(parsed.get("doc_id"), str) else None
        response_text = completion.text
        elapsed_seconds = completion.elapsed_seconds
        prompt_tokens = completion.prompt_tokens or 0
        generated_tokens = completion.generated_tokens or 0
    except ModelUnavailable as exc:
        selected = None
        runtime_error = str(exc)
        response_text = json.dumps({"runtime_error": runtime_error})
        elapsed_seconds = client.timeout_seconds
        prompt_tokens = generated_tokens = 0
    lookup = {record["doc_id"]: record for record in pool}
    selected_record = lookup.get(selected)
    result = {
        "expected_doc_id": correct_id,
        "selected_doc_id": selected,
        "correct": selected == correct_id,
        "wrong_but_related": bool(selected_record and selected_record["kind"] in {"related_wrong", "terminology_distractor"}),
        "hallucinated_document_id": selected is not None and selected not in lookup,
        "selected_kind": selected_record["kind"] if selected_record else None,
        "elapsed_seconds": elapsed_seconds,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "runtime_error": runtime_error,
        "cached_resume": False,
    }
    store.record_run(
        kind=kind,
        prompt=prompt,
        response=response_text,
        elapsed_seconds=elapsed_seconds,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        passed=result["correct"],
        metadata={**result, "pool_size": len(pool), "retrieval_prompt_hash": RETRIEVAL_PROMPT_HASH_011, "model_seed": model_seed},
    )
    return result


def _cached_learning_attempts_012(store: ExperimentStore, family: PythonFamily009, learner_seed: int) -> tuple[RepairAttempt009, ...]:
    kind = f"air-009:{family.family_id}:proposal"
    with store.connect() as connection:
        rows = connection.execute("SELECT response, metadata_json FROM runs WHERE kind = ? ORDER BY id", (kind,)).fetchall()
    attempts: list[RepairAttempt009] = []
    for row in rows:
        metadata = json.loads(row["metadata_json"])
        if metadata.get("seed") != learner_seed:
            continue
        parsed = parse_response(row["response"])
        code = parsed.get("code") if isinstance(parsed, dict) and isinstance(parsed.get("code"), str) else None
        static_data = metadata.get("static", {})
        gate_data = metadata.get("public_gate", {})
        attempts.append(
            RepairAttempt009(
                int(metadata.get("attempt", len(attempts) + 1)),
                code,
                StaticCheck009(bool(static_data.get("passed")), str(static_data.get("reason", "cached attempt"))),
                PythonGate009(str(gate_data.get("condition", "discovery")), int(gate_data.get("correct", 0)), int(gate_data.get("total", len(family.discovery))), float(gate_data.get("accuracy", 0.0))),
                str(metadata.get("feedback", "cached attempt")),
            )
        )
    return tuple(attempts)


def _learn_family_resumable_012(client: LlamaCppClient, store: ExperimentStore, family: PythonFamily009, skill_id: str, learner_seed: int, max_attempts: int = 3) -> tuple[PythonSkillArtifact009 | None, tuple[RepairAttempt009, ...]]:
    attempts = list(_cached_learning_attempts_012(store, family, learner_seed))
    for item in attempts:
        if item.static.passed and item.public_gate.accuracy == 1.0 and item.code:
            artifact = PythonSkillArtifact009(skill_id, family.family_id, 1, "value: str", "str", item.code, "0009 generic frozen learner; public-gated and hidden-validated")
            return artifact, tuple(attempts)
    if any(item.static.reason.startswith("model runtime unavailable:") for item in attempts):
        return None, tuple(attempts)
    if len(attempts) >= max_attempts:
        return None, tuple(attempts[:max_attempts])
    previous_code = attempts[-1].code if attempts else None
    feedback = attempts[-1].feedback if attempts else None
    for attempt_number in range(len(attempts) + 1, max_attempts + 1):
        prompt = generic_learning_prompt(family, previous_code, feedback)
        runtime_error: str | None = None
        try:
            completion = client.chat_json(prompt, max_tokens=512, seed=learner_seed)
            response_text = completion.text
            elapsed_seconds = completion.elapsed_seconds
            prompt_tokens = completion.prompt_tokens
            generated_tokens = completion.generated_tokens
            code = _extract_code(completion.text)
        except ModelUnavailable as exc:
            runtime_error = str(exc)
            response_text = json.dumps({"runtime_error": runtime_error})
            elapsed_seconds = client.timeout_seconds
            prompt_tokens = generated_tokens = 0
            code = None
        static = static_check_python_009(code, family) if code else StaticCheck009(False, runtime_error or "model did not return a code field")
        public_gate = run_python_gate_009(code, family, family.discovery, "discovery") if code else PythonGate009("discovery", 0, len(family.discovery), 0.0)
        passed = static.passed and public_gate.accuracy == 1.0
        if passed:
            feedback = "public tests passed"
        elif runtime_error:
            feedback = runtime_error
        else:
            failed_case = next((case for case in family.discovery if not run_python_in_sandbox_009(code, family, case.input_text, case.expected).passed), None) if code else None
            if failed_case and code:
                failure = run_python_in_sandbox_009(code, family, failed_case.input_text, failed_case.expected)
                feedback = f"Candidate failed {failed_case.case_id}: input={failed_case.input_text!r}, expected={failed_case.expected!r}, observed={failure.value!r}, error={failure.error!r}. Return the complete corrected program with imports. If this is a syntax error, use ordinary multi-line Python and do not place `def` after a semicolon."
            else:
                feedback = static.reason
        item = RepairAttempt009(attempt_number, code, static, public_gate, feedback)
        attempts.append(item)
        store.record_run(
            kind=f"air-009:{family.family_id}:proposal",
            prompt=prompt,
            response=response_text,
            elapsed_seconds=elapsed_seconds,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            passed=passed,
            metadata={"attempt": attempt_number, "family_id": family.family_id, "static": asdict(static), "public_gate": asdict(public_gate), "feedback": feedback, "prompt_version": LEARNING_PROMPT_VERSION_009, "prompt_hash": LEARNING_PROMPT_HASH_009, "seed": learner_seed, "resumable_012": True, "runtime_error": runtime_error},
        )
        if runtime_error:
            return None, tuple(attempts)
        if passed and code:
            artifact = PythonSkillArtifact009(skill_id, family.family_id, 1, "value: str", "str", code, "0009 generic frozen learner; public-gated and hidden-validated")
            return artifact, tuple(attempts)
        previous_code = code
    return None, tuple(attempts)


def _family_selected_doc_012(family: PythonFamily009, pool: tuple[dict[str, str], ...], selected_id: str | None) -> PythonFamily009:
    selected = next((record["text"] for record in pool if record["doc_id"] == selected_id), "")
    return replace(family, api_docs=selected)


def _base_matches_012(family: PythonFamily009) -> list[str]:
    return [artifact.skill_id for artifact in BASE_PYTHON_LIBRARY_010 if run_python_gate_009(artifact.code, family, family.discovery, "existing").accuracy == 1.0]


_PRIOR_0008_CODE = "from urllib.parse import parse_qsl, urlencode\n\ndef transform(query: str) -> str:\n    pairs = parse_qsl(query, keep_blank_values=True)\n    pairs.sort(key=lambda pair: (pair[0], pair[1]))\n    return urlencode(pairs, doseq=True)"


def run_robustness_block_012(client: LlamaCppClient, store: ExperimentStore, heldout_limit: int | None = None) -> dict[str, Any]:
    module_path = Path(air_synth_012.__file__)
    source_before = hashlib.sha256(module_path.read_bytes()).hexdigest()
    base_snapshot = {artifact.skill_id: artifact.to_dict() for artifact in BASE_PYTHON_LIBRARY_010}
    results: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    for pool_size in POOL_SIZES_012:
        for run_index, seed in enumerate(ROBUSTNESS_SEEDS_012, 1):
            regressions.append({"pool_size": pool_size, "seed": seed, "prior_0008_accuracy": run_python_gate(_PRIOR_0008_CODE, HELD_OUT_008, "prior-0008").accuracy})
            for family_index, family in enumerate(make_robustness_families_012(seed), 1):
                pool, correct_id = make_document_pool_012(family, pool_size, seed)
                retrieval = _retrieve_012(client, store, family, pool, correct_id, seed + pool_size * 10 + family_index)
                selected_family = _family_selected_doc_012(family, pool, retrieval["selected_doc_id"])
                matches = _base_matches_012(family)
                diagnosis = {"status": "covered" if matches else "gap_detected", "matching_skill_ids": matches, "evaluations": len(BASE_PYTHON_LIBRARY_010)}
                skill, attempts = _learn_family_resumable_012(
                    client,
                    store,
                    selected_family,
                    f"py-skill-012-{seed}-{pool_size}-{family_index}",
                    seed * 100 + pool_size + family_index,
                )
                discovery = run_python_gate_009(skill.code, family, family.discovery, "discovery") if skill else PythonGate009("discovery", 0, len(family.discovery), 0.0)
                validation = run_python_gate_009(skill.code, family, family.validation, "validation") if skill else PythonGate009("validation", 0, len(family.validation), 0.0)
                edge = run_python_gate_009(skill.code, family, family.edge, "edge") if skill else PythonGate009("edge", 0, len(family.edge), 0.0)
                passed_all_gates = bool(skill and discovery.accuracy == validation.accuracy == edge.accuracy == 1.0)
                active = bool(retrieval["correct"] and diagnosis["status"] == "gap_detected" and passed_all_gates)
                heldout = family.heldout[:heldout_limit] if heldout_limit is not None else family.heldout
                heldout_gate = run_python_gate_009(skill.code, family, heldout, "heldout") if active and skill else PythonGate009("heldout", 0, len(heldout), 0.0)
                unsafe = static_check_python_009("import os\ndef transform(value: str) -> str:\n    return os.getcwd()\n", family)
                semantic = run_python_gate_009("def transform(value: str) -> str:\n    return value\n", family, family.validation, "semantic_wrong")
                results.append(
                    {
                        "pool_size": pool_size,
                        "seed": seed,
                        "family_id": family.family_id,
                        "operation_kind": air_synth_012.FAMILY_KINDS_012[family_index - 1],
                        "retrieval": retrieval,
                        "gap_detection": diagnosis,
                        "learning": {
                            "proposal_count": len(attempts),
                            "repair_count": max(0, len(attempts) - 1),
                            "accepted_public_skill": skill.to_dict() if skill else None,
                            "attempts": [
                                {"attempt": item.attempt, "code": item.code, "static": asdict(item.static), "public_gate": asdict(item.public_gate), "feedback": item.feedback}
                                for item in attempts
                            ],
                            "discovery_gate": asdict(discovery),
                            "hidden_validation_gate": asdict(validation),
                            "edge_gate": asdict(edge),
                        },
                        "activation": {
                            "active": active,
                            "wrong_activation": bool(not retrieval["correct"] and active),
                            "wrong_retrieval_candidate_passed_gates_but_blocked": bool(not retrieval["correct"] and passed_all_gates),
                            "heldout_reuse": asdict(heldout_gate),
                        },
                        "controls": {
                            "unsafe_rejected": not unsafe.passed,
                            "unsafe_reason": unsafe.reason,
                            "semantic_wrong_rejected": semantic.accuracy < 0.9,
                            "semantic_wrong_gate": asdict(semantic),
                            "base_skill_immutable": base_snapshot == {artifact.skill_id: artifact.to_dict() for artifact in BASE_PYTHON_LIBRARY_010},
                        },
                    }
                )
    source_after = hashlib.sha256(module_path.read_bytes()).hexdigest()
    by_pool: dict[str, dict[str, Any]] = {}
    for pool_size in POOL_SIZES_012:
        group = [item for item in results if item["pool_size"] == pool_size]
        successful_retrieval_calls = [item for item in group if not item["retrieval"].get("runtime_error")]
        correct_retrievals = sum(item["retrieval"]["correct"] for item in group)
        active_count = sum(item["activation"]["active"] for item in group)
        by_pool[str(pool_size)] = {
            "runs": len(group),
            "correct_retrieval": correct_retrievals,
            "retrieval_rate": correct_retrievals / len(group),
            "wrong_but_related": sum(item["retrieval"]["wrong_but_related"] for item in group),
            "hallucinated_document_ids": sum(item["retrieval"]["hallucinated_document_id"] for item in group),
            "skills_activated": active_count,
            "activation_given_correct_retrieval": active_count / correct_retrievals if correct_retrievals else 0.0,
            "wrong_activations": sum(item["activation"]["wrong_activation"] for item in group),
            "heldout_correct": sum(item["activation"]["heldout_reuse"]["correct"] for item in group),
            "heldout_total_active": sum(item["activation"]["heldout_reuse"]["total"] for item in group if item["activation"]["active"]),
            "proposals": sum(item["learning"]["proposal_count"] for item in group),
            "repairs": sum(item["learning"]["repair_count"] for item in group),
            "context_sustainability": {
                "successful_model_calls": len(successful_retrieval_calls),
                "runtime_timeouts": len(group) - len(successful_retrieval_calls),
                "mean_prompt_tokens_successful": statistics.mean(item["retrieval"]["prompt_tokens"] for item in successful_retrieval_calls) if successful_retrieval_calls else 0.0,
                "max_prompt_tokens_successful": max((item["retrieval"]["prompt_tokens"] for item in successful_retrieval_calls), default=0),
                "max_fraction_of_4096_context": max((item["retrieval"]["prompt_tokens"] for item in successful_retrieval_calls), default=0) / 4096,
                "mean_retrieval_seconds_successful": statistics.mean(item["retrieval"]["elapsed_seconds"] for item in successful_retrieval_calls) if successful_retrieval_calls else 0.0,
            },
        }
    by_seed: dict[str, dict[str, Any]] = {}
    for seed in ROBUSTNESS_SEEDS_012:
        group = [item for item in results if item["seed"] == seed]
        by_seed[str(seed)] = {
            "runs": len(group),
            "correct_retrieval": sum(item["retrieval"]["correct"] for item in group),
            "skills_activated": sum(item["activation"]["active"] for item in group),
            "families_with_at_least_one_active_repeat": sum(any(item["activation"]["active"] for item in group if item["operation_kind"] == kind) for kind in air_synth_012.FAMILY_KINDS_012),
        }
    return {
        "protocol": {
            "model": MODEL_IDENTITY_010,
            "learner_prompt_version": LEARNING_PROMPT_VERSION_009,
            "learner_prompt_sha256": LEARNING_PROMPT_HASH_009,
            "generic_learner_frozen": True,
            "family_specific_prompt_patches": False,
            "retrieval_prompt_version": RETRIEVAL_PROMPT_VERSION_011,
            "retrieval_prompt_sha256": RETRIEVAL_PROMPT_HASH_011,
            "seeds": list(ROBUSTNESS_SEEDS_012),
            "families_per_seed": len(air_synth_012.FAMILY_KINDS_012),
            "pool_sizes": list(POOL_SIZES_012),
            "result_patching_after_observation": False,
        },
        "results": results,
        "summary_by_pool_size": by_pool,
        "summary_by_seed": by_seed,
        "regression": regressions,
        "synthetic_api_source": {"path": str(module_path), "sha256_before": source_before, "sha256_after": source_after, "immutable": source_before == source_after},
        "base_skill_immutability_all": all(item["controls"]["base_skill_immutable"] for item in results),
    }


def _make_rule_spec_012() -> dict[str, Any]:
    rng = random.Random(12012)
    symbols = rng.sample(tuple("ABCDEFGHJKLMNPRSTUVWXYZ"), 5)
    values = rng.sample((2, 3, 5, 7, 11, 13, 17, 19), 5)
    return {"namespace": "Talven-12", "version": 1, "symbols": dict(zip(symbols, values)), "even_add": 5, "odd_multiply": 3}


RULE_SPEC_012 = _make_rule_spec_012()
RULE_HASH_012 = hashlib.sha256(canonical_json_bytes(RULE_SPEC_012)).hexdigest()
RULE_DOC_012 = (
    f"{RULE_SPEC_012['namespace']} v1 is a new experiment-local rule. Input is <symbol><integer>. "
    f"Symbol map: {RULE_SPEC_012['symbols']}. For even n use n + {RULE_SPEC_012['even_add']}; "
    f"for odd n use n * {RULE_SPEC_012['odd_multiply']}. Return symbol_value * transformed integer as a decimal string. Unknown symbols are invalid."
)


def _rule_reference_012(value: str) -> str:
    symbol = value[0]
    number = int(value[1:])
    transformed = number + RULE_SPEC_012["even_add"] if number % 2 == 0 else number * RULE_SPEC_012["odd_multiply"]
    return str(RULE_SPEC_012["symbols"][symbol] * transformed)


RULE_SOURCE_012 = """def transform(value: str) -> str:
    symbol_map = {symbols}
    symbol = value[0]
    number = int(value[1:])
    factor = symbol_map[symbol]
    adjusted = number + {even_add} if number % 2 == 0 else number * {odd_multiply}
    return str(factor * adjusted)
""".format(symbols=RULE_SPEC_012["symbols"], even_add=RULE_SPEC_012["even_add"], odd_multiply=RULE_SPEC_012["odd_multiply"])


def _rule_cases_012() -> tuple[tuple[FamilyCase009, ...], tuple[FamilyCase009, ...], tuple[FamilyCase009, ...], tuple[FamilyCase009, ...]]:
    symbols = tuple(RULE_SPEC_012["symbols"])
    tokens = [f"{symbols[index % len(symbols)]}{number}" for index, number in enumerate((2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 26, 27))]
    def cases(split: str, values: list[str]) -> tuple[FamilyCase009, ...]:
        return tuple(FamilyCase009(f"talven-{split}-{index:02d}", value, _rule_reference_012(value), split) for index, value in enumerate(values, 1))
    return cases("discovery", tokens[:4]), cases("validation", tokens[4:7]), cases("edge", tokens[7:10]), cases("heldout", tokens[10:18])


RULE_DISCOVERY_012, RULE_VALIDATION_012, RULE_EDGE_012, RULE_HELDOUT_012 = _rule_cases_012()
RULE_FAMILY_012 = PythonFamily009(
    family_id="talven-12-rule",
    title="Talven-12 generated rule",
    api_docs=RULE_DOC_012,
    contract="Define transform(value: str) -> str and apply the Talven-12 rule.",
    allowed_imports=frozenset(),
    allowed_import_members=frozenset(),
    allowed_call_names=frozenset({"int", "str"}),
    allowed_attrs=frozenset(),
    discovery=RULE_DISCOVERY_012,
    validation=RULE_VALIDATION_012,
    edge=RULE_EDGE_012,
    heldout=RULE_HELDOUT_012,
)


def _trusted_python_loader(source: str) -> Callable[[str], str]:
    namespace: dict[str, Any] = {}
    exec(source, {"__builtins__": {"int": int, "str": str}}, namespace)
    return namespace["transform"]


def _representation_metrics(name: str, payload: bytes, loader: Callable[[bytes], Any], executor: Callable[[Any, str], str], cases: tuple[FamilyCase009, ...], python_bytes: int, json_bytes: int, index_bytes: int) -> dict[str, Any]:
    load_metrics, loaded = _measure(lambda: loader(payload))
    case_index = 0
    def execute_next() -> str:
        nonlocal case_index
        case = cases[case_index % len(cases)]
        case_index += 1
        return executor(loaded, case.input_text)
    execute_metrics, _ = _measure(execute_next)
    correct = sum(executor(loaded, case.input_text) == case.expected for case in cases)
    return {
        "representation": name,
        "serialized_bytes_per_skill": len(payload),
        "total_bytes_for_n_skills": {str(count): len(payload) * count for count in (10, 100, 1_000, 10_000)},
        "index_bytes_per_skill": index_bytes,
        "load_deserialize": load_metrics,
        "execution": execute_metrics,
        "memory_footprint_proxy_bytes": deep_size(loaded),
        "semantic_equivalence": {"correct": correct, "total": len(cases), "passed": correct == len(cases)},
        "compression_ratio_vs_python": len(payload) / python_bytes,
        "compression_ratio_vs_json_ast": len(payload) / json_bytes,
    }


def run_storage_block_012() -> dict[str, Any]:
    source_before = RULE_SOURCE_012
    program = compile_python_rule_subset(RULE_SOURCE_012)
    json_ast = serialize_json_ast(program)
    compact_ir = serialize_compact_ir(program)
    binary_ir = serialize_binary_ir(program)
    python_payload = RULE_SOURCE_012.encode("utf-8")
    equivalence_cases = RULE_DISCOVERY_012 + RULE_VALIDATION_012 + RULE_EDGE_012 + RULE_HELDOUT_012
    provenance = DerivedArtifactProvenance(
        source_skill_id="talven-12-readable-python",
        source_version=1,
        source_sha256=source_sha256(RULE_SOURCE_012),
        ir_version=IR_VERSION,
        compiler_version=COMPILER_VERSION,
        semantic_equivalence_test_ids=tuple(case.case_id for case in equivalence_cases),
        created_at="2026-08-31T00:00:00+00:00",
        activation_status="active",
        source_immutable=True,
    )
    cold = ColdState(
        raw_experiences=tuple({"input": case.input_text, "expected": case.expected} for case in RULE_DISCOVERY_012),
        documentation=RULE_DOC_012,
        provenance=asdict(provenance),
        validation_history=({"validation": "3/3", "edge": "3/3"},),
        edge_tests=tuple({"input": case.input_text, "expected": case.expected} for case in RULE_EDGE_012),
        source_blob=RULE_SOURCE_012,
    )
    warm = WarmState(("talven", "symbol", "integer", "parity"), 0, 0, tuple(), 0.0, "synthetic-rule")
    formats = {
        "readable_python": (python_payload, lambda data: _trusted_python_loader(data.decode("utf-8")), lambda loaded, value: loaded(value)),
        "json_typed_ast": (json_ast, deserialize_json_ast, execute_program),
        "compact_air_ir": (compact_ir, deserialize_compact_ir, execute_program),
        "binary_air_ir": (binary_ir, deserialize_binary_ir, execute_program),
    }
    metrics: list[dict[str, Any]] = []
    layers: dict[str, Any] = {}
    for name, (payload, loader, executor) in formats.items():
        hot = HotState("talven-12-skill", "str", "str", "synthetic-rule", "verified", "active", 1, "talven parity transform", name, f"artifact://{name}")
        layered = LayeredSkillState(hot, warm, cold, payload)
        layer_sizes = layered.layer_bytes()
        layers[name] = {**layer_sizes, "runtime_loads_cold": False, "hot_plus_warm": layer_sizes["hot"] + layer_sizes["warm"]}
        metrics.append(_representation_metrics(name, payload, loader, executor, equivalence_cases, len(python_payload), len(json_ast), len(canonical_json_bytes(asdict(hot))) + len(canonical_json_bytes(asdict(warm)))))

    safety: dict[str, Any] = {}
    controls: list[tuple[str, Callable[[], Any]]] = []
    controls.append(("malformed_binary", lambda: deserialize_binary_ir(binary_ir[:-3])))
    unknown = bytearray(binary_ir)
    unknown[7] = 255
    controls.append(("unknown_opcode", lambda: deserialize_binary_ir(bytes(unknown))))
    wrong_version = bytearray(binary_ir)
    wrong_version[4] = 99
    controls.append(("wrong_version", lambda: deserialize_binary_ir(bytes(wrong_version))))
    type_invalid = AIRProgram(IR_VERSION, (Instruction("PARSE_TOKEN", (slot("symbol"), slot("number"))), Instruction("MUL_INT", (slot("product"), slot("symbol"), slot("number"))), Instruction("TO_STR", (slot("result"), slot("product"))), Instruction("RETURN", (slot("result"),))))
    controls.append(("type_invalid", lambda: validate_program(type_invalid)))
    for name, control in controls:
        try:
            control()
            safety[name] = {"rejected": False, "error": None}
        except (IRValidationError, UnicodeDecodeError) as exc:
            safety[name] = {"rejected": True, "error": str(exc)}
    wrong_symbols = dict(RULE_SPEC_012["symbols"])
    first_symbol = next(iter(wrong_symbols))
    wrong_symbols[first_symbol] += 1
    semantic_wrong = build_rule_program(wrong_symbols, RULE_SPEC_012["even_add"], RULE_SPEC_012["odd_multiply"])
    wrong_correct = sum(execute_program(semantic_wrong, case.input_text) == case.expected for case in RULE_VALIDATION_012 + RULE_EDGE_012)
    safety["semantic_wrong"] = {"rejected": wrong_correct < len(RULE_VALIDATION_012 + RULE_EDGE_012), "correct": wrong_correct, "total": len(RULE_VALIDATION_012 + RULE_EDGE_012)}
    equivalent, test_ids = semantic_equivalence(program, _rule_reference_012, ((case.case_id, case.input_text) for case in equivalence_cases))
    return {
        "architecture": {
            "version": 0,
            "hot": "identity, typed contract, category, trust/status, version, tiny retrieval descriptor, artifact pointer or compact executable",
            "warm": "lexical terms, usage/success statistics, relations, rank metadata, operation family",
            "cold": "raw experiences, documentation, provenance, validation history, edge tests, readable source blob",
            "cold_loaded_during_normal_query": False,
        },
        "air_ir": {
            "version": IR_VERSION,
            "compiler_version": COMPILER_VERSION,
            "opcodes": [instruction.opcode for instruction in program.instructions],
            "general_python": False,
            "unsupported_artifact_policy": "retain readable Python artifact",
            "deterministic_serialization": True,
            "model_independent_execution": True,
        },
        "layers_by_representation": layers,
        "representations": metrics,
        "semantic_equivalence": {"passed": equivalent, "test_ids": test_ids},
        "derived_artifact_provenance": asdict(provenance),
        "safety_controls": safety,
        "source_immutable": source_before == RULE_SOURCE_012,
    }


def _artifact_load_execute_metrics(binary_ir: bytes) -> dict[str, float]:
    symbol = next(iter(RULE_SPEC_012["symbols"]))
    value = f"{symbol}12"
    return _measure(lambda: execute_program(deserialize_binary_ir(binary_ir), value), repeats=50)[0]


def run_scaling_block_012(binary_ir: bytes) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    previous_sls: int | None = None
    for count in SCALE_POINTS_012:
        records = generate_skill_records(count, binary_ir)
        stored_bytes = sum(record.stored_bytes() for record in records)
        hot_warm_bytes = sum(record.hot_warm_bytes() for record in records)
        cold_bytes = sum(record.cold_bytes for record in records)
        query = query_for_record(records[-1])
        naive = benchmark_naive_retrieval(records, query)
        with tempfile.TemporaryDirectory(prefix=f"air-012-{count}-") as directory:
            index_path = Path(directory) / "skills.sqlite3"
            index = SQLiteSkillIndex(index_path)
            started = time.perf_counter()
            index.insert(records)
            build_seconds = time.perf_counter() - started
            indexed = index.benchmark(query)
            index_bytes = index.index_bytes()
            sqlite_file_bytes = index_path.stat().st_size
            index.close()
        artifact_cost = _artifact_load_execute_metrics(binary_ir)
        naive["total_latency_p50_us"] = naive["retrieval_latency"]["p50_us"] + artifact_cost["p50_us"]
        indexed["total_latency_p50_us"] = indexed["retrieval_latency"]["p50_us"] + artifact_cost["p50_us"]
        point = {
            "skills": count,
            "stored_learned_state_bytes": stored_bytes,
            "stored_hot_warm_bytes": hot_warm_bytes,
            "stored_cold_bytes": cold_bytes,
            "sqlite_index_bytes": index_bytes,
            "sqlite_file_bytes": sqlite_file_bytes,
            "index_build_seconds": build_seconds,
            "active_learned_state_bytes_indexed": indexed["active_learned_state_bytes"],
            "stored_to_active_ratio": stored_bytes / indexed["active_learned_state_bytes"] if indexed["active_learned_state_bytes"] else None,
            "sls_growth_vs_previous": stored_bytes / previous_sls if previous_sls else 1.0,
            "naive": naive,
            "indexed": indexed,
            "artifact_load_plus_execution": artifact_cost,
            "retrieval_correct": indexed["found_skill_ids"] == [records[-1].skill_id] and naive["found_skill_ids"] == [records[-1].skill_id],
            "composition_candidate_explosion": composition_candidate_counts(records),
        }
        points.append(point)
        previous_sls = stored_bytes
    return {
        "definitions": {
            "SLS": "all persistent HOT+WARM+COLD metadata and executable bytes, excluding SQLite page overhead",
            "ALS": "HOT+WARM metadata and full executable bytes materialized for returned top-K skills",
            "bytes_read_query": "logical serialized bytes scanned/returned plus an indexed B-tree page traversal proxy; not physical disk I/O telemetry",
            "ram_working_set": "recursive Python object-size proxy for active returned records, not process RSS",
        },
        "scale_points": points,
        "full_library_loaded_into_prompt": False,
        "cold_state_loaded_during_query": False,
        "vector_database_dependency": False,
        "maximum_scale_completed": max(point["skills"] for point in points),
    }


def _rule_prompt_012(case: FamilyCase009) -> str:
    return f"Return exactly one JSON object {{\"result\":\"...\"}}. Apply this full rule document: {RULE_DOC_012}\nInput: {json.dumps(case.input_text)}. No explanation."


def _raw_rule_condition_012(client: LlamaCppClient, store: ExperimentStore, cases: tuple[FamilyCase009, ...]) -> dict[str, Any]:
    correct = input_tokens = output_tokens = 0
    latencies: list[float] = []
    for case in cases:
        prompt = _rule_prompt_012(case)
        runtime_error: str | None = None
        try:
            completion = client.chat_json(prompt, max_tokens=96, seed=1212)
            response_text = completion.text
            elapsed_seconds = completion.elapsed_seconds
            prompt_count = completion.prompt_tokens or 0
            generated_count = completion.generated_tokens or 0
            parsed = parse_response(completion.text)
            observed = parsed.get("result") if isinstance(parsed, dict) else None
        except ModelUnavailable as exc:
            runtime_error = str(exc)
            response_text = json.dumps({"runtime_error": runtime_error})
            elapsed_seconds = client.timeout_seconds
            prompt_count = generated_count = 0
            observed = None
        passed = isinstance(observed, str) and observed == case.expected
        correct += int(passed)
        input_tokens += prompt_count
        output_tokens += generated_count
        latencies.append(elapsed_seconds)
        store.record_run(kind="air-012:efficiency:raw-rule-context", prompt=prompt, response=response_text, elapsed_seconds=elapsed_seconds, prompt_tokens=prompt_count, generated_tokens=generated_count, passed=passed, metadata={"case_id": case.case_id, "expected": case.expected, "runtime_error": runtime_error})
    latency_us = _percentiles_us(latencies)
    return {
        "condition": "full_raw_rule_document",
        "correct": correct,
        "total": len(cases),
        "accuracy": correct / len(cases),
        "latency_ms": {key.replace("_us", "_ms"): value / 1000 for key, value in latency_us.items()},
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_calls": len(cases),
        "attempts": len(cases),
        "bytes_loaded_per_query": len(RULE_DOC_012.encode("utf-8")),
        "total_bytes_loaded": len(RULE_DOC_012.encode("utf-8")) * len(cases),
    }


def _python_artifact_condition_012(code: str | None, cases: tuple[FamilyCase009, ...]) -> dict[str, Any]:
    correct = 0
    samples: list[float] = []
    for case in cases:
        result = run_python_in_sandbox_009(code, RULE_FAMILY_012, case.input_text, case.expected) if code else SandboxResult009(False, None, "no artifact", 0.0)
        correct += int(result.passed)
        samples.append(result.elapsed_seconds)
    return {
        "condition": "readable_python_artifact",
        "correct": correct,
        "total": len(cases),
        "accuracy": correct / len(cases),
        "latency_ms": {key.replace("_us", "_ms"): value / 1000 for key, value in _percentiles_us(samples).items()},
        "input_tokens": 0,
        "output_tokens": 0,
        "model_calls": 0,
        "attempts": len(cases),
        "bytes_loaded_per_query": len(code.encode("utf-8")) if code else 0,
        "total_bytes_loaded": len(code.encode("utf-8")) * len(cases) if code else 0,
        "execution_boundary": "isolated Python subprocess",
    }


def _ir_condition_012(name: str, payload: bytes | None, loader: Callable[[bytes], AIRProgram], cases: tuple[FamilyCase009, ...]) -> dict[str, Any]:
    correct = 0
    load_samples: list[float] = []
    execution_samples: list[float] = []
    total_samples: list[float] = []
    for case in cases:
        if payload is None:
            continue
        started = time.perf_counter()
        load_started = time.perf_counter()
        program = loader(payload)
        load_samples.append(time.perf_counter() - load_started)
        execution_started = time.perf_counter()
        try:
            observed = execute_program(program, case.input_text)
        except IRExecutionError:
            observed = None
        execution_samples.append(time.perf_counter() - execution_started)
        total_samples.append(time.perf_counter() - started)
        correct += int(observed == case.expected)
    return {
        "condition": name,
        "correct": correct,
        "total": len(cases),
        "accuracy": correct / len(cases),
        "latency": _percentiles_us(total_samples),
        "load_deserialize": _percentiles_us(load_samples),
        "execution": _percentiles_us(execution_samples),
        "input_tokens": 0,
        "output_tokens": 0,
        "model_calls": 0,
        "attempts": len(cases),
        "bytes_loaded_per_query": len(payload) if payload else 0,
        "total_bytes_loaded": len(payload) * len(cases) if payload else 0,
        "execution_boundary": "typed allowlisted AIR IR interpreter",
    }


def run_efficiency_block_012(client: LlamaCppClient, store: ExperimentStore) -> dict[str, Any]:
    source_family_hash = hashlib.sha256(RULE_DOC_012.encode()).hexdigest()
    base_matches = _base_matches_012(RULE_FAMILY_012)
    skill, attempts = _learn_family_resumable_012(client, store, RULE_FAMILY_012, "py-skill-012-talven", 12120)
    discovery = run_python_gate_009(skill.code, RULE_FAMILY_012, RULE_DISCOVERY_012, "discovery") if skill else PythonGate009("discovery", 0, len(RULE_DISCOVERY_012), 0.0)
    validation = run_python_gate_009(skill.code, RULE_FAMILY_012, RULE_VALIDATION_012, "validation") if skill else PythonGate009("validation", 0, len(RULE_VALIDATION_012), 0.0)
    edge = run_python_gate_009(skill.code, RULE_FAMILY_012, RULE_EDGE_012, "edge") if skill else PythonGate009("edge", 0, len(RULE_EDGE_012), 0.0)
    active = bool(not base_matches and skill and discovery.accuracy == validation.accuracy == edge.accuracy == 1.0)
    source_hash_before = source_sha256(skill.code) if active and skill else None
    program: AIRProgram | None = None
    compile_error: str | None = None
    equivalence_ids: list[str] = []
    if active and skill:
        try:
            candidate = compile_python_rule_subset(skill.code)
            equivalent, equivalence_ids = semantic_equivalence(candidate, _rule_reference_012, ((case.case_id, case.input_text) for case in RULE_VALIDATION_012 + RULE_EDGE_012))
            if equivalent:
                program = candidate
            else:
                compile_error = "compiled IR failed semantic equivalence"
        except IRValidationError as exc:
            compile_error = str(exc)
    json_ast = serialize_json_ast(program) if program else None
    compact_json = serialize_compact_ir(program) if program else None
    binary_ir = serialize_binary_ir(program) if program else None
    conditions = {
        "raw_context": _raw_rule_condition_012(client, store, RULE_HELDOUT_012),
        "readable_python": _python_artifact_condition_012(skill.code if active and skill else None, RULE_HELDOUT_012),
        "json_ast": _ir_condition_012("json_typed_ast", json_ast, deserialize_json_ast, RULE_HELDOUT_012),
        "compact_json_ir": _ir_condition_012("compact_air_ir", compact_json, deserialize_compact_ir, RULE_HELDOUT_012),
        "binary_ir": _ir_condition_012("binary_air_ir", binary_ir, deserialize_binary_ir, RULE_HELDOUT_012),
    }
    unknown = f"Z7"
    ir_unknown_rejected = True
    if program:
        try:
            execute_program(program, unknown)
            ir_unknown_rejected = False
        except IRExecutionError:
            pass
    python_unknown = run_python_in_sandbox_009(skill.code, RULE_FAMILY_012, unknown, "__invalid_unknown_token__") if active and skill else SandboxResult009(False, None, "no artifact", 0.0)
    provenance = DerivedArtifactProvenance(
        source_skill_id=skill.skill_id if skill else "none",
        source_version=skill.version if skill else 0,
        source_sha256=source_hash_before or "",
        ir_version=IR_VERSION,
        compiler_version=COMPILER_VERSION,
        semantic_equivalence_test_ids=tuple(equivalence_ids),
        created_at=datetime.now(UTC).isoformat(),
        activation_status="active" if program else "rejected",
        source_immutable=bool(active and skill and source_hash_before == source_sha256(skill.code)),
    )
    return {
        "rule_system": {"name": RULE_SPEC_012["namespace"], "version": 1, "rule_sha256": RULE_HASH_012, "documentation_sha256": source_family_hash},
        "acquisition": {
            "gap_detected": not base_matches,
            "proposal_count": len(attempts),
            "repair_count": max(0, len(attempts) - 1),
            "accepted_skill": skill.to_dict() if skill else None,
            "discovery_gate": asdict(discovery),
            "validation_gate": asdict(validation),
            "edge_gate": asdict(edge),
            "python_active": active,
            "ir_compiled_and_active": program is not None,
            "compile_error": compile_error,
        },
        "conditions": conditions,
        "provenance": asdict(provenance),
        "controls": {
            "unknown_token": {"input": unknown, "python_safe_rejection": not python_unknown.passed, "ir_safe_rejection": ir_unknown_rejected},
            "source_immutable": bool(active and skill and source_hash_before == source_sha256(skill.code)),
            "artifact_is_model_parameter": False,
        },
    }


def run_exp012(*, client: LlamaCppClient, store: ExperimentStore, report_directory: str, heldout_limit: int | None = None) -> dict[str, Any]:
    block_a = run_robustness_block_012(client, store, heldout_limit)
    block_b = run_storage_block_012()
    binary_metric = next(item for item in block_b["representations"] if item["representation"] == "binary_air_ir")
    canonical_program = compile_python_rule_subset(RULE_SOURCE_012)
    block_c = run_scaling_block_012(serialize_binary_ir(canonical_program))
    block_d = run_efficiency_block_012(client, store)
    report = {
        "benchmark": "air-012-robust-learning-storage-scaling",
        "created_at": datetime.now(UTC).isoformat(),
        "model_runtime": MODEL_IDENTITY_010,
        "block_a_multi_api_robustness": block_a,
        "block_b_learned_state_storage_v0": block_b,
        "block_c_skill_library_scaling": block_c,
        "block_d_learned_state_efficiency_control": block_d,
        "cross_block_interpretation": {
            "bounded_experiment_not_general_autonomy": True,
            "learned_state_is_external_model_independent_executable_state": True,
            "model_parameter_update": False,
            "binary_ir_bytes_per_skill": binary_metric["serialized_bytes_per_skill"],
            "brain_transplant_tested": False,
        },
    }
    directory = Path(report_directory)
    directory.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(UTC).strftime("air-012-%Y%m%dT%H%M%SZ.json")
    path = directory / filename
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_file"] = str(path)
    return report
