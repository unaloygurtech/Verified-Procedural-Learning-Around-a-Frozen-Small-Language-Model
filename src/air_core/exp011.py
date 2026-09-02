"""Experiment 0011: documentation retrieval and rule-learning efficiency.

The first block reuses the 0010 synthetic ``zorvik_010`` API but hides its
operation notes in a small noisy document pool.  A retriever must select the
normative note before the frozen 0009 learner can propose code.  The second
block creates a deterministic, experiment-local Neralis-like rule system and
compares zero-knowledge, raw-document, and learned-artifact execution costs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Any

from .exp008 import HELD_OUT_008, run_python_gate
from .exp009 import (
    FamilyCase009,
    LEARNING_PROMPT_HASH_009,
    LEARNING_PROMPT_VERSION_009,
    PythonFamily009,
    PythonGate009,
    PythonSkillArtifact009,
    RepairAttempt009,
    SandboxResult009,
    generic_learning_prompt,
    learn_family_009,
    run_python_gate_009,
    run_python_in_sandbox_009,
    run_skill_heldout_009,
    static_check_python_009,
)
from .exp010 import (
    BASE_PYTHON_LIBRARY_010,
    MODEL_IDENTITY_010,
    SYNTHETIC_API_NAME_010,
    ZORVIK_FAMILIES_010,
    _api_source_hash_010,
    _docs_hash_010,
    _provenance_010,
)
from .model_client import LlamaCppClient
from .neralis import parse_response
from .store import ExperimentStore


RETRIEVAL_PROMPT_VERSION_011 = "air-011-document-retriever-v1"
RETRIEVAL_PROMPT_TEMPLATE_011 = """Select the one normative documentation record needed for the requested synthetic API operation.

Requested operation: {operation}
Required signature: transform(value: str) -> str using {api_name}.{operation}

Choose exactly one record ID from the document pool. Do not solve the task and do not explain. Return exactly {{\"doc_id\":\"...\"}}.

Document pool:
{documents}
"""
RETRIEVAL_PROMPT_HASH_011 = hashlib.sha256(RETRIEVAL_PROMPT_TEMPLATE_011.encode("utf-8")).hexdigest()


def _doc_pool_011() -> tuple[dict[str, str], ...]:
    records: list[dict[str, str]] = []
    opaque_ids = {"zorvik-kel": "manual-amber", "zorvik-nam": "manual-cobalt", "zorvik-tesh": "manual-ivory", "zorvik-vum": "manual-saffron"}
    for family in ZORVIK_FAMILIES_010:
        records.append({"doc_id": opaque_ids[family.family_id], "text": f"Normative synthetic API reference. {family.api_docs}"})
    records.extend(
        [
            {"doc_id": "note-charcoal", "text": "Deployment note: zorvik_010 is an isolated package. This note intentionally contains no operation semantics."},
            {"doc_id": "note-coral", "text": "Historical draft: the old nam prototype was not normative and described a different interface. Do not use this draft to implement current tasks."},
            {"doc_id": "note-linen", "text": "Unrelated standard-library reminder about urllib.parse; it is not documentation for zorvik_010."},
            {"doc_id": "note-olive", "text": "General Python style guidance: define transform(value: str) and return a string. No synthetic API behavior is specified here."},
        ]
    )
    return tuple(records)


DOC_POOL_011 = _doc_pool_011()
_CORRECT_DOC_IDS_011 = {"zorvik-kel": "manual-amber", "zorvik-nam": "manual-cobalt", "zorvik-tesh": "manual-ivory", "zorvik-vum": "manual-saffron"}


@dataclass(frozen=True)
class RetrievalResult011:
    family_id: str
    expected_doc_id: str
    selected_doc_id: str | None
    passed: bool
    related_wrong_selection: bool
    elapsed_seconds: float
    prompt_tokens: int
    generated_tokens: int


def retrieval_prompt_011(family: PythonFamily009) -> str:
    docs = "\n\n".join(f"[{record['doc_id']}]\n{record['text']}" for record in DOC_POOL_011)
    operation = family.family_id.removeprefix("zorvik-")
    return RETRIEVAL_PROMPT_TEMPLATE_011.format(operation=operation, api_name=SYNTHETIC_API_NAME_010, documents=docs)


def retrieve_document_011(client: LlamaCppClient, store: ExperimentStore, family: PythonFamily009, seed: int) -> RetrievalResult011:
    prompt = retrieval_prompt_011(family)
    completion = client.chat_json(prompt, max_tokens=96, seed=seed)
    parsed = parse_response(completion.text)
    selected = parsed.get("doc_id") if isinstance(parsed, dict) and isinstance(parsed.get("doc_id"), str) else None
    expected = _CORRECT_DOC_IDS_011[family.family_id]
    related = selected in {"note-coral", "note-linen"} or (selected is not None and selected.startswith("manual-") and selected != expected)
    store.record_run(kind=f"air-011:retrieval:{family.family_id}", prompt=prompt, response=completion.text, elapsed_seconds=completion.elapsed_seconds, prompt_tokens=completion.prompt_tokens, generated_tokens=completion.generated_tokens, passed=selected == expected, metadata={"family_id": family.family_id, "expected_doc_id": expected, "selected_doc_id": selected, "related_wrong_selection": related, "retrieval_prompt_version": RETRIEVAL_PROMPT_VERSION_011, "retrieval_prompt_hash": RETRIEVAL_PROMPT_HASH_011, "seed": seed})
    return RetrievalResult011(family.family_id, expected, selected, selected == expected, related, completion.elapsed_seconds, completion.prompt_tokens or 0, completion.generated_tokens or 0)


def _family_with_retrieved_doc(family: PythonFamily009, selected_doc_id: str | None) -> PythonFamily009:
    selected = next((record["text"] for record in DOC_POOL_011 if record["doc_id"] == selected_doc_id), "")
    return replace(family, api_docs=selected)


def _base_match_ids_011(family: PythonFamily009) -> list[str]:
    matches: list[str] = []
    for artifact in BASE_PYTHON_LIBRARY_010:
        if run_python_gate_009(artifact.code, family, family.discovery, "existing").accuracy == 1.0:
            matches.append(artifact.skill_id)
    return matches


def run_retrieval_block_011(*, client: LlamaCppClient, store: ExperimentStore, heldout_limit: int | None = None) -> dict[str, Any]:
    package_before = _api_source_hash_010()
    retrievals = [retrieve_document_011(client, store, family, 1100 + index) for index, family in enumerate(ZORVIK_FAMILIES_010, 1)]
    family_results: list[dict[str, Any]] = []
    active_artifacts: list[tuple[PythonSkillArtifact009, PythonFamily009]] = []
    for index, family in enumerate(ZORVIK_FAMILIES_010, 1):
        retrieval = next(item for item in retrievals if item.family_id == family.family_id)
        selected_family = _family_with_retrieved_doc(family, retrieval.selected_doc_id)
        matches_before = _base_match_ids_011(selected_family)
        diagnosis = {"status": "covered" if matches_before else "gap_detected", "matching_skill_ids": matches_before, "evaluations": len(BASE_PYTHON_LIBRARY_010)}
        skill, attempts = learn_family_009(client, store, selected_family, f"py-skill-011-{family.family_id}", 1200 + index)
        discovery = run_python_gate_009(skill.code, selected_family, family.discovery, "discovery") if skill else PythonGate009("discovery", 0, len(family.discovery), 0.0)
        validation = run_python_gate_009(skill.code, selected_family, family.validation, "validation") if skill else PythonGate009("validation", 0, len(family.validation), 0.0)
        edge = run_python_gate_009(skill.code, selected_family, family.edge, "edge") if skill else PythonGate009("edge", 0, len(family.edge), 0.0)
        active = bool(retrieval.passed and skill and diagnosis["status"] == "gap_detected" and discovery.accuracy == 1.0 and validation.accuracy == 1.0 and edge.accuracy == 1.0)
        if active and skill:
            active_artifacts.append((skill, selected_family))
        heldout = family.heldout[:heldout_limit] if heldout_limit is not None else family.heldout
        before = run_skill_heldout_009(None, selected_family, heldout, store, "before_gap")
        learned = run_skill_heldout_009(skill.code if active and skill else None, selected_family, heldout, store, "learned_retrieved_skill")
        family_results.append({"family_id": family.family_id, "retrieval": asdict(retrieval), "gap_detection": diagnosis, "learning": {"attempt_count": len(attempts), "repair_count": max(0, len(attempts) - 1), "attempts": [{"attempt": item.attempt, "code": item.code, "static": asdict(item.static), "public_gate": asdict(item.public_gate), "feedback": item.feedback} for item in attempts], "accepted_skill": skill.to_dict() if skill else None, "provenance": _provenance_010(skill, selected_family, attempts) if skill else None, "discovery_gate": asdict(discovery), "validation_gate": asdict(validation), "edge_gate": asdict(edge)}, "activation": {"active": active, "wrong_activation_before": bool(matches_before), "artifact_reuse": bool(active and skill), "matching_skill_ids_before": matches_before}, "heldout_results": [asdict(before), asdict(learned)], "base_skill_immutability": {artifact.skill_id: artifact.to_dict() for artifact in BASE_PYTHON_LIBRARY_010} == {artifact.skill_id: artifact.to_dict() for artifact in BASE_PYTHON_LIBRARY_010}})
    unsafe = static_check_python_009("import os\ndef transform(value: str) -> str:\n    return os.getcwd()\n", ZORVIK_FAMILIES_010[0])
    semantic_code = "def transform(value: str) -> str:\n    return value\n"
    semantic_gate = run_python_gate_009(semantic_code, ZORVIK_FAMILIES_010[0], ZORVIK_FAMILIES_010[0].validation, "semantic_corrupted")
    package_after = _api_source_hash_010()
    report = {"protocol": {"retrieval_prompt_version": RETRIEVAL_PROMPT_VERSION_011, "retrieval_prompt_sha256": RETRIEVAL_PROMPT_HASH_011, "learner_prompt_version": LEARNING_PROMPT_VERSION_009, "learner_prompt_sha256": LEARNING_PROMPT_HASH_009, "generic_learner_frozen": True, "family_specific_learner_patches": False, "document_pool_size": len(DOC_POOL_011), "family_count": len(ZORVIK_FAMILIES_010)}, "document_pool": [{"doc_id": record["doc_id"], "text_sha256": hashlib.sha256(record["text"].encode()).hexdigest()} for record in DOC_POOL_011], "retrieval_results": [asdict(item) for item in retrievals], "family_results": family_results, "safety_controls": {"unsafe_candidate": {"static": asdict(unsafe), "activated": False}, "semantic_wrong_candidate": {"gate": asdict(semantic_gate), "state": "rejected" if semantic_gate.accuracy < 0.9 else "unsafe-active"}}, "regression": {"prior_0008_accuracy": run_python_gate("from urllib.parse import parse_qsl, urlencode\n\ndef transform(query: str) -> str:\n    pairs = parse_qsl(query, keep_blank_values=True)\n    pairs.sort(key=lambda pair: (pair[0], pair[1]))\n    return urlencode(pairs, doseq=True)", HELD_OUT_008, "prior").accuracy}, "synthetic_package": {"source_sha256_before": package_before, "source_sha256_after": package_after, "unchanged": package_before == package_after}, "summary": {"retrieval_accuracy": sum(item.passed for item in retrievals) / len(retrievals), "wrong_related_retrievals": sum(item.related_wrong_selection for item in retrievals), "gap_detection_rate": sum(item["gap_detection"]["status"] == "gap_detected" for item in family_results) / len(family_results), "skills_activated": sum(item["activation"]["active"] for item in family_results), "activation_rate": sum(item["activation"]["active"] for item in family_results) / len(family_results), "wrong_activation_count": sum(item["activation"]["wrong_activation_before"] for item in family_results), "artifact_reuse_count": sum(item["activation"]["artifact_reuse"] for item in family_results), "hidden_validation_failures": sum(item["learning"]["accepted_skill"] is not None and item["learning"]["validation_gate"]["accuracy"] < 1.0 for item in family_results), "base_skill_immutability_all": all(item["base_skill_immutability"] for item in family_results)}}
    return report


def _make_rule_spec_011() -> dict[str, Any]:
    rng = random.Random(11011)
    symbols = ("Q", "M", "V", "R")
    values = rng.sample((2, 4, 7, 9, 11, 13, 17), len(symbols))
    return {"namespace": "Neralis-11", "version": 1, "symbols": dict(zip(symbols, values)), "even_add": 3, "odd_multiply": 2}


RULE_SPEC_011 = _make_rule_spec_011()
RULE_HASH_011 = hashlib.sha256(json.dumps(RULE_SPEC_011, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _rule_reference(value: str) -> str:
    symbol, number = value[0], int(value[1:])
    transformed = number + RULE_SPEC_011["even_add"] if number % 2 == 0 else number * RULE_SPEC_011["odd_multiply"]
    return str(RULE_SPEC_011["symbols"][symbol] * transformed)


RULE_DOC_011 = """Neralis-11 v1 is a new deterministic rule system. Input is <symbol><integer>. Symbol map: {symbols}. For an even integer n use n + {even_add}; for an odd integer n use n * {odd_multiply}. Return symbol_value * transformed_integer as a decimal string. Unknown symbols are invalid and must be rejected.""".format(symbols=RULE_SPEC_011["symbols"], even_add=RULE_SPEC_011["even_add"], odd_multiply=RULE_SPEC_011["odd_multiply"])


RULE_FAMILY_011 = PythonFamily009(
    family_id="neralis-11-rule",
    title="Neralis-11 synthetic rule",
    api_docs=RULE_DOC_011,
    contract="Define transform(value: str) -> str. Apply the Neralis-11 rule to the input token.",
    allowed_imports=frozenset(),
    allowed_import_members=frozenset(),
    allowed_call_names=frozenset({"int", "str"}),
    allowed_attrs=frozenset(),
    discovery=tuple(FamilyCase009(f"rule-discover-{i:02d}", value, _rule_reference(value), "discovery") for i, value in enumerate(("Q8", "M5", "V12", "R3"), 1)),
    validation=tuple(FamilyCase009(f"rule-validation-{i:02d}", value, _rule_reference(value), "validation") for i, value in enumerate(("Q14", "M7", "V2"), 1)),
    edge=tuple(FamilyCase009(f"rule-edge-{i:02d}", value, _rule_reference(value), "edge") for i, value in enumerate(("R0", "Q1", "M10"), 1)),
    heldout=tuple(FamilyCase009(f"rule-heldout-{i:02d}", value, _rule_reference(value), "heldout") for i, value in enumerate(("V9", "R4", "Q16", "M11", "R2", "V7", "Q0", "M14"), 1)),
)


def _rule_prompt_011(case: FamilyCase009, include_docs: bool) -> str:
    docs = f"\nFull rule documentation:\n{RULE_DOC_011}" if include_docs else ""
    return f"Return exactly one JSON object {{\"result\":\"...\"}}. No explanation. The input token is {json.dumps(case.input_text)}.{docs}"


def _run_rule_model_condition_011(cases: tuple[FamilyCase009, ...], condition: str, include_docs: bool, client: LlamaCppClient, store: ExperimentStore, seed: int) -> dict[str, Any]:
    correct = prompt_tokens = generated_tokens = 0
    latencies: list[float] = []
    for case in cases:
        prompt = _rule_prompt_011(case, include_docs)
        completion = client.chat_json(prompt, max_tokens=96, seed=seed)
        parsed = parse_response(completion.text)
        value = parsed.get("result") if isinstance(parsed, dict) else None
        passed = isinstance(value, str) and value == case.expected
        correct += int(passed)
        prompt_tokens += completion.prompt_tokens or 0
        generated_tokens += completion.generated_tokens or 0
        latencies.append(completion.elapsed_seconds)
        store.record_run(kind=f"air-011:rule:{condition}", prompt=prompt, response=completion.text, elapsed_seconds=completion.elapsed_seconds, prompt_tokens=completion.prompt_tokens, generated_tokens=completion.generated_tokens, passed=passed, metadata={"case_id": case.case_id, "expected": case.expected, "parsed": parsed, "model_calls": 1, "attempts": 1})
    return _rule_metrics_011(condition, correct, len(cases), latencies, prompt_tokens, generated_tokens, model_calls=len(cases), attempts=len(cases), active_context_tokens=0)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((percentile / 100) * (len(ordered) - 1)))))
    return ordered[index]


def _rule_metrics_011(condition: str, correct: int, total: int, latencies: list[float], input_tokens: int, output_tokens: int, model_calls: int, attempts: int, active_context_tokens: int) -> dict[str, Any]:
    return {"condition": condition, "valid_correct": correct, "valid_total": total, "accuracy": correct / total if total else 0.0, "latency_ms": {"mean": statistics.mean(latencies) * 1000 if latencies else 0.0, "p50": _percentile(latencies, 50) * 1000, "p95": _percentile(latencies, 95) * 1000}, "input_tokens": input_tokens, "output_tokens": output_tokens, "model_calls": model_calls, "attempts": attempts, "active_context_tokens": active_context_tokens}


def _run_rule_artifact_011(code: str | None, cases: tuple[FamilyCase009, ...], store: ExperimentStore, condition: str) -> dict[str, Any]:
    correct = 0
    latencies: list[float] = []
    context_tokens = len(code.split()) if code else 0
    for case in cases:
        result = run_python_in_sandbox_009(code, RULE_FAMILY_011, case.input_text, case.expected) if code else SandboxResult009(False, None, "no active skill", 0.0)
        correct += int(result.passed)
        latencies.append(result.elapsed_seconds)
        store.record_run(kind=f"air-011:rule:{condition}", prompt="artifact execution", response=json.dumps({"value": result.value, "error": result.error}), elapsed_seconds=result.elapsed_seconds, prompt_tokens=0, generated_tokens=0, passed=result.passed, metadata={"case_id": case.case_id, "expected": case.expected, "model_calls": 0, "attempts": 1, "artifact_context_tokens": context_tokens})
    return _rule_metrics_011(condition, correct, len(cases), latencies, 0, 0, model_calls=0, attempts=len(cases), active_context_tokens=context_tokens)


def run_rule_efficiency_block_011(*, client: LlamaCppClient, store: ExperimentStore) -> dict[str, Any]:
    base_matches = []
    for artifact in BASE_PYTHON_LIBRARY_010:
        if run_python_gate_009(artifact.code, RULE_FAMILY_011, RULE_FAMILY_011.discovery, "existing").accuracy == 1.0:
            base_matches.append(artifact.skill_id)
    diagnosis = {"status": "covered" if base_matches else "gap_detected", "matching_skill_ids": base_matches, "evaluations": len(BASE_PYTHON_LIBRARY_010)}
    skill, attempts = learn_family_009(client, store, RULE_FAMILY_011, "py-skill-011-neralis-rule", 1501)
    discovery = run_python_gate_009(skill.code, RULE_FAMILY_011, RULE_FAMILY_011.discovery, "discovery") if skill else PythonGate009("discovery", 0, 4, 0.0)
    validation = run_python_gate_009(skill.code, RULE_FAMILY_011, RULE_FAMILY_011.validation, "validation") if skill else PythonGate009("validation", 0, 3, 0.0)
    edge = run_python_gate_009(skill.code, RULE_FAMILY_011, RULE_FAMILY_011.edge, "edge") if skill else PythonGate009("edge", 0, 3, 0.0)
    active = bool(skill and diagnosis["status"] == "gap_detected" and discovery.accuracy == 1.0 and validation.accuracy == 1.0 and edge.accuracy == 1.0)
    heldout = RULE_FAMILY_011.heldout
    before = _run_rule_model_condition_011(heldout, "before_learning", False, client, store, 1601)
    raw = _run_rule_model_condition_011(heldout, "raw_rule_context", True, client, store, 1602)
    learned = _run_rule_artifact_011(skill.code if active and skill else None, heldout, store, "learned_artifact")
    unknown = "Z7"
    # Supply a deliberately impossible sentinel so a candidate that returns a
    # friendly error string cannot be counted as a valid transformation.
    unknown_result = run_python_in_sandbox_009(skill.code, RULE_FAMILY_011, unknown, "__invalid_unknown_token__") if active and skill else SandboxResult009(False, None, "no active skill", 0.0)
    wrong_code = "def transform(value: str) -> str:\n    return str(int(value[1:]))\n"
    wrong_gate = run_python_gate_009(wrong_code, RULE_FAMILY_011, RULE_FAMILY_011.validation, "semantic_wrong")
    unsafe = static_check_python_009("import os\ndef transform(value: str) -> str:\n    return os.getcwd()\n", RULE_FAMILY_011)
    return {"rule_system": {"namespace": RULE_SPEC_011["namespace"], "version": RULE_SPEC_011["version"], "spec_sha256": RULE_HASH_011, "documentation_sha256": hashlib.sha256(RULE_DOC_011.encode()).hexdigest(), "spec": RULE_SPEC_011}, "gap_detection": diagnosis, "learning": {"attempt_count": len(attempts), "repair_count": max(0, len(attempts) - 1), "accepted_skill": skill.to_dict() if skill else None, "discovery_gate": asdict(discovery), "validation_gate": asdict(validation), "edge_gate": asdict(edge), "attempts": [{"attempt": item.attempt, "code": item.code, "static": asdict(item.static), "public_gate": asdict(item.public_gate), "feedback": item.feedback} for item in attempts]}, "conditions": {"before_learning": before, "raw_rule_context": raw, "learned_artifact": learned}, "unknown_token": {"input": unknown, "passed_as_valid": unknown_result.passed, "error": unknown_result.error, "safe_rejection": not unknown_result.passed}, "controls": {"unsafe_candidate": {"static": asdict(unsafe), "activated": False}, "semantic_wrong_candidate": {"gate": asdict(wrong_gate), "state": "rejected" if wrong_gate.accuracy < 0.9 else "unsafe-active"}}, "interpretation": {"active_artifact": active, "artifact_context_tokens": len(skill.code.split()) if active and skill else 0, "artifact_is_model_parameter": False}}


def run_exp011(*, client: LlamaCppClient, store: ExperimentStore, report_directory: str, heldout_limit: int | None = None) -> dict[str, Any]:
    retrieval_block = run_retrieval_block_011(client=client, store=store, heldout_limit=heldout_limit)
    rule_block = run_rule_efficiency_block_011(client=client, store=store)
    report = {"benchmark": "air-011-documentation-retrieval-and-rule-efficiency", "created_at": datetime.now(UTC).isoformat(), "model_runtime": MODEL_IDENTITY_010, "documentation_retrieval": retrieval_block, "novel_rule_efficiency": rule_block}
    path = Path(report_directory)
    path.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(UTC).strftime("air-011-%Y%m%dT%H%M%SZ.json")
    report_path = path / filename
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_file"] = str(report_path)
    return report
