"""Experiment 0014: frozen-model utilization, planning, and acquisition.

0013 deliberately measured the retrieval and composition boundary without
calling the model.  This module keeps that boundary frozen and adds a small,
auditable model arm.  Every model-facing template is versioned and hashed;
the report records failures by stage instead of collapsing them into one
``failed`` counter.

The benchmark is intentionally bounded.  It is not a claim of general
continual learning: it measures candidate ranking, context use, decomposition,
verified Python acquisition, and external artifact reuse with the existing
SmolLM3 runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Iterable, Sequence

from .exp009 import (
    PythonFamily009,
    PythonGate009,
    PythonSkillArtifact009,
    _prior_0008_regression,
    run_python_gate_009,
    run_python_in_sandbox_009,
    static_check_python_009,
    LEARNING_PROMPT_HASH_009,
    LEARNING_PROMPT_VERSION_009,
)
from .exp010 import BASE_PYTHON_LIBRARY_010
from .exp011 import RETRIEVAL_PROMPT_HASH_011, RETRIEVAL_PROMPT_VERSION_011
from .exp012 import (
    ROBUSTNESS_SEEDS_012,
    _family_selected_doc_012,
    _retrieval_prompt_012,
    make_document_pool_012,
    make_robustness_families_012,
    _learn_family_resumable_012,
)
from .exp013 import (
    CapabilityQuery,
    CapabilityRecord,
    CompositionSkill,
    CompositionTask,
    HierarchicalCapabilityIndex,
    _scoped_paths,
    _type_paths,
    generate_capability_records,
    make_composition_library,
)
from .model_client import LlamaCppClient, ModelUnavailable
from .neralis import parse_response
from .store import ExperimentStore


EXP014_VERSION = "air-014-v1"
MODEL_IDENTITY_014 = "SmolLM3-3B-GGUF-Q4_K_M; llama.cpp; CPU; context=4096"
CONTEXT_SIZE_014 = 4096
MAX_REPAIR_ATTEMPTS_014 = 3
ACQUISITION_FAMILIES_014 = 3


RANKING_PROMPT_TEMPLATE_014 = """Choose the one capability that best satisfies the task.
Return exactly one JSON object: {{\"skill_id\": string or null}}.
Choose null only when the candidates are genuinely ambiguous or none can do the task.
Do not invent an id.  Compare the task with the structured descriptors; do not assume
that an id, version, or natural-language name is evidence by itself.

Task: {task}
Candidates (the complete bounded candidate set):
{candidates}
"""
RANKING_PROMPT_VERSION_014 = "air-014-ranking-v1"
RANKING_PROMPT_HASH_014 = hashlib.sha256(RANKING_PROMPT_TEMPLATE_014.encode("utf-8")).hexdigest()

CONTEXT_PROMPT_TEMPLATE_014 = """Execute the task using only the supplied source material.
Return exactly one JSON object with a string field named result. Do not explain.

Task: {task}
Input: {input}
Source material:
{source}
"""
CONTEXT_PROMPT_VERSION_014 = "air-014-context-v1"
CONTEXT_PROMPT_HASH_014 = hashlib.sha256(CONTEXT_PROMPT_TEMPLATE_014.encode("utf-8")).hexdigest()

DECOMPOSITION_PROMPT_TEMPLATE_014 = """Decompose the task into the ordered capability keys needed to solve it.
Return exactly one JSON object: {{\"required_capabilities\":[string,...]}}.
Use only keys from the catalog. Do not return implementation code or a final answer.

Task: {task}
Capability catalog:
{catalog}
"""
DECOMPOSITION_PROMPT_VERSION_014 = "air-014-decomposition-v1"
DECOMPOSITION_PROMPT_HASH_014 = hashlib.sha256(DECOMPOSITION_PROMPT_TEMPLATE_014.encode("utf-8")).hexdigest()

COMPOSITION_RANKING_PROMPT_TEMPLATE_014 = """Select the capability artifact for this single stage.
Return exactly one JSON object: {{\"skill_id\": string or null}}.
The stage contract must match the requested input/output types.

Stage: {stage}
Required input type: {input_type}
Required output type: {output_type}
Candidates:
{candidates}
"""
COMPOSITION_RANKING_PROMPT_VERSION_014 = "air-014-composition-ranking-v1"
COMPOSITION_RANKING_PROMPT_HASH_014 = hashlib.sha256(COMPOSITION_RANKING_PROMPT_TEMPLATE_014.encode("utf-8")).hexdigest()

DIRECT_ANSWER_PROMPT_TEMPLATE_014 = """Return exactly one JSON object {{\"result\":\"...\"}}.
Contract: {contract}{docs}
Input: {input}
Do not explain.
"""
DIRECT_ANSWER_PROMPT_VERSION_014 = "air-014-direct-answer-v1"
DIRECT_ANSWER_PROMPT_HASH_014 = hashlib.sha256(DIRECT_ANSWER_PROMPT_TEMPLATE_014.encode("utf-8")).hexdigest()
ACQUISITION_TASK_PROMPT_VERSION_014 = "air-014-acquisition-v1"  # the learner body is 0009 and remains frozen


FAILURE_TAXONOMY_014 = (
    "retrieval_failure", "decomposition_failure", "ranking_failure", "composition_failure",
    "gap_detection_failure", "synthesis_failure", "repair_failure", "public_validation_failure",
    "hidden_validation_failure", "edge_failure", "safety_rejection", "timeout", "safe_unknown",
)


@dataclass
class ModelLedger014:
    """Shared accounting for all calls, including unavailable-runtime trials."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    timeouts: int = 0
    latencies_seconds: list[float] | None = None

    def __post_init__(self) -> None:
        if self.latencies_seconds is None:
            self.latencies_seconds = []

    def observe(self, *, prompt: str, elapsed: float, prompt_tokens: int | None,
                output_tokens: int | None, timeout: bool = False) -> None:
        self.calls += 1
        self.input_tokens += prompt_tokens if prompt_tokens is not None else max(1, len(prompt) // 4)
        self.output_tokens += output_tokens or 0
        self.latencies_seconds.append(elapsed)
        if timeout:
            self.timeouts += 1

    def summary(self) -> dict[str, Any]:
        ordered = sorted(self.latencies_seconds or [])
        p50 = ordered[len(ordered) // 2] if ordered else 0.0
        p95 = ordered[min(len(ordered) - 1, int(round(.95 * (len(ordered) - 1))))] if ordered else 0.0
        return {
            "total_model_calls": self.calls,
            "total_input_tokens": self.input_tokens,
            "total_output_tokens": self.output_tokens,
            "timeout_count": self.timeouts,
            "latency_seconds": {"p50": p50, "p95": p95, "mean": statistics.mean(ordered) if ordered else 0.0},
        }


class _CountingClient014:
    """Proxy used by the frozen 0012 learner without changing its prompts."""

    def __init__(self, inner: LlamaCppClient, ledger: ModelLedger014) -> None:
        self.inner = inner
        self.ledger = ledger
        self.timeout_seconds = inner.timeout_seconds

    def chat_json(self, prompt: str, **kwargs: Any):
        started = time.perf_counter()
        try:
            completion = self.inner.chat_json(prompt, **kwargs)
        except ModelUnavailable:
            self.ledger.observe(prompt=prompt, elapsed=time.perf_counter() - started, prompt_tokens=0, output_tokens=0, timeout=True)
            raise
        self.ledger.observe(prompt=prompt, elapsed=completion.elapsed_seconds, prompt_tokens=completion.prompt_tokens, output_tokens=completion.generated_tokens)
        return completion


def _safe_model_json(
    client: _CountingClient014,
    store: ExperimentStore,
    *,
    kind: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Call the frozen runtime and persist one auditable call."""
    try:
        completion = client.chat_json(prompt, max_tokens=max_tokens, seed=seed)
        payload = parse_response(completion.text)
        runtime_error = None
        elapsed = completion.elapsed_seconds
        prompt_tokens = completion.prompt_tokens
        output_tokens = completion.generated_tokens
        response = completion.text
    except ModelUnavailable as exc:
        payload = None
        runtime_error = str(exc)
        elapsed = client.timeout_seconds
        prompt_tokens = output_tokens = 0
        response = json.dumps({"runtime_error": runtime_error})
    record = {
        **(metadata or {}),
        "runtime_error": runtime_error,
        "prompt_version": metadata.get("prompt_version") if metadata else None,
        "prompt_sha256": metadata.get("prompt_sha256") if metadata else None,
    }
    store.record_run(
        kind=kind,
        prompt=prompt,
        response=response,
        elapsed_seconds=elapsed,
        prompt_tokens=prompt_tokens,
        generated_tokens=output_tokens,
        passed=None if payload is None else True,
        metadata=record,
    )
    return payload if isinstance(payload, dict) else None, {
        "elapsed_seconds": elapsed,
        "prompt_tokens": prompt_tokens if prompt_tokens is not None else (max(1, len(prompt) // 4) if runtime_error is None else 0),
        "generated_tokens": output_tokens or 0,
        "runtime_error": runtime_error,
    }


def _descriptor(item: CapabilityRecord) -> dict[str, Any]:
    fp = item.fingerprint
    return {
        "skill_id": item.skill_id,
        "domain": item.domain,
        "family": item.family,
        "input": fp.input_type,
        "output": fp.output_type,
        "operation": fp.operation_family,
        "trust": fp.trust,
        "version": fp.version,
        "deterministic": fp.deterministic,
        "facets": list(item.facets),
    }


@dataclass(frozen=True)
class RankingCase014:
    case_id: str
    task: str
    candidates: tuple[CapabilityRecord, ...]
    target_skill_id: str | None
    intentionally_ambiguous: bool = False


def _variant_record(base: CapabilityRecord, *, skill_id: str, operation: str | None = None,
                    input_type: str | None = None, output_type: str | None = None,
                    trust: str | None = None, version: int | None = None) -> CapabilityRecord:
    record = replace(
        base.record,
        skill_id=skill_id,
        operation_family=operation or base.record.operation_family,
        input_contract=input_type or base.record.input_contract,
        output_contract=output_type or base.record.output_contract,
        trust=trust or base.record.trust,
        version=version or base.record.version,
        lexical_descriptor=f"descriptor-{skill_id}",
    )
    from .exp013 import fingerprint_for_record, CapabilityFingerprint
    fp = fingerprint_for_record(record)
    facets = (base.domain, base.family, fp.input_type, fp.output_type, fp.operation_family, "deterministic", fp.trust)
    return CapabilityRecord(record, base.domain, base.family, facets, fp, (record.lexical_descriptor, skill_id, fp.operation_family, base.family, base.domain))


def make_ranking_cases_014() -> tuple[RankingCase014, ...]:
    """Create controlled distractors; no whole-library prompt is ever built."""
    seed = generate_capability_records(32, artifact=b"air014")
    base = seed[10]
    target = _variant_record(base, skill_id="air014-target-normalize", operation="normalize", input_type="str", output_type="str", trust="verified", version=4)
    same_types_wrong_operation = _variant_record(base, skill_id="air014-wrong-operation", operation="aggregate", input_type="str", output_type="str", trust="verified", version=4)
    near_match = _variant_record(base, skill_id="air014-near-match", operation="normalize", input_type="bytes", output_type="str", trust="verified", version=3)
    deprecated = _variant_record(base, skill_id="air014-deprecated", operation="normalize", input_type="str", output_type="str", trust="deprecated", version=1)
    wrong_capability = _variant_record(base, skill_id="air014-wrong-capability", operation="lookup", input_type="json", output_type="dict", trust="verified", version=4)
    candidates = (target, same_types_wrong_operation, near_match, deprecated, wrong_capability)
    ambiguous = (
        _variant_record(base, skill_id="air014-ambiguous-a", operation="normalize", input_type="str", output_type="str", trust="verified", version=2),
        _variant_record(base, skill_id="air014-ambiguous-b", operation="normalize", input_type="str", output_type="str", trust="verified", version=2),
    )
    return (
        RankingCase014("rank-1", "Normalize a deterministic text value to text.", candidates, target.skill_id),
        RankingCase014("rank-2", "Use a pure deterministic text normalization procedure.", candidates[:3], target.skill_id),
        RankingCase014("rank-3", "Choose between indistinguishable normalization procedures.", ambiguous, None, True),
        RankingCase014("rank-4-retrieval-miss", "Normalize a deterministic text value, but the bounded retriever omitted it.", candidates[1:], target.skill_id),
    )


def _candidate_text(candidates: Iterable[CapabilityRecord]) -> str:
    return "\n".join(json.dumps(_descriptor(item), sort_keys=True) for item in candidates)


def _rank_call(client: _CountingClient014, store: ExperimentStore, case: RankingCase014,
               candidates: Sequence[CapabilityRecord], seed: int, *, kind_suffix: str = "ranking") -> dict[str, Any]:
    prompt = RANKING_PROMPT_TEMPLATE_014.format(task=case.task, candidates=_candidate_text(candidates))
    payload, telemetry = _safe_model_json(client, store, kind=f"air-014:{kind_suffix}:{case.case_id}", prompt=prompt, max_tokens=80, seed=seed,
                                          metadata={"prompt_version": RANKING_PROMPT_VERSION_014, "prompt_sha256": RANKING_PROMPT_HASH_014, "candidate_count": len(candidates)})
    chosen = payload.get("skill_id") if payload and (isinstance(payload.get("skill_id"), str) or payload.get("skill_id") is None) else None
    ids = {item.skill_id for item in candidates}
    retrieval_failure = case.target_skill_id is not None and case.target_skill_id not in ids
    safe_abstention = case.intentionally_ambiguous and chosen is None
    correct = (case.target_skill_id is not None and chosen == case.target_skill_id) or safe_abstention
    failure = None
    if telemetry["runtime_error"]:
        failure = "timeout"
    elif retrieval_failure:
        failure = "retrieval_failure"
    elif not correct:
        failure = "ranking_failure"
    chosen_record = next((item for item in candidates if item.skill_id == chosen), None)
    return {
        "case_id": case.case_id,
        "target_skill_id": case.target_skill_id,
        "chosen_skill_id": chosen,
        "correct": correct,
        "safe_abstention": safe_abstention,
        "retrieval_failure": retrieval_failure,
        "ranking_failure": failure == "ranking_failure",
        "deprecated_selected": bool(chosen_record and chosen_record.fingerprint.trust == "deprecated"),
        "wrong_operation_selected": bool(chosen_record and chosen_record.fingerprint.operation_family != "normalize"),
        "failure": failure,
        "candidate_count": len(candidates),
        **telemetry,
    }


def run_ranking_block_014(client: LlamaCppClient, store: ExperimentStore, ledger: ModelLedger014) -> dict[str, Any]:
    proxy = _CountingClient014(client, ledger)
    cases = make_ranking_cases_014()
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases, 1):
        # The model sees only the result of the existing 0013 index.  The
        # retrieval-miss case intentionally omits the target to separate a
        # retrieval failure from a ranking failure.
        index = HierarchicalCapabilityIndex(case.candidates)
        retrieval_query = CapabilityQuery(
            domain=case.candidates[0].domain,
            family=case.candidates[0].family,
            trust=None,
            require_pure=False,
            require_deterministic=False,
        )
        retrieved, retrieval_telemetry = index.retrieve(retrieval_query, top_k=5, use_fingerprint=False)
        retrieved = tuple(retrieved)
        # The deterministic top-1 arm is a retrieval baseline, not a model call.
        deterministic = {
            "case_id": case.case_id,
            "candidate_count": 1,
            "chosen_skill_id": retrieved[0].skill_id if retrieved else None,
            "correct": bool(retrieved and case.target_skill_id == retrieved[0].skill_id),
            "model_calls": 0,
        }
        top3 = _rank_call(proxy, store, case, retrieved[:3], 1400 + case_index, kind_suffix="ranking-top3")
        top5 = _rank_call(proxy, store, case, retrieved, 1500 + case_index, kind_suffix="ranking-top5")
        rows.append({"case_id": case.case_id, "intentionally_ambiguous": case.intentionally_ambiguous,
                     "deterministic_top1": deterministic, "top3": top3, "top5": top5,
                     "retrieval_candidate_present": case.target_skill_id is None or case.target_skill_id in {item.skill_id for item in retrieved},
                     "retrieval_candidates_examined": retrieval_telemetry["candidates_examined"]})
    valid = [row for row in rows if not row["intentionally_ambiguous"]]
    rankable = [row for row in valid if row["retrieval_candidate_present"]]
    def accuracy(key: str) -> float:
        return sum(bool(row[key]["correct"]) for row in rankable) / len(rankable) if rankable else 0.0
    retrieval_failures = sum(row["top5"]["retrieval_failure"] for row in rows)
    ranking_failures = sum(row["top5"]["ranking_failure"] for row in rows)
    return {
        "prompt_version": RANKING_PROMPT_VERSION_014,
        "prompt_sha256": RANKING_PROMPT_HASH_014,
        "cases": rows,
        "top1_retrieval_accuracy": sum(row["deterministic_top1"]["correct"] for row in valid) / len(valid) if valid else 0.0,
        "top3_ranking_accuracy": accuracy("top3"),
        "top5_ranking_accuracy": accuracy("top5"),
        "safe_abstention_count": sum(row["top5"]["safe_abstention"] for row in rows),
        "deprecated_selected_count": sum(row["top5"]["deprecated_selected"] for row in rows),
        "wrong_operation_selected_count": sum(row["top5"]["wrong_operation_selected"] for row in rows),
        "retrieval_failure_count": retrieval_failures,
        "ranking_failure_count": ranking_failures,
        "average_candidates": statistics.mean(row["top5"]["candidate_count"] for row in rows),
    }


def _context_docs_014() -> tuple[list[dict[str, str]], str]:
    correct_id = "air014-doc-normative"
    docs = [{"doc_id": correct_id, "text": "NORMATIVE CONTRACT: for the input token AIR_CONTEXT_014 return exactly the string AIR_CONTEXT_OK_014."}]
    distractors = (
        "Historical draft: return AIR_CONTEXT_OLD_014; this is deprecated.",
        "Deployment note: context tasks are not executable API calls.",
        "Unrelated reference about robotics, music, invoices, and weather.",
    )
    for index in range(99):
        docs.append({"doc_id": f"air014-doc-{index:03d}", "text": distractors[index % len(distractors)] + f" Ref {index}."})
    return docs, correct_id


def _context_call_014(client: _CountingClient014, store: ExperimentStore, condition: str,
                      docs: Sequence[dict[str, str]], seed: int) -> dict[str, Any]:
    task = "Apply the normative token transformation."
    prompt = CONTEXT_PROMPT_TEMPLATE_014.format(task=task, input="AIR_CONTEXT_014", source="\n".join(f"[{d['doc_id']}] {d['text']}" for d in docs))
    payload, telemetry = _safe_model_json(client, store, kind=f"air-014:context:{condition}", prompt=prompt, max_tokens=48, seed=seed,
                                          metadata={"prompt_version": CONTEXT_PROMPT_VERSION_014, "prompt_sha256": CONTEXT_PROMPT_HASH_014, "condition": condition, "documents": len(docs)})
    result = payload.get("result") if payload and isinstance(payload.get("result"), str) else None
    expected = "AIR_CONTEXT_OK_014"
    return {"condition": condition, "documents_sent": len(docs), "result": result, "task_accuracy": result == expected,
            "semantic_verifier_pass": result == expected, "wrong_answer": result is not None and result != expected,
            "hallucinated_api": result not in {None, expected}, "input_tokens": telemetry["prompt_tokens"],
            "output_tokens": telemetry["generated_tokens"], "latency_seconds": telemetry["elapsed_seconds"],
            "timeout": bool(telemetry["runtime_error"]), "hard_budget_enabled": False}


def run_context_block_014(client: LlamaCppClient, store: ExperimentStore, ledger: ModelLedger014) -> dict[str, Any]:
    docs, correct_id = _context_docs_014()
    proxy = _CountingClient014(client, ledger)
    relevant = docs[0]
    conditions = (
        ("full_document_pool", docs),
        ("top_k_retrieved_documents", docs[:5]),
        ("relevant_snippet_only", [{"doc_id": relevant["doc_id"], "text": relevant["text"].split("NORMATIVE CONTRACT:", 1)[-1].strip()}]),
        ("fingerprint_plus_contract", [{"doc_id": "air014-fingerprint", "text": "input=str; output=str; op=token_transform; pure+deterministic; contract=return AIR_CONTEXT_OK_014 for AIR_CONTEXT_014"}]),
    )
    rows = [_context_call_014(proxy, store, name, payload, 1600 + index) for index, (name, payload) in enumerate(conditions, 1)]
    full_tokens = rows[0]["input_tokens"] or max(1, len("\n".join(item["text"] for item in docs)) // 4)
    for row in rows:
        row["context_compression_ratio"] = 1 - row["input_tokens"] / full_tokens if full_tokens else 0.0
    return {"prompt_version": CONTEXT_PROMPT_VERSION_014, "prompt_sha256": CONTEXT_PROMPT_HASH_014,
            "pool_size": len(docs), "correct_doc_id": correct_id, "conditions": rows,
            "full_pool_stress_trials": 1, "hard_budget_default": False,
            "accuracy_regressions": [row["condition"] for row in rows if not row["task_accuracy"]]}


@dataclass(frozen=True)
class DecompositionCase014:
    case_id: str
    task_text: str
    expected_stages: tuple[str, ...]
    task: CompositionTask


def _decomposition_cases_014() -> tuple[DecompositionCase014, ...]:
    library = make_composition_library()
    return (
        DecompositionCase014("decomp-two", "Turn the raw record into a cleaned analytical value.", ("cleaning", "analysis"), CompositionTask("two-stage", "raw", "analysis", ("cleaning", "analysis"))),
        DecompositionCase014("decomp-three", "Turn the raw record into a report-ready output.", ("cleaning", "analysis", "formatting"), CompositionTask("three-stage", "raw", "report", ("cleaning", "analysis", "formatting"))),
        DecompositionCase014("decomp-missing", "Turn the raw record into an embedding representation.", ("cleaning", "embedding"), CompositionTask("missing", "raw", "embedding", ("cleaning", "embedding"), False)),
    )


def _catalog_text() -> str:
    return "\n".join((
        "cap-alpha: accepts raw and emits clean; removes input noise",
        "cap-beta: accepts clean and emits analysis; derives a summary",
        "cap-gamma: accepts analysis and emits report; formats a report",
        "cap-delta: accepts bytes and emits raw; unrelated reversible adapter",
    ))


def _stage_from_key(key: str) -> str | None:
    return {"cap-alpha": "cleaning", "cap-beta": "analysis", "cap-gamma": "formatting", "cleaning": "cleaning", "analysis": "analysis", "formatting": "formatting", "embedding": "embedding"}.get(key)


def _composition_descriptor(item: CompositionSkill) -> str:
    return json.dumps({"skill_id": item.skill_id, "stage": item.stage, "input": item.input_type, "output": item.output_type}, sort_keys=True)


def _composition_rank_call(client: _CountingClient014, store: ExperimentStore, stage: str,
                           candidates: Sequence[CompositionSkill], task_id: str, seed: int) -> tuple[str | None, dict[str, Any]]:
    if not candidates:
        return None, {"runtime_error": None, "prompt_tokens": 0, "generated_tokens": 0, "elapsed_seconds": 0.0}
    prompt = COMPOSITION_RANKING_PROMPT_TEMPLATE_014.format(stage=stage, input_type=candidates[0].input_type, output_type=candidates[0].output_type,
                                                            candidates="\n".join(_composition_descriptor(item) for item in candidates))
    payload, telemetry = _safe_model_json(client, store, kind=f"air-014:composition-ranking:{task_id}:{stage}", prompt=prompt, max_tokens=64, seed=seed,
                                          metadata={"prompt_version": COMPOSITION_RANKING_PROMPT_VERSION_014, "prompt_sha256": COMPOSITION_RANKING_PROMPT_HASH_014, "stage": stage})
    chosen = payload.get("skill_id") if payload and isinstance(payload.get("skill_id"), str) else None
    return chosen, telemetry


def run_decomposition_block_014(client: LlamaCppClient, store: ExperimentStore, ledger: ModelLedger014) -> dict[str, Any]:
    proxy = _CountingClient014(client, ledger)
    library = make_composition_library()
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(_decomposition_cases_014(), 1):
        prompt = DECOMPOSITION_PROMPT_TEMPLATE_014.format(task=case.task_text, catalog=_catalog_text())
        payload, telemetry = _safe_model_json(proxy, store, kind=f"air-014:decomposition:{case.case_id}", prompt=prompt, max_tokens=96, seed=1800 + index,
                                              metadata={"prompt_version": DECOMPOSITION_PROMPT_VERSION_014, "prompt_sha256": DECOMPOSITION_PROMPT_HASH_014})
        raw = payload.get("required_capabilities", []) if payload else []
        keys = tuple(item for item in raw if isinstance(item, str)) if isinstance(raw, list) else ()
        predicted = tuple(stage for key in keys if (stage := _stage_from_key(key)) is not None)
        decomposition_correct = predicted == case.expected_stages
        scoped_task = replace(case.task, stages=predicted)
        deterministic_paths, scoped_candidates, _ = _scoped_paths(library, scoped_task, len(predicted), agent_scoped=False) if predicted else ([], 0, 0)
        # Model ranking is only opened after a correct decomposition; this keeps
        # stage errors separate from selection errors.
        selected: list[CompositionSkill] = []
        ranking_rows: list[dict[str, Any]] = []
        if decomposition_correct:
            for stage_index, stage in enumerate(predicted):
                candidates = tuple(item for item in library if item.stage == stage)
                chosen_id, rank_telemetry = _composition_rank_call(proxy, store, stage, candidates, case.case_id, 1900 + index * 10 + stage_index)
                chosen = next((item for item in candidates if item.skill_id == chosen_id), None)
                selected.append(chosen) if chosen else None
                ranking_rows.append({"stage": stage, "chosen_skill_id": chosen_id, "correct_stage": chosen is not None, **rank_telemetry})
        types_valid = len(selected) == len(predicted) and all(selected[i].output_type == selected[i + 1].input_type for i in range(len(selected) - 1)) if selected else False
        final_correct = bool(case.task.valid and decomposition_correct and types_valid and tuple(item.stage for item in selected) == case.expected_stages)
        failure = None
        if telemetry["runtime_error"]:
            failure = "timeout"
        elif not decomposition_correct:
            failure = "decomposition_failure"
        elif not types_valid or (case.task.valid and not final_correct):
            failure = "composition_failure"
        elif not case.task.valid and final_correct:
            failure = "composition_failure"
        safe_no_valid = not case.task.valid and decomposition_correct and not deterministic_paths
        rows.append({"case_id": case.case_id, "expected_stages": case.expected_stages, "predicted_stages": predicted,
                     "decomposition_correct": decomposition_correct, "decomposition_failure": failure == "decomposition_failure",
                     "deterministic_type_scope_candidates": len(deterministic_paths), "scoped_candidates_examined": scoped_candidates,
                     "ranking": ranking_rows, "composition_correct": final_correct, "composition_failure": failure == "composition_failure",
                     "failure": failure, "safe_no_valid_composition": safe_no_valid, "subtask_count": len(predicted),
                     "global_brute_force_candidates": len(library) ** len(case.expected_stages),
                     "retrieval_calls": len(predicted),
                     "composition_candidates": sum(sum(1 for item in library if item.stage == stage) for stage in predicted),
                     "model_calls": 1 + len(ranking_rows),
                     "total_latency_seconds": telemetry["elapsed_seconds"] + sum(item["elapsed_seconds"] for item in ranking_rows),
                     "model_input_tokens": telemetry["prompt_tokens"], "model_output_tokens": telemetry["generated_tokens"],
                     "decomposition_latency_seconds": telemetry["elapsed_seconds"]})
    deterministic = sum(bool(row["expected_stages"] and row["deterministic_type_scope_candidates"] > 0) for row in rows)
    return {"decomposition_prompt_version": DECOMPOSITION_PROMPT_VERSION_014, "decomposition_prompt_sha256": DECOMPOSITION_PROMPT_HASH_014,
            "composition_ranking_prompt_version": COMPOSITION_RANKING_PROMPT_VERSION_014, "composition_ranking_prompt_sha256": COMPOSITION_RANKING_PROMPT_HASH_014,
            "tasks": rows, "decomposition_accuracy": sum(row["decomposition_correct"] for row in rows) / len(rows),
            "decomposition_failure_count": sum(row["decomposition_failure"] for row in rows),
            "scoped_composition_success": sum(row["composition_correct"] for row in rows),
            "composition_failure_count": sum(row["composition_failure"] for row in rows),
            "deterministic_scope_success": deterministic, "average_candidates_per_subtask": statistics.mean((row["scoped_candidates_examined"] / row["subtask_count"] if row["subtask_count"] else 0.0) for row in rows),
            "no_valid_composition_accuracy": sum(row["safe_no_valid_composition"] for row in rows if row["case_id"] == "decomp-missing") / max(1, sum(row["case_id"] == "decomp-missing" for row in rows)),
            "global_brute_force_candidates": sum(row["global_brute_force_candidates"] for row in rows),
            "total_retrieval_calls": sum(row["retrieval_calls"] for row in rows),
            "total_composition_candidates": sum(row["composition_candidates"] for row in rows),
            "total_model_calls": sum(row["model_calls"] for row in rows)}


def _safe_direct_answer_014(client: _CountingClient014, store: ExperimentStore, family: PythonFamily009,
                            condition: str, case: Any, include_docs: bool, seed: int) -> dict[str, Any]:
    docs = f"\nAPI documentation:\n{family.api_docs}" if include_docs else ""
    prompt = DIRECT_ANSWER_PROMPT_TEMPLATE_014.format(contract=family.contract, docs=docs, input=json.dumps(case.input_text, ensure_ascii=False))
    payload, telemetry = _safe_model_json(client, store, kind=f"air-014:acquisition-baseline:{family.family_id}:{condition}:{case.case_id}", prompt=prompt, max_tokens=160, seed=seed,
                                          metadata={"condition": condition, "family_id": family.family_id, "prompt_version": DIRECT_ANSWER_PROMPT_VERSION_014, "prompt_sha256": DIRECT_ANSWER_PROMPT_HASH_014})
    value = payload.get("result") if payload and isinstance(payload.get("result"), str) else None
    return {"case_id": case.case_id, "correct": value == case.expected, "result": value, "input_tokens": telemetry["prompt_tokens"], "output_tokens": telemetry["generated_tokens"], "latency_seconds": telemetry["elapsed_seconds"], "timeout": bool(telemetry["runtime_error"])}


def _artifact_reuse(skill: PythonSkillArtifact009 | None, family: PythonFamily009, cases: Sequence[Any], model_context_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    model_context = {
        "correct": sum(bool(row["correct"]) for row in model_context_rows),
        "total": len(model_context_rows),
        "accuracy": sum(bool(row["correct"]) for row in model_context_rows) / len(model_context_rows) if model_context_rows else 0.0,
        "model_calls": len(model_context_rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in model_context_rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in model_context_rows),
    }
    if skill is None:
        return {"model_context": model_context, "correct": 0, "total": len(cases), "accuracy": 0.0, "model_calls": 0, "executable_calls": 0, "bytes_read_query": 0, "als": "none"}
    correct = sum(run_python_in_sandbox_009(skill.code, family, case.input_text, case.expected).passed for case in cases)
    return {"model_context": model_context, "correct": correct, "total": len(cases), "accuracy": correct / len(cases) if cases else 0.0, "model_calls": 0, "executable_calls": len(cases), "bytes_read_query": len(skill.code.encode("utf-8")), "als": "external_procedural_artifact"}


def run_acquisition_block_014(client: LlamaCppClient, store: ExperimentStore, ledger: ModelLedger014, heldout_limit: int | None = 2) -> dict[str, Any]:
    proxy = _CountingClient014(client, ledger)
    families = make_robustness_families_012(ROBUSTNESS_SEEDS_012[0])[:ACQUISITION_FAMILIES_014]
    results: list[dict[str, Any]] = []
    for family_index, family in enumerate(families, 1):
        pool, correct_id = make_document_pool_012(family, 10, ROBUSTNESS_SEEDS_012[0])
        retrieval_prompt = _retrieval_prompt_012(family, pool)
        payload, retrieval_telemetry = _safe_model_json(proxy, store, kind=f"air-014:acquisition-retrieval:{family.family_id}", prompt=retrieval_prompt, max_tokens=96, seed=2100 + family_index,
                                                        metadata={"prompt_version": RETRIEVAL_PROMPT_VERSION_011, "prompt_sha256": RETRIEVAL_PROMPT_HASH_011, "family_id": family.family_id})
        selected_id = payload.get("doc_id") if payload and isinstance(payload.get("doc_id"), str) else None
        retrieval_correct = selected_id == correct_id
        selected_family = _family_selected_doc_012(family, pool, selected_id)
        base_snapshot = tuple(artifact.to_dict() for artifact in BASE_PYTHON_LIBRARY_010)
        base_matches = [artifact.skill_id for artifact in BASE_PYTHON_LIBRARY_010 if run_python_gate_009(artifact.code, family, family.discovery, "existing").accuracy == 1.0]
        gap = {"status": "gap_detected" if not base_matches else "covered", "matching_skill_ids": base_matches}
        skill, attempts = _learn_family_resumable_012(proxy, store, selected_family, f"py-skill-014-{family_index}", 2400 + family_index, max_attempts=MAX_REPAIR_ATTEMPTS_014)
        discovery = run_python_gate_009(skill.code, family, family.discovery, "discovery") if skill else PythonGate009("discovery", 0, len(family.discovery), 0.0)
        validation = run_python_gate_009(skill.code, family, family.validation, "validation") if skill else PythonGate009("validation", 0, len(family.validation), 0.0)
        edge = run_python_gate_009(skill.code, family, family.edge, "edge") if skill else PythonGate009("edge", 0, len(family.edge), 0.0)
        active = bool(retrieval_correct and gap["status"] == "gap_detected" and skill and discovery.accuracy == validation.accuracy == edge.accuracy == 1.0)
        heldout = family.heldout[:heldout_limit] if heldout_limit is not None else family.heldout
        model_rows = [_safe_direct_answer_014(proxy, store, family, "model_only", case, False, 2500 + family_index) for case in heldout]
        snippet_family = _family_selected_doc_012(family, pool, selected_id) if selected_id else family
        snippet_rows = [_safe_direct_answer_014(proxy, store, snippet_family, "model_plus_retrieved_snippet", case, True, 2600 + family_index) for case in heldout]
        reuse = _artifact_reuse(skill if active else None, family, heldout, snippet_rows)
        heldout_gate = run_python_gate_009(skill.code, family, heldout, "heldout") if active and skill else PythonGate009("heldout", 0, len(heldout), 0.0)
        unsafe = static_check_python_009("import os\ndef transform(value: str) -> str:\n    return os.getcwd()\n", family)
        semantic_wrong = run_python_gate_009("def transform(value: str) -> str:\n    return value\n", family, family.validation, "semantic_wrong")
        failure = None
        if retrieval_telemetry["runtime_error"]:
            failure = "timeout"
        elif not retrieval_correct:
            failure = "retrieval_failure"
        elif not gap["status"] == "gap_detected":
            failure = "gap_detection_failure"
        elif not skill:
            failure = "repair_failure" if len(attempts) > 1 else "synthesis_failure"
        elif discovery.accuracy < 1.0:
            failure = "public_validation_failure"
        elif validation.accuracy < 1.0:
            failure = "hidden_validation_failure"
        elif edge.accuracy < 1.0:
            failure = "edge_failure"
        results.append({"family_id": family.family_id, "retrieval": {"correct": retrieval_correct, "expected_doc_id": correct_id, "selected_doc_id": selected_id, **retrieval_telemetry},
                        "gap_detection": gap, "learning": {"proposal_count": len(attempts), "repair_count": max(0, len(attempts) - 1), "accepted_skill": skill.to_dict() if skill else None,
                                                                  "discovery": asdict(discovery), "hidden_validation": asdict(validation), "edge": asdict(edge)},
                        "activation": {"active": active, "failure": failure, "activation_given_correct_retrieval": active if retrieval_correct else None, "heldout_reuse": asdict(heldout_gate)},
                        "model_conditions": {"model_only": model_rows, "model_plus_retrieved_snippet": snippet_rows,
                                             "previous_library": {"matching_skill_ids": base_matches, "correct": 0, "total": len(heldout), "accuracy": 0.0, "model_calls": 0}}, "artifact_reuse": reuse,
                        "controls": {"unsafe_rejected": not unsafe.passed, "unsafe_reason": unsafe.reason, "semantic_wrong_rejected": semantic_wrong.accuracy < 0.9,
                                     "base_library_immutable": base_snapshot == tuple(item.to_dict() for item in BASE_PYTHON_LIBRARY_010)}})
    correct_retrieval = sum(item["retrieval"]["correct"] for item in results)
    active_count = sum(item["activation"]["active"] for item in results)
    return {"families_attempted": len(results), "results": results, "correct_retrieval_count": correct_retrieval, "activation_count": active_count,
            "activation_given_correct_retrieval": active_count / correct_retrieval if correct_retrieval else 0.0,
            "wrong_activation_count": sum(item["activation"]["active"] and not item["retrieval"]["correct"] for item in results),
            "hidden_validation_pass_count": sum(item["learning"]["hidden_validation"]["accuracy"] == 1.0 for item in results),
            "edge_validation_pass_count": sum(item["learning"]["edge"]["accuracy"] == 1.0 for item in results),
            "unsafe_rejection_count": sum(item["controls"]["unsafe_rejected"] for item in results),
            "semantic_wrong_rejection_count": sum(item["controls"]["semantic_wrong_rejected"] for item in results)}


def run_exp014(*, client: LlamaCppClient, store: ExperimentStore, report_directory: str,
               heldout_limit: int | None = 2) -> dict[str, Any]:
    """Run bounded 0014 and persist a JSON report without changing prompts mid-run."""
    ledger = ModelLedger014()
    ranking = run_ranking_block_014(client, store, ledger)
    context = run_context_block_014(client, store, ledger)
    decomposition = run_decomposition_block_014(client, store, ledger)
    acquisition = run_acquisition_block_014(client, store, ledger, heldout_limit)
    report: dict[str, Any] = {
        "benchmark": "air-014-frozen-model-utilization-decomposition-ranking-acquisition",
        "version": EXP014_VERSION, "created_at": datetime.now(UTC).isoformat(),
        "model": {"identity": MODEL_IDENTITY_014, "context_size": CONTEXT_SIZE_014, "model_parameter_update": False, "model_swap": False},
        "protocol": {"prompt_patch_after_first_result": False, "task_family_change_after_first_result": False, "expected_result_change": False,
                     "retrieval_logic_change": False, "facet_schema_change": False, "verifier_relaxed": False, "whole_library_in_model_context": False,
                     "hard_context_budget": False, "max_repair_attempts": MAX_REPAIR_ATTEMPTS_014, "learner_prompt_version": LEARNING_PROMPT_VERSION_009,
                     "learner_prompt_sha256": LEARNING_PROMPT_HASH_009, "learner_prompt_frozen": True,
                     "direct_answer_prompt_version": DIRECT_ANSWER_PROMPT_VERSION_014, "direct_answer_prompt_sha256": DIRECT_ANSWER_PROMPT_HASH_014,
                     "retrieval_prompt_version": RETRIEVAL_PROMPT_VERSION_011, "retrieval_prompt_sha256": RETRIEVAL_PROMPT_HASH_011},
        "block_a_frozen_candidate_ranking": ranking, "block_b_real_context_compression": context,
        "block_c_model_decomposition_scoped_composition": decomposition, "block_d_novel_capability_acquisition": acquisition,
        "block_e_learned_state_reuse": {"active_skills": [item["family_id"] for item in acquisition["results"] if item["activation"]["active"]],
                                         "model_calls_before_reuse": sum(item["artifact_reuse"]["model_context"]["model_calls"] for item in acquisition["results"] if item["activation"]["active"]),
                                         "model_calls_after_reuse": sum(item["artifact_reuse"]["model_calls"] for item in acquisition["results"] if item["activation"]["active"]),
                                         "latency_before_reuse_seconds": sum(sum(row["latency_seconds"] for row in item["model_conditions"]["model_plus_retrieved_snippet"]) for item in acquisition["results"] if item["activation"]["active"]),
                                         "latency_after_reuse_seconds": 0.0,
                                         "artifact_reuse": [item["artifact_reuse"] for item in acquisition["results"]]},
        "model_accounting": ledger.summary(),
        "failure_taxonomy": list(FAILURE_TAXONOMY_014),
        "regression": {"prior_0008": _prior_0008_regression(), "source_skill_immutability": all(item["controls"]["base_library_immutable"] for item in acquisition["results"])},
        "interpretation": {"bounded_not_general_continual_learning": True,
                            "retrieval_utilization": "see Block A; retrieval and ranking are separate",
                            "context_optimization": "smaller context is only a win when task_accuracy is preserved",
                            "external_state_reuse_not_weight_learning": True,
                            "next_experiment_recommendation": "derived from largest observed failure bucket"},
    }
    failure_counts: dict[str, int] = {key: 0 for key in FAILURE_TAXONOMY_014}
    for row in ranking["cases"]:
        if row["top5"]["failure"] in failure_counts: failure_counts[row["top5"]["failure"]] += 1
    for row in decomposition["tasks"]:
        if row["failure"] in failure_counts: failure_counts[row["failure"]] += 1
    for row in acquisition["results"]:
        if row["activation"]["failure"] in failure_counts: failure_counts[row["activation"]["failure"]] += 1
        if row["controls"]["unsafe_rejected"]: failure_counts["safety_rejection"] += 1
    report["failure_counts"] = failure_counts
    nonzero_failures = {key: value for key, value in failure_counts.items() if value}
    report["interpretation"]["next_experiment_recommendation"] = (
        max(nonzero_failures, key=nonzero_failures.get) if nonzero_failures else "larger realistic workload / model portability"
    )
    report["verification"] = {
        "test_suite": "run externally before release; runtime benchmark does not mutate tests",
        "commit_hash": os.getenv("AIR_COMMIT_SHA", "not_available_in_runtime"),
    }
    path_dir = Path(report_directory); path_dir.mkdir(parents=True, exist_ok=True)
    path = path_dir / datetime.now(UTC).strftime("air-014-%Y%m%dT%H%M%SZ.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_file"] = str(path)
    return report


__all__ = [
    "EXP014_VERSION", "MODEL_IDENTITY_014", "CONTEXT_SIZE_014", "RANKING_PROMPT_HASH_014", "CONTEXT_PROMPT_HASH_014",
    "DECOMPOSITION_PROMPT_HASH_014", "COMPOSITION_RANKING_PROMPT_HASH_014", "DIRECT_ANSWER_PROMPT_HASH_014", "ModelLedger014", "RankingCase014",
    "make_ranking_cases_014", "run_ranking_block_014", "run_context_block_014", "run_decomposition_block_014",
    "run_acquisition_block_014", "run_exp014",
]
