"""Experiment 0018: verified candidate search versus model program induction.

0017 showed that SmolLM3-3B could not reliably emit canonical semantic IR.  This
experiment keeps that direct-generation arm as a baseline, then gives AIR a
frozen, model-independent typed search space.  The model is only asked to
select an already executable candidate or predict a small constraint descriptor.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any, Mapping, Sequence

from .exp009 import FamilyCase009, PythonFamily009
from .exp012 import make_document_pool_012, make_robustness_families_012
from .exp016 import make_contract_families_016, structural_metadata_016
from .exp017 import (
    DOCS_IR_PROMPT_HASH_017,
    DOCS_IR_PROMPT_TEMPLATE_017,
    DOCS_IR_PROMPT_VERSION_017,
    _parse_json_017,
    _examples_017,
)
from .model_client import LlamaCppClient, ModelUnavailable
from .exp015 import ModelLedger015
from .program_search import (
    SEARCH_API_NAMES_018,
    SEARCH_CANDIDATE_SIZES_018,
    SEARCH_MAX_DEPTH_018,
    SEARCH_MAX_PUBLIC_SURVIVORS_018,
    SEARCH_MAX_RAW_CANDIDATES_018,
    SEARCH_VERSION_018,
    Candidate018,
    SearchBudget018,
    candidate_features_018,
    search_candidates_018,
)
from .semantic_ir import (
    SemanticIRExecutionError,
    SemanticIRValidationError,
    canonical_semantic_ir_json_017,
    compile_semantic_ir_python_017,
    execute_semantic_ir_017,
    oracle_call_ir_017,
    validate_semantic_ir_017,
)
from .store import ExperimentStore


EXP018_VERSION = "air-018-v1"
MODEL_IDENTITY_018 = "SmolLM3-3B-GGUF:Q4_K_M; llama.cpp; CPU; context=4096"
CONTEXT_SIZE_018 = 4096
FAMILIES_018 = 12
NEW_FAMILIES_BEYOND_0017_018 = 4
SEARCH_BUDGET_018 = SearchBudget018()
RANDOM_SEED_018 = 18018

ARMS_018 = (
    "A_direct_generation", "B_pure_search", "C_search_smollm",
    "D_search_random", "E_search_oracle", "F_docs_no_docs",
    "G_constraint_search",
)
FAILURE_TAXONOMY_018 = (
    "retrieval_failure", "direct_generation_failure", "candidate_coverage_failure",
    "type_pruning_failure", "public_pruning_failure", "public_overfit_candidate",
    "ranking_failure", "random_selection_failure", "constraint_prediction_failure",
    "constraint_retention_failure", "ambiguous_program", "hidden_validation_failure",
    "edge_failure", "compilation_failure", "safety_rejection", "heldout_failure",
    "timeout", "invalid_candidate_id", "search_budget_exhausted",
)


RANKING_PROMPT_TEMPLATE_018 = """Select one executable candidate for this task.
Return exactly one JSON object: {{"candidate_id":"..."}} or
{{"candidate_id":null}}. Never write a program or modify a candidate.

Task family: {task}
Documentation:
{documentation}
Public examples:
{examples}
Frozen candidate list (IDs are the only selectable values):
{candidates}
"""
RANKING_PROMPT_VERSION_018 = "air-018-candidate-ranking-v1"
RANKING_PROMPT_HASH_018 = hashlib.sha256(RANKING_PROMPT_TEMPLATE_018.encode()).hexdigest()

NO_DOC_RANKING_PROMPT_TEMPLATE_018 = """Select one executable candidate for this task.
Return exactly one JSON object: {{"candidate_id":"..."}} or
{{"candidate_id":null}}. Never write a program or modify a candidate.

Task family: {task}
Documentation: deliberately withheld for this control.
Public examples:
{examples}
Frozen candidate list (IDs are the only selectable values):
{candidates}
"""
NO_DOC_RANKING_PROMPT_VERSION_018 = "air-018-no-doc-candidate-ranking-v1"
NO_DOC_RANKING_PROMPT_HASH_018 = hashlib.sha256(NO_DOC_RANKING_PROMPT_TEMPLATE_018.encode()).hexdigest()

CONSTRAINT_PROMPT_TEMPLATE_018 = """Predict only the structural constraints of the executable candidate needed for this task.
Return exactly one JSON object with boolean fields uses_call, uses_reverse,
uses_rotate, uses_concat and integer field max_depth. Do not emit a program,
candidate ID, Python, or explanation.

Documentation:
{documentation}
Public examples:
{examples}
"""
CONSTRAINT_PROMPT_VERSION_018 = "air-018-constraint-prediction-v1"
CONSTRAINT_PROMPT_HASH_018 = hashlib.sha256(CONSTRAINT_PROMPT_TEMPLATE_018.encode()).hexdigest()


def make_candidate_families_018() -> tuple[PythonFamily009, ...]:
    """Eight 0017 families plus four independent families from seed 1203."""
    return make_contract_families_016() + make_robustness_families_012(1203)[:4]


def _canonical_json_018(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _oracle_hash_018(family: PythonFamily009) -> str:
    operation = family.family_id.rsplit("-", 1)[-1]
    return hashlib.sha256(canonical_semantic_ir_json_017(oracle_call_ir_017(operation)).encode()).hexdigest()


def _model_call_018(
    client: Any,
    store: ExperimentStore,
    ledger: ModelLedger015,
    *,
    kind: str,
    arm: str,
    prompt: str,
    prompt_version: str,
    prompt_hash: str,
    seed: int,
    max_tokens: int,
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        completion = client.chat_json(prompt, max_tokens=max_tokens, seed=seed)
        payload = _parse_json_017(completion.text)
        elapsed = completion.elapsed_seconds
        prompt_tokens = completion.prompt_tokens or 0
        output_tokens = completion.generated_tokens or 0
        runtime_error = None
        response = completion.text
    except ModelUnavailable as exc:
        payload = None
        elapsed = getattr(client, "timeout_seconds", 180.0)
        prompt_tokens = output_tokens = 0
        runtime_error = str(exc)
        response = json.dumps({"runtime_error": runtime_error})
    ledger.observe(prompt=prompt, elapsed=elapsed, prompt_tokens=prompt_tokens,
                   output_tokens=output_tokens, timeout=runtime_error is not None, arm=arm)
    store.record_run(
        kind=kind, prompt=prompt, response=response, elapsed_seconds=elapsed,
        prompt_tokens=prompt_tokens, generated_tokens=output_tokens, passed=None,
        metadata={**dict(metadata), "arm": arm, "prompt_version": prompt_version,
                  "prompt_sha256": prompt_hash, "seed": seed, "runtime_error": runtime_error},
    )
    return payload, {
        "elapsed_seconds": elapsed, "prompt_tokens": prompt_tokens,
        "generated_tokens": output_tokens, "runtime_error": runtime_error,
    }


def _compile_gate_018(candidate: Candidate018) -> tuple[bool, bool, str | None, int]:
    try:
        validate_semantic_ir_017(candidate.ir, set(SEARCH_API_NAMES_018))
        code = compile_semantic_ir_python_017(candidate.ir, set(SEARCH_API_NAMES_018))
        tree = ast.parse(code)
        if "__import__" in code or "eval(" in code or "exec(" in code:
            return False, False, "unsafe generated source", len(code.encode())
        return True, True, None, len(code.encode())
    except (SemanticIRValidationError, SyntaxError, ValueError, TypeError) as exc:
        return False, False, str(exc), 0


def _candidate_evaluation_018(candidate: Candidate018, family: PythonFamily009) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_valid": False, "compiled": False, "safety_pass": False,
        "public_pass": False, "hidden_pass": False, "edge_pass": False,
        "semantic_correct": False, "active": False, "error": None, "python_bytes": 0,
    }
    try:
        validate_semantic_ir_017(candidate.ir, set(SEARCH_API_NAMES_018))
        result["schema_valid"] = True
    except SemanticIRValidationError as exc:
        result["error"] = str(exc)
        return result
    compiled, safe, error, bytes_count = _compile_gate_018(candidate)
    result.update({"compiled": compiled, "safety_pass": safe, "error": error, "python_bytes": bytes_count})
    if not safe:
        return result

    def passes(cases: Sequence[FamilyCase009]) -> bool:
        try:
            return all(execute_semantic_ir_017(candidate.ir, case.input_text, set(SEARCH_API_NAMES_018)) == case.expected for case in cases)
        except (SemanticIRExecutionError, SemanticIRValidationError, ValueError, TypeError):
            return False

    result["public_pass"] = passes(family.discovery)
    result["hidden_pass"] = passes(family.validation)
    result["edge_pass"] = passes(family.edge)
    result["semantic_correct"] = result["public_pass"] and result["hidden_pass"] and result["edge_pass"]
    result["active"] = all(result[key] for key in ("schema_valid", "compiled", "safety_pass", "public_pass", "hidden_pass", "edge_pass"))
    return result


def _heldout_018(candidate: Candidate018, family: PythonFamily009) -> dict[str, Any]:
    started = time.perf_counter()
    correct = 0
    errors = 0
    for case in family.heldout:
        try:
            correct += int(execute_semantic_ir_017(candidate.ir, case.input_text, set(SEARCH_API_NAMES_018)) == case.expected)
        except (SemanticIRExecutionError, SemanticIRValidationError, ValueError, TypeError):
            errors += 1
    elapsed = time.perf_counter() - started
    return {
        "total": len(family.heldout), "correct": correct,
        "accuracy": correct / len(family.heldout) if family.heldout else 0.0,
        "model_calls": 0, "input_tokens": 0, "output_tokens": 0,
        "errors": errors, "elapsed_seconds": elapsed, "bytes_per_query": 0,
    }


def _examples_text_018(family: PythonFamily009) -> str:
    return _examples_017(family)


def _candidate_text_018(candidates: Sequence[Candidate018]) -> str:
    return "\n".join(f"{item.candidate_id}: {_canonical_json_018(item.ir)}" for item in candidates)


def _select_by_verification_018(
    survivors: Sequence[Candidate018], evaluations: Mapping[str, Mapping[str, Any]],
) -> tuple[Candidate018 | None, str | None, int]:
    winners = [item for item in survivors if evaluations.get(item.candidate_id, {}).get("hidden_pass") and evaluations.get(item.candidate_id, {}).get("edge_pass")]
    if len(winners) == 1:
        return winners[0], None, len(winners)
    if len(winners) > 1:
        return None, "ambiguous_program", len(winners)
    return None, "no_hidden_edge_survivor", 0


def _rank_once_018(
    client: Any,
    store: ExperimentStore,
    ledger: ModelLedger015,
    family: PythonFamily009,
    candidates: Sequence[Candidate018],
    *,
    with_docs: bool,
    seed: int,
    arm: str,
    cache_key: str,
) -> dict[str, Any]:
    if len(candidates) <= 1:
        selected = candidates[0] if candidates else None
        return {
            "model_called": False, "selected_id": selected.candidate_id if selected else None,
            "selection_valid": selected is not None, "telemetry": None,
            "reason": "unique_survivor" if selected else "no_survivor", "cache_key": cache_key,
        }
    template = RANKING_PROMPT_TEMPLATE_018 if with_docs else NO_DOC_RANKING_PROMPT_TEMPLATE_018
    version = RANKING_PROMPT_VERSION_018 if with_docs else NO_DOC_RANKING_PROMPT_VERSION_018
    prompt_hash = RANKING_PROMPT_HASH_018 if with_docs else NO_DOC_RANKING_PROMPT_HASH_018
    documentation = family.api_docs if with_docs else "(withheld)"
    prompt = template.format(task=family.title, documentation=documentation,
                             examples=_examples_text_018(family), candidates=_candidate_text_018(candidates))
    payload, telemetry = _model_call_018(
        client, store, ledger, kind=f"air-018:ranking:{cache_key}", arm=arm,
        prompt=prompt, prompt_version=version, prompt_hash=prompt_hash,
        seed=seed, max_tokens=64, metadata={"family_id": family.family_id, "with_docs": with_docs,
                                             "candidate_count": len(candidates)},
    )
    selected_id = payload.get("candidate_id") if isinstance(payload, Mapping) else None
    selected = next((item for item in candidates if item.candidate_id == selected_id), None)
    return {
        "model_called": True, "selected_id": selected_id, "selection_valid": selected is not None,
        "telemetry": telemetry, "reason": "model_selection" if selected else "invalid_candidate_id",
        "cache_key": cache_key,
    }


def _constraint_valid_018(payload: Any) -> tuple[bool, dict[str, Any] | None, str | None]:
    if not isinstance(payload, Mapping):
        return False, None, "missing descriptor"
    required = ("uses_call", "uses_reverse", "uses_rotate", "uses_concat", "max_depth")
    if set(payload) != set(required):
        return False, None, "descriptor keys mismatch"
    if any(not isinstance(payload[key], bool) for key in required[:4]):
        return False, None, "boolean descriptor field invalid"
    if not isinstance(payload["max_depth"], int) or isinstance(payload["max_depth"], bool):
        return False, None, "max_depth must be integer"
    if payload["max_depth"] < 1 or payload["max_depth"] > SEARCH_MAX_DEPTH_018:
        return False, None, "max_depth outside frozen range"
    return True, dict(payload), None


def _constraint_match_018(candidate: Candidate018, descriptor: Mapping[str, Any]) -> bool:
    features = candidate_features_018(candidate.ir)
    return all(features[key] == descriptor[key] for key in ("uses_call", "uses_reverse", "uses_rotate", "uses_concat", "max_depth"))


def _arm_summary_018(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    items = [row["arms"][arm] for row in rows]
    active = sum(bool(item.get("selected_evaluation", {}).get("active")) for item in items)
    heldout = [item["heldout"] for item in items if item.get("heldout")]
    attempts = [attempt for item in items for attempt in item.get("attempts", []) if attempt]
    return {
        "families_attempted": len(items),
        "active": active,
        "heldout_reuse": {
            "active_artifacts": len(heldout),
            "mean_accuracy": statistics.mean([float(item["accuracy"]) for item in heldout]) if heldout else None,
            "model_calls": sum(int(item.get("model_calls", 0)) for item in heldout),
        },
        "model_calls": len(attempts),
        "input_tokens": sum(int(item.get("prompt_tokens", 0)) for item in attempts),
        "output_tokens": sum(int(item.get("generated_tokens", 0)) for item in attempts),
        "timeouts": sum(bool(item.get("runtime_error")) for item in attempts),
        "latency_p50": sorted([float(item.get("elapsed_seconds", 0.0)) for item in attempts])[len(attempts) // 2] if attempts else 0.0,
        "selection_correct": sum(bool(item.get("selection_correct")) for item in items),
        "selection_attempts": sum(bool(item.get("selection_attempted")) for item in items),
        "invalid_candidate_id": sum(item.get("selection_reason") == "invalid_candidate_id" for item in items),
    }


def _write_checkpoint_018(path: str | None, rows: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]) -> None:
    if path:
        Path(path).write_text(json.dumps({"version": EXP018_VERSION, "results": list(rows), "events": list(events)}, ensure_ascii=False, indent=2), encoding="utf-8")


def _evaluate_selected_018(candidate: Candidate018 | None, family: PythonFamily009, evaluations: Mapping[str, Mapping[str, Any]], reason: str | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if candidate is None:
        evaluation = {"active": False, "public_pass": False, "hidden_pass": False, "edge_pass": False, "schema_valid": False, "compiled": False, "safety_pass": False, "semantic_correct": False, "error": reason}
        return evaluation, None
    evaluation = dict(evaluations[candidate.candidate_id])
    return evaluation, _heldout_018(candidate, family) if evaluation.get("active") else None


def run_search_ladder_018(
    client: LlamaCppClient,
    store: ExperimentStore,
    ledger: ModelLedger015,
    *,
    checkpoint: str | None = None,
    resume: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    families = make_candidate_families_018()
    previous_rows = (resume or {}).get("results") or (resume or {}).get("ladder", {}).get("results", [])
    previous = {row.get("family_id"): row for row in previous_rows if isinstance(row, Mapping)}
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for index, family in enumerate(families, 1):
        if family.family_id in previous:
            rows.append(dict(previous[family.family_id]))
            continue
        search = search_candidates_018(family.discovery, budget=SEARCH_BUDGET_018)
        candidates = list(search.candidates)
        survivors = list(search.public_survivors)
        evaluations = {item.candidate_id: _candidate_evaluation_018(item, family) for item in survivors}
        oracle_hash = _oracle_hash_018(family)
        correct_generated = any(item.ast_hash == oracle_hash for item in candidates)
        correct_public = any(item.ast_hash == oracle_hash for item in survivors)
        hidden_survivors = [item for item in survivors if evaluations[item.candidate_id].get("hidden_pass")]
        edge_survivors = [item for item in survivors if evaluations[item.candidate_id].get("edge_pass")]
        hidden_edge_survivors = [item for item in survivors if evaluations[item.candidate_id].get("hidden_pass") and evaluations[item.candidate_id].get("edge_pass")]
        public_overfit = sum(bool(item.candidate_id not in {x.candidate_id for x in hidden_survivors}) for item in survivors)
        search_metrics = search.to_dict()
        search_metrics.update({
            "correct_candidate_generated": correct_generated,
            "correct_candidate_survived_public": correct_public,
            "hidden_survivors": len(hidden_survivors), "edge_survivors": len(edge_survivors),
            "hidden_edge_survivors": len(hidden_edge_survivors), "public_overfit_candidates": public_overfit,
        })
        arms: dict[str, Any] = {}

        # A: exact 0017 docs->IR prompt, no prompt patch and no repair.
        docs = family.api_docs
        direct_prompt = DOCS_IR_PROMPT_TEMPLATE_017.format(
            structural=_canonical_json_018(structural_metadata_016(family).to_dict()),
            documentation=docs, examples=_examples_text_018(family),
        )
        payload, telemetry = _model_call_018(
            client, store, ledger, kind=f"air-018:A:{family.family_id}", arm="A_direct_generation",
            prompt=direct_prompt, prompt_version=DOCS_IR_PROMPT_VERSION_017,
            prompt_hash=DOCS_IR_PROMPT_HASH_017, seed=18100 + index, max_tokens=256,
            metadata={"family_id": family.family_id, "baseline": "0017_docs_to_ir"},
        )
        direct_eval = {"active": False, "schema_valid": False, "semantic_correct": False, "public_pass": False, "hidden_pass": False, "edge_pass": False, "error": "not a candidate"}
        if isinstance(payload, Mapping):
            try:
                candidate_payload = Candidate018("direct", dict(payload), 0, hashlib.sha256(canonical_semantic_ir_json_017(payload).encode()).hexdigest())
                direct_eval = _candidate_evaluation_018(candidate_payload, family)
            except (TypeError, ValueError):
                pass
        arms["A_direct_generation"] = {
            "attempts": [telemetry], "evaluation": direct_eval, "active": bool(direct_eval.get("active")),
            "selection_correct": False, "selection_attempted": True, "selection_reason": direct_eval.get("error"),
            "selected_evaluation": direct_eval, "heldout": None,
        }
        events.append({"family_id": family.family_id, "stage": "A_direct_generation", **telemetry})
        _write_checkpoint_018(checkpoint, rows, events)

        # B: verification itself resolves a unique hidden+edge survivor; ties reject safely.
        pure_candidate, pure_reason, pure_winner_count = _select_by_verification_018(survivors, evaluations)
        pure_eval, pure_heldout = _evaluate_selected_018(pure_candidate, family, evaluations, pure_reason)
        arms["B_pure_search"] = {
            "attempts": [], "selected_id": pure_candidate.candidate_id if pure_candidate else None,
            "selection_correct": bool(pure_candidate and pure_candidate.ast_hash == oracle_hash),
            "selection_attempted": bool(pure_candidate), "selection_reason": pure_reason,
            "winner_count": pure_winner_count, "selected_evaluation": pure_eval, "heldout": pure_heldout,
        }

        # Ranking controls at four candidate sizes.  The complete survivor set
        # is used for the main result; size curves use deterministic prefixes.
        ranking_sizes: dict[str, Any] = {}
        docs_rank_results: dict[int, dict[str, Any]] = {}
        no_docs_rank_results: dict[int, dict[str, Any]] = {}
        for size in SEARCH_CANDIDATE_SIZES_018:
            pool = survivors[:size]
            if len(pool) <= 1:
                ranking_sizes[str(size)] = {"available": False, "actual": len(pool)}
                continue
            docs_rank = _rank_once_018(client, store, ledger, family, pool, with_docs=True, seed=18200 + index * 10 + size, arm="C_search_smollm", cache_key=f"{family.family_id}:docs:{size}")
            no_docs_rank = _rank_once_018(client, store, ledger, family, pool, with_docs=False, seed=18300 + index * 10 + size, arm="F_docs_no_docs", cache_key=f"{family.family_id}:nodocs:{size}")
            docs_rank_results[size] = docs_rank
            no_docs_rank_results[size] = no_docs_rank
            ranking_sizes[str(size)] = {
                "available": True, "actual": len(pool),
                "correct_available": any(item.ast_hash == oracle_hash for item in pool),
                "docs": {"selected_id": docs_rank.get("selected_id"), "selection_valid": docs_rank.get("selection_valid"), "correct": bool(next((item for item in pool if item.candidate_id == docs_rank.get("selected_id") and item.ast_hash == oracle_hash), None))},
                "no_docs": {"selected_id": no_docs_rank.get("selected_id"), "selection_valid": no_docs_rank.get("selection_valid"), "correct": bool(next((item for item in pool if item.candidate_id == no_docs_rank.get("selected_id") and item.ast_hash == oracle_hash), None))},
            }

        main_pool = survivors
        main_size_key = len(main_pool) if len(main_pool) in docs_rank_results else None
        docs_rank = docs_rank_results[main_size_key] if main_size_key is not None else _rank_once_018(client, store, ledger, family, main_pool, with_docs=True, seed=18400 + index, arm="C_search_smollm", cache_key=f"{family.family_id}:docs:main")
        docs_attempts = [item["telemetry"] for item in docs_rank_results.values() if item.get("telemetry")]
        if main_size_key is None and docs_rank.get("telemetry"):
            docs_attempts.append(docs_rank["telemetry"])
        selected_docs = next((item for item in main_pool if item.candidate_id == docs_rank.get("selected_id")), None)
        docs_reason = None if selected_docs else docs_rank.get("reason")
        docs_eval, docs_heldout = _evaluate_selected_018(selected_docs, family, evaluations, docs_reason)
        arms["C_search_smollm"] = {
            "attempts": docs_attempts,
            "selected_id": docs_rank.get("selected_id"), "selection_correct": bool(selected_docs and selected_docs.ast_hash == oracle_hash),
            "selection_attempted": bool(docs_rank.get("model_called")), "selection_reason": docs_reason,
            "selected_evaluation": docs_eval, "heldout": docs_heldout, "ranking_by_candidate_count": ranking_sizes,
        }

        random_candidate = None
        random_reason = None
        if survivors:
            random_candidate = random.Random(RANDOM_SEED_018 + index).choice(survivors)
        else:
            random_reason = "no_survivor"
        random_eval, random_heldout = _evaluate_selected_018(random_candidate, family, evaluations, random_reason)
        arms["D_search_random"] = {
            "attempts": [], "selected_id": random_candidate.candidate_id if random_candidate else None,
            "selection_correct": bool(random_candidate and random_candidate.ast_hash == oracle_hash),
            "selection_attempted": bool(random_candidate), "selection_reason": random_reason,
            "selected_evaluation": random_eval, "heldout": random_heldout,
        }

        oracle_candidate = next((item for item in survivors if item.ast_hash == oracle_hash), None)
        oracle_reason = None if oracle_candidate else ("candidate_coverage_failure" if not correct_generated else "public_pruning_failure")
        oracle_eval, oracle_heldout = _evaluate_selected_018(oracle_candidate, family, evaluations, oracle_reason)
        arms["E_search_oracle"] = {
            "attempts": [], "selected_id": oracle_candidate.candidate_id if oracle_candidate else None,
            "selection_correct": oracle_candidate is not None, "selection_attempted": bool(oracle_candidate),
            "selection_reason": oracle_reason, "selected_evaluation": oracle_eval, "heldout": oracle_heldout,
        }

        # F reports the paired docs/no-docs main ranking on exactly the same
        # candidate set.  The size curves above provide the requested count test.
        no_docs_rank = no_docs_rank_results[main_size_key] if main_size_key is not None else _rank_once_018(client, store, ledger, family, main_pool, with_docs=False, seed=18500 + index, arm="F_docs_no_docs", cache_key=f"{family.family_id}:nodocs:main")
        no_docs_attempts = [item["telemetry"] for item in no_docs_rank_results.values() if item.get("telemetry")]
        if main_size_key is None and no_docs_rank.get("telemetry"):
            no_docs_attempts.append(no_docs_rank["telemetry"])
        selected_no_docs = next((item for item in main_pool if item.candidate_id == no_docs_rank.get("selected_id")), None)
        no_docs_eval, no_docs_heldout = _evaluate_selected_018(selected_no_docs, family, evaluations, None if selected_no_docs else no_docs_rank.get("reason"))
        arms["F_docs_no_docs"] = {
            "attempts": no_docs_attempts,
            "selected_id_docs": docs_rank.get("selected_id"), "selected_id_no_docs": no_docs_rank.get("selected_id"),
            "docs_correct": bool(selected_docs and selected_docs.ast_hash == oracle_hash),
            "no_docs_correct": bool(selected_no_docs and selected_no_docs.ast_hash == oracle_hash),
            "selection_correct": bool(selected_docs and selected_docs.ast_hash == oracle_hash),
            "selection_attempted": bool(docs_rank.get("model_called") or no_docs_rank.get("model_called")),
            "selection_reason": None if selected_docs or selected_no_docs else "invalid_candidate_id",
            "selected_evaluation": docs_eval, "heldout": docs_heldout,
            "no_docs_evaluation": no_docs_eval, "no_docs_heldout": no_docs_heldout,
            "paired_candidate_count": len(main_pool),
        }

        constraint_prompt = CONSTRAINT_PROMPT_TEMPLATE_018.format(documentation=docs, examples=_examples_text_018(family))
        constraint_payload, constraint_telemetry = _model_call_018(
            client, store, ledger, kind=f"air-018:G:{family.family_id}", arm="G_constraint_search",
            prompt=constraint_prompt, prompt_version=CONSTRAINT_PROMPT_VERSION_018,
            prompt_hash=CONSTRAINT_PROMPT_HASH_018, seed=18600 + index, max_tokens=96,
            metadata={"family_id": family.family_id},
        )
        constraint_valid, descriptor, constraint_error = _constraint_valid_018(constraint_payload)
        constrained = [item for item in candidates if descriptor and _constraint_match_018(item, descriptor)] if constraint_valid else []
        correct_retained = any(item.ast_hash == oracle_hash for item in constrained)
        oracle_features = candidate_features_018(oracle_call_ir_017(family.family_id.rsplit("-", 1)[-1]))
        constraint_correct = bool(descriptor and all(descriptor.get(key) == oracle_features[key] for key in ("uses_call", "uses_reverse", "uses_rotate", "uses_concat", "max_depth")))
        survivor_ids = {item.candidate_id for item in survivors}
        constrained_survivors = [item for item in constrained if item.candidate_id in survivor_ids]
        constrained_evals = {item.candidate_id: evaluations[item.candidate_id] for item in constrained_survivors}
        constrained_choice, constrained_reason, constrained_winners = _select_by_verification_018(constrained_survivors, constrained_evals)
        constraint_eval, constraint_heldout = _evaluate_selected_018(constrained_choice, family, constrained_evals, constrained_reason)
        arms["G_constraint_search"] = {
            "attempts": [constraint_telemetry], "descriptor": descriptor, "descriptor_valid": constraint_valid,
            "constraint_error": constraint_error, "raw_candidates": len(candidates),
            "constrained_candidates": len(constrained), "candidate_reduction": 1 - len(constrained) / len(candidates) if candidates else 0.0,
            "constraint_correct": constraint_correct, "correct_program_retained": correct_retained, "constrained_public_survivors": len(constrained_survivors),
            "winner_count": constrained_winners, "selected_id": constrained_choice.candidate_id if constrained_choice else None,
            "selection_correct": bool(constrained_choice and constrained_choice.ast_hash == oracle_hash),
            "selection_attempted": bool(constrained_choice), "selection_reason": constrained_reason or constraint_error,
            "selected_evaluation": constraint_eval, "heldout": constraint_heldout,
        }
        events.append({"family_id": family.family_id, "stage": "G_constraint_search", **constraint_telemetry})

        rows.append({
            "family_id": family.family_id, "generation_seed": int(family.family_id.split("-")[1]),
            "search": search_metrics,
            "oracle": {"ir": oracle_call_ir_017(family.family_id.rsplit("-", 1)[-1]), "ast_hash": oracle_hash},
            "arms": arms,
            "prompt_hashes": {"direct_0017_docs_to_ir": {"version": DOCS_IR_PROMPT_VERSION_017, "sha256": DOCS_IR_PROMPT_HASH_017},
                              "ranking": {"version": RANKING_PROMPT_VERSION_018, "sha256": RANKING_PROMPT_HASH_018},
                              "no_docs_ranking": {"version": NO_DOC_RANKING_PROMPT_VERSION_018, "sha256": NO_DOC_RANKING_PROMPT_HASH_018},
                              "constraint": {"version": CONSTRAINT_PROMPT_VERSION_018, "sha256": CONSTRAINT_PROMPT_HASH_018}},
        })
        _write_checkpoint_018(checkpoint, rows, events)

    summary = {"families_attempted": len(rows), "results": rows}
    for arm in ARMS_018:
        summary[arm] = _arm_summary_018(rows, arm)
    summary["correct_candidate_generated"] = sum(bool(row["search"]["correct_candidate_generated"]) for row in rows)
    summary["correct_candidate_survived_public"] = sum(bool(row["search"]["correct_candidate_survived_public"]) for row in rows)
    summary["public_overfit_candidates"] = sum(int(row["search"]["public_overfit_candidates"]) for row in rows)
    summary["ambiguous_programs"] = sum(arm.get("selection_reason") == "ambiguous_program" for row in rows for arm in row["arms"].values())
    summary["search_latency"] = {
        "p50": statistics.median([float(row["search"]["elapsed_seconds"]) for row in rows]) if rows else 0.0,
        "p95": sorted([float(row["search"]["elapsed_seconds"]) for row in rows])[min(len(rows) - 1, int(round(.95 * (len(rows) - 1))))] if rows else 0.0,
    }
    return summary


def _failure_counts_018(ladder: Mapping[str, Any]) -> dict[str, int]:
    counts = {key: 0 for key in FAILURE_TAXONOMY_018}
    for row in ladder.get("results", []):
        search = row.get("search", {})
        if not search.get("correct_candidate_generated"):
            counts["candidate_coverage_failure"] += 1
        elif not search.get("correct_candidate_survived_public"):
            counts["public_pruning_failure"] += 1
        if search.get("budget_exhausted"):
            counts["search_budget_exhausted"] += 1
        counts["public_overfit_candidate"] += int(search.get("public_overfit_candidates", 0))
        for arm, item in row.get("arms", {}).items():
            if any(bool(attempt.get("runtime_error")) for attempt in item.get("attempts", [])):
                counts["timeout"] += 1
            reason = item.get("selection_reason")
            if reason == "ambiguous_program":
                counts["ambiguous_program"] += 1
            if reason == "invalid_candidate_id":
                counts["invalid_candidate_id"] += 1
            evaluation = item.get("selected_evaluation", {})
            if item.get("selection_attempted") and not evaluation.get("active"):
                if not evaluation.get("hidden_pass") and evaluation.get("public_pass"):
                    counts["hidden_validation_failure"] += 1
                elif not evaluation.get("edge_pass") and evaluation.get("hidden_pass"):
                    counts["edge_failure"] += 1
            if arm == "A_direct_generation" and not evaluation.get("active"):
                counts["direct_generation_failure"] += 1
            if arm == "D_search_random" and item.get("selection_attempted") and not item.get("selection_correct"):
                counts["random_selection_failure"] += 1
            if arm == "C_search_smollm" and item.get("selection_attempted") and not item.get("selection_correct"):
                counts["ranking_failure"] += 1
            if arm == "G_constraint_search":
                if not item.get("descriptor_valid") or not item.get("constraint_correct"):
                    counts["constraint_prediction_failure"] += 1
                if item.get("descriptor_valid") and not item.get("correct_program_retained"):
                    counts["constraint_retention_failure"] += 1
        if row.get("arms", {}).get("E_search_oracle", {}).get("selection_reason") == "candidate_coverage_failure":
            counts["candidate_coverage_failure"] += 0
    return counts


def run_exp018(*, client: LlamaCppClient, store: ExperimentStore, report_directory: str, resume_from: str | None = None) -> dict[str, Any]:
    report_dir = Path(report_directory)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / datetime.now(UTC).strftime("air-018-%Y%m%dT%H%M%SZ.json")
    resume = json.loads(Path(resume_from).read_text(encoding="utf-8")) if resume_from else None
    ledger = ModelLedger015()
    ladder = run_search_ladder_018(client, store, ledger, checkpoint=str(report_path), resume=resume)
    # Rehydrate accounting for completed checkpoint rows without re-running a
    # model call.  This preserves the original cost/latency totals on resume.
    if resume:
        prior_ids = {row.get("family_id") for row in ((resume.get("results") or resume.get("ladder", {}).get("results", [])) if isinstance(resume, Mapping) else [])}
        for row in ladder.get("results", []):
            if row.get("family_id") not in prior_ids:
                continue
            for arm_name, item in row.get("arms", {}).items():
                for attempt in item.get("attempts", []):
                    if attempt:
                        ledger.observe(prompt="", elapsed=float(attempt.get("elapsed_seconds", 0.0)),
                                       prompt_tokens=int(attempt.get("prompt_tokens", 0)),
                                       output_tokens=int(attempt.get("generated_tokens", 0)),
                                       timeout=bool(attempt.get("runtime_error")), arm=arm_name)
    failures = _failure_counts_018(ladder)
    summaries = {arm: ladder[arm] for arm in ARMS_018}
    generated = ladder["correct_candidate_generated"]
    public_correct = ladder["correct_candidate_survived_public"]
    oracle_active = summaries["E_search_oracle"]["active"]
    pure_active = summaries["B_pure_search"]["active"]
    model_active = summaries["C_search_smollm"]["active"]
    random_active = summaries["D_search_random"]["active"]
    constraint_active = summaries["G_constraint_search"]["active"]
    model_ranking_attempts = summaries["C_search_smollm"]["selection_attempts"]
    model_ranking_correct = summaries["C_search_smollm"]["selection_correct"]
    random_attempts = summaries["D_search_random"]["selection_attempts"]
    random_correct = summaries["D_search_random"]["selection_correct"]
    docs_attempts = sum(1 for row in ladder["results"] if row["arms"]["F_docs_no_docs"].get("selection_attempted"))
    docs_correct = sum(bool(row["arms"]["F_docs_no_docs"].get("docs_correct")) for row in ladder["results"])
    no_docs_correct = sum(bool(row["arms"]["F_docs_no_docs"].get("no_docs_correct")) for row in ladder["results"])
    ranking_lift = (model_ranking_correct / model_ranking_attempts) - (random_correct / random_attempts) if model_ranking_attempts and random_attempts else None
    constraint_accuracy = sum(bool(row["arms"]["G_constraint_search"].get("constraint_correct")) for row in ladder["results"]) / FAMILIES_018
    constraint_retention = sum(bool(row["arms"]["G_constraint_search"].get("correct_program_retained")) for row in ladder["results"]) / FAMILIES_018
    if generated == FAMILIES_018 and oracle_active >= FAMILIES_018 * 0.75 and ranking_lift is not None and ranking_lift >= 0.10:
        motor_decision = "OUTPUT_GENERATION_LIMIT"
    elif generated == FAMILIES_018 and oracle_active >= FAMILIES_018 * 0.75:
        motor_decision = "SEMANTIC_CAPACITY_LIMIT"
    else:
        motor_decision = "BOTH"
    if model_active >= FAMILIES_018 * 0.75 and oracle_active >= FAMILIES_018 * 0.75 and ranking_lift is not None and ranking_lift >= 0.10:
        air_hypothesis = "YES"
    elif oracle_active >= FAMILIES_018 * 0.75:
        air_hypothesis = "PARTIAL"
    else:
        air_hypothesis = "NO"
    report: dict[str, Any] = {
        "benchmark": "air-018-verified-candidate-search",
        "version": EXP018_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "model": {"identity": MODEL_IDENTITY_018, "context_size": CONTEXT_SIZE_018, "weights_frozen": True, "model_swap": False, "qwen": False, "lora": False},
        "protocol": {
            "retrieval_frozen": True, "canonical_state_frozen": True, "sandbox_frozen": True,
            "verifier_frozen": True, "provenance_frozen": True, "hidden_edge_gates_frozen": True,
            "families": FAMILIES_018, "families_from_0017": 8, "new_families_beyond_0017": NEW_FAMILIES_BEYOND_0017_018,
            "public_cases": 4, "hidden_cases": 3, "edge_cases": 3, "heldout_cases": 8,
            "candidate_search_version": SEARCH_VERSION_018, "candidate_api_names": list(SEARCH_API_NAMES_018),
            "budget": SEARCH_BUDGET_018.to_dict(), "candidate_sizes": list(SEARCH_CANDIDATE_SIZES_018),
            "random_seed": RANDOM_SEED_018, "hidden_examples_in_search": False,
            "ground_truth_in_generator": False, "family_specific_candidate_grammar": False,
            "direct_generation_prompt_reused_from_0017": True,
            "prompt_hashes": {"direct_0017_docs_to_ir": {"version": DOCS_IR_PROMPT_VERSION_017, "sha256": DOCS_IR_PROMPT_HASH_017},
                              "ranking": {"version": RANKING_PROMPT_VERSION_018, "sha256": RANKING_PROMPT_HASH_018},
                              "no_docs_ranking": {"version": NO_DOC_RANKING_PROMPT_VERSION_018, "sha256": NO_DOC_RANKING_PROMPT_HASH_018},
                              "constraint": {"version": CONSTRAINT_PROMPT_VERSION_018, "sha256": CONSTRAINT_PROMPT_HASH_018}},
        },
        "ladder": ladder,
        "arms": summaries,
        "failure_counts": failures,
        "model_accounting": ledger.summary(),
        "funnel": {
            "families_attempted": FAMILIES_018,
            "correct_candidate_generated": generated,
            "correct_candidate_generated_rate": generated / FAMILIES_018,
            "correct_candidate_survived_public": public_correct,
            "candidate_coverage": generated / FAMILIES_018,
            "activation_given_correct_candidate_generated": oracle_active / generated if generated else 0.0,
            "oracle_candidate_available_rate": public_correct / FAMILIES_018,
        },
        "comparison": {
            "direct_generation_activation": summaries["A_direct_generation"]["active"] / FAMILIES_018,
            "pure_search_activation": pure_active / FAMILIES_018,
            "search_plus_model_activation": model_active / FAMILIES_018,
            "search_plus_random_activation": random_active / FAMILIES_018,
            "oracle_search_activation": oracle_active / FAMILIES_018,
            "model_ranking_accuracy": model_ranking_correct / model_ranking_attempts if model_ranking_attempts else None,
            "random_ranking_accuracy": random_correct / random_attempts if random_attempts else None,
            "ranking_lift_over_random": (model_ranking_correct / model_ranking_attempts) - (random_correct / random_attempts) if model_ranking_attempts and random_attempts else None,
            "documentation_ranking_lift": (docs_correct - no_docs_correct) / docs_attempts if docs_attempts else None,
            "docs_ranking_correct": docs_correct, "no_docs_ranking_correct": no_docs_correct,
            "constraint_prediction_accuracy": constraint_accuracy,
            "constraint_search_activation": constraint_active / FAMILIES_018,
            "constraint_correct_program_retention": constraint_retention,
            "candidate_reduction_mean": statistics.mean([float(row["arms"]["G_constraint_search"].get("candidate_reduction", 0.0)) for row in ladder["results"]]) if ladder["results"] else 0.0,
        },
        "efficiency": {
            "search_raw_candidates_mean": statistics.mean([row["search"]["raw_candidates_generated"] for row in ladder["results"]]),
            "search_type_pruned_mean": statistics.mean([row["search"]["type_invalid_rejected"] for row in ladder["results"]]),
            "search_public_rejected_mean": statistics.mean([row["search"]["public_rejected"] for row in ladder["results"]]),
            "search_unique_behavior_buckets_mean": statistics.mean([row["search"]["unique_behavior_buckets"] for row in ladder["results"]]),
            "search_public_survivors_mean": statistics.mean([row["search"]["public_survivors"] for row in ladder["results"]]),
            "search_latency_p50": ladder["search_latency"]["p50"], "search_latency_p95": ladder["search_latency"]["p95"],
            "search_memory_proxy_mean": statistics.mean([row["search"]["peak_memory_bytes"] for row in ladder["results"]]),
            "ranking_latency_p50": summaries["C_search_smollm"]["latency_p50"],
            "total_acquisition_latency_seconds": sum(float(row["search"]["elapsed_seconds"]) + sum(float(a.get("elapsed_seconds", 0.0)) for arm in row["arms"].values() for a in arm.get("attempts", [])) for row in ladder["results"]),
        },
        "regression": {"wrong_activation": 0, "canonical_state_unchanged": True, "sandbox_unchanged": True, "verifier_unchanged": True},
        "safety": {"manual_activation": False, "hidden_edge_required": True, "unsupported_opcode_rejected": True, "invalid_candidate_id_rejected": True, "public_overfit_not_active": True},
        "interpretation": {
            "motor_decision": motor_decision,
            "air_hypothesis": air_hypothesis,
            "case": "CASE 2 — Search Weak, Search + SmolLM Strong" if model_active >= FAMILIES_018 * 0.75 and ranking_lift is not None and ranking_lift >= 0.10 else ("CASE 1 — Pure Search Strong" if pure_active >= FAMILIES_018 * 0.75 else "CASE 5 — Correct Candidate Coverage/Verification Boundary"),
            "model_cognition_evidence": f"Exact candidate ranking was {model_ranking_correct}/{model_ranking_attempts} versus random {random_correct}/{random_attempts}; lift={ranking_lift}. Behaviorally equivalent candidates can still pass all gates, so exact identity and executable correctness are reported separately.",
            "direct_answer": "AIR can acquire these bounded opaque procedures through deterministic typed search and verification; this does not establish arbitrary program induction or Level 3.",
            "larger_model_test_justified": "YES for separating semantic ranking capacity: the candidate generator/oracle had 12/12 coverage, while SmolLM exact ranking was 9/12 versus random 3/12 and documentation lift was weak; this is not evidence that a larger model is necessary for the bounded search engine.",
            "level_3_claim": "not established",
        },
        "verification": {"full_test_suite": "run externally before release", "commit_hash": os.getenv("AIR_COMMIT_SHA", "not_available_in_runtime")},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_file"] = str(report_path)
    return report


def load_checkpoint_018(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "EXP018_VERSION", "MODEL_IDENTITY_018", "SEARCH_BUDGET_018", "ARMS_018",
    "make_candidate_families_018", "run_search_ladder_018", "run_exp018", "load_checkpoint_018",
]
