"""Deterministic behavioral canonicalization for Experiment 0019.

Search remains public-only.  Hidden and edge cases are consumed only after
candidate generation, as activation gates and equivalence witnesses.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Mapping, Sequence

from .exp009 import FamilyCase009
from .program_search import Candidate018, SEARCH_API_NAMES_018
from .semantic_ir import (
    SemanticIRExecutionError,
    SemanticIRValidationError,
    canonical_semantic_ir_json_017,
    execute_semantic_ir_017,
)


CANONICALIZATION_VERSION_019 = "air-019-behavioral-canonicalization-v1"
CANONICAL_CHOICE_ORDER_019 = (
    "minimum_semantic_depth",
    "minimum_operation_count",
    "minimum_serialized_ir_bytes",
    "minimum_estimated_execution_cost",
    "lexical_canonical_ir",
)


@dataclass(frozen=True)
class EquivalenceClass019:
    signature: tuple[str, ...]
    members: tuple[Candidate018, ...]
    canonical: Candidate018

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature_sha256": hashlib.sha256("\n".join(self.signature).encode()).hexdigest(),
            "member_ids": [item.candidate_id for item in self.members],
            "canonical_id": self.canonical.candidate_id,
            "canonical_cost": list(canonical_cost_key_019(self.canonical)),
        }


@dataclass(frozen=True)
class CanonicalizationResult019:
    selected: Candidate018 | None
    verified_candidates: tuple[Candidate018, ...]
    equivalence_classes: tuple[EquivalenceClass019, ...]
    reason: str | None
    stable: bool
    elapsed_seconds: float
    level_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_id": self.selected.candidate_id if self.selected else None,
            "verified_candidates": len(self.verified_candidates),
            "equivalence_class_count": len(self.equivalence_classes),
            "classes": [item.to_dict() for item in self.equivalence_classes],
            "reason": self.reason,
            "stable": self.stable,
            "elapsed_seconds": self.elapsed_seconds,
            "level_counts": dict(self.level_counts),
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalized_expr_019(expr: Mapping[str, Any]) -> dict[str, Any]:
    opcode = expr.get("op")
    if opcode in {"INPUT", "INT"}:
        return dict(expr)
    if opcode == "CALL":
        return {"op": "CALL", "api": expr["api"], "args": [_normalized_expr_019(expr["args"][0])]}
    if opcode == "REVERSE":
        value = _normalized_expr_019(expr["value"])
        if value.get("op") == "REVERSE":
            return _normalized_expr_019(value["value"])
        return {"op": "REVERSE", "value": value}
    if opcode == "ROTATE":
        value = _normalized_expr_019(expr["value"])
        amount = _normalized_expr_019(expr["amount"])
        if amount.get("op") == "INT" and amount.get("value") == 0:
            return value
        return {"op": "ROTATE", "value": value, "amount": amount}
    if opcode == "CONCAT":
        return {"op": "CONCAT", "values": [_normalized_expr_019(item) for item in expr["values"]]}
    if opcode == "RETURN":
        return {"op": "RETURN", "value": _normalized_expr_019(expr["value"])}
    return dict(expr)


def normalized_ir_019(candidate: Candidate018 | Mapping[str, Any]) -> dict[str, Any]:
    ir = candidate.ir if isinstance(candidate, Candidate018) else candidate
    result = dict(ir)
    result["expr"] = _normalized_expr_019(ir["expr"])
    return result


def normalized_ast_hash_019(candidate: Candidate018) -> str:
    return hashlib.sha256(_canonical_json(normalized_ir_019(candidate)).encode()).hexdigest()


def _expr_depth(expr: Mapping[str, Any]) -> int:
    children: list[Mapping[str, Any]] = []
    for key in ("value", "amount"):
        child = expr.get(key)
        if isinstance(child, Mapping):
            children.append(child)
    for key in ("args", "values"):
        values = expr.get(key)
        if isinstance(values, list):
            children.extend(item for item in values if isinstance(item, Mapping))
    return 1 + max((_expr_depth(item) for item in children), default=0)


def _operation_count(expr: Mapping[str, Any]) -> int:
    count = 0 if expr.get("op") in {"INPUT", "INT", "RETURN"} else 1
    for key in ("value", "amount"):
        child = expr.get(key)
        if isinstance(child, Mapping):
            count += _operation_count(child)
    for key in ("args", "values"):
        values = expr.get(key)
        if isinstance(values, list):
            count += sum(_operation_count(item) for item in values if isinstance(item, Mapping))
    return count


_COSTS = {"INPUT": 0, "INT": 0, "RETURN": 0, "CALL": 10, "REVERSE": 2, "ROTATE": 3, "CONCAT": 2}


def _execution_cost(expr: Mapping[str, Any]) -> int:
    cost = _COSTS.get(str(expr.get("op")), 100)
    for key in ("value", "amount"):
        child = expr.get(key)
        if isinstance(child, Mapping):
            cost += _execution_cost(child)
    for key in ("args", "values"):
        values = expr.get(key)
        if isinstance(values, list):
            cost += sum(_execution_cost(item) for item in values if isinstance(item, Mapping))
    return cost


def canonical_cost_key_019(candidate: Candidate018) -> tuple[int, int, int, int, str]:
    normalized = normalized_ir_019(candidate)
    serialized = canonical_semantic_ir_json_017(candidate.ir)
    return (
        _expr_depth(normalized["expr"]),
        _operation_count(normalized["expr"]),
        len(serialized.encode()),
        _execution_cost(normalized["expr"]),
        serialized,
    )


def behavior_signature_019(candidate: Candidate018, cases: Sequence[FamilyCase009]) -> tuple[str, ...]:
    result: list[str] = []
    for case in cases:
        try:
            output = execute_semantic_ir_017(candidate.ir, case.input_text, set(SEARCH_API_NAMES_018))
            result.append("ok:" + output)
        except (SemanticIRExecutionError, SemanticIRValidationError, TypeError, ValueError) as exc:
            result.append("error:" + type(exc).__name__)
    return tuple(result)


def _passes_expected(candidate: Candidate018, cases: Sequence[FamilyCase009]) -> bool:
    signature = behavior_signature_019(candidate, cases)
    return all(value == "ok:" + case.expected for value, case in zip(signature, cases))


def equivalence_classes_019(
    candidates: Sequence[Candidate018], cases: Sequence[FamilyCase009],
) -> tuple[EquivalenceClass019, ...]:
    buckets: dict[tuple[str, ...], list[Candidate018]] = {}
    for candidate in candidates:
        buckets.setdefault(behavior_signature_019(candidate, cases), []).append(candidate)
    classes: list[EquivalenceClass019] = []
    for signature, members in sorted(buckets.items(), key=lambda item: item[0]):
        ordered = tuple(sorted(members, key=canonical_cost_key_019))
        classes.append(EquivalenceClass019(signature, ordered, ordered[0]))
    return tuple(classes)


def _level_count(candidates: Sequence[Candidate018], key: Any) -> int:
    return len({key(item) for item in candidates})


def canonicalize_verified_019(
    candidates: Sequence[Candidate018],
    public: Sequence[FamilyCase009],
    hidden: Sequence[FamilyCase009],
    edge: Sequence[FamilyCase009],
) -> CanonicalizationResult019:
    started = time.perf_counter()
    verified = tuple(item for item in candidates if (
        _passes_expected(item, public) and _passes_expected(item, hidden) and _passes_expected(item, edge)
    ))
    all_cases = tuple(public) + tuple(hidden) + tuple(edge)
    classes = equivalence_classes_019(verified, all_cases)
    selected = classes[0].canonical if len(classes) == 1 else None
    reason = None if selected else ("no_verified_candidate" if not classes else "distinct_verified_behaviors")
    reverse_classes = equivalence_classes_019(tuple(reversed(verified)), all_cases)
    reverse_selected = reverse_classes[0].canonical if len(reverse_classes) == 1 else None
    stable = (selected.ast_hash if selected else None) == (reverse_selected.ast_hash if reverse_selected else None)
    public_classes = equivalence_classes_019(verified, public)
    hidden_classes = equivalence_classes_019(verified, hidden)
    edge_classes = equivalence_classes_019(verified, edge)
    level_counts = {
        "exact_canonical_ir": _level_count(verified, lambda item: item.ast_hash),
        "normalized_ast": _level_count(verified, normalized_ast_hash_019),
        "public_behavior": len(public_classes),
        "hidden_behavior": len(hidden_classes),
        "edge_behavior": len(edge_classes),
        "combined_behavior": len(classes),
    }
    return CanonicalizationResult019(
        selected, verified, classes, reason, stable, time.perf_counter() - started, level_counts,
    )


def canonical_representatives_by_normalized_ast_019(
    candidates: Sequence[Candidate018],
) -> tuple[Candidate018, ...]:
    buckets: dict[str, list[Candidate018]] = {}
    for candidate in candidates:
        buckets.setdefault(normalized_ast_hash_019(candidate), []).append(candidate)
    representatives = [min(items, key=canonical_cost_key_019) for items in buckets.values()]
    return tuple(sorted(representatives, key=canonical_cost_key_019))


__all__ = [
    "CANONICALIZATION_VERSION_019", "CANONICAL_CHOICE_ORDER_019",
    "EquivalenceClass019", "CanonicalizationResult019", "normalized_ir_019",
    "normalized_ast_hash_019", "canonical_cost_key_019", "behavior_signature_019",
    "equivalence_classes_019", "canonicalize_verified_019",
    "canonical_representatives_by_normalized_ast_019",
]
