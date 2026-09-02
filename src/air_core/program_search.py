"""Model-independent typed candidate search for Experiment 0018.

The generator deliberately knows only the frozen semantic IR grammar and the
opaque operation namespace.  It never receives a family ground-truth rule or
hidden/edge examples.  Public examples are used only by the caller's filtering
stage; search itself is deterministic and bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import time
from typing import Any, Iterable, Mapping, Sequence

import air_synth_012

from .exp009 import FamilyCase009
from .semantic_ir import (
    SEMANTIC_IR_FORMAT_017,
    SEMANTIC_IR_VERSION_017,
    SemanticIRExecutionError,
    SemanticIRValidationError,
    canonical_semantic_ir_json_017,
    execute_semantic_ir_017,
    validate_semantic_ir_017,
)


SEARCH_VERSION_018 = "air-018-search-v1"
SEARCH_MAX_DEPTH_018 = 3
SEARCH_MAX_RAW_CANDIDATES_018 = 5_000
SEARCH_MAX_PUBLIC_SURVIVORS_018 = 32
SEARCH_CANDIDATE_SIZES_018 = (2, 3, 5, 8)
SEARCH_INT_LITERALS_018 = (-2, -1, 0, 1, 2)
SEARCH_API_NAMES_018 = tuple(sorted(air_synth_012.__all__))


@dataclass(frozen=True)
class SearchBudget018:
    max_depth: int = SEARCH_MAX_DEPTH_018
    max_raw_candidates: int = SEARCH_MAX_RAW_CANDIDATES_018
    max_public_survivors: int = SEARCH_MAX_PUBLIC_SURVIVORS_018

    def to_dict(self) -> dict[str, int]:
        return {
            "max_depth": self.max_depth,
            "max_raw_candidates": self.max_raw_candidates,
            "max_public_survivors": self.max_public_survivors,
        }


@dataclass(frozen=True)
class Candidate018:
    candidate_id: str
    ir: dict[str, Any]
    depth: int
    ast_hash: str
    behavior_signature: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "ir": self.ir,
            "depth": self.depth,
            "ast_hash": self.ast_hash,
            "behavior_signature": list(self.behavior_signature),
        }


@dataclass(frozen=True)
class SearchResult018:
    candidates: tuple[Candidate018, ...]
    public_survivors: tuple[Candidate018, ...]
    behavior_buckets: tuple[tuple[str, ...], ...]
    raw_candidates_generated: int
    type_invalid_rejected: int
    semantic_dedup_rejected: int
    behavior_equivalent_alternatives: int
    public_rejected: int
    budget_exhausted: bool
    elapsed_seconds: float
    peak_memory_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_candidates_generated": self.raw_candidates_generated,
            "type_invalid_rejected": self.type_invalid_rejected,
            "semantic_dedup_rejected": self.semantic_dedup_rejected,
            "behavior_equivalent_alternatives": self.behavior_equivalent_alternatives,
            "public_rejected": self.public_rejected,
            "public_survivors": len(self.public_survivors),
            "unique_behavior_buckets": len(self.behavior_buckets),
            "behavior_buckets": [list(bucket) for bucket in self.behavior_buckets],
            "budget_exhausted": self.budget_exhausted,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "candidates": [item.to_dict() for item in self.candidates],
            "survivor_candidates": [item.to_dict() for item in self.public_survivors],
        }


def _root(expr: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": SEMANTIC_IR_FORMAT_017,
        "version": SEMANTIC_IR_VERSION_017,
        "input_type": "str",
        "output_type": "str",
        "expr": {"op": "RETURN", "value": dict(expr)},
    }


def _ast_hash(ir: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_semantic_ir_json_017(ir).encode("utf-8")).hexdigest()


def _depth(expr: Mapping[str, Any]) -> int:
    children: list[Mapping[str, Any]] = []
    for key in ("value", "amount"):
        child = expr.get(key)
        if isinstance(child, Mapping):
            children.append(child)
    for child in expr.get("args", []) if isinstance(expr.get("args"), list) else []:
        if isinstance(child, Mapping):
            children.append(child)
    for child in expr.get("values", []) if isinstance(expr.get("values"), list) else []:
        if isinstance(child, Mapping):
            children.append(child)
    return 1 + max((_depth(child) for child in children), default=0)


def candidate_features_018(ir: Mapping[str, Any]) -> dict[str, Any]:
    opcodes: list[str] = []

    def walk(expr: Any) -> None:
        if not isinstance(expr, Mapping):
            return
        opcode = expr.get("op")
        if isinstance(opcode, str):
            opcodes.append(opcode)
        for key in ("value", "amount"):
            walk(expr.get(key))
        for key in ("args", "values"):
            values = expr.get(key)
            if isinstance(values, list):
                for item in values:
                    walk(item)

    walk(ir.get("expr"))
    return {
        "uses_call": "CALL" in opcodes,
        "uses_reverse": "REVERSE" in opcodes,
        "uses_rotate": "ROTATE" in opcodes,
        "uses_concat": "CONCAT" in opcodes,
        "max_depth": _depth(ir.get("expr", {})),
    }


def _candidate_exprs_018(budget: SearchBudget018) -> Iterable[tuple[dict[str, Any], int]]:
    """Yield a deterministic, typed, bounded expression language.

    The first layer contains every generic opaque API call, ensuring coverage
    is not accidentally dependent on hash order.  Later layers compose frozen
    string operators around those calls.  No family ID, expected operation, or
    validation case is consulted.
    """
    input_expr = {"op": "INPUT"}
    yield input_expr, 1
    for api in SEARCH_API_NAMES_018:
        yield {"op": "CALL", "api": api, "args": [input_expr]}, 2

    # A small generic vocabulary of transformations.  Duplicates are removed
    # by the caller's canonical AST hash, not by family-specific knowledge.
    current: list[tuple[dict[str, Any], int]] = [(input_expr, 1)]
    current.extend(({"op": "CALL", "api": api, "args": [input_expr]}, 2) for api in SEARCH_API_NAMES_018)
    for _level in range(2, budget.max_depth + 1):
        next_level: list[tuple[dict[str, Any], int]] = []
        for expr, expr_depth in current:
            if expr_depth + 1 <= budget.max_depth:
                next_level.append(({"op": "REVERSE", "value": expr}, expr_depth + 1))
                for amount in SEARCH_INT_LITERALS_018:
                    next_level.append((
                        {"op": "ROTATE", "value": expr, "amount": {"op": "INT", "value": amount}},
                        expr_depth + 1,
                    ))
            # CALL accepts one string expression; wrapping every expression is
            # useful for a generic program search and remains type-safe.
            if expr_depth + 1 <= budget.max_depth:
                for api in SEARCH_API_NAMES_018:
                    next_level.append(({"op": "CALL", "api": api, "args": [expr]}, expr_depth + 1))
        # Pair only shallow expressions to keep the frozen search bounded.
        shallow = [(expr, depth) for expr, depth in current if depth <= budget.max_depth - 1]
        for left, right in product(shallow, repeat=2):
            depth = max(left[1], right[1]) + 1
            if depth <= budget.max_depth:
                next_level.append((
                    {"op": "CONCAT", "values": [left[0], right[0]]}, depth
                ))
        for item in next_level:
            yield item
            # The hard cap is enforced by search_candidates_018, where the
            # attempted count is recorded as an auditable budget metric.
        current = next_level


def _safe_behavior_018(candidate: Candidate018, cases: Sequence[FamilyCase009]) -> tuple[str, ...]:
    outputs: list[str] = []
    for case in cases:
        try:
            outputs.append("ok:" + execute_semantic_ir_017(candidate.ir, case.input_text, SEARCH_API_NAMES_018))
        except (SemanticIRExecutionError, SemanticIRValidationError, ValueError, TypeError) as exc:
            outputs.append("error:" + type(exc).__name__)
    return tuple(outputs)


def search_candidates_018(
    public_cases: Sequence[FamilyCase009],
    *,
    budget: SearchBudget018 = SearchBudget018(),
) -> SearchResult018:
    started = time.perf_counter()
    candidates: list[Candidate018] = []
    seen_ast: set[str] = set()
    type_invalid = 0
    dedup_rejected = 0
    raw_attempts = 0
    budget_exhausted = False
    for expr, depth in _candidate_exprs_018(budget):
        raw_attempts += 1
        if raw_attempts > budget.max_raw_candidates:
            budget_exhausted = True
            break
        ir = _root(expr)
        try:
            validate_semantic_ir_017(ir, set(SEARCH_API_NAMES_018))
        except SemanticIRValidationError:
            type_invalid += 1
            continue
        ast_hash = _ast_hash(ir)
        if ast_hash in seen_ast:
            dedup_rejected += 1
            continue
        seen_ast.add(ast_hash)
        candidates.append(Candidate018(f"candidate_{len(candidates) + 1}", ir, depth, ast_hash))
    candidates_with_behavior: list[Candidate018] = []
    behavior_buckets: dict[tuple[str, ...], list[str]] = {}
    public_rejected = 0
    for candidate in candidates:
        signature = _safe_behavior_018(candidate, public_cases)
        enriched = Candidate018(candidate.candidate_id, candidate.ir, candidate.depth, candidate.ast_hash, signature)
        behavior_buckets.setdefault(signature, []).append(enriched.candidate_id)
        if all(item.startswith("ok:") and item[3:] == case.expected for item, case in zip(signature, public_cases)):
            candidates_with_behavior.append(enriched)
        else:
            public_rejected += 1
    survivors = tuple(candidates_with_behavior[: budget.max_public_survivors])
    if len(candidates_with_behavior) > budget.max_public_survivors:
        budget_exhausted = True
    elapsed = time.perf_counter() - started
    # Python has no portable peak RSS in the standard library.  Record a
    # conservative process-local proxy rather than pretending it is RSS.
    memory_proxy = len(candidates) * 512 + len(survivors) * 1024
    return SearchResult018(
        candidates=tuple(candidates), public_survivors=survivors,
        behavior_buckets=tuple(tuple(ids) for ids in behavior_buckets.values()),
        raw_candidates_generated=min(raw_attempts, budget.max_raw_candidates),
        type_invalid_rejected=type_invalid, semantic_dedup_rejected=dedup_rejected,
        behavior_equivalent_alternatives=sum(max(0, len(ids) - 1) for ids in behavior_buckets.values()),
        public_rejected=public_rejected, budget_exhausted=budget_exhausted,
        elapsed_seconds=elapsed, peak_memory_bytes=memory_proxy,
    )


def candidate_matches_public_018(candidate: Candidate018, public_cases: Sequence[FamilyCase009]) -> bool:
    signature = _safe_behavior_018(candidate, public_cases)
    return all(item.startswith("ok:") and item[3:] == case.expected for item, case in zip(signature, public_cases))


def candidate_matches_cases_018(candidate: Candidate018, cases: Sequence[FamilyCase009]) -> bool:
    return candidate_matches_public_018(candidate, cases)


__all__ = [
    "SEARCH_VERSION_018", "SEARCH_MAX_DEPTH_018", "SEARCH_MAX_RAW_CANDIDATES_018",
    "SEARCH_MAX_PUBLIC_SURVIVORS_018", "SEARCH_CANDIDATE_SIZES_018", "SEARCH_API_NAMES_018",
    "SearchBudget018", "Candidate018", "SearchResult018", "candidate_features_018",
    "search_candidates_018", "candidate_matches_public_018", "candidate_matches_cases_018",
]
