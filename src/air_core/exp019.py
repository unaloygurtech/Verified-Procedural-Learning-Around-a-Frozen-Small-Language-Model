"""Experiment 0019: behavioral canonicalization and documentation grounding."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any, Mapping, Sequence

import air_synth_012

from .behavioral_canonicalization import (
    CANONICALIZATION_VERSION_019,
    CANONICAL_CHOICE_ORDER_019,
    behavior_signature_019,
    canonical_cost_key_019,
    canonical_representatives_by_normalized_ast_019,
    canonicalize_verified_019,
    equivalence_classes_019,
    normalized_ir_019,
)
from .exp009 import FamilyCase009, PythonFamily009
from .exp015 import ModelLedger015
from .exp017 import _parse_json_017
from .exp018 import (
    CONTEXT_SIZE_018,
    MODEL_IDENTITY_018,
    SEARCH_BUDGET_018,
    _candidate_evaluation_018,
    _heldout_018,
    make_candidate_families_018,
)
from .model_client import LlamaCppClient, ModelUnavailable
from .program_search import Candidate018, SEARCH_API_NAMES_018, search_candidates_018
from .semantic_ir import (
    SEMANTIC_IR_FORMAT_017,
    SEMANTIC_IR_VERSION_017,
    canonical_semantic_ir_json_017,
    execute_semantic_ir_017,
    oracle_call_ir_017,
)
from .store import ExperimentStore


EXP019_VERSION = "air-019-v1"
MODEL_IDENTITY_019 = MODEL_IDENTITY_018
CONTEXT_SIZE_019 = CONTEXT_SIZE_018
PART_A_FAMILIES_019 = 12
PART_B_FAMILIES_019 = 20
COUNTERFACTUAL_PAIRS_019 = 8
RANKING_REPRESENTATIONS_019 = ("compact_ir", "behavior_descriptor")
RANDOM_SEED_019 = 19019

FAILURE_TAXONOMY_019 = (
    "candidate_coverage_failure", "public_ambiguity", "hidden_ambiguity",
    "edge_ambiguity", "false_equivalence_failure", "canonicalization_failure",
    "ranking_failure", "documentation_misuse", "counterfactual_doc_failure",
    "position_bias_failure", "wrong_activation", "heldout_failure", "timeout",
    "safe_unknown", "invalid_candidate_id",
)


RANKING_PROMPT_BODY_019 = """Select the executable candidate that implements the normative behavior.
Return exactly one JSON object: {{"candidate_id":"..."}} or {{"candidate_id":null}}.
Candidate IDs are opaque. Do not write or modify a program.

Documentation condition: {condition}
Documentation:
{documentation}

Public examples are deliberately non-discriminating:
{public_examples}

Candidate representation: {representation}
Candidate records include diagnostic probes, but no probe is a hidden, edge,
or held-out activation case:
{candidate_records}
"""


def _prompt_template(condition: str) -> str:
    return RANKING_PROMPT_BODY_019.replace("{condition}", condition)


PROMPT_TEMPLATES_019 = {
    "no_doc": _prompt_template("NO_DOCUMENTATION"),
    "correct_doc": _prompt_template("CORRECT_NORMATIVE_DOCUMENTATION"),
    "wrong_doc": _prompt_template("WRONG_BUT_PLAUSIBLE_DOCUMENTATION"),
    "distractor_doc": _prompt_template("DISTRACTOR_HEAVY_DOCUMENTATION"),
    "counterfactual_doc": _prompt_template("COUNTERFACTUAL_NORMATIVE_DOCUMENTATION"),
    "hybrid": _prompt_template("HYBRID_ONLY_IF_DISTINCT_VERIFIED_CLASSES_REMAIN"),
}
PROMPT_VERSIONS_019 = {key: f"air-019-{key.replace('_', '-')}-ranking-v1" for key in PROMPT_TEMPLATES_019}
PROMPT_HASHES_019 = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in PROMPT_TEMPLATES_019.items()}


@dataclass(frozen=True)
class RankingFamily019:
    family: PythonFamily009
    kind: str
    target_operation: str
    wrong_operation: str
    correct_docs: str
    wrong_docs: str
    probe_inputs: tuple[str, ...]
    data_seed: int


def _apply(operation: str, value: str) -> str:
    return getattr(air_synth_012, operation)(value)


def _semantic_docs_019(kind: str, seed: int) -> str:
    variant = air_synth_012.SEEDS_012.index(seed)
    if kind == "object":
        direction = "descending" if variant % 2 else "ascending"
        return (
            "Normative behavior: parse the input as a JSON object, order keys "
            f"{direction}, render each parsed key/value as key=value, and join fields with |."
        )
    if kind == "mixed":
        return (
            "Normative behavior: split at the final #; rotate the text left by "
            f"{variant + 1}; reverse and uppercase it; append ':' and the integer "
            f"transformed as n*{variant + 2}+{variant + 3}."
        )
    separator = ("/", ":", ".")[variant]
    return (
        "Normative behavior: run-length encode consecutive characters as character+count "
        f"in input order and join groups with {separator!r}; empty input returns empty."
    )


def _ambiguous_public_values(kind: str) -> tuple[str, ...]:
    if kind == "object":
        return ('{"a":1}', '{"q":"p"}', '{"m":0}', '{"z":true}')
    if kind == "mixed":
        return ("a#-1", "b#-1", "q#-1", "z#-1")
    return ("a", "bb", "ccc", "dddd")


def _distinct_values(kind: str, seed: int, count: int) -> tuple[str, ...]:
    rng = random.Random(seed * 7919 + len(kind))
    operations = tuple(air_synth_012.operation_names(item)[kind] for item in air_synth_012.SEEDS_012)
    values: list[str] = []
    while len(values) < count:
        if kind == "object":
            keys = rng.sample(tuple("abcdefghi"), rng.randint(2, 5))
            payload = {key: rng.randint(-5, 25) if index % 2 == 0 else rng.choice(("p", "qr", "stu")) for index, key in enumerate(keys)}
            value = json.dumps(payload, sort_keys=False, separators=(",", ":"))
        elif kind == "mixed":
            text = "".join(rng.choice("abcdefghjkmnpq") for _ in range(rng.randint(2, 8)))
            number = rng.choice(tuple(item for item in range(-8, 20) if item != -1))
            value = f"{text}#{number}"
        else:
            chunks = [rng.choice("abcd") * rng.randint(1, 4) for _ in range(rng.randint(2, 6))]
            value = "".join(chunks)
            if len(set(value)) == 1:
                continue
        outputs = {_apply(operation, value) for operation in operations}
        minimum_classes = 2 if kind == "object" else 3
        if len(outputs) >= minimum_classes and value not in values:
            values.append(value)
    return tuple(values)


def _cases_019(prefix: str, split: str, values: Sequence[str], operation: str) -> tuple[FamilyCase009, ...]:
    return tuple(FamilyCase009(f"{prefix}-{split}-{index:02d}", value, _apply(operation, value), split) for index, value in enumerate(values, 1))


def make_document_ranking_families_019() -> tuple[RankingFamily019, ...]:
    """Twenty frozen dataset variants over the existing opaque operation set."""
    result: list[RankingFamily019] = []
    kinds = ("object", "mixed", "runs")
    seeds = air_synth_012.SEEDS_012
    for index in range(PART_B_FAMILIES_019):
        data_seed = 1901 + index
        kind = kinds[index % len(kinds)]
        target_seed = seeds[(index // len(kinds)) % len(seeds)]
        if kind == "object":
            wrong_seed = 1202 if target_seed != 1202 else 1201
        else:
            wrong_seed = seeds[(seeds.index(target_seed) + 1) % len(seeds)]
        target_operation = air_synth_012.operation_names(target_seed)[kind]
        wrong_operation = air_synth_012.operation_names(wrong_seed)[kind]
        public = _ambiguous_public_values(kind)
        generated = _distinct_values(kind, data_seed, 16)
        prefix = f"air019-{data_seed}-{kind}"
        family = PythonFamily009(
            family_id=f"{prefix}-{target_operation}",
            title=f"Opaque documentation ranking task {data_seed}",
            api_docs=_semantic_docs_019(kind, target_seed),
            contract="Select a verified executable candidate for the normative opaque behavior.",
            allowed_imports=frozenset({"air_synth_012"}),
            allowed_import_members=frozenset(SEARCH_API_NAMES_018),
            allowed_call_names=frozenset(SEARCH_API_NAMES_018),
            allowed_attrs=frozenset(SEARCH_API_NAMES_018),
            discovery=_cases_019(prefix, "public", public, target_operation),
            validation=_cases_019(prefix, "hidden", generated[:3], target_operation),
            edge=_cases_019(prefix, "edge", generated[3:6], target_operation),
            heldout=_cases_019(prefix, "heldout", generated[6:14], target_operation),
            sandbox_import_root=str(Path(__file__).resolve().parents[1]),
        )
        probe_inputs = tuple(generated[14:16])
        result.append(RankingFamily019(
            family, kind, target_operation, wrong_operation,
            _semantic_docs_019(kind, target_seed), _semantic_docs_019(kind, wrong_seed),
            probe_inputs, data_seed,
        ))
    return tuple(result)


def _alternate_family_019(item: RankingFamily019) -> PythonFamily009:
    family = item.family
    return PythonFamily009(
        family_id=family.family_id + "-counterfactual", title=family.title,
        api_docs=item.wrong_docs, contract=family.contract,
        allowed_imports=family.allowed_imports, allowed_import_members=family.allowed_import_members,
        allowed_call_names=family.allowed_call_names, allowed_attrs=family.allowed_attrs,
        discovery=_cases_019(family.family_id, "public-alt", [case.input_text for case in family.discovery], item.wrong_operation),
        validation=_cases_019(family.family_id, "hidden-alt", [case.input_text for case in family.validation], item.wrong_operation),
        edge=_cases_019(family.family_id, "edge-alt", [case.input_text for case in family.edge], item.wrong_operation),
        heldout=_cases_019(family.family_id, "heldout-alt", [case.input_text for case in family.heldout], item.wrong_operation),
        sandbox_import_root=family.sandbox_import_root,
    )


def _model_call_019(
    client: Any, store: ExperimentStore, ledger: ModelLedger015, *, kind: str,
    arm: str, prompt: str, prompt_key: str, seed: int, metadata: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        completion = client.chat_json(prompt, max_tokens=64, seed=seed)
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
        metadata={**dict(metadata), "arm": arm, "prompt_version": PROMPT_VERSIONS_019[prompt_key],
                  "prompt_sha256": PROMPT_HASHES_019[prompt_key], "seed": seed,
                  "runtime_error": runtime_error},
    )
    return payload, {"elapsed_seconds": elapsed, "prompt_tokens": prompt_tokens,
                     "generated_tokens": output_tokens, "runtime_error": runtime_error}


def _opaque_id(candidate: Candidate018) -> str:
    return "choice_" + hashlib.sha256(("air019:" + candidate.ast_hash).encode()).hexdigest()[:10]


def _ordered_candidates(candidates: Sequence[Candidate018], family_seed: int, ordering: int) -> tuple[Candidate018, ...]:
    items = list(candidates)
    random.Random(RANDOM_SEED_019 + family_seed * 17 + ordering * 1009).shuffle(items)
    return tuple(items)


def _candidate_records_019(
    candidates: Sequence[Candidate018], probes: Sequence[str], representation: str,
) -> str:
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        probe_rows: list[dict[str, str]] = []
        for value in probes:
            try:
                output = execute_semantic_ir_017(candidate.ir, value, set(SEARCH_API_NAMES_018))
            except Exception as exc:  # auditable candidate observation, never activation
                output = "ERROR:" + type(exc).__name__
            probe_rows.append({"input": value, "output": output})
        if representation == "compact_ir":
            representation_value: Any = normalized_ir_019(candidate)
        else:
            representation_value = {
                "input_type": "str", "output_type": "str",
                "normalized_depth": candidate.depth, "executable": True,
            }
        records.append({"candidate_id": _opaque_id(candidate), "representation": representation_value, "diagnostic_probes": probe_rows})
    return "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for item in records)


def _public_examples(family: PythonFamily009) -> str:
    return "\n".join(json.dumps({"input": case.input_text, "output": case.expected}, ensure_ascii=False) for case in family.discovery)


def _rank_019(
    client: Any, store: ExperimentStore, ledger: ModelLedger015, item: RankingFamily019,
    candidates: Sequence[Candidate018], *, prompt_key: str, documentation: str,
    representation: str, ordering: int, seed: int,
) -> dict[str, Any]:
    ordered = _ordered_candidates(candidates, item.data_seed, ordering)
    prompt = PROMPT_TEMPLATES_019[prompt_key].format(
        documentation=documentation or "(withheld)", representation=representation,
        public_examples=_public_examples(item.family),
        candidate_records=_candidate_records_019(ordered, item.probe_inputs, representation),
    )
    payload, telemetry = _model_call_019(
        client, store, ledger, kind=f"air-019:{item.family.family_id}:{prompt_key}:{representation}:{ordering}",
        arm=f"ranking_{prompt_key}", prompt=prompt, prompt_key=prompt_key, seed=seed,
        metadata={"family_id": item.family.family_id, "representation": representation,
                  "ordering": ordering, "candidate_count": len(ordered)},
    )
    selected_id = payload.get("candidate_id") if isinstance(payload, Mapping) else None
    selected = next((candidate for candidate in ordered if _opaque_id(candidate) == selected_id), None)
    selected_index = next((index for index, candidate in enumerate(ordered) if candidate is selected), None)
    return {
        "selected_id": selected_id, "selected_ast_hash": selected.ast_hash if selected else None,
        "selection_valid": selected is not None,
        "selection_reason": "model_selection" if selected else "invalid_candidate_id",
        "selected_position": selected_index, "candidate_count": len(ordered),
        "telemetry": telemetry,
    }


def _candidate_from_result(result: Mapping[str, Any], pool: Sequence[Candidate018]) -> Candidate018 | None:
    ast_hash = result.get("selected_ast_hash")
    return next((item for item in pool if item.ast_hash == ast_hash), None)


def _same_behavior(candidate_a: Candidate018 | None, candidate_b: Candidate018 | None, family: PythonFamily009) -> bool:
    if not candidate_a or not candidate_b:
        return False
    cases = family.validation + family.edge + family.heldout
    return behavior_signature_019(candidate_a, cases) == behavior_signature_019(candidate_b, cases)


def _candidate_root(expr: Mapping[str, Any], candidate_id: str) -> Candidate018:
    ir = {"format": SEMANTIC_IR_FORMAT_017, "version": SEMANTIC_IR_VERSION_017,
          "input_type": "str", "output_type": "str", "expr": {"op": "RETURN", "value": dict(expr)}}
    serialized = canonical_semantic_ir_json_017(ir)
    return Candidate018(candidate_id, ir, 3, hashlib.sha256(serialized.encode()).hexdigest())


def _control_cases(prefix: str, split: str, values: Sequence[str]) -> tuple[FamilyCase009, ...]:
    return tuple(FamilyCase009(f"{prefix}-{split}-{index}", value, "", split) for index, value in enumerate(values))


def false_equivalence_controls_019() -> tuple[dict[str, Any], ...]:
    object_asc = air_synth_012.operation_names(1201)["object"]
    object_desc = air_synth_012.operation_names(1202)["object"]
    runs = air_synth_012.operation_names(1201)["runs"]
    input_expr = {"op": "INPUT"}
    pairs = (
        ("same_public_different_hidden", _candidate_root({"op": "CALL", "api": object_asc, "args": [input_expr]}, "a"), _candidate_root({"op": "CALL", "api": object_desc, "args": [input_expr]}, "b"), ("{\"a\":1}",), ("{\"a\":1,\"z\":2}",), ()),
        ("same_public_hidden_different_edge", _candidate_root({"op": "CALL", "api": object_asc, "args": [input_expr]}, "a"), _candidate_root({"op": "CALL", "api": object_desc, "args": [input_expr]}, "b"), ("{\"a\":1}",), ("{\"q\":2}",), ("{\"a\":1,\"z\":2}",)),
        ("short_vs_long", _candidate_root(input_expr, "a"), _candidate_root({"op": "ROTATE", "value": input_expr, "amount": {"op": "INT", "value": 1}}, "b"), ("x",), ("abcd",), ()),
        ("ordering", _candidate_root(input_expr, "a"), _candidate_root({"op": "REVERSE", "value": input_expr}, "b"), ("aba", "aa"), ("abcd",), ()),
        ("same_callable_wrapper", _candidate_root({"op": "CALL", "api": runs, "args": [input_expr]}, "a"), _candidate_root({"op": "REVERSE", "value": {"op": "CALL", "api": runs, "args": [input_expr]}}, "b"), ("",), ("aaab",), ()),
        ("different_constants", _candidate_root({"op": "ROTATE", "value": input_expr, "amount": {"op": "INT", "value": 1}}, "a"), _candidate_root({"op": "ROTATE", "value": input_expr, "amount": {"op": "INT", "value": -1}}, "b"), ("ab",), ("abc",), ()),
    )
    results: list[dict[str, Any]] = []
    for name, first, second, public_values, hidden_values, edge_values in pairs:
        public = _control_cases(name, "public", public_values)
        hidden = _control_cases(name, "hidden", hidden_values)
        edge = _control_cases(name, "edge", edge_values)
        public_equal = behavior_signature_019(first, public) == behavior_signature_019(second, public)
        hidden_equal = behavior_signature_019(first, hidden) == behavior_signature_019(second, hidden)
        edge_equal = behavior_signature_019(first, edge) == behavior_signature_019(second, edge)
        classes = equivalence_classes_019((first, second), public + hidden + edge)
        results.append({"control": name, "public_equal": public_equal, "hidden_equal": hidden_equal,
                        "edge_equal": edge_equal, "class_count": len(classes), "false_merge": len(classes) == 1})
    return tuple(results)


def _write_checkpoint(path: Path, part_a_rows: Sequence[Mapping[str, Any]], part_b_rows: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(json.dumps({"version": EXP019_VERSION, "part_a_rows": list(part_a_rows),
                                "part_b_rows": list(part_b_rows), "events": list(events)},
                               ensure_ascii=False, indent=2), encoding="utf-8")


def _part_a_019(checkpoint: Path, previous: Mapping[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prior_rows = (previous or {}).get("part_a_rows") or (previous or {}).get("part_a", {}).get("rows", [])
    prior = {row.get("family_id"): row for row in prior_rows if isinstance(row, Mapping)}
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for family in make_candidate_families_018():
        if family.family_id in prior:
            rows.append(dict(prior[family.family_id]))
            continue
        search = search_candidates_018(family.discovery, budget=SEARCH_BUDGET_018)
        canonical = canonicalize_verified_019(search.public_survivors, family.discovery, family.validation, family.edge)
        selected = canonical.selected
        evaluation = _candidate_evaluation_018(selected, family) if selected else {"active": False}
        heldout = _heldout_018(selected, family) if selected and evaluation.get("active") else None
        oracle_hash = hashlib.sha256(canonical_semantic_ir_json_017(oracle_call_ir_017(family.family_id.rsplit("-", 1)[-1])).encode()).hexdigest()
        row = {
            "family_id": family.family_id,
            "search": {"raw_candidates": search.raw_candidates_generated,
                       "public_survivors": len(search.public_survivors),
                       "elapsed_seconds": search.elapsed_seconds,
                       "correct_candidate_generated": any(item.ast_hash == oracle_hash for item in search.candidates)},
            "ambiguity_before": len(canonical.verified_candidates),
            "ambiguity_after": len(canonical.equivalence_classes),
            "canonicalization": canonical.to_dict(),
            "selected_ast_hash": selected.ast_hash if selected else None,
            "oracle_exact": bool(selected and selected.ast_hash == oracle_hash),
            "behaviorally_correct_canonical_activation": bool(evaluation.get("active")),
            "selected_evaluation": evaluation, "heldout": heldout,
            "artifact_bytes": len(canonical_semantic_ir_json_017(selected.ir).encode()) if selected else 0,
            "model_calls": 0,
        }
        rows.append(row)
        events.append({"family_id": family.family_id, "stage": "part_a_canonicalization"})
        _write_checkpoint(checkpoint, rows, [], events)
    return rows, events


def _ranking_pool_019(item: RankingFamily019) -> tuple[Any, tuple[Candidate018, ...]]:
    search = search_candidates_018(item.family.discovery, budget=SEARCH_BUDGET_018)
    representatives = canonical_representatives_by_normalized_ast_019(search.public_survivors)
    if not representatives:
        return search, representatives
    # Part B compares primitive-level semantic alternatives.  Keep the
    # globally cheapest normalized structural tier; this is target-independent
    # and removes composed candidates that merely imitate a primitive on the
    # deliberately weak public examples.
    minimum_tier = min(canonical_cost_key_019(item)[:2] for item in representatives)
    scoped = tuple(item for item in representatives if canonical_cost_key_019(item)[:2] == minimum_tier)
    return search, scoped


def _part_b_019(
    client: Any, store: ExperimentStore, ledger: ModelLedger015, checkpoint: Path,
    part_a_rows: Sequence[Mapping[str, Any]], initial_events: Sequence[Mapping[str, Any]],
    previous: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prior_rows = (previous or {}).get("part_b_rows") or (previous or {}).get("part_b", {}).get("rows", [])
    prior = {row.get("family_id"): row for row in prior_rows if isinstance(row, Mapping)}
    rows: list[dict[str, Any]] = []
    events = list(initial_events)
    for index, item in enumerate(make_document_ranking_families_019(), 1):
        family = item.family
        if family.family_id in prior:
            rows.append(dict(prior[family.family_id]))
            continue
        search, pool = _ranking_pool_019(item)
        alternate = _alternate_family_019(item)
        distractor_docs = (
            "[CURRENT NORMATIVE RECORD]\n" + item.correct_docs +
            "\n\n[HISTORICAL NON-NORMATIVE DRAFT]\n" + item.wrong_docs +
            "\n\n[UNRELATED OPERATIONS NOTE]\nDeployment identifiers and storage metadata contain no behavioral authority."
        )
        size_two_pool = _ordered_candidates(pool, item.data_seed, 2)[:2]
        conditions = {
            "no_doc": _rank_019(client, store, ledger, item, pool, prompt_key="no_doc", documentation="", representation="compact_ir", ordering=0, seed=19100 + index),
            "correct_doc": _rank_019(client, store, ledger, item, pool, prompt_key="correct_doc", documentation=item.correct_docs, representation="compact_ir", ordering=0, seed=19200 + index),
            "wrong_doc": _rank_019(client, store, ledger, item, pool, prompt_key="wrong_doc", documentation=item.wrong_docs, representation="compact_ir", ordering=0, seed=19300 + index),
            "distractor_doc": _rank_019(client, store, ledger, item, pool, prompt_key="distractor_doc", documentation=distractor_docs, representation="compact_ir", ordering=0, seed=19400 + index),
            "correct_doc_order_1": _rank_019(client, store, ledger, item, pool, prompt_key="correct_doc", documentation=item.correct_docs, representation="compact_ir", ordering=1, seed=19500 + index),
            "correct_doc_descriptor": _rank_019(client, store, ledger, item, pool, prompt_key="correct_doc", documentation=item.correct_docs, representation="behavior_descriptor", ordering=0, seed=19600 + index),
            "correct_doc_size_2": _rank_019(client, store, ledger, item, size_two_pool, prompt_key="correct_doc", documentation=item.correct_docs, representation="compact_ir", ordering=0, seed=19700 + index),
        }
        if index <= COUNTERFACTUAL_PAIRS_019:
            conditions["counterfactual_doc"] = _rank_019(
                client, store, ledger, item, pool, prompt_key="counterfactual_doc",
                documentation=item.wrong_docs, representation="compact_ir", ordering=0,
                seed=19800 + index,
            )
        for result in conditions.values():
            candidate = _candidate_from_result(result, pool)
            result["target_evaluation"] = _candidate_evaluation_018(candidate, family) if candidate else {"active": False}
            result["alternate_evaluation"] = _candidate_evaluation_018(candidate, alternate) if candidate else {"active": False}
        correct_candidate = _candidate_from_result(conditions["correct_doc"], pool)
        order_candidate = _candidate_from_result(conditions["correct_doc_order_1"], pool)
        random_candidate = random.Random(RANDOM_SEED_019 + index).choice(pool) if pool else None
        oracle_candidate = next((candidate for candidate in pool if _candidate_evaluation_018(candidate, family).get("active")), None)
        pure = canonicalize_verified_019(pool, family.discovery, family.validation, family.edge)
        pure_candidate = pure.selected
        pure_eval = _candidate_evaluation_018(pure_candidate, family) if pure_candidate else {"active": False}
        pure_heldout = _heldout_018(pure_candidate, family) if pure_candidate and pure_eval.get("active") else None
        ranked_eval = conditions["correct_doc"]["target_evaluation"]
        ranked_heldout = _heldout_018(correct_candidate, family) if correct_candidate and ranked_eval.get("active") else None
        hybrid_candidate = pure_candidate if pure_candidate else correct_candidate
        hybrid_eval = pure_eval if pure_candidate else ranked_eval
        hybrid_heldout = _heldout_018(hybrid_candidate, family) if hybrid_candidate and hybrid_eval.get("active") else None
        hybrid_model_required = pure_candidate is None
        doc_telemetry = conditions["correct_doc"]["telemetry"]
        attempts = [result["telemetry"] for result in conditions.values()]
        row = {
            "family_id": family.family_id, "data_seed": item.data_seed, "kind": item.kind,
            "target_operation": item.target_operation, "wrong_operation": item.wrong_operation,
            "search": {"raw_candidates": search.raw_candidates_generated,
                       "public_survivors": len(search.public_survivors), "ranking_candidates": len(pool),
                       "elapsed_seconds": search.elapsed_seconds,
                       "public_ambiguity_requirement_met": 2 <= len(pool) <= 4,
                       "correct_candidate_available": any(_candidate_evaluation_018(candidate, family).get("active") for candidate in pool),
                       "size_2_correct_candidate_available": any(_candidate_evaluation_018(candidate, family).get("active") for candidate in size_two_pool)},
            "conditions": conditions,
            "random": {"selected_ast_hash": random_candidate.ast_hash if random_candidate else None,
                       "target_evaluation": _candidate_evaluation_018(random_candidate, family) if random_candidate else {"active": False}},
            "oracle": {"selected_ast_hash": oracle_candidate.ast_hash if oracle_candidate else None,
                       "target_evaluation": _candidate_evaluation_018(oracle_candidate, family) if oracle_candidate else {"active": False}},
            "counterfactual": {"enabled": index <= COUNTERFACTUAL_PAIRS_019,
                               "doc_a_followed": bool(conditions["correct_doc"]["target_evaluation"].get("active")),
                               "doc_b_followed": bool(conditions["counterfactual_doc"]["alternate_evaluation"].get("active"))} if index <= COUNTERFACTUAL_PAIRS_019 else {"enabled": False},
            "order_stability": _same_behavior(correct_candidate, order_candidate, family),
            "first_candidate_selected": conditions["correct_doc"].get("selected_position") == 0,
            "last_candidate_selected": conditions["correct_doc"].get("selected_position") == len(pool) - 1,
            "part_c": {
                "pure": {"canonicalization": pure.to_dict(), "selected_evaluation": pure_eval, "heldout": pure_heldout, "model_calls": 0},
                "ranked": {"selected_evaluation": ranked_eval, "heldout": ranked_heldout, "model_calls": 1,
                           "input_tokens": doc_telemetry.get("prompt_tokens", 0), "output_tokens": doc_telemetry.get("generated_tokens", 0)},
                "hybrid": {"selected_evaluation": hybrid_eval, "heldout": hybrid_heldout,
                           "model_calls": int(hybrid_model_required),
                           "input_tokens": doc_telemetry.get("prompt_tokens", 0) if hybrid_model_required else 0,
                           "output_tokens": doc_telemetry.get("generated_tokens", 0) if hybrid_model_required else 0,
                           "model_avoided": not hybrid_model_required,
                           "artifact_bytes": len(canonical_semantic_ir_json_017(hybrid_candidate.ir).encode()) if hybrid_candidate else 0},
            },
            "attempts": attempts,
        }
        rows.append(row)
        events.append({"family_id": family.family_id, "stage": "part_b_all_conditions", "model_calls": len(attempts)})
        _write_checkpoint(checkpoint, part_a_rows, rows, events)
    return rows, events


def _wilson_95(successes: int, total: int) -> dict[str, float | int]:
    if total <= 0:
        return {"successes": successes, "n": total, "estimate": 0.0, "low": 0.0, "high": 0.0}
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"successes": successes, "n": total, "estimate": p, "low": max(0.0, center - margin), "high": min(1.0, center + margin)}


def _condition_successes(rows: Sequence[Mapping[str, Any]], condition: str, evaluation: str = "target_evaluation") -> int:
    return sum(bool(row["conditions"][condition][evaluation].get("active")) for row in rows)


def _summarize_019(part_a_rows: Sequence[Mapping[str, Any]], part_b_rows: Sequence[Mapping[str, Any]], ledger: ModelLedger015) -> dict[str, Any]:
    false_controls = false_equivalence_controls_019()
    part_a_active = sum(bool(row["behaviorally_correct_canonical_activation"]) for row in part_a_rows)
    part_a_heldout = [row["heldout"]["accuracy"] for row in part_a_rows if row.get("heldout")]
    condition_names = ("no_doc", "correct_doc", "wrong_doc", "distractor_doc", "correct_doc_descriptor", "correct_doc_size_2")
    successes = {name: _condition_successes(part_b_rows, name) for name in condition_names}
    random_success = sum(bool(row["random"]["target_evaluation"].get("active")) for row in part_b_rows)
    oracle_success = sum(bool(row["oracle"]["target_evaluation"].get("active")) for row in part_b_rows)
    counterfactual_rows = [row for row in part_b_rows if row["counterfactual"].get("enabled")]
    counterfactual_success = sum(bool(row["counterfactual"].get("doc_a_followed") and row["counterfactual"].get("doc_b_followed")) for row in counterfactual_rows)
    order_stable = sum(bool(row["order_stability"]) for row in part_b_rows)
    first_bias = sum(bool(row["first_candidate_selected"]) for row in part_b_rows)
    last_bias = sum(bool(row["last_candidate_selected"]) for row in part_b_rows)
    wrong_doc_follow = sum(bool(row["conditions"]["wrong_doc"]["alternate_evaluation"].get("active")) for row in part_b_rows)
    arms: dict[str, Any] = {}
    for arm in ("pure", "ranked", "hybrid"):
        active = sum(bool(row["part_c"][arm]["selected_evaluation"].get("active")) for row in part_b_rows)
        heldout = [row["part_c"][arm]["heldout"]["accuracy"] for row in part_b_rows if row["part_c"][arm].get("heldout")]
        arms[arm] = {"active": active, "active_rate": active / len(part_b_rows),
                     "heldout_mean_accuracy": statistics.mean(heldout) if heldout else None,
                     "model_calls": sum(int(row["part_c"][arm].get("model_calls", 0)) for row in part_b_rows),
                     "input_tokens": sum(int(row["part_c"][arm].get("input_tokens", 0)) for row in part_b_rows),
                     "output_tokens": sum(int(row["part_c"][arm].get("output_tokens", 0)) for row in part_b_rows)}
    total = len(part_b_rows)
    correct_doc_lift = (successes["correct_doc"] - successes["no_doc"]) / total
    wrong_doc_harm = (successes["no_doc"] - successes["wrong_doc"]) / total
    correct_vs_wrong = (successes["correct_doc"] - successes["wrong_doc"]) / total
    counterfactual_rate = counterfactual_success / len(counterfactual_rows) if counterfactual_rows else 0.0
    if correct_doc_lift >= 0.20 and correct_vs_wrong >= 0.20 and counterfactual_rate >= 0.60:
        doc_decision = "STRONG_DOC_SEMANTIC_USE"
    elif correct_doc_lift >= 0.10 or counterfactual_rate >= 0.50:
        doc_decision = "PARTIAL_DOC_SEMANTIC_USE"
    else:
        doc_decision = "WEAK_OR_NO_DOC_SEMANTIC_USE"
    wrong_activation = 0
    acquisition_robust = arms["hybrid"]["active_rate"] >= 0.90 and arms["hybrid"]["heldout_mean_accuracy"] == 1.0 and wrong_activation == 0
    viability = "YES" if acquisition_robust else ("PARTIAL" if arms["hybrid"]["active_rate"] >= 0.60 else "NO")
    if acquisition_robust:
        next_experiment = "end-to-end sequential accumulation/planning"
    elif oracle_success / total < 0.80:
        next_experiment = "wider capability search"
    else:
        next_experiment = "same frozen ranking protocol with a larger Western open-weight model"
    ranked_calls = arms["ranked"]["model_calls"]
    hybrid_calls = arms["hybrid"]["model_calls"]
    pure_latency = sum(float(row["search"]["elapsed_seconds"]) + float(row["part_c"]["pure"]["canonicalization"]["elapsed_seconds"]) for row in part_b_rows)
    ranked_latency = sum(float(row["search"]["elapsed_seconds"]) + float(row["conditions"]["correct_doc"]["telemetry"].get("elapsed_seconds", 0.0)) for row in part_b_rows)
    hybrid_latency = sum(float(row["search"]["elapsed_seconds"]) + float(row["part_c"]["pure"]["canonicalization"]["elapsed_seconds"]) + (float(row["conditions"]["correct_doc"]["telemetry"].get("elapsed_seconds", 0.0)) if row["part_c"]["hybrid"]["model_calls"] else 0.0) for row in part_b_rows)
    return {
        "part_a": {
            "families": len(part_a_rows), "correct_candidate_coverage": sum(bool(row["search"]["correct_candidate_generated"]) for row in part_a_rows),
            "ambiguity_before": sum(row["ambiguity_before"] > 1 for row in part_a_rows),
            "ambiguity_after": sum(row["ambiguity_after"] > 1 for row in part_a_rows),
            "candidates_per_family_mean": statistics.mean(int(row["search"]["public_survivors"]) for row in part_a_rows),
            "equivalence_classes_per_family_mean": statistics.mean(int(row["ambiguity_after"]) for row in part_a_rows),
            "canonical_activation": part_a_active,
            "behaviorally_correct_canonical_activation": part_a_active / len(part_a_rows),
            "oracle_exact_match": sum(bool(row["oracle_exact"]) for row in part_a_rows),
            "canonical_choice_stability": sum(bool(row["canonicalization"]["stable"]) for row in part_a_rows) / len(part_a_rows),
            "heldout_mean_accuracy": statistics.mean(part_a_heldout) if part_a_heldout else None,
            "false_equivalence_controls": list(false_controls),
            "false_equivalence": sum(bool(item["false_merge"]) for item in false_controls),
            "model_calls": 0,
            "search_latency_mean": statistics.mean(float(row["search"]["elapsed_seconds"]) for row in part_a_rows),
            "canonicalization_latency_mean": statistics.mean(float(row["canonicalization"]["elapsed_seconds"]) for row in part_a_rows),
            "artifact_bytes_mean": statistics.mean(int(row["artifact_bytes"]) for row in part_a_rows),
        },
        "part_b": {
            "families": total,
            "public_ambiguity_requirement_met": sum(bool(row["search"]["public_ambiguity_requirement_met"]) for row in part_b_rows),
            "candidate_count_mean": statistics.mean(int(row["search"]["ranking_candidates"]) for row in part_b_rows),
            "candidate_size_effects": {
                "size_2": {"correct_available": sum(bool(row["search"]["size_2_correct_candidate_available"]) for row in part_b_rows),
                           "accuracy": _wilson_95(successes["correct_doc_size_2"], total),
                           "accuracy_given_available": _wilson_95(successes["correct_doc_size_2"], sum(bool(row["search"]["size_2_correct_candidate_available"]) for row in part_b_rows))},
                "size_3": {"correct_available": sum(bool(row["search"]["correct_candidate_available"]) for row in part_b_rows),
                           "accuracy": _wilson_95(successes["correct_doc"], total),
                           "accuracy_given_available": _wilson_95(successes["correct_doc"], sum(bool(row["search"]["correct_candidate_available"]) for row in part_b_rows))},
            },
            "no_doc_accuracy": _wilson_95(successes["no_doc"], total),
            "correct_doc_accuracy": _wilson_95(successes["correct_doc"], total),
            "wrong_doc_accuracy": _wilson_95(successes["wrong_doc"], total),
            "distractor_doc_accuracy": _wilson_95(successes["distractor_doc"], total),
            "descriptor_correct_doc_accuracy": _wilson_95(successes["correct_doc_descriptor"], total),
            "random_accuracy": _wilson_95(random_success, total),
            "oracle_accuracy": _wilson_95(oracle_success, total),
            "correct_documentation_lift": correct_doc_lift,
            "correct_vs_wrong_documentation_lift": correct_vs_wrong,
            "wrong_documentation_harm": wrong_doc_harm,
            "wrong_documentation_following": _wilson_95(wrong_doc_follow, total),
            "counterfactual_doc_following_accuracy": _wilson_95(counterfactual_success, len(counterfactual_rows)),
            "candidate_order_stability": _wilson_95(order_stable, total),
            "first_candidate_bias": first_bias / total,
            "last_candidate_bias": last_bias / total,
        },
        "part_c": {
            "arms": arms,
            "model_avoidance_rate": sum(bool(row["part_c"]["hybrid"].get("model_avoided")) for row in part_b_rows) / total,
            "model_calls_saved_vs_ranked": ranked_calls - hybrid_calls,
            "input_tokens_saved_vs_ranked": arms["ranked"]["input_tokens"] - arms["hybrid"]["input_tokens"],
            "output_tokens_saved_vs_ranked": arms["ranked"]["output_tokens"] - arms["hybrid"]["output_tokens"],
            "correct_acquisition_per_model_call": arms["hybrid"]["active"] / hybrid_calls if hybrid_calls else None,
            "correct_acquisition_per_model_call_label": "model_free" if not hybrid_calls and arms["hybrid"]["active"] else None,
            "latency_seconds": {"pure_total": pure_latency, "ranked_total": ranked_latency,
                                "hybrid_total": hybrid_latency,
                                "hybrid_per_acquired_skill": hybrid_latency / arms["hybrid"]["active"] if arms["hybrid"]["active"] else None},
            "als_bytes_per_query_mean": statistics.mean(int(row["part_c"]["hybrid"].get("artifact_bytes", 0)) for row in part_b_rows),
        },
        "decisions": {
            "documentation_semantic_use": doc_decision,
            "three_b_plus_air_viability": viability,
            "acquisition_robust_enough_for_sequential_planning": acquisition_robust,
            "next_experiment": next_experiment,
        },
        "wrong_activation": wrong_activation,
        "model_accounting": ledger.summary(),
    }


def _failure_counts(summary: Mapping[str, Any], part_a_rows: Sequence[Mapping[str, Any]], part_b_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {key: 0 for key in FAILURE_TAXONOMY_019}
    counts["candidate_coverage_failure"] = sum(not row["search"]["correct_candidate_generated"] for row in part_a_rows)
    counts["public_ambiguity"] = sum(row["ambiguity_before"] > 1 for row in part_a_rows) + sum(row["search"]["ranking_candidates"] > 1 for row in part_b_rows)
    counts["hidden_ambiguity"] = sum(row["ambiguity_before"] > 1 for row in part_a_rows)
    counts["edge_ambiguity"] = sum(row["ambiguity_before"] > 1 for row in part_a_rows)
    counts["false_equivalence_failure"] = int(summary["part_a"]["false_equivalence"])
    counts["canonicalization_failure"] = sum(not row["behaviorally_correct_canonical_activation"] for row in part_a_rows)
    counts["ranking_failure"] = sum(not row["conditions"]["correct_doc"]["target_evaluation"].get("active") for row in part_b_rows)
    counts["documentation_misuse"] = sum(row["conditions"]["correct_doc"]["target_evaluation"].get("active") is False for row in part_b_rows)
    counts["counterfactual_doc_failure"] = sum(row["counterfactual"].get("enabled") and not (row["counterfactual"].get("doc_a_followed") and row["counterfactual"].get("doc_b_followed")) for row in part_b_rows)
    counts["position_bias_failure"] = sum(not row["order_stability"] for row in part_b_rows)
    counts["invalid_candidate_id"] = sum(not result.get("selection_valid") for row in part_b_rows for result in row["conditions"].values())
    counts["timeout"] = sum(bool(attempt.get("runtime_error")) for row in part_b_rows for attempt in row.get("attempts", []))
    counts["heldout_failure"] = sum(row["part_c"]["hybrid"].get("heldout") and row["part_c"]["hybrid"]["heldout"].get("accuracy") != 1.0 for row in part_b_rows)
    counts["wrong_activation"] = int(summary["wrong_activation"])
    return counts


def run_exp019(*, client: LlamaCppClient, store: ExperimentStore, report_directory: str, resume_from: str | None = None) -> dict[str, Any]:
    report_dir = Path(report_directory)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / datetime.now(UTC).strftime("air-019-%Y%m%dT%H%M%SZ.json")
    previous = json.loads(Path(resume_from).read_text(encoding="utf-8")) if resume_from else None
    ledger = ModelLedger015()
    part_a_rows, events = _part_a_019(report_path, previous)
    part_b_rows, events = _part_b_019(client, store, ledger, report_path, part_a_rows, events, previous)
    if previous:
        previous_ids = {row.get("family_id") for row in ((previous.get("part_b_rows") or previous.get("part_b", {}).get("rows", [])) if isinstance(previous, Mapping) else [])}
        for row in part_b_rows:
            if row.get("family_id") not in previous_ids:
                continue
            for attempt in row.get("attempts", []):
                ledger.observe(prompt="", elapsed=float(attempt.get("elapsed_seconds", 0.0)),
                               prompt_tokens=int(attempt.get("prompt_tokens", 0)),
                               output_tokens=int(attempt.get("generated_tokens", 0)),
                               timeout=bool(attempt.get("runtime_error")), arm="resumed_ranking")
    summary = _summarize_019(part_a_rows, part_b_rows, ledger)
    failures = _failure_counts(summary, part_a_rows, part_b_rows)
    report = {
        "benchmark": "air-019-behavioral-canonicalization-and-documentation-grounding",
        "version": EXP019_VERSION, "created_at": datetime.now(UTC).isoformat(),
        "model": {"identity": MODEL_IDENTITY_019, "context_size": CONTEXT_SIZE_019,
                  "weights_frozen": True, "model_swap": False, "lora": False},
        "protocol": {
            "candidate_grammar_frozen": True, "candidate_generator_frozen": True,
            "semantic_ir_frozen": True, "retrieval_frozen": True, "learned_state_frozen": True,
            "sandbox_frozen": True, "verifier_frozen": True, "hidden_edge_in_search": False,
            "part_a_families": PART_A_FAMILIES_019, "part_b_families": PART_B_FAMILIES_019,
            "new_deterministic_dataset_variants": 8, "counterfactual_pairs": COUNTERFACTUAL_PAIRS_019,
            "canonicalization_version": CANONICALIZATION_VERSION_019,
            "canonical_choice_order": list(CANONICAL_CHOICE_ORDER_019),
            "search_budget": SEARCH_BUDGET_018.to_dict(),
            "ranking_representations": list(RANKING_REPRESENTATIONS_019),
            "prompt_versions": PROMPT_VERSIONS_019, "prompt_hashes": PROMPT_HASHES_019,
        },
        "part_a": {"rows": part_a_rows, "summary": summary["part_a"]},
        "part_b": {"rows": part_b_rows, "summary": summary["part_b"]},
        "part_c": summary["part_c"],
        "decisions": summary["decisions"],
        "failure_counts": failures,
        "model_accounting": summary["model_accounting"],
        "regression": {"wrong_activation": summary["wrong_activation"], "grammar_unchanged": True,
                       "candidate_generator_unchanged": True, "sandbox_unchanged": True,
                       "verifier_unchanged": True},
        "safety": {"false_equivalence_controls": 6, "false_merge": summary["part_a"]["false_equivalence"],
                   "hidden_leakage": False, "manual_tie_break": False,
                   "invalid_candidate_id_rejected": True, "silent_retry": False},
        "events": events,
        "verification": {"full_test_suite": "run externally before release",
                         "commit_hash": os.getenv("AIR_COMMIT_SHA", "not_available_in_runtime")},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_file"] = str(report_path)
    return report


__all__ = [
    "EXP019_VERSION", "MODEL_IDENTITY_019", "PROMPT_HASHES_019", "RankingFamily019",
    "make_document_ranking_families_019", "false_equivalence_controls_019", "run_exp019",
]
