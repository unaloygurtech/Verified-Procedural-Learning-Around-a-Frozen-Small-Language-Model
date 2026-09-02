"""Experiment 0007: capability-gap detection and missing-skill synthesis.

The 0006 library is deliberately frozen while a new task family is presented.
AIR must first diagnose that no existing composition works, then learn one
missing intermediate behavior from an independent experience set, validate it,
and finally re-run composition search over the extended artifact library.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import itertools
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .exp005 import Expr, STRING, TypeCheckError, binary, execute_expr, field, literal, type_of, unary
from .exp006 import (
    CompositionCase,
    CompositionPlan,
    PrimitiveSkill,
    _all_ids,
    _binary_ids,
    _no_valid,
    _ok,
    _skill_map,
    check_plan_types,
    execute_composition,
    learn_primitive_library,
    raw_primitive_experiences_006,
)
from .model_client import LlamaCppClient
from .neralis import parse_response
from .store import ExperimentStore


@dataclass(frozen=True)
class MissingSkillCase:
    case_id: str
    payload: dict[str, str]
    expected: str


MISSING_SKILL_TRAINING_007 = (
    MissingSkillCase("f-01", {"input": "Alpha Beta"}, "Alpha_Beta"),
    MissingSkillCase("f-02", {"input": "MiXeD Case"}, "MiXeD_Case"),
    MissingSkillCase("f-03", {"input": "  Edge Case  "}, "__Edge_Case__"),
    MissingSkillCase("f-04", {"input": "Under_score Test"}, "Under_score_Test"),
)

MISSING_SKILL_VALIDATION_007 = (
    MissingSkillCase("fv-01", {"input": "New York"}, "New_York"),
    MissingSkillCase("fv-02", {"input": "  Two Words  "}, "__Two_Words__"),
    MissingSkillCase("fv-03", {"input": "A  B"}, "A__B"),
)

MISSING_SKILL_EDGE_007 = (
    MissingSkillCase("fe-01", {"input": ""}, ""),
    MissingSkillCase("fe-02", {"input": "Tab\tSeparated"}, "Tab\tSeparated"),
    MissingSkillCase("fe-03", {"input": "  Several  Internal Spaces  "}, "__Several__Internal_Spaces__"),
)


GAP_VALIDATION_007 = (
    CompositionCase("gv-01", "g1", "  Alpha Beta  ", "unused", _ok("alpha_beta "), "gap-depth-3"),
    CompositionCase("gv-02", "g2", "  Alpha Beta  ", " Gamma Delta ", _ok("alpha_beta/gamma_delta"), "gap-depth-5"),
    CompositionCase("gv-03", "g3", "  Alpha Beta  ", " Gamma Delta ", _ok("gamma_delta/alpha_beta"), "gap-depth-5"),
    CompositionCase("gx-01", "gx", "  Unknown Task  ", "unused", _no_valid(), "impossible"),
    CompositionCase("gx-02", "gx", "Another Input", "unused", _no_valid(), "impossible"),
)

GAP_HELD_OUT_007 = (
    CompositionCase("gh-01", "g1", "  November Rain  ", "unused", _ok("november_rain "), "gap-depth-3"),
    CompositionCase("gh-02", "g1", "Oscar Hotel", "unused", _ok("oscar_hotel "), "gap-depth-3"),
    CompositionCase("gh-03", "g1", "  Papa  ", "unused", _ok("papa "), "gap-depth-3"),
    CompositionCase("gh-04", "g2", "  Quebec Tango  ", " Sierra Uniform ", _ok("quebec_tango/sierra_uniform"), "gap-depth-5"),
    CompositionCase("gh-05", "g2", "Victor Whiskey", "  Xray Yankee  ", _ok("victor_whiskey/xray_yankee"), "gap-depth-5"),
    CompositionCase("gh-06", "g2", "Zulu Alpha", "Bravo Charlie", _ok("zulu_alpha/bravo_charlie"), "gap-depth-5"),
    CompositionCase("gh-07", "g3", "  Delta Echo  ", " Foxtrot Golf ", _ok("foxtrot_golf/delta_echo"), "gap-depth-5"),
    CompositionCase("gh-08", "g3", "Hotel India", "  Juliet Kilo  ", _ok("juliet_kilo/hotel_india"), "gap-depth-5"),
    CompositionCase("gh-09", "g3", "Lima Mike", "November Oscar", _ok("november_oscar/lima_mike"), "gap-depth-5"),
    CompositionCase("gh-10", "gz", "  No Such  ", "Procedure", _no_valid(), "impossible"),
    CompositionCase("gh-11", "gz", "Unknown", " Combination ", _no_valid(), "impossible"),
    CompositionCase("gh-12", "gz", "Missing", "Skill", _no_valid(), "impossible"),
)


@dataclass(frozen=True)
class MissingSkillSynthesisStats:
    candidate_search_count: int
    type_invalid_candidate_count: int
    semantic_training_rejected_count: int
    ambiguous_candidate_count: int


def _replace_spaces(source: Expr) -> Expr:
    return Expr("REPLACE", args=(source, literal(" "), literal("_")))


def _missing_skill_candidates() -> tuple[Expr, ...]:
    source = field("input")
    candidates = (
        source,
        unary("TRIM", source),
        unary("LOWER", source),
        unary("LOWER", unary("TRIM", source)),
        _replace_spaces(source),
        unary("LOWER", _replace_spaces(source)),
        _replace_spaces(unary("TRIM", source)),
        _replace_spaces(unary("LOWER", unary("TRIM", source))),
        binary("ADD", source, literal("_")),
    )
    unique: dict[str, Expr] = {}
    for candidate in candidates:
        key = json.dumps(candidate.to_dict(), sort_keys=True, separators=(",", ":"))
        unique.setdefault(key, candidate)
    return tuple(unique.values())


def synthesize_missing_skill_007(skill_id: str = "skill-6") -> tuple[PrimitiveSkill, MissingSkillSynthesisStats]:
    matches: list[Expr] = []
    type_invalid = semantic_rejected = 0
    candidates = _missing_skill_candidates()
    for expression in candidates:
        try:
            if type_of(expression, {"input": STRING}) != STRING:
                type_invalid += 1
                continue
        except TypeCheckError:
            type_invalid += 1
            continue
        try:
            if all(execute_expr(expression, case.payload) == case.expected for case in MISSING_SKILL_TRAINING_007):
                matches.append(expression)
            else:
                semantic_rejected += 1
        except TypeCheckError:
            semantic_rejected += 1
    if len(matches) != 1:
        raise ValueError(f"missing skill has {len(matches)} valid explanations")
    skill = PrimitiveSkill(
        skill_id=skill_id,
        version=1,
        input_fields=("input",),
        output_type=STRING,
        expression=matches[0],
        source_set_id="missing-set-007",
        source_case_ids=tuple(case.case_id for case in MISSING_SKILL_TRAINING_007),
    )
    return skill, MissingSkillSynthesisStats(len(candidates), type_invalid, semantic_rejected, 0)


@dataclass(frozen=True)
class SkillGate007:
    condition: str
    correct: int
    total: int
    accuracy: float


def _skill_gate(condition: str, skill: PrimitiveSkill, cases: tuple[MissingSkillCase, ...]) -> SkillGate007:
    correct = 0
    for case in cases:
        try:
            correct += int(execute_expr(skill.expression, case.payload) == case.expected)
        except TypeCheckError:
            pass
    return SkillGate007(condition, correct, len(cases), correct / len(cases) if cases else 0.0)


@dataclass(frozen=True)
class GapSearchStats:
    candidate_composition_count: int
    candidate_evaluation_count: int
    type_invalid_composition_count: int
    semantic_validation_rejected_count: int
    ambiguous_composition_count: int


def iter_gap_composition_candidates(skills: tuple[PrimitiveSkill, ...]) -> Iterable[CompositionPlan]:
    """Enumerate bounded chains up to three unary steps or two per binary branch."""
    all_ids = _all_ids(skills)
    for length in (1, 2, 3):
        for chain in itertools.product(all_ids, repeat=length):
            yield CompositionPlan(left_chain=chain)
    branch_options = ((),) + tuple((skill_id,) for skill_id in all_ids)
    branch_options += tuple(itertools.product(all_ids, repeat=2))
    for left, right, binary_skill in itertools.product(branch_options, branch_options, all_ids):
        yield CompositionPlan(left_chain=left, right_chain=right, binary_skill=binary_skill)


def search_gap_compositions(
    skills: tuple[PrimitiveSkill, ...], cases: tuple[CompositionCase, ...]
) -> tuple[dict[str, CompositionPlan], dict[str, str], GapSearchStats]:
    grouped: dict[str, list[CompositionCase]] = {}
    for case in cases:
        grouped.setdefault(case.task_token, []).append(case)
    candidates = tuple(iter_gap_composition_candidates(skills))
    plans: dict[str, CompositionPlan] = {}
    no_valid: dict[str, str] = {}
    evaluations = type_invalid = semantic_rejected = ambiguous = 0
    for task_token, task_cases in sorted(grouped.items()):
        matches: list[CompositionPlan] = []
        for plan in candidates:
            evaluations += 1
            try:
                check_plan_types(plan, skills)
            except (KeyError, TypeCheckError):
                type_invalid += 1
                continue
            try:
                matches_case = all(
                    case.expected is not None and execute_composition(plan, skills, case) == case.expected
                    for case in task_cases
                )
            except (KeyError, TypeCheckError):
                matches_case = False
            if matches_case:
                matches.append(plan)
            else:
                semantic_rejected += 1
        if len(matches) == 1:
            plans[task_token] = matches[0]
        elif len(matches) > 1:
            ambiguous += 1
            no_valid[task_token] = "ambiguous composition"
        else:
            no_valid[task_token] = "no valid composition"
    return plans, no_valid, GapSearchStats(len(candidates), evaluations, type_invalid, semantic_rejected, ambiguous)


@dataclass(frozen=True)
class GapDiagnosis:
    task_token: str
    status: str
    reason: str


def diagnose_capability_gaps(
    cases: tuple[CompositionCase, ...], plans: dict[str, CompositionPlan], no_valid: dict[str, str]
) -> tuple[GapDiagnosis, ...]:
    grouped: dict[str, list[CompositionCase]] = {}
    for case in cases:
        grouped.setdefault(case.task_token, []).append(case)
    diagnoses: list[GapDiagnosis] = []
    for task_token, task_cases in sorted(grouped.items()):
        positive = any(case.expected and case.expected.get("status") == "ok" for case in task_cases)
        if task_token in plans:
            diagnoses.append(GapDiagnosis(task_token, "covered", "existing composition matched validation evidence"))
        elif positive and task_token in no_valid:
            diagnoses.append(GapDiagnosis(task_token, "gap_detected", "positive task evidence has no valid existing composition"))
        else:
            diagnoses.append(GapDiagnosis(task_token, "safe_unknown", "no valid composition and no positive evidence"))
    return tuple(diagnoses)


def raw_missing_experiences_007() -> str:
    lines = ["Verified independent input/output experiences (operation name omitted):"]
    for case in MISSING_SKILL_TRAINING_007:
        lines.append(json.dumps({"input": case.payload, "verified_output": case.expected}, sort_keys=True))
    return "\n".join(lines)


def raw_gap_context_007() -> str:
    return raw_primitive_experiences_006() + "\n\n" + raw_missing_experiences_007()


def gap_prompt_007(case: CompositionCase, context: str | None = None) -> str:
    sections = []
    if context:
        sections.append(context)
    sections.extend(
        (
            "Solve this task using verified behaviors if possible.",
            "If no valid solution exists, return exactly {\"status\":\"no_valid_composition\"}.",
            f"input={json.dumps(case.payload(), sort_keys=True)}",
            "For a solved task return exactly {\"status\":\"ok\",\"value\":\"...\"}.",
        )
    )
    return "\n\n".join(sections)


@dataclass(frozen=True)
class Result007:
    condition: str
    valid_correct: int
    valid_total: int
    valid_accuracy: float
    impossible_rejected: int
    impossible_total: int
    safe_rejection_rate: float
    total_prompt_tokens: int
    total_generated_tokens: int
    average_seconds: float


def _result007(
    condition: str,
    valid_correct: int,
    valid_total: int,
    rejected: int,
    impossible_total: int,
    prompt_tokens: int = 0,
    generated_tokens: int = 0,
    elapsed: list[float] | None = None,
) -> Result007:
    return Result007(
        condition,
        valid_correct,
        valid_total,
        valid_correct / valid_total if valid_total else 0.0,
        rejected,
        impossible_total,
        rejected / impossible_total if impossible_total else 0.0,
        prompt_tokens,
        generated_tokens,
        mean(elapsed) if elapsed else 0.0,
    )


def run_gap_compositions(
    *, store: ExperimentStore, condition: str, cases: tuple[CompositionCase, ...], plans: dict[str, CompositionPlan], skills: tuple[PrimitiveSkill, ...], phase: str,
) -> Result007:
    valid = [case for case in cases if case.expected and case.expected.get("status") == "ok"]
    impossible = [case for case in cases if case.expected and case.expected.get("status") == "no_valid_composition"]
    correct = rejected = 0
    for case in cases:
        if case.task_token not in plans:
            parsed = _no_valid()
        else:
            try:
                parsed = execute_composition(plans[case.task_token], skills, case)
            except (KeyError, TypeCheckError):
                parsed = _no_valid()
        passed = parsed == case.expected
        correct += int(case in valid and passed)
        rejected += int(case in impossible and parsed.get("status") == "no_valid_composition")
        store.record_run(
            kind=f"air-007:{phase}:{condition}",
            prompt=f"deterministic gap input={json.dumps(case.payload(), sort_keys=True)}",
            response=json.dumps(parsed, sort_keys=True),
            elapsed_seconds=0.0,
            prompt_tokens=0,
            generated_tokens=0,
            passed=passed,
            metadata={"case_id": case.case_id, "task_token": case.task_token, "expected": case.expected, "parsed": parsed},
        )
    return _result007(condition, correct, len(valid), rejected, len(impossible))


def run_model_007(
    *, cases: tuple[CompositionCase, ...], condition: str, context: str | None, client: LlamaCppClient, store: ExperimentStore, phase: str,
) -> Result007:
    valid = [case for case in cases if case.expected and case.expected.get("status") == "ok"]
    impossible = [case for case in cases if case.expected and case.expected.get("status") == "no_valid_composition"]
    correct = rejected = 0
    elapsed: list[float] = []
    prompt_tokens: list[int] = []
    generated_tokens: list[int] = []
    for case in cases:
        prompt = gap_prompt_007(case, context)
        completion = client.chat_json(prompt, max_tokens=96)
        parsed = parse_response(completion.text)
        passed = parsed == case.expected
        correct += int(case in valid and passed)
        rejected += int(case in impossible and isinstance(parsed, dict) and parsed.get("status") == "no_valid_composition")
        elapsed.append(completion.elapsed_seconds)
        prompt_tokens.append(completion.prompt_tokens or 0)
        generated_tokens.append(completion.generated_tokens or 0)
        store.record_run(
            kind=f"air-007:{phase}:{condition}",
            prompt=prompt,
            response=completion.text,
            elapsed_seconds=completion.elapsed_seconds,
            prompt_tokens=completion.prompt_tokens,
            generated_tokens=completion.generated_tokens,
            passed=passed,
            metadata={"case_id": case.case_id, "task_token": case.task_token, "expected": case.expected, "parsed": parsed},
        )
    return _result007(condition, correct, len(valid), rejected, len(impossible), sum(prompt_tokens), sum(generated_tokens), elapsed)


def _flat_gap_candidates() -> tuple[Expr, ...]:
    x, y = field("x"), field("y")
    nx, ny = unary("LOWER", unary("TRIM", x)), unary("LOWER", unary("TRIM", y))
    rx, ry = _replace_spaces(nx), _replace_spaces(ny)
    candidates = (
        binary("CONCAT", rx, literal(" ")),
        binary("CONCAT", binary("CONCAT", rx, literal("/")), ry),
        binary("CONCAT", binary("CONCAT", ry, literal("/")), rx),
    )
    unique: dict[str, Expr] = {}
    for candidate in candidates:
        unique.setdefault(json.dumps(candidate.to_dict(), sort_keys=True, separators=(",", ":")), candidate)
    return tuple(unique.values())


def search_flat_gap_programs(cases: tuple[CompositionCase, ...]) -> tuple[dict[str, Expr], dict[str, str]]:
    grouped: dict[str, list[CompositionCase]] = {}
    for case in cases:
        grouped.setdefault(case.task_token, []).append(case)
    found: dict[str, Expr] = {}
    rejected: dict[str, str] = {}
    for task_token, task_cases in sorted(grouped.items()):
        matches: list[Expr] = []
        for expression in _flat_gap_candidates():
            try:
                if type_of(expression, {"x": STRING, "y": STRING}) != STRING:
                    continue
                if all(
                    case.expected is not None
                    and _ok(str(execute_expr(expression, {"x": case.x, "y": case.y}))) == case.expected
                    for case in task_cases
                ):
                    matches.append(expression)
            except TypeCheckError:
                continue
        if len(matches) == 1:
            found[task_token] = matches[0]
        else:
            rejected[task_token] = "no valid flat program" if not matches else "ambiguous flat program"
    return found, rejected


def run_flat_gap(
    *, cases: tuple[CompositionCase, ...], expressions: dict[str, Expr], store: ExperimentStore, phase: str
) -> Result007:
    valid = [case for case in cases if case.expected and case.expected.get("status") == "ok"]
    impossible = [case for case in cases if case.expected and case.expected.get("status") == "no_valid_composition"]
    correct = rejected = 0
    for case in cases:
        if case.task_token in expressions:
            try:
                parsed = _ok(str(execute_expr(expressions[case.task_token], {"x": case.x, "y": case.y})))
            except TypeCheckError:
                parsed = _no_valid()
        else:
            parsed = _no_valid()
        passed = parsed == case.expected
        correct += int(case in valid and passed)
        rejected += int(case in impossible and parsed.get("status") == "no_valid_composition")
        store.record_run(
            kind=f"air-007:{phase}:flat_synthesized",
            prompt=f"flat AST input={json.dumps(case.payload(), sort_keys=True)}",
            response=json.dumps(parsed, sort_keys=True),
            elapsed_seconds=0.0,
            prompt_tokens=0,
            generated_tokens=0,
            passed=passed,
            metadata={"case_id": case.case_id, "expected": case.expected, "parsed": parsed},
        )
    return _result007("flat_synthesized", correct, len(valid), rejected, len(impossible))


def _corrupted_gap_plans(plans: dict[str, CompositionPlan], skills: tuple[PrimitiveSkill, ...]) -> dict[str, CompositionPlan]:
    corrupted = dict(plans)
    if "g1" in plans:
        plan = plans["g1"]
        corrupted["g1"] = CompositionPlan(left_chain=tuple(reversed(plan.left_chain)))
    binary_ids = _binary_ids(skills)
    if "g2" in plans and len(binary_ids) > 1:
        plan = plans["g2"]
        alternatives = [item for item in binary_ids if item != plan.binary_skill]
        if alternatives:
            corrupted["g2"] = CompositionPlan(plan.left_chain, plan.right_chain, alternatives[0], plan.post_chain)
    return corrupted


def run_exp007(*, client: LlamaCppClient, store: ExperimentStore, report_directory: str, heldout_limit: int | None = None) -> dict[str, object]:
    base_skills, base_stats = learn_primitive_library()
    base_snapshot = {skill.skill_id: skill.to_dict() for skill in base_skills}
    before_plans, before_no_valid, before_stats = search_gap_compositions(base_skills, GAP_VALIDATION_007)
    diagnoses = diagnose_capability_gaps(GAP_VALIDATION_007, before_plans, before_no_valid)
    missing_skill, missing_stats = synthesize_missing_skill_007()
    missing_train = _skill_gate("missing_skill_training", missing_skill, MISSING_SKILL_TRAINING_007)
    missing_validation = _skill_gate("missing_skill_validation", missing_skill, MISSING_SKILL_VALIDATION_007)
    missing_edge = _skill_gate("missing_skill_edge", missing_skill, MISSING_SKILL_EDGE_007)
    extended_skills = base_skills + (missing_skill,)
    after_plans, after_no_valid, after_stats = search_gap_compositions(extended_skills, GAP_VALIDATION_007)
    flat_programs, flat_rejected = search_flat_gap_programs(GAP_VALIDATION_007)
    gap_valid = tuple(case for case in GAP_VALIDATION_007 if case.expected and case.expected.get("status") == "ok")
    gap_impossible = tuple(case for case in GAP_VALIDATION_007 if case.expected and case.expected.get("status") == "no_valid_composition")
    validation = run_gap_compositions(store=store, condition="learned_composition", cases=gap_valid, plans=after_plans, skills=extended_skills, phase="validation")
    impossible = run_gap_compositions(store=store, condition="learned_composition", cases=gap_impossible, plans=after_plans, skills=extended_skills, phase="impossible-validation")
    corrupted = run_gap_compositions(store=store, condition="corrupted_composition", cases=gap_valid, plans=_corrupted_gap_plans(after_plans, extended_skills), skills=extended_skills, phase="corrupted-validation")
    positive_tokens = {case.task_token for case in gap_valid}
    active = (
        all(item.status == "gap_detected" for item in diagnoses if item.task_token in positive_tokens)
        and missing_validation.accuracy == 1.0
        and missing_edge.accuracy == 1.0
        and validation.valid_accuracy == 1.0
        and impossible.safe_rejection_rate == 1.0
        and corrupted.valid_accuracy < 0.9
    )
    for skill in extended_skills:
        state = "active" if active else "candidate"
        store.upsert_skill(name=f"air-007-{skill.skill_id}", body=json.dumps(skill.to_dict(), sort_keys=True), state=state)
    composition_artifacts: dict[str, dict[str, object]] = {}
    for task_token, plan in after_plans.items():
        provenance = {
            "skill_ids": plan.left_chain + plan.right_chain + ((plan.binary_skill,) if plan.binary_skill else ()) + plan.post_chain,
            "order": plan.to_dict(),
            "source_versions": {skill.skill_id: skill.version for skill in extended_skills},
            "validation_case_ids": [case.case_id for case in gap_valid if case.task_token == task_token],
            "activation_reason": "gap skill validation, composition validation, edge, and impossible-task gates passed" if active else "not active",
        }
        store.upsert_skill(name=f"air-007-composed-{task_token}", body=json.dumps({"plan": plan.to_dict(), "provenance": provenance}, sort_keys=True), state="active" if active else "rejected")
        composition_artifacts[task_token] = provenance
    heldout = GAP_HELD_OUT_007[:heldout_limit] if heldout_limit is not None else GAP_HELD_OUT_007
    results = [
        run_model_007(cases=heldout, condition="model", context=None, client=client, store=store, phase="heldout"),
        run_model_007(cases=heldout, condition="prior_raw", context=raw_primitive_experiences_006(), client=client, store=store, phase="heldout"),
        run_model_007(cases=heldout, condition="all_raw", context=raw_gap_context_007(), client=client, store=store, phase="heldout"),
        run_flat_gap(cases=heldout, expressions=flat_programs, store=store, phase="heldout"),
    ]
    before_result = run_gap_compositions(store=store, condition="before_missing_skill", cases=heldout, plans=before_plans, skills=base_skills, phase="heldout")
    results.append(before_result)
    if active:
        results.append(run_gap_compositions(store=store, condition="learned_composition", cases=heldout, plans=after_plans, skills=extended_skills, phase="heldout"))
    base_after = {skill.skill_id: skill.to_dict() for skill in extended_skills if skill.skill_id in base_snapshot}
    report = {
        "benchmark": "air-007-capability-gap-missing-skill",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "base_experiment": "0006",
            "base_skill_count": len(base_skills),
            "missing_skill_training_cases": len(MISSING_SKILL_TRAINING_007),
            "missing_skill_validation_cases": len(MISSING_SKILL_VALIDATION_007),
            "missing_skill_edge_cases": len(MISSING_SKILL_EDGE_007),
            "gap_validation_cases": len(GAP_VALIDATION_007),
            "heldout_cases": len(GAP_HELD_OUT_007),
            "heldout_is_new": True,
            "composed_examples_in_missing_skill_training": False,
            "family_names_exposed_to_model": False,
        },
        "base_library": {"skills": [skill.to_dict() for skill in base_skills], "synthesis": asdict(base_stats)},
        "capability_gap_before_learning": {"diagnoses": [asdict(item) for item in diagnoses], "search": {"stats": asdict(before_stats), "accepted_plans": {token: plan.to_dict() for token, plan in before_plans.items()}, "no_valid_composition": before_no_valid}},
        "missing_skill_discovery": {"skill": missing_skill.to_dict(), "synthesis": asdict(missing_stats), "training": asdict(missing_train), "validation": asdict(missing_validation), "edge": asdict(missing_edge)},
        "composition_after_learning": {"stats": asdict(after_stats), "accepted_plans": {token: plan.to_dict() for token, plan in after_plans.items()}, "no_valid_composition": after_no_valid, "flat_rejected": flat_rejected, "flat_programs": {token: expr.to_dict() for token, expr in flat_programs.items()}},
        "validation": {"composition": asdict(validation), "impossible": asdict(impossible), "corrupted": asdict(corrupted), "active": active},
        "provenance": composition_artifacts,
        "immutability": {"base_skills_unchanged": base_snapshot == base_after},
        "baselines": {
            "model": {"source": "direct LLM completion", "raw_context": False, "artifact_reuse": False, "new_composed_program": False},
            "prior_raw": {"source": "LLM with 0006 independent primitive pairs only", "raw_context": True, "artifact_reuse": False, "new_composed_program": False},
            "all_raw": {"source": "LLM with base and missing-skill independent pairs", "raw_context": True, "artifact_reuse": False, "new_composed_program": False},
            "flat_synthesized": {"source": "monolithic REPLACE-containing typed AST selected from gap validation", "raw_context": False, "artifact_reuse": False, "new_composed_program": True},
            "before_missing_skill": {"source": "existing 0006 artifact search before learning", "raw_context": False, "artifact_reuse": True, "new_composed_program": False},
            "learned_composition": {"source": "new skill artifact plus existing artifacts, selected by bounded typed search", "raw_context": False, "artifact_reuse": True, "new_composed_program": True},
        },
        "heldout_results": [asdict(result) for result in results],
    }
    report_path = Path(report_directory)
    report_path.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(UTC).strftime("air-007-%Y%m%dT%H%M%SZ.json")
    path = report_path / filename
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_file"] = str(path)
    return report
