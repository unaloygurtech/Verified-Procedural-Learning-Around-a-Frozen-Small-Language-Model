"""Experiment 0006: compositional reuse of independently learned skills.

Primitive skills are synthesized from separate experience sets.  A planner then
searches compositions over the immutable typed artifacts, while a deterministic
executor runs the accepted plan.  No composed input/output examples are present
in the primitive training data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import itertools
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .exp005 import (
    Expr,
    INT,
    STRING,
    TypeCheckError,
    binary,
    check_program,
    execute_expr,
    field,
    literal,
    type_of,
    unary,
)
from .model_client import LlamaCppClient
from .neralis import parse_response
from .store import ExperimentStore


@dataclass(frozen=True)
class PrimitiveCase:
    case_id: str
    payload: dict[str, str]
    expected: str


@dataclass(frozen=True)
class PrimitiveSet:
    set_id: str
    input_fields: tuple[str, ...]
    cases: tuple[PrimitiveCase, ...]


PRIMITIVE_SETS_006 = (
    PrimitiveSet(
        "set-a",
        ("input",),
        (
            PrimitiveCase("a-01", {"input": "  Alpha "}, "Alpha"),
            PrimitiveCase("a-02", {"input": "Beta  "}, "Beta"),
            PrimitiveCase("a-03", {"input": "  Gamma"}, "Gamma"),
        ),
    ),
    PrimitiveSet(
        "set-b",
        ("input",),
        (
            PrimitiveCase("b-01", {"input": " MiXeD "}, "mixed"),
            PrimitiveCase("b-02", {"input": "Camel"}, "camel"),
            PrimitiveCase("b-03", {"input": "UP "}, "up"),
        ),
    ),
    PrimitiveSet(
        "set-c",
        ("input",),
        (
            PrimitiveCase("c-01", {"input": "Alpha"}, "Alpha "),
            PrimitiveCase("c-02", {"input": "Beta "}, "Beta  "),
            PrimitiveCase("c-03", {"input": "MiXeD"}, "MiXeD "),
        ),
    ),
    PrimitiveSet(
        "set-d",
        ("left", "right"),
        (
            PrimitiveCase("d-01", {"left": "A ", "right": "b"}, "A /b"),
            PrimitiveCase("d-02", {"left": "Left", "right": " RIGHT "}, "Left/ RIGHT "),
            PrimitiveCase("d-03", {"left": "MiX", "right": "Ed"}, "MiX/Ed"),
        ),
    ),
    PrimitiveSet(
        "set-e",
        ("left", "right"),
        (
            PrimitiveCase("e-01", {"left": "A ", "right": "b"}, "b/A "),
            PrimitiveCase("e-02", {"left": "Left", "right": " RIGHT "}, " RIGHT /Left"),
            PrimitiveCase("e-03", {"left": "MiX", "right": "Ed"}, "Ed/MiX"),
        ),
    ),
)


@dataclass(frozen=True)
class PrimitiveSkill:
    skill_id: str
    version: int
    input_fields: tuple[str, ...]
    output_type: str
    expression: Expr
    source_set_id: str
    source_case_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "input_fields": self.input_fields,
            "output_type": self.output_type,
            "expression": self.expression.to_dict(),
            "source_set_id": self.source_set_id,
            "source_case_ids": self.source_case_ids,
        }


@dataclass(frozen=True)
class PrimitiveSynthesisStats:
    candidate_search_count: int
    type_invalid_candidate_count: int
    ambiguous_candidate_count: int


def _expr_key(expr: Expr) -> str:
    return json.dumps(expr.to_dict(), sort_keys=True, separators=(",", ":"))


def _primitive_candidates(input_fields: tuple[str, ...]) -> tuple[Expr, ...]:
    if input_fields == ("input",):
        source = field("input")
        candidates = [
            source,
            unary("TRIM", source),
            unary("LOWER", source),
            unary("LOWER", unary("TRIM", source)),
            binary("CONCAT", source, literal(" ")),
            binary("CONCAT", unary("LOWER", source), literal(" ")),
        ]
    else:
        left, right = field("left"), field("right")
        candidates = [
            binary("ADD", left, right),
            binary("CONCAT", binary("CONCAT", left, literal("/")), right),
            binary("CONCAT", binary("CONCAT", right, literal("/")), left),
            binary("CONCAT", binary("CONCAT", unary("TRIM", left), literal("/")), unary("TRIM", right)),
            binary("CONCAT", binary("CONCAT", unary("LOWER", left), literal("/")), unary("LOWER", right)),
        ]
    unique: dict[str, Expr] = {}
    for candidate in candidates:
        unique.setdefault(_expr_key(candidate), candidate)
    return tuple(unique.values())


def synthesize_primitive(set_spec: PrimitiveSet, skill_id: str) -> tuple[PrimitiveSkill, PrimitiveSynthesisStats]:
    field_types = {name: STRING for name in set_spec.input_fields}
    candidates = _primitive_candidates(set_spec.input_fields)
    search_count = 0
    type_invalid = 0
    matches: list[Expr] = []
    for expression in candidates:
        search_count += 1
        try:
            if type_of(expression, field_types) != STRING:
                type_invalid += 1
                continue
        except TypeCheckError:
            type_invalid += 1
            continue
        try:
            if all(execute_expr(expression, case.payload) == case.expected for case in set_spec.cases):
                matches.append(expression)
        except TypeCheckError:
            continue
    if len(matches) != 1:
        raise ValueError(f"primitive {set_spec.set_id} has {len(matches)} valid explanations")
    skill = PrimitiveSkill(
        skill_id,
        1,
        set_spec.input_fields,
        STRING,
        matches[0],
        set_spec.set_id,
        tuple(case.case_id for case in set_spec.cases),
    )
    return skill, PrimitiveSynthesisStats(search_count, type_invalid, 0)


def learn_primitive_library() -> tuple[tuple[PrimitiveSkill, ...], PrimitiveSynthesisStats]:
    skills: list[PrimitiveSkill] = []
    counts = [0, 0, 0]
    for index, set_spec in enumerate(PRIMITIVE_SETS_006, 1):
        skill, stats = synthesize_primitive(set_spec, f"skill-{index}")
        skills.append(skill)
        counts[0] += stats.candidate_search_count
        counts[1] += stats.type_invalid_candidate_count
        counts[2] += stats.ambiguous_candidate_count
    return tuple(skills), PrimitiveSynthesisStats(*counts)


@dataclass(frozen=True)
class CompositionCase:
    case_id: str
    task_token: str
    x: str
    y: str
    expected: dict[str, object] | None
    task_kind: str

    def payload(self) -> dict[str, object]:
        return {"task": self.task_token, "x": self.x, "y": self.y}


def _ok(value: str) -> dict[str, object]:
    return {"status": "ok", "value": value}


def _no_valid() -> dict[str, object]:
    return {"status": "no_valid_composition"}


VALID_COMPOSITION_VALIDATION_006 = (
    CompositionCase("v-a1", "m2", "  MiXeD ", "unused", _ok("mixed "), "depth-2"),
    CompositionCase("v-a2", "m2", "  Other ", "ignored", _ok("other "), "depth-2"),
    CompositionCase("v-b1", "m3", "  Alpha ", " BeTa ", _ok("alpha/beta"), "depth-3"),
    CompositionCase("v-b2", "m3", " Gamma", "DELTA ", _ok("gamma/delta"), "depth-3"),
    CompositionCase("v-c1", "m4", "  Alpha ", " BeTa ", _ok("beta/alpha"), "depth-3"),
    CompositionCase("v-c2", "m4", " Gamma", "DELTA ", _ok("delta/gamma"), "depth-3"),
)

IMPOSSIBLE_VALIDATION_006 = (
    CompositionCase("v-x1", "mx", "  Alpha ", "Beta", _no_valid(), "impossible"),
    CompositionCase("v-x2", "mx", "Gamma", " Delta ", _no_valid(), "impossible"),
)

VALIDATION_006 = VALID_COMPOSITION_VALIDATION_006 + IMPOSSIBLE_VALIDATION_006

HELD_OUT_006 = (
    CompositionCase("h-a1", "m2", "  NOVEMBER ", "unused", _ok("november "), "depth-2"),
    CompositionCase("h-a2", "m2", "  Oscar", "ignored", _ok("oscar "), "depth-2"),
    CompositionCase("h-a3", "m2", "PAPA  ", "ignored", _ok("papa "), "depth-2"),
    CompositionCase("h-b1", "m3", "  Quebec ", " ROMEO", _ok("quebec/romeo"), "depth-3"),
    CompositionCase("h-b2", "m3", "Sierra", "  TANGO  ", _ok("sierra/tango"), "depth-3"),
    CompositionCase("h-b3", "m3", "Uniform ", "Victor", _ok("uniform/victor"), "depth-3"),
    CompositionCase("h-c1", "m4", "  Whiskey ", "Xray", _ok("xray/whiskey"), "depth-3"),
    CompositionCase("h-c2", "m4", "Yankee", "  Zulu  ", _ok("zulu/yankee"), "depth-3"),
    CompositionCase("h-c3", "m4", "Alpha Beta", " Gamma Delta ", _ok("gamma delta/alpha beta"), "depth-3"),
    CompositionCase("h-x1", "mz", "  New ", "Task", _no_valid(), "impossible"),
    CompositionCase("h-x2", "mz", "Unknown", " Input ", _no_valid(), "impossible"),
    CompositionCase("h-x3", "mz", "NoSuch", "Procedure", _no_valid(), "impossible"),
)

EDGE_006 = (
    CompositionCase("e-a1", "m2", "  edge ", "unused", _ok("edge "), "depth-2"),
    CompositionCase("e-b1", "m3", "  Alpha Beta  ", " GAMMA ", _ok("alpha beta/gamma"), "depth-3"),
    CompositionCase("e-c1", "m4", "  MiXeD ", "Case Test ", _ok("case test/mixed"), "depth-3"),
    CompositionCase("e-x1", "mz", "Anything", "Else", _no_valid(), "impossible"),
)


def raw_primitive_experiences_006() -> str:
    lines = ["Verified independent primitive experiences (skill names omitted):"]
    for set_spec in PRIMITIVE_SETS_006:
        for case in set_spec.cases:
            lines.append(json.dumps({"input": case.payload, "verified_output": case.expected}, sort_keys=True))
    return "\n".join(lines)


def composition_prompt(case: CompositionCase, context: str | None = None) -> str:
    sections = []
    if context:
        sections.append(context)
    sections.extend(
        [
            "Solve this task by reusing independent learned behaviors if possible.",
            "If no valid composition exists, return exactly {\"status\":\"no_valid_composition\"}.",
            f"input={json.dumps(case.payload(), sort_keys=True)}",
            "For a solved task return exactly {\"status\":\"ok\",\"value\":\"...\"}.",
        ]
    )
    return "\n\n".join(sections)


@dataclass(frozen=True)
class CompositionPlan:
    left_chain: tuple[str, ...] = ()
    right_chain: tuple[str, ...] = ()
    binary_skill: str | None = None
    post_chain: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "left_chain": self.left_chain,
            "right_chain": self.right_chain,
            "binary_skill": self.binary_skill,
            "post_chain": self.post_chain,
            "depth": self.depth(),
        }

    def depth(self) -> int:
        return len(self.left_chain) + len(self.right_chain) + len(self.post_chain) + int(self.binary_skill is not None)


@dataclass(frozen=True)
class CompositionSearchStats:
    candidate_composition_count: int
    candidate_evaluation_count: int
    type_invalid_composition_count: int
    semantic_validation_rejected_count: int
    ambiguous_composition_count: int


def _skill_map(skills: tuple[PrimitiveSkill, ...]) -> dict[str, PrimitiveSkill]:
    return {skill.skill_id: skill for skill in skills}


def _unary_ids(skills: tuple[PrimitiveSkill, ...]) -> tuple[str, ...]:
    return tuple(skill.skill_id for skill in skills if len(skill.input_fields) == 1)


def _binary_ids(skills: tuple[PrimitiveSkill, ...]) -> tuple[str, ...]:
    return tuple(skill.skill_id for skill in skills if len(skill.input_fields) == 2)


def _all_ids(skills: tuple[PrimitiveSkill, ...]) -> tuple[str, ...]:
    return tuple(skill.skill_id for skill in skills)


def iter_composition_candidates(skills: tuple[PrimitiveSkill, ...]) -> Iterable[CompositionPlan]:
    all_ids = _all_ids(skills)
    # Include all artifact IDs in each slot; static checks remove arity-invalid
    # paths before any runtime execution.
    for length in (1, 2):
        for chain in itertools.product(all_ids, repeat=length):
            yield CompositionPlan(left_chain=chain)
    chain_options = ((),) + tuple((skill_id,) for skill_id in all_ids)
    post_options = ((),) + tuple((skill_id,) for skill_id in all_ids)
    for left, right, binary_skill in itertools.product(chain_options, chain_options, all_ids):
        # A post-chain models binary-first ordering, but is kept separate from
        # branch-normalized plans to avoid duplicate equivalent programs.
        yield CompositionPlan(left_chain=left, right_chain=right, binary_skill=binary_skill)
        if not left and not right:
            for post in post_options:
                if post:
                    yield CompositionPlan(binary_skill=binary_skill, post_chain=post)


def check_plan_types(plan: CompositionPlan, skills: tuple[PrimitiveSkill, ...]) -> None:
    skill_by_id = _skill_map(skills)
    for chain in (plan.left_chain, plan.right_chain, plan.post_chain):
        for skill_id in chain:
            skill = skill_by_id[skill_id]
            if len(skill.input_fields) != 1 or skill.output_type != STRING:
                raise TypeCheckError(f"unary slot received {skill_id}")
            type_of(skill.expression, {"input": STRING})
    if plan.binary_skill is not None:
        binary_skill = skill_by_id[plan.binary_skill]
        if len(binary_skill.input_fields) != 2 or binary_skill.output_type != STRING:
            raise TypeCheckError(f"binary slot received {plan.binary_skill}")
        type_of(binary_skill.expression, {"left": STRING, "right": STRING})


def _apply_unary_chain(value: str, chain: tuple[str, ...], skills: dict[str, PrimitiveSkill]) -> str:
    for skill_id in chain:
        result = execute_expr(skills[skill_id].expression, {"input": value})
        if not isinstance(result, str):
            raise TypeCheckError("unary composition produced non-string")
        value = result
    return value


def execute_composition(plan: CompositionPlan, skills: tuple[PrimitiveSkill, ...], case: CompositionCase) -> dict[str, object]:
    check_plan_types(plan, skills)
    skill_by_id = _skill_map(skills)
    if plan.binary_skill is None:
        value = _apply_unary_chain(case.x, plan.left_chain, skill_by_id)
    else:
        left = _apply_unary_chain(case.x, plan.left_chain, skill_by_id)
        right = _apply_unary_chain(case.y, plan.right_chain, skill_by_id)
        value = execute_expr(skill_by_id[plan.binary_skill].expression, {"left": left, "right": right})
        if not isinstance(value, str):
            raise TypeCheckError("binary composition produced non-string")
        value = _apply_unary_chain(value, plan.post_chain, skill_by_id)
    return _ok(value)


def search_compositions(
    skills: tuple[PrimitiveSkill, ...],
    cases: tuple[CompositionCase, ...],
) -> tuple[dict[str, CompositionPlan], dict[str, str], CompositionSearchStats]:
    grouped: dict[str, list[CompositionCase]] = {}
    for case in cases:
        grouped.setdefault(case.task_token, []).append(case)
    plans: dict[str, CompositionPlan] = {}
    no_valid: dict[str, str] = {}
    evaluation_count = type_invalid = semantic_rejected = ambiguous = 0
    candidates = tuple(iter_composition_candidates(skills))
    for task_token, task_cases in sorted(grouped.items()):
        matches: list[CompositionPlan] = []
        for plan in candidates:
            evaluation_count += 1
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
        if len(matches) > 1:
            ambiguous += 1
            no_valid[task_token] = "ambiguous composition"
        elif len(matches) == 1:
            plans[task_token] = matches[0]
        else:
            no_valid[task_token] = "no valid composition"
    return plans, no_valid, CompositionSearchStats(len(candidates), evaluation_count, type_invalid, semantic_rejected, ambiguous)


def _flat_expr_candidates() -> tuple[Expr, ...]:
    x, y = field("x"), field("y")
    nx, ny = unary("LOWER", unary("TRIM", x)), unary("LOWER", unary("TRIM", y))
    candidates = [
        unary("LOWER", unary("TRIM", x)),
        binary("CONCAT", unary("LOWER", unary("TRIM", x)), literal(" ")),
        binary("CONCAT", binary("CONCAT", nx, literal("/")), ny),
        binary("CONCAT", binary("CONCAT", ny, literal("/")), nx),
        binary("CONCAT", binary("CONCAT", x, literal("/")), y),
        unary("TRIM", binary("CONCAT", binary("CONCAT", x, literal("/")), y)),
        unary("LOWER", unary("TRIM", binary("CONCAT", binary("CONCAT", y, literal("/")), x))),
    ]
    unique: dict[str, Expr] = {}
    for expr in candidates:
        unique.setdefault(_expr_key(expr), expr)
    return tuple(unique.values())


def search_flat_programs(cases: tuple[CompositionCase, ...]) -> tuple[dict[str, Expr], dict[str, str]]:
    grouped: dict[str, list[CompositionCase]] = {}
    for case in cases:
        grouped.setdefault(case.task_token, []).append(case)
    found: dict[str, Expr] = {}
    rejected: dict[str, str] = {}
    for task_token, task_cases in sorted(grouped.items()):
        matches = []
        for expr in _flat_expr_candidates():
            try:
                if type_of(expr, {"x": STRING, "y": STRING}) != STRING:
                    continue
                if all(case.expected is not None and _ok(str(execute_expr(expr, {"x": case.x, "y": case.y}))) == case.expected for case in task_cases):
                    matches.append(expr)
            except TypeCheckError:
                continue
        if len(matches) == 1:
            found[task_token] = matches[0]
        else:
            rejected[task_token] = "no valid flat program" if not matches else "ambiguous flat program"
    return found, rejected


def execute_flat(expr: Expr, case: CompositionCase) -> dict[str, object]:
    return _ok(str(execute_expr(expr, {"x": case.x, "y": case.y})))


@dataclass(frozen=True)
class Result006:
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


def _result006(condition: str, valid_correct: int, valid_total: int, rejected: int, impossible_total: int, prompt_tokens: int = 0, generated_tokens: int = 0, elapsed: list[float] | None = None) -> Result006:
    return Result006(condition, valid_correct, valid_total, valid_correct / valid_total if valid_total else 0.0, rejected, impossible_total, rejected / impossible_total if impossible_total else 0.0, prompt_tokens, generated_tokens, mean(elapsed) if elapsed else 0.0)


def run_executable_compositions(
    *, store: ExperimentStore, condition: str, cases: tuple[CompositionCase, ...], plans: dict[str, CompositionPlan], skills: tuple[PrimitiveSkill, ...], phase: str,
) -> Result006:
    valid = [case for case in cases if case.expected and case.expected.get("status") == "ok"]
    impossible = [case for case in cases if case.expected and case.expected.get("status") == "no_valid_composition"]
    correct = rejected = 0
    for case in cases:
        parsed: dict[str, object]
        if case.task_token not in plans:
            parsed = _no_valid()
        else:
            try:
                parsed = execute_composition(plans[case.task_token], skills, case)
            except (KeyError, TypeCheckError):
                parsed = _no_valid()
        passed = parsed == case.expected
        if case in valid:
            correct += int(passed)
        if case in impossible:
            rejected += int(parsed.get("status") == "no_valid_composition")
        store.record_run(kind=f"air-006:{phase}:{condition}", prompt=f"deterministic composition input={json.dumps(case.payload(), sort_keys=True)}", response=json.dumps(parsed, sort_keys=True), elapsed_seconds=0.0, prompt_tokens=0, generated_tokens=0, passed=passed, metadata={"case_id": case.case_id, "task_token": case.task_token, "task_kind": case.task_kind, "expected": case.expected, "parsed": parsed})
    return _result006(condition, correct, len(valid), rejected, len(impossible))


def run_flat(cases: tuple[CompositionCase, ...], expressions: dict[str, Expr], store: ExperimentStore, phase: str) -> Result006:
    valid = [case for case in cases if case.expected and case.expected.get("status") == "ok"]
    impossible = [case for case in cases if case.expected and case.expected.get("status") == "no_valid_composition"]
    correct = rejected = 0
    for case in cases:
        parsed = execute_flat(expressions[case.task_token], case) if case.task_token in expressions else _no_valid()
        passed = parsed == case.expected
        correct += int(passed and case in valid)
        rejected += int(case in impossible and parsed.get("status") == "no_valid_composition")
        store.record_run(kind=f"air-006:{phase}:flat_synthesized", prompt=f"flat AST input={json.dumps(case.payload(), sort_keys=True)}", response=json.dumps(parsed, sort_keys=True), elapsed_seconds=0.0, prompt_tokens=0, generated_tokens=0, passed=passed, metadata={"case_id": case.case_id, "expected": case.expected, "parsed": parsed})
    return _result006("flat_synthesized", correct, len(valid), rejected, len(impossible))


def run_model(cases: tuple[CompositionCase, ...], condition: str, context: str | None, client: LlamaCppClient, store: ExperimentStore, phase: str) -> Result006:
    valid = [case for case in cases if case.expected and case.expected.get("status") == "ok"]
    impossible = [case for case in cases if case.expected and case.expected.get("status") == "no_valid_composition"]
    correct = rejected = 0
    elapsed: list[float] = []
    prompt_tokens: list[int] = []
    generated_tokens: list[int] = []
    for case in cases:
        prompt = composition_prompt(case, context)
        completion = client.chat_json(prompt, max_tokens=96)
        parsed = parse_response(completion.text)
        passed = parsed == case.expected
        correct += int(passed and case in valid)
        rejected += int(case in impossible and isinstance(parsed, dict) and parsed.get("status") == "no_valid_composition")
        elapsed.append(completion.elapsed_seconds)
        prompt_tokens.append(completion.prompt_tokens or 0)
        generated_tokens.append(completion.generated_tokens or 0)
        store.record_run(kind=f"air-006:{phase}:{condition}", prompt=prompt, response=completion.text, elapsed_seconds=completion.elapsed_seconds, prompt_tokens=completion.prompt_tokens, generated_tokens=completion.generated_tokens, passed=passed, metadata={"case_id": case.case_id, "task_token": case.task_token, "expected": case.expected, "parsed": parsed})
    return _result006(condition, correct, len(valid), rejected, len(impossible), sum(prompt_tokens), sum(generated_tokens), elapsed)


def _corrupted_plan(plans: dict[str, CompositionPlan], skills: tuple[PrimitiveSkill, ...]) -> dict[str, CompositionPlan]:
    corrupted = dict(plans)
    # Semantic order corruption: reverse the accepted depth-2 sequence.
    if "m2" in plans:
        plan = plans["m2"]
        corrupted["m2"] = CompositionPlan(left_chain=tuple(reversed(plan.left_chain)))
    # Binary direction corruption: replace forward join with the other learned join.
    binary_ids = _binary_ids(skills)
    if "m3" in plans and len(binary_ids) > 1:
        plan = plans["m3"]
        alternatives = [item for item in binary_ids if item != plan.binary_skill]
        if alternatives:
            corrupted["m3"] = CompositionPlan(plan.left_chain, plan.right_chain, alternatives[0], plan.post_chain)
    return corrupted


def run_exp006(*, client: LlamaCppClient, store: ExperimentStore, report_directory: str, heldout_limit: int | None = None) -> dict[str, object]:
    skills, primitive_stats = learn_primitive_library()
    source_serialized = {skill.skill_id: skill.to_dict() for skill in skills}
    plans, no_valid, composition_stats = search_compositions(skills, VALIDATION_006)
    flat_programs, flat_rejected = search_flat_programs(VALIDATION_006)
    edge = run_executable_compositions(store=store, condition="learned_composition", cases=EDGE_006, plans=plans, skills=skills, phase="edge-validation")
    valid_validation = run_executable_compositions(store=store, condition="learned_composition", cases=VALID_COMPOSITION_VALIDATION_006, plans=plans, skills=skills, phase="validation")
    impossible_validation = run_executable_compositions(store=store, condition="learned_composition", cases=IMPOSSIBLE_VALIDATION_006, plans=plans, skills=skills, phase="impossible-validation")
    corrupt = run_executable_compositions(store=store, condition="corrupted_composition", cases=VALID_COMPOSITION_VALIDATION_006, plans=_corrupted_plan(plans, skills), skills=skills, phase="corrupted-validation")
    active = (
        valid_validation.valid_accuracy == 1.0
        and edge.valid_accuracy == 1.0
        and impossible_validation.safe_rejection_rate == 1.0
        and corrupt.valid_accuracy < 0.9
    )
    for skill in skills:
        store.upsert_skill(name=f"air-006-{skill.skill_id}", body=json.dumps(skill.to_dict(), sort_keys=True), state="active" if active else "candidate")
    composition_artifacts = {}
    for task_token, plan in plans.items():
        artifact_name = f"air-006-composed-{task_token}"
        provenance = {"skill_ids": plan.left_chain + plan.right_chain + ((plan.binary_skill,) if plan.binary_skill else ()) + plan.post_chain, "order": plan.to_dict(), "source_versions": {skill.skill_id: skill.version for skill in skills}, "validation_case_ids": [case.case_id for case in VALID_COMPOSITION_VALIDATION_006 if case.task_token == task_token], "edge_case_ids": [case.case_id for case in EDGE_006 if case.task_token == task_token], "activation_reason": "validation and edge semantic pass" if active else "not active"}
        store.upsert_skill(name=artifact_name, body=json.dumps({"plan": plan.to_dict(), "provenance": provenance}, sort_keys=True), state="active" if active else "rejected")
        composition_artifacts[task_token] = provenance
    heldout = HELD_OUT_006[:heldout_limit] if heldout_limit is not None else HELD_OUT_006
    results = [
        run_model(heldout, "model", None, client, store, "heldout"),
        run_model(heldout, "raw_primitives", raw_primitive_experiences_006(), client, store, "heldout"),
        run_flat(heldout, flat_programs, store, "heldout"),
    ]
    if active:
        results.append(run_executable_compositions(store=store, condition="learned_composition", cases=heldout, plans=plans, skills=skills, phase="heldout"))
    source_after = {skill.skill_id: skill.to_dict() for skill in skills}
    raw_text = raw_primitive_experiences_006()
    report = {
        "benchmark": "air-006-compositional-skill-reuse",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": {"primitive_sets": len(PRIMITIVE_SETS_006), "primitive_cases_per_set": 3, "validation_cases": len(VALIDATION_006), "edge_cases": len(EDGE_006), "heldout_cases": len(HELD_OUT_006), "heldout_is_new": True, "composed_examples_in_primitive_training": False, "family_names_exposed_to_model": False},
        "discovery": {"learned_skills": [skill.to_dict() for skill in skills], "primitive_synthesis": asdict(primitive_stats)},
        "composition_search": {"stats": asdict(composition_stats), "accepted_plans": {token: plan.to_dict() for token, plan in plans.items()}, "no_valid_composition": no_valid, "flat_rejected": flat_rejected, "flat_programs": {token: expr.to_dict() for token, expr in flat_programs.items()}},
        "validation": {"valid": asdict(valid_validation), "edge": asdict(edge), "impossible": asdict(impossible_validation), "active": active},
        "corrupted_composition": {"result": asdict(corrupt), "state": "rejected" if corrupt.valid_accuracy < 0.9 else "unsafe-active"},
        "provenance": composition_artifacts,
        "immutability": {"source_skills_unchanged": source_serialized == source_after},
        "baselines": {
            "model": {"source": "direct LLM completion", "raw_context": False, "artifact_reuse": False, "new_composed_program": False},
            "raw_primitives": {"source": "direct LLM completion with independent primitive pairs", "raw_context": True, "artifact_reuse": False, "new_composed_program": False},
            "flat_synthesized": {"source": "monolithic typed AST selected from validation evidence", "raw_context": False, "artifact_reuse": False, "new_composed_program": True},
            "learned_composition": {"source": "deterministic execution of accepted composition plans", "raw_context": False, "artifact_reuse": True, "new_composed_program": True},
        },
        "measurements": {"raw_primitive_chars": len(raw_text), "raw_primitive_bytes": len(raw_text.encode("utf-8")), "skill_count": len(skills), "composition_candidate_count": composition_stats.candidate_composition_count, "composition_candidate_evaluation_count": composition_stats.candidate_evaluation_count, "scaling_note": "Unique candidate count grows with unary_chain slots, binary choices, and post-chain choices; evaluation count additionally multiplies by task groups."},
        "heldout_results": [asdict(result) for result in results],
    }
    report_path = Path(report_directory)
    report_path.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(UTC).strftime("air-006-%Y%m%dT%H%M%SZ.json")
    path = report_path / filename
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_file"] = str(path)
    return report
