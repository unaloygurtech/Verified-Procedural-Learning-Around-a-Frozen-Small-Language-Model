"""Experiment 0005: typed executable skill synthesis.

0004 showed that a short prose rule can lose type and formatting semantics.  In
0005 the consolidator searches a small typed AST language and the accepted skill
is executed directly; the LLM is not asked to reinterpret the synthesized AST.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .model_client import LlamaCppClient
from .neralis import parse_response
from .store import ExperimentStore


class TypeCheckError(ValueError):
    """Raised when an AST violates the typed DSL contracts."""


class SynthesisError(ValueError):
    """Raised when evidence is ambiguous, conflicting, or unsupported."""


STRING = "String"
INT = "Int"
FIELD_TYPES = {"recipe": STRING, "x": STRING, "y": STRING}


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple["Expr", ...] = ()
    value: str | int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"op": self.op}
        if self.value is not None:
            result["value"] = self.value
        if self.args:
            result["args"] = [arg.to_dict() for arg in self.args]
        return result


@dataclass(frozen=True)
class Program:
    value: Expr
    label: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value.to_dict(), "label": self.label.to_dict()}


def field(name: str) -> Expr:
    return Expr("FIELD", value=name)


def literal(value: str | int) -> Expr:
    return Expr("LITERAL", value=value)


def unary(op: str, child: Expr) -> Expr:
    return Expr(op, args=(child,))


def binary(op: str, left: Expr, right: Expr) -> Expr:
    return Expr(op, args=(left, right))


def type_of(expr: Expr, field_types: dict[str, str] = FIELD_TYPES) -> str:
    if expr.op == "FIELD":
        if not isinstance(expr.value, str) or expr.value not in field_types:
            raise TypeCheckError(f"unknown field: {expr.value!r}")
        return field_types[expr.value]
    if expr.op == "LITERAL":
        if isinstance(expr.value, bool) or not isinstance(expr.value, (str, int)):
            raise TypeCheckError("literal must be String or Int")
        return INT if isinstance(expr.value, int) else STRING
    if expr.op in {"TRIM", "LOWER"}:
        if len(expr.args) != 1 or type_of(expr.args[0], field_types) != STRING:
            raise TypeCheckError(f"{expr.op}(String) required")
        return STRING
    if expr.op == "PARSE_INT":
        if len(expr.args) != 1 or type_of(expr.args[0], field_types) != STRING:
            raise TypeCheckError("PARSE_INT(String) required")
        return INT
    if expr.op in {"ADD", "SUB"}:
        if len(expr.args) != 2 or any(type_of(arg, field_types) != INT for arg in expr.args):
            raise TypeCheckError(f"{expr.op}(Int, Int) required")
        return INT
    if expr.op == "CONCAT":
        if len(expr.args) != 2 or any(type_of(arg, field_types) != STRING for arg in expr.args):
            raise TypeCheckError("CONCAT(String, String) required")
        return STRING
    if expr.op == "REPLACE":
        if len(expr.args) != 3 or any(type_of(arg, field_types) != STRING for arg in expr.args):
            raise TypeCheckError("REPLACE(String, String, String) required")
        return STRING
    raise TypeCheckError(f"unknown operation: {expr.op}")


def check_program(program: Program) -> dict[str, str]:
    value_type = type_of(program.value)
    label_type = type_of(program.label)
    if label_type != STRING:
        raise TypeCheckError("program label must be String")
    return {"value": value_type, "label": label_type}


def execute_expr(expr: Expr, payload: dict[str, object]) -> str | int:
    if expr.op == "FIELD":
        value = payload.get(expr.value)
        if not isinstance(value, str):
            raise TypeCheckError(f"field {expr.value!r} is not a String at runtime")
        return value
    if expr.op == "LITERAL":
        if isinstance(expr.value, bool) or not isinstance(expr.value, (str, int)):
            raise TypeCheckError("invalid literal at runtime")
        return expr.value
    if expr.op == "TRIM":
        return str(execute_expr(expr.args[0], payload)).strip()
    if expr.op == "LOWER":
        return str(execute_expr(expr.args[0], payload)).lower()
    if expr.op == "PARSE_INT":
        value = execute_expr(expr.args[0], payload)
        if not isinstance(value, str):
            raise TypeCheckError("PARSE_INT received non-string at runtime")
        try:
            return int(value.strip())
        except ValueError as exc:
            raise TypeCheckError(f"cannot parse integer: {value!r}") from exc
    if expr.op in {"ADD", "SUB"}:
        left = execute_expr(expr.args[0], payload)
        right = execute_expr(expr.args[1], payload)
        if not isinstance(left, int) or isinstance(left, bool) or not isinstance(right, int) or isinstance(right, bool):
            raise TypeCheckError(f"{expr.op} received non-integer at runtime")
        return left + right if expr.op == "ADD" else left - right
    if expr.op == "CONCAT":
        left = execute_expr(expr.args[0], payload)
        right = execute_expr(expr.args[1], payload)
        if not isinstance(left, str) or not isinstance(right, str):
            raise TypeCheckError("CONCAT received non-string at runtime")
        return left + right
    if expr.op == "REPLACE":
        source = execute_expr(expr.args[0], payload)
        old = execute_expr(expr.args[1], payload)
        new = execute_expr(expr.args[2], payload)
        if not isinstance(source, str) or not isinstance(old, str) or not isinstance(new, str):
            raise TypeCheckError("REPLACE received non-string at runtime")
        return source.replace(old, new)
    raise TypeCheckError(f"unknown operation: {expr.op}")


def execute_program(program: Program, payload: dict[str, object]) -> dict[str, str | int]:
    check_program(program)
    value = execute_expr(program.value, payload)
    label = execute_expr(program.label, payload)
    if not isinstance(label, str):
        raise TypeCheckError("label result is not String")
    return {"value": value, "label": label}


@dataclass(frozen=True)
class RecipeCase005:
    case_id: str
    family: str
    recipe: str
    x: str
    y: str
    output: dict[str, str | int]

    def payload(self) -> dict[str, object]:
        return {"recipe": self.recipe, "x": self.x, "y": self.y}


def _ref_sum(x: str, y: str) -> dict[str, str | int]:
    return {"value": int(x) + int(y), "label": "total"}


def _ref_difference(x: str, y: str) -> dict[str, str | int]:
    return {"value": int(x) - int(y), "label": "gap"}


def _ref_join(x: str, y: str) -> dict[str, str | int]:
    return {"value": f"{x.strip().lower()}/{y.strip().lower()}", "label": "path"}


def _ref_reverse(x: str, y: str) -> dict[str, str | int]:
    return {"value": f"{y.strip().lower()}/{x.strip().lower()}", "label": "reverse-path"}


REFERENCE = {"sum": _ref_sum, "difference": _ref_difference, "join": _ref_join, "reverse": _ref_reverse}


def make_case(case_id: str, family: str, recipe: str, x: str, y: str) -> RecipeCase005:
    return RecipeCase005(case_id, family, recipe, x, y, REFERENCE[recipe](x, y))


# New partitions: none of these held-out IDs or values reuse Experiment 0004's
# held-out set. Family names remain harness metadata and never enter raw text.
TRAIN_005 = (
    make_case("train-005-01", "arithmetic-total", "sum", "7", "5"),
    make_case("train-005-02", "arithmetic-total", "sum", "12", "3"),
    make_case("train-005-03", "arithmetic-gap", "difference", "12", "5"),
    make_case("train-005-04", "arithmetic-gap", "difference", "4", "9"),
    make_case("train-005-05", "text-forward", "join", " Alpha ", "Beta"),
    make_case("train-005-06", "text-forward", "join", "Gamma", " Delta "),
    make_case("train-005-07", "text-reverse", "reverse", "North", "Star"),
    make_case("train-005-08", "text-reverse", "reverse", "Red ", " Fox"),
)

VALIDATION_005 = (
    make_case("validation-005-01", "arithmetic-total", "sum", "20", "22"),
    make_case("validation-005-02", "arithmetic-total", "sum", "3", "18"),
    make_case("validation-005-03", "arithmetic-gap", "difference", "5", "14"),
    make_case("validation-005-04", "arithmetic-gap", "difference", "30", "11"),
    make_case("validation-005-05", "text-forward", "join", " Echo ", "Foxtrot "),
    make_case("validation-005-06", "text-forward", "join", "Hotel", " INDIA"),
    make_case("validation-005-07", "text-reverse", "reverse", "Juliet", "Kilo"),
    make_case("validation-005-08", "text-reverse", "reverse", "Lima", "Mike"),
)

HELD_OUT_005 = (
    make_case("heldout-005-01", "arithmetic-total", "sum", "101", "-4"),
    make_case("heldout-005-02", "arithmetic-total", "sum", "0", "37"),
    make_case("heldout-005-03", "arithmetic-total", "sum", "-8", "-12"),
    make_case("heldout-005-04", "arithmetic-gap", "difference", "-2", "7"),
    make_case("heldout-005-05", "arithmetic-gap", "difference", "40", "-5"),
    make_case("heldout-005-06", "arithmetic-gap", "difference", "9", "21"),
    make_case("heldout-005-07", "text-forward", "join", "November", "Oscar"),
    make_case("heldout-005-08", "text-forward", "join", "Papa ", " Quebec"),
    make_case("heldout-005-09", "text-forward", "join", "Romeo", "sierra"),
    make_case("heldout-005-10", "text-reverse", "reverse", "Tango", "Uniform"),
    make_case("heldout-005-11", "text-reverse", "reverse", "Victor ", " Whiskey"),
    make_case("heldout-005-12", "text-reverse", "reverse", "Xray", "Yankee"),
)

# Independently authored adversarial checks. They are not generated from a
# synthesized program, so they can expose a candidate that merely memorizes the
# normal validation rows.
EDGE_005 = (
    make_case("edge-zero", "arithmetic-total", "sum", "0", "0"),
    make_case("edge-negative-mix", "arithmetic-total", "sum", "-7", "2"),
    make_case("edge-equal", "arithmetic-gap", "difference", "9", "9"),
    make_case("edge-reversed-sign", "arithmetic-gap", "difference", "4", "-6"),
    make_case("edge-spaces", "text-forward", "join", "  Alpha Beta  ", " GAMMA "),
    make_case("edge-forward-space", "text-forward", "join", "X Y", " z"),
    make_case("edge-reverse-case", "text-reverse", "reverse", "  MiXeD ", "Case Test "),
    make_case("edge-reverse-spaces", "text-reverse", "reverse", "A B", " C D "),
)


def raw_experiences_005() -> str:
    lines = ["Verified AIR recipe experiences (procedure names omitted):"]
    for case in TRAIN_005:
        lines.append(json.dumps({"input": case.payload(), "verified_output": case.output}, sort_keys=True))
    return "\n".join(lines)


COMPACT_PROSE_SKILL_005 = """# Compact AIR recipe skill

Inputs have `recipe`, `x`, `y`; return `value`, `label`.
- sum: parse x and y as integers, add them, label `total`.
- difference: parse x and y as integers, subtract y from x, label `gap`.
- join: trim and lowercase x and y, join x then y with `/`, label `path`.
- reverse: trim and lowercase y and x, join y then x with `/`, label `reverse-path`.

Compute final values exactly. Return one JSON object only.
"""


def task_prompt_005(case: RecipeCase005, context: str | None = None) -> str:
    sections = []
    if context:
        sections.append(context)
    sections.extend(
        [
            "Apply the procedure selected by the recipe value.",
            "Compute numbers fully, preserve exact string casing/whitespace rules, and never return an expression.",
            f"input={json.dumps(case.payload(), sort_keys=True)}",
            "Return only one JSON object with exactly the keys value and label.",
        ]
    )
    return "\n\n".join(sections)


def _expr_key(expr: Expr) -> str:
    return json.dumps(expr.to_dict(), sort_keys=True, separators=(",", ":"))


def _candidate_value_exprs() -> tuple[Expr, ...]:
    x, y = field("x"), field("y")
    px, py = unary("PARSE_INT", x), unary("PARSE_INT", y)
    tx, ty = unary("TRIM", x), unary("TRIM", y)
    lx, ly = unary("LOWER", x), unary("LOWER", y)
    tlx, tly = unary("LOWER", tx), unary("LOWER", ty)
    # The first group contains intentionally type-invalid arithmetic/string
    # combinations; the checker counts and rejects them before execution.
    candidates = [
        binary("ADD", x, y),
        binary("SUB", x, y),
        binary("CONCAT", px, py),
        binary("ADD", px, py),
        binary("SUB", px, py),
        binary("SUB", py, px),
    ]
    for left, right in ((x, y), (tx, ty), (lx, ly), (tlx, tly)):
        candidates.extend(
            [
                binary("CONCAT", binary("CONCAT", left, literal("/")), right),
                binary("CONCAT", binary("CONCAT", right, literal("/")), left),
            ]
        )
    # Include the correct transform in either operand order and a few incomplete
    # normalization variants so the synthesis is a real search, not a lookup.
    unique: dict[str, Expr] = {}
    for expr in candidates:
        unique.setdefault(_expr_key(expr), expr)
    return tuple(unique.values())


@dataclass(frozen=True)
class SynthesisStats:
    candidate_search_count: int
    type_invalid_candidate_count: int
    ambiguous_candidate_count: int


@dataclass(frozen=True)
class Skill005Candidate:
    name: str
    programs: dict[str, Program]
    stats: SynthesisStats
    method: str = "typed-bounded-program-synthesis"

    def serialized(self) -> str:
        body = {
            "version": 1,
            "input_types": FIELD_TYPES,
            "routes": {recipe: program.to_dict() for recipe, program in sorted(self.programs.items())},
        }
        return json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def synthesize_005(cases: tuple[RecipeCase005, ...] = TRAIN_005) -> Skill005Candidate:
    groups: dict[str, list[RecipeCase005]] = {}
    for case in cases:
        groups.setdefault(case.recipe, []).append(case)
    if set(groups) != {"sum", "difference", "join", "reverse"}:
        raise SynthesisError(f"expected four discovered recipe groups, got {sorted(groups)}")

    value_exprs = _candidate_value_exprs()
    labels = sorted({str(case.output["label"]) for case in cases})
    programs: dict[str, Program] = {}
    search_count = 0
    type_invalid = 0
    ambiguous = 0
    for recipe, recipe_cases in sorted(groups.items()):
        matches: list[Program] = []
        for expr in value_exprs:
            for label in labels:
                search_count += 1
                program = Program(expr, literal(label))
                try:
                    check_program(program)
                except TypeCheckError:
                    type_invalid += 1
                    continue
                try:
                    if all(execute_program(program, case.payload()) == case.output for case in recipe_cases):
                        matches.append(program)
                except TypeCheckError:
                    continue
        if len(matches) > 1:
            ambiguous += 1
            raise SynthesisError(f"recipe {recipe!r} has {len(matches)} equally valid programs")
        if not matches:
            raise SynthesisError(f"no typed program explains recipe {recipe!r}")
        programs[recipe] = matches[0]
    return Skill005Candidate(
        "air-005-typed-generated",
        programs,
        SynthesisStats(search_count, type_invalid, ambiguous),
    )


def _routed_prose_context(case: RecipeCase005, context: str) -> str:
    for line in context.splitlines():
        if line.startswith(f"- {case.recipe}:"):
            return line
    raise ValueError(f"no prose rule for recipe {case.recipe!r}")


def _routed_program(candidate: Skill005Candidate, case: RecipeCase005) -> Program:
    try:
        return candidate.programs[case.recipe]
    except KeyError as exc:
        raise SynthesisError(f"router has no recipe {case.recipe!r}") from exc


@dataclass(frozen=True)
class Result005:
    condition: str
    correct: int
    total: int
    accuracy: float
    type_failures: int
    total_prompt_tokens: int
    total_generated_tokens: int
    average_seconds: float


def run_executable_cases_005(
    *,
    store: ExperimentStore,
    condition: str,
    cases: tuple[RecipeCase005, ...],
    phase: str,
    programs: dict[str, Program],
) -> Result005:
    correct = 0
    type_failures = 0
    for case in cases:
        passed = False
        parsed: object = None
        try:
            program = programs[case.recipe]
            parsed = execute_program(program, case.payload())
            passed = parsed == case.output and type(parsed["value"]) is type(case.output["value"]) and type(parsed["label"]) is str
        except (KeyError, TypeCheckError):
            type_failures += 1
        store.record_run(
            kind=f"air-005:{phase}:{condition}",
            prompt=f"deterministic-executor input={json.dumps(case.payload(), sort_keys=True)}",
            response=json.dumps(parsed, ensure_ascii=False, sort_keys=True),
            elapsed_seconds=0.0,
            prompt_tokens=0,
            generated_tokens=0,
            passed=passed,
            metadata={"case_id": case.case_id, "recipe": case.recipe, "expected": case.output, "parsed": parsed, "type_failure": not passed and parsed is None},
        )
        correct += int(passed)
    return Result005(condition, correct, len(cases), correct / len(cases) if cases else 0.0, type_failures, 0, 0, 0.0)


def run_model_cases_005(
    *,
    client: LlamaCppClient,
    store: ExperimentStore,
    condition: str,
    cases: tuple[RecipeCase005, ...],
    phase: str,
    context: str | None,
) -> Result005:
    correct = 0
    elapsed: list[float] = []
    prompt_tokens: list[int] = []
    generated_tokens: list[int] = []
    for case in cases:
        routed = _routed_prose_context(case, context) if context and condition == "compact_prose" else context
        prompt = task_prompt_005(case, routed)
        completion = client.chat_json(prompt, max_tokens=96)
        parsed = parse_response(completion.text)
        passed = parsed == case.output and isinstance(parsed, dict) and type(parsed.get("value")) is type(case.output["value"])
        correct += int(passed)
        elapsed.append(completion.elapsed_seconds)
        prompt_tokens.append(completion.prompt_tokens or 0)
        generated_tokens.append(completion.generated_tokens or 0)
        store.record_run(
            kind=f"air-005:{phase}:{condition}",
            prompt=prompt,
            response=completion.text,
            elapsed_seconds=completion.elapsed_seconds,
            prompt_tokens=completion.prompt_tokens,
            generated_tokens=completion.generated_tokens,
            passed=passed,
            metadata={"case_id": case.case_id, "recipe": case.recipe, "expected": case.output, "parsed": parsed},
        )
    return Result005(condition, correct, len(cases), correct / len(cases) if cases else 0.0, 0, sum(prompt_tokens), sum(generated_tokens), mean(elapsed) if elapsed else 0.0)


def corrupted_programs_005() -> dict[str, Program]:
    x, y = unary("PARSE_INT", field("x")), unary("PARSE_INT", field("y"))
    lower_x = unary("LOWER", unary("TRIM", field("x")))
    lower_y = unary("LOWER", unary("TRIM", field("y")))
    return {
        "sum": Program(binary("SUB", x, y), literal("total")),
        "difference": Program(binary("SUB", y, x), literal("gap")),
        "join": Program(binary("CONCAT", binary("CONCAT", lower_y, literal("/")), lower_x), literal("path")),
        "reverse": Program(binary("CONCAT", binary("CONCAT", field("y"), literal("/")), field("x")), literal("reverse-path")),
    }


def run_exp005(
    *,
    client: LlamaCppClient,
    store: ExperimentStore,
    report_directory: str,
    heldout_limit: int | None = None,
    threshold: float = 0.9,
) -> dict[str, object]:
    candidate = synthesize_005()
    validation = run_executable_cases_005(store=store, condition="typed_executable", cases=VALIDATION_005, phase="validation", programs=candidate.programs)
    edge = run_executable_cases_005(store=store, condition="typed_executable", cases=EDGE_005, phase="edge-validation", programs=candidate.programs)
    active = validation.accuracy >= threshold and edge.accuracy == 1.0
    store.upsert_skill(name=candidate.name, body=candidate.serialized(), state="candidate")
    store.set_skill_state(name=candidate.name, state="active" if active else "rejected")

    corrupted = run_executable_cases_005(store=store, condition="corrupted_candidate", cases=VALIDATION_005, phase="corrupted-validation", programs=corrupted_programs_005())
    corrupted_rejected = corrupted.accuracy < threshold

    heldout = HELD_OUT_005[:heldout_limit]
    results = [
        run_model_cases_005(client=client, store=store, condition="model", cases=heldout, phase="heldout", context=None),
        run_model_cases_005(client=client, store=store, condition="raw", cases=heldout, phase="heldout", context=raw_experiences_005()),
        run_model_cases_005(client=client, store=store, condition="compact_prose", cases=heldout, phase="heldout", context=COMPACT_PROSE_SKILL_005),
    ]
    if active:
        results.append(run_executable_cases_005(store=store, condition="typed_executable", cases=heldout, phase="heldout", programs=candidate.programs))

    raw_chars = len(raw_experiences_005())
    prose_chars = len(COMPACT_PROSE_SKILL_005)
    executable_chars = len(candidate.serialized())
    report = {
        "benchmark": "air-005-typed-executable-synthesis",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "same_input_schema": True,
            "same_output_schema": True,
            "family_labels_exposed_to_model": False,
            "training_cases": len(TRAIN_005),
            "validation_cases": len(VALIDATION_005),
            "edge_cases": len(EDGE_005),
            "heldout_cases": len(HELD_OUT_005),
            "heldout_is_new_for_0005": True,
        },
        "candidate": {
            "name": candidate.name,
            "method": candidate.method,
            "serialized_program": json.loads(candidate.serialized()),
            "discovered_recipe_groups": sorted(candidate.programs),
            "synthesized_programs": {recipe: program.to_dict() for recipe, program in sorted(candidate.programs.items())},
            "search": asdict(candidate.stats),
            "threshold": threshold,
            "validation": asdict(validation),
            "edge_validation": asdict(edge),
            "state": "active" if active else "rejected",
        },
        "corrupted_candidate": {
            "corruption": "typed but semantic: ADD/SUB swap, reverse operands, or missing trim/lower",
            "validation": asdict(corrupted),
            "state": "rejected" if corrupted_rejected else "unsafe-active",
        },
        "measurements": {
            "raw_experience_chars": raw_chars,
            "raw_experience_bytes": len(raw_experiences_005().encode("utf-8")),
            "compact_prose_chars": prose_chars,
            "executable_serialized_chars": executable_chars,
            "raw_to_prose_compression_ratio": round(raw_chars / prose_chars, 3),
            "raw_to_executable_compression_ratio": round(raw_chars / executable_chars, 3),
        },
        "heldout_results": [asdict(item) for item in results],
    }
    report_path = Path(report_directory)
    report_path.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(UTC).strftime("air-005-%Y%m%dT%H%M%SZ.json")
    path = report_path / filename
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_file"] = str(path)
    return report
