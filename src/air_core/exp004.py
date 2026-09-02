"""Experiment 0004: content-discriminator routing and negative candidates.

All four procedures deliberately share the same input and output schema.  The
consolidator must infer which normal ``recipe`` value selects which bounded DSL
program; input/output field names no longer identify the procedure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import mean

from .model_client import LlamaCppClient
from .neralis import parse_response
from .store import ExperimentStore


@dataclass(frozen=True)
class RecipeCase:
    case_id: str
    family: str
    recipe: str
    x: str
    y: str
    output: dict[str, object]

    def payload(self) -> dict[str, object]:
        return {"recipe": self.recipe, "x": self.x, "y": self.y}


def _numeric_output(recipe: str, x: str, y: str) -> dict[str, object]:
    left, right = int(x), int(y)
    if recipe == "sum":
        return {"value": left + right, "label": "total"}
    if recipe == "difference":
        return {"value": left - right, "label": "gap"}
    raise ValueError(recipe)


def _text_output(recipe: str, x: str, y: str) -> dict[str, object]:
    if recipe == "join":
        return {"value": f"{x.strip().lower()}/{y.strip().lower()}", "label": "path"}
    if recipe == "reverse":
        return {"value": f"{y.strip().lower()}/{x.strip().lower()}", "label": "reverse-path"}
    raise ValueError(recipe)


def _case(case_id: str, family: str, recipe: str, x: str, y: str) -> RecipeCase:
    output = _numeric_output(recipe, x, y) if recipe in {"sum", "difference"} else _text_output(recipe, x, y)
    return RecipeCase(case_id, family, recipe, x, y, output)


# The family label is test-harness-only.  Every model-facing row has exactly
# the same fields: recipe, x, y -> value, label.
TRAIN_004 = (
    _case("train-01", "arithmetic-total", "sum", "7", "5"),
    _case("train-02", "arithmetic-total", "sum", "12", "3"),
    _case("train-03", "arithmetic-gap", "difference", "12", "5"),
    _case("train-04", "arithmetic-gap", "difference", "4", "9"),
    _case("train-05", "text-forward", "join", " Alpha ", "Beta"),
    _case("train-06", "text-forward", "join", "Gamma", " Delta "),
    _case("train-07", "text-reverse", "reverse", "North", "Star"),
    _case("train-08", "text-reverse", "reverse", "Red", "Fox"),
)

VALIDATION_004 = (
    _case("validation-01", "arithmetic-total", "sum", "20", "22"),
    _case("validation-02", "arithmetic-total", "sum", "3", "18"),
    _case("validation-03", "arithmetic-gap", "difference", "5", "14"),
    _case("validation-04", "arithmetic-gap", "difference", "30", "11"),
    _case("validation-05", "text-forward", "join", " Echo ", "Foxtrot "),
    _case("validation-06", "text-forward", "join", "Hotel", " INDIA"),
    _case("validation-07", "text-reverse", "reverse", "Juliet", "Kilo"),
    _case("validation-08", "text-reverse", "reverse", "Lima", "Mike"),
)

HELD_OUT_004 = (
    _case("heldout-01", "arithmetic-total", "sum", "101", "-4"),
    _case("heldout-02", "arithmetic-total", "sum", "0", "37"),
    _case("heldout-03", "arithmetic-total", "sum", "-8", "-12"),
    _case("heldout-04", "arithmetic-gap", "difference", "-2", "7"),
    _case("heldout-05", "arithmetic-gap", "difference", "40", "-5"),
    _case("heldout-06", "arithmetic-gap", "difference", "9", "21"),
    _case("heldout-07", "text-forward", "join", "November", "Oscar"),
    _case("heldout-08", "text-forward", "join", "Papa ", " Quebec"),
    _case("heldout-09", "text-forward", "join", "Romeo", "sierra"),
    _case("heldout-10", "text-reverse", "reverse", "Tango", "Uniform"),
    _case("heldout-11", "text-reverse", "reverse", "Victor ", " Whiskey"),
    _case("heldout-12", "text-reverse", "reverse", "Xray", "Yankee"),
)


def _without_family(case: RecipeCase) -> dict[str, object]:
    return {"input": case.payload(), "verified_output": case.output}


def raw_experiences_004() -> str:
    lines = ["Verified AIR recipe experiences (family labels intentionally omitted):"]
    for case in TRAIN_004:
        lines.append(json.dumps(_without_family(case), sort_keys=True))
    return "\n".join(lines)


def task_prompt_004(case: RecipeCase, context: str | None = None) -> str:
    sections = []
    if context:
        sections.append(context)
    sections.extend(
        [
            "Apply the procedure selected by the recipe value to this input.",
            "Compute numbers fully and preserve exact strings. Never return arithmetic expressions or add spaces.",
            f"input={json.dumps(case.payload(), sort_keys=True)}",
            "Return only one JSON object with exactly the keys value and label.",
        ]
    )
    return "\n\n".join(sections)


MANUAL_SKILL_004 = """# AIR recipe-dispatch verified skill

All inputs have fields `recipe`, `x`, and `y`, and all outputs have `value` and
`label`. Select by the literal recipe value:

- `sum`: parse x and y as integers; value = x + y; label = `total`.
- `difference`: parse x and y as integers; value = x - y; label = `gap`.
- `join`: value = trimmed lowercase x + `/` + trimmed lowercase y; label = `path`.
- `reverse`: value = trimmed lowercase y + `/` + trimmed lowercase x; label = `reverse-path`.

Compute the final value and return exactly one JSON object, with no explanation.
"""


@dataclass(frozen=True)
class RecipeRule:
    recipe: str
    operation: str
    label: str


def _render_operation(rule: RecipeRule, case: RecipeCase) -> object:
    if rule.operation == "add":
        return int(case.x) + int(case.y)
    if rule.operation == "subtract":
        return int(case.x) - int(case.y)
    if rule.operation == "join-forward":
        return f"{case.x.strip().lower()}/{case.y.strip().lower()}"
    if rule.operation == "join-reverse":
        return f"{case.y.strip().lower()}/{case.x.strip().lower()}"
    raise ValueError(rule.operation)


def _infer_rule(recipe: str, cases: list[RecipeCase]) -> RecipeRule:
    candidates = (
        RecipeRule(recipe, "add", "total"),
        RecipeRule(recipe, "subtract", "gap"),
        RecipeRule(recipe, "join-forward", "path"),
        RecipeRule(recipe, "join-reverse", "reverse-path"),
    )
    matching = []
    for rule in candidates:
        try:
            matches = all(
                {"value": _render_operation(rule, case), "label": rule.label} == case.output
                for case in cases
            )
        except (TypeError, ValueError):
            matches = False
        if matches:
            matching.append(rule)
    if len(matching) != 1:
        raise ValueError(f"recipe {recipe!r} is ambiguous or unsupported: {matching}")
    return matching[0]


def discover_rules_004(cases: tuple[RecipeCase, ...] = TRAIN_004) -> tuple[RecipeRule, ...]:
    groups: dict[str, list[RecipeCase]] = {}
    for case in cases:
        groups.setdefault(case.recipe, []).append(case)
    rules = tuple(_infer_rule(recipe, items) for recipe, items in sorted(groups.items()))
    if len(rules) != 4:
        raise ValueError(f"expected four recipe procedures, got {len(rules)}")
    return rules


def _rule_text(rule: RecipeRule) -> str:
    operations = {
        "add": "value=int(x)+int(y); label=total",
        "subtract": "value=int(x)-int(y); label=gap",
        "join-forward": "value=trim(lower(x))+'/'+trim(lower(y)); label=path",
        "join-reverse": "value=trim(lower(y))+'/'+trim(lower(x)); label=reverse-path; example x=Juliet,y=Kilo => value=kilo/juliet",
    }
    return f"recipe={rule.recipe}: {operations[rule.operation]}."


@dataclass(frozen=True)
class Skill004Candidate:
    name: str
    body: str
    rules: tuple[RecipeRule, ...]
    method: str = "content-discriminator-and-bounded-dsl"


def generate_skill_004(cases: tuple[RecipeCase, ...] = TRAIN_004) -> Skill004Candidate:
    rules = discover_rules_004(cases)
    body = (
        "# Automatically discovered AIR recipe skill\n\n"
        "All procedures share fields recipe,x,y and return value,label. Select by the literal recipe value:\n"
        + "\n".join(f"- {_rule_text(rule)}" for rule in rules)
        + "\n\nCompute the final value and return exactly one JSON object with keys value and label."
    )
    return Skill004Candidate("air-004-generated", body, rules)


def routed_context_004(case: RecipeCase, context: str | None, condition: str) -> str | None:
    if context is None or condition not in {"manual_skill", "generated_skill", "negative_candidate"}:
        return context
    if condition == "manual_skill":
        return {
            "sum": "recipe=sum: value=int(x)+int(y); label=total.",
            "difference": "recipe=difference: value=int(x)-int(y); label=gap.",
            "join": "recipe=join: value=trim(lower(x))+'/'+trim(lower(y)); label=path.",
            "reverse": "recipe=reverse: value=trim(lower(y))+'/'+trim(lower(x)); label=reverse-path.",
        }[case.recipe]
    for line in context.splitlines():
        if line.startswith("- recipe=") and line.startswith(f"- recipe={case.recipe}:"):
            return line[2:]
    raise ValueError(f"router could not match recipe {case.recipe!r}")


@dataclass(frozen=True)
class Result004:
    condition: str
    correct: int
    total: int
    accuracy: float
    total_prompt_tokens: int
    total_generated_tokens: int
    average_seconds: float


def run_cases_004(
    *,
    client: LlamaCppClient,
    store: ExperimentStore,
    condition: str,
    cases: tuple[RecipeCase, ...],
    phase: str,
    context: str | None,
) -> Result004:
    elapsed: list[float] = []
    prompt_tokens: list[int] = []
    generated_tokens: list[int] = []
    correct = 0
    for case in cases:
        prompt = task_prompt_004(case, routed_context_004(case, context, condition))
        completion = client.chat_json(prompt, max_tokens=100)
        parsed = parse_response(completion.text)
        passed = parsed == case.output
        correct += int(passed)
        elapsed.append(completion.elapsed_seconds)
        prompt_tokens.append(completion.prompt_tokens or 0)
        generated_tokens.append(completion.generated_tokens or 0)
        store.record_run(
            kind=f"air-004:{phase}:{condition}",
            prompt=prompt,
            response=completion.text,
            elapsed_seconds=completion.elapsed_seconds,
            prompt_tokens=completion.prompt_tokens,
            generated_tokens=completion.generated_tokens,
            passed=passed,
            metadata={"case_id": case.case_id, "recipe": case.recipe, "expected": case.output, "parsed": parsed},
        )
    return Result004(
        condition,
        correct,
        len(cases),
        correct / len(cases) if cases else 0.0,
        sum(prompt_tokens),
        sum(generated_tokens),
        mean(elapsed) if elapsed else 0.0,
    )


def run_exp004(
    *,
    client: LlamaCppClient,
    store: ExperimentStore,
    report_directory: str,
    heldout_limit: int | None = None,
    threshold: float = 0.8,
) -> dict[str, object]:
    candidate = generate_skill_004()
    store.upsert_skill(name=candidate.name, body=candidate.body, state="candidate")
    validation = run_cases_004(
        client=client, store=store, condition="generated_skill", cases=VALIDATION_004,
        phase="validation", context=candidate.body,
    )
    active = validation.accuracy >= threshold
    store.set_skill_state(name=candidate.name, state="active" if active else "rejected")

    bad_body = candidate.body.replace("label=total", "label=wrong").replace("label=gap", "label=wrong")
    negative_validation = run_cases_004(
        client=client, store=store, condition="negative_candidate", cases=VALIDATION_004,
        phase="negative-validation", context=bad_body,
    )
    negative_rejected = negative_validation.accuracy < threshold

    heldout = HELD_OUT_004[:heldout_limit]
    results = [
        run_cases_004(client=client, store=store, condition="model", cases=heldout, phase="heldout", context=None),
        run_cases_004(client=client, store=store, condition="raw", cases=heldout, phase="heldout", context=raw_experiences_004()),
        run_cases_004(client=client, store=store, condition="manual_skill", cases=heldout, phase="heldout", context=MANUAL_SKILL_004),
    ]
    if active:
        results.append(run_cases_004(client=client, store=store, condition="generated_skill", cases=heldout, phase="heldout", context=candidate.body))
    report = {
        "benchmark": "air-004-content-discriminator",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "same_input_schema": True,
            "same_output_schema": True,
            "recipe_values": sorted({case.recipe for case in TRAIN_004}),
            "family_labels_exposed_to_model": False,
            "training_cases": len(TRAIN_004),
            "validation_cases": len(VALIDATION_004),
            "heldout_cases": len(HELD_OUT_004),
        },
        "candidate": {
            "name": candidate.name,
            "method": candidate.method,
            "body": candidate.body,
            "discovered_rules": [asdict(rule) for rule in candidate.rules],
            "threshold": threshold,
            "validation": asdict(validation),
            "state": "active" if active else "rejected",
        },
        "negative_candidate": {
            "validation": asdict(negative_validation),
            "state": "rejected" if negative_rejected else "unsafe-active",
        },
        "heldout_results": [asdict(item) for item in results],
    }
    report_path = Path(report_directory)
    report_path.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(UTC).strftime("air-004-%Y%m%dT%H%M%SZ.json")
    path = report_path / filename
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_file"] = str(path)
    return report
