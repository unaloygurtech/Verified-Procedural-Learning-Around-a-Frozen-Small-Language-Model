from __future__ import annotations

from dataclasses import dataclass
import json

from .model_client import Completion, LlamaCppClient
from .neralis import TRAIN_CASES, parse_response, raw_experiences


@dataclass(frozen=True)
class SkillCandidate:
    name: str
    body: str
    elapsed_seconds: float
    prompt_tokens: int
    generated_tokens: int
    method: str


def consolidation_prompt() -> str:
    return f"""You are AIR's experience consolidator.

Infer one reusable deterministic procedure from the verified experiences below.
Use only these experiences. Do not assume access to validation or test cases.
Do not copy the full example list into the procedure.

{raw_experiences()}

Return one JSON object with exactly these fields:
- `name`: the string `neralis-3-generated`
- `body`: a concise standalone skill that states the exact rules needed to
  transform a new Neralis-3 input into keys `key`, `score`, and `label`.
"""


def _candidate_from_completion(completion: Completion) -> SkillCandidate:
    payload = parse_response(completion.text)
    if payload is None:
        raise ValueError("consolidator did not return a JSON object")
    name = payload.get("name")
    body = payload.get("body")
    if name != "neralis-3-generated":
        raise ValueError("consolidator returned an unexpected skill name")
    if not isinstance(body, str) or not 80 <= len(body) <= 3000:
        raise ValueError("consolidator returned an invalid skill body")
    return SkillCandidate(
        name=name,
        body=body.strip(),
        elapsed_seconds=completion.elapsed_seconds,
        prompt_tokens=completion.prompt_tokens or 0,
        generated_tokens=completion.generated_tokens or 0,
        method="one-shot",
    )


def generate_skill(client: LlamaCppClient) -> SkillCandidate:
    completion = client.chat_json(consolidation_prompt(), max_tokens=512)
    return _candidate_from_completion(completion)


def _ask_object(client: LlamaCppClient, prompt: str) -> tuple[dict[str, object], Completion]:
    completion = client.chat(prompt, max_tokens=512, thinking=True)
    payload = parse_response(completion.text)
    if payload is None:
        raise ValueError("decomposed consolidator did not return a JSON object")
    return payload, completion


def _signal_examples(signal: str) -> str:
    rows = []
    for case in TRAIN_CASES:
        if case.signal == signal:
            rows.append(
                f"input value={case.value} -> verified score={case.expected()['score']}, "
                f"verified label={case.expected()['label']}"
            )
    return "\n".join(rows)


def generate_skill_decomposed(client: LlamaCppClient) -> SkillCandidate:
    completions: list[Completion] = []
    key_examples = "\n".join(
        json.dumps(
            {
                "code": case.code,
                "tag": case.tag,
                "verified_key": case.expected()["key"],
            },
            sort_keys=True,
        )
        for case in TRAIN_CASES[:6]
    )
    key_payload, completion = _ask_object(
        client,
        f"""Infer the key template shared by all verified examples.
{key_examples}
Choose exactly one template from: code:tag, code-tag, tag:code, tag-code.
Return {{"template":"chosen-template"}}.""",
    )
    completions.append(completion)
    template = key_payload.get("template")
    if template not in {"code:tag", "code-tag", "tag:code", "tag-code"}:
        raise ValueError("invalid generated key template")

    signal_rules: dict[str, tuple[str, int, str]] = {}
    for signal in ("amber", "cobalt", "ivory"):
        payload, completion = _ask_object(
            client,
            f"""Infer the exact score and label rule for signal `{signal}`.
{_signal_examples(signal)}
Compare each input value directly with its corresponding verified score. Do not
analyze the sequence of output scores by itself.
Choose operation from add, subtract, multiply, identity. `operand` must be an
integer. The formula applies the operation to the input value. Return exactly:
{{"operation":"...","operand":0,"label":"..."}}.""",
        )
        completions.append(completion)
        operation = payload.get("operation")
        operand = payload.get("operand")
        label = payload.get("label")
        if operation not in {"add", "subtract", "multiply", "identity"}:
            raise ValueError(f"invalid operation for {signal}")
        if not isinstance(operand, int) or not isinstance(label, str) or not label:
            raise ValueError(f"invalid generated rule for {signal}")
        signal_rules[signal] = (operation, operand, label)

    formula_text = {
        "add": "value + {operand}",
        "subtract": "value - {operand}",
        "multiply": "value * {operand}",
        "identity": "value",
    }
    rule_lines = []
    for signal, (operation, operand, label) in signal_rules.items():
        formula = formula_text[operation].format(operand=operand)
        rule_lines.append(f"- {signal}: score = {formula}; label = `{label}`")
    body = (
        "# Generated Neralis-3 skill\n\n"
        f"- key template: `{template}`\n"
        + "\n".join(rule_lines)
        + "\n- Return exactly one JSON object with keys `key`, `score`, and `label`."
    )
    return SkillCandidate(
        name="neralis-3-generated-decomposed",
        body=body,
        elapsed_seconds=sum(item.elapsed_seconds for item in completions),
        prompt_tokens=sum(item.prompt_tokens or 0 for item in completions),
        generated_tokens=sum(item.generated_tokens or 0 for item in completions),
        method="decomposed-dsl",
    )


def _apply_score(operation: str, operand: int, value: int) -> int:
    if operation == "add":
        return value + operand
    if operation == "subtract":
        return value - operand
    if operation == "multiply":
        return value * operand
    if operation == "identity":
        return value
    raise ValueError(f"unknown operation: {operation}")


def generate_skill_symbolic() -> SkillCandidate:
    key_hypotheses = {
        "code:tag": lambda case: f"{case.code}:{case.tag}",
        "code-tag": lambda case: f"{case.code}-{case.tag}",
        "tag:code": lambda case: f"{case.tag}:{case.code}",
        "tag-code": lambda case: f"{case.tag}-{case.code}",
    }
    matching_keys = [
        name
        for name, render in key_hypotheses.items()
        if all(render(case) == case.expected()["key"] for case in TRAIN_CASES)
    ]
    if len(matching_keys) != 1:
        raise ValueError(f"key hypothesis is not unique: {matching_keys}")

    hypotheses = [("identity", 0)]
    hypotheses.extend(("add", operand) for operand in range(1, 21))
    hypotheses.extend(("subtract", operand) for operand in range(1, 21))
    hypotheses.extend(("multiply", operand) for operand in range(2, 11))

    signal_rules: dict[str, tuple[str, int, str]] = {}
    for signal in ("amber", "cobalt", "ivory"):
        cases = [case for case in TRAIN_CASES if case.signal == signal]
        matching_scores = [
            (operation, operand)
            for operation, operand in hypotheses
            if all(
                _apply_score(operation, operand, case.value) == case.expected()["score"]
                for case in cases
            )
        ]
        labels = {str(case.expected()["label"]) for case in cases}
        if len(matching_scores) != 1 or len(labels) != 1:
            raise ValueError(
                f"signal hypothesis is not unique for {signal}: {matching_scores}, {labels}"
            )
        operation, operand = matching_scores[0]
        signal_rules[signal] = (operation, operand, labels.pop())

    formula_text = {
        "add": "value + {operand}",
        "subtract": "value - {operand}",
        "multiply": "value * {operand}",
        "identity": "value",
    }
    score_lines = []
    label_parts = []
    for signal, (operation, operand, label) in signal_rules.items():
        formula = formula_text[operation].format(operand=operand)
        score_lines.append(f"   - {signal}: `{formula}`")
        label_parts.append(f"{signal} maps to `{label}`")
    key_template = matching_keys[0]
    key_parts = key_template.split(":")
    body = (
        "# Neralis-3 normalization skill\n\n"
        "Input fields are `code` (text), `value` (integer), `signal`, and `tag`.\n\n"
        f"1. `key`: join `{key_parts[0]}`, a colon (`:`), and `{key_parts[1]}`, "
        "preserving their characters.\n"
        "2. `score`:\n"
        + "\n".join(score_lines)
        + "\n3. `label`: "
        + ", ".join(label_parts[:-1])
        + f", and {label_parts[-1]}.\n"
        "4. Return exactly one JSON object with keys `key`, `score`, and `label`.\n"
    )
    return SkillCandidate(
        name="neralis-3-generated-symbolic",
        body=body,
        elapsed_seconds=0.0,
        prompt_tokens=0,
        generated_tokens=0,
        method="symbolic-hypothesis-search",
    )
