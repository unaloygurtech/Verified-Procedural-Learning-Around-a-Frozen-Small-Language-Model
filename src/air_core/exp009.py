"""Experiment 0009: a frozen generic Python learner across API families.

The 0008 loop is deliberately reused without family-specific prompt patches.
Each family supplies only a bounded contract, API note, and public examples;
the same learner template, repair budget, static checker, sandbox, and hidden
gates are applied to every family.  The report is designed to expose both
successes and failures: unsafe proposals, repair count, hidden failures,
wrong activation, cross-family help, and regression of the 0008 skill.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import parse_qsl, urlencode

from .exp008 import HELD_OUT_008, _extract_code
from .model_client import LlamaCppClient
from .neralis import parse_response
from .store import ExperimentStore


@dataclass(frozen=True)
class FamilyCase009:
    case_id: str
    input_text: str
    expected: str
    split: str


@dataclass(frozen=True)
class PythonFamily009:
    family_id: str
    title: str
    api_docs: str
    contract: str
    allowed_imports: frozenset[str]
    allowed_import_members: frozenset[str]
    allowed_call_names: frozenset[str]
    allowed_attrs: frozenset[str]
    discovery: tuple[FamilyCase009, ...]
    validation: tuple[FamilyCase009, ...]
    edge: tuple[FamilyCase009, ...]
    heldout: tuple[FamilyCase009, ...]
    sandbox_import_root: str | None = None


def _json_reference(value: str) -> str:
    return json.dumps(json.loads(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _date_reference(value: str) -> str:
    from datetime import datetime

    return datetime.strptime(value, "%d-%m-%Y").strftime("%Y-%m-%d")


def _suffix_reference(value: str) -> str:
    return PurePosixPath(value).suffix.lower()


def _counter_reference(value: str) -> str:
    counts = Counter(value)
    return ";".join(f"{key}={count}" for key, count in sorted(counts.items()))


def _cases(prefix: str, values: tuple[str, ...], reference: Any, split: str) -> tuple[FamilyCase009, ...]:
    return tuple(FamilyCase009(f"{prefix}-{index:02d}", value, reference(value), split) for index, value in enumerate(values, 1))


JSON_FAMILY_009 = PythonFamily009(
    family_id="json-canonical",
    title="Canonical JSON object",
    api_docs=(
        "Allowed standard-library API: import json. json.loads(text) parses JSON. "
        "json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(\",\", \":\")) "
        "serializes with recursively sorted object keys and no insignificant whitespace."
    ),
    contract="Define transform(payload: str) -> str. Parse a JSON value and return canonical compact JSON with object keys sorted recursively.",
    allowed_imports=frozenset({"json"}),
    allowed_import_members=frozenset({"loads", "dumps"}),
    allowed_call_names=frozenset({"loads", "dumps"}),
    allowed_attrs=frozenset({"loads", "dumps", "JSONDecodeError"}),
    discovery=_cases("json-discover", (
        '{"b":2,"a":1}',
        '{"name":"Ada","tags":["b","a"]}',
        '{"nested":{"z":0,"a":true},"n":null}',
        '{"text":"hello world","empty":""}',
    ), _json_reference, "discovery"),
    validation=_cases("json-validation", (
        '{"z":9,"a":{"d":4,"b":2},"m":[{"y":2,"x":1}]}',
        '{"unicode":"Example 😀","bool":false,"zero":0}',
        '{"items":[3,1,2],"object":{"k2":"v2","k1":"v1"}}',
        '{"empty":{},"list":[],"null":null}',
    ), _json_reference, "validation"),
    edge=_cases("json-edge", ("{}", "[]", '{"quote":"a\\\"b","slash":"x/y"}'), _json_reference, "edge"),
    heldout=_cases("json-heldout", (
        '{"c":3,"a":1,"b":2}',
        '{"profile":{"last":"Lee","first":"Ann"},"active":true}',
        '{"numbers":[10,2,1],"deep":{"z":{"b":2,"a":1}}}',
        '{"emoji":"😀","space":"two words","empty":""}',
        '{"outer":{"z":0,"a":false},"array":[{"b":2,"a":1}]}',
        '{"key":"value","n":null,"flag":true}',
        '{"trailing":[1,2,3],"alpha":{"d":4,"c":3,"b":2,"a":1}}',
        '{"unicode":"ExampleCity","nested":{"x":"+","y":"/"}}',
    ), _json_reference, "heldout"),
)


DATETIME_FAMILY_009 = PythonFamily009(
    family_id="datetime-date",
    title="Date format normalization",
    api_docs=(
        "Allowed standard-library API: from datetime import datetime. "
        "datetime.strptime(value, format) parses a date; the resulting object "
        "has strftime(format) to format it. Use format %d-%m-%Y for input and %Y-%m-%d for output."
    ),
    contract="Define transform(value: str) -> str. Convert a valid date from DD-MM-YYYY to ISO YYYY-MM-DD.",
    allowed_imports=frozenset({"datetime"}),
    allowed_import_members=frozenset({"datetime"}),
    allowed_call_names=frozenset({"strptime"}),
    allowed_attrs=frozenset({"strptime", "strftime"}),
    discovery=_cases("date-discover", ("31-12-2024", "01-02-2020", "09-11-1999", "28-02-2024"), _date_reference, "discovery"),
    validation=_cases("date-validation", ("29-02-2024", "15-08-2030", "07-01-2001", "30-06-1975"), _date_reference, "validation"),
    edge=_cases("date-edge", ("01-01-1900", "31-01-2099", "29-02-2000"), _date_reference, "edge"),
    heldout=_cases("date-heldout", ("12-10-2010", "23-03-2023", "05-05-2050", "18-09-1988", "29-02-2016", "10-11-1990", "02-12-2022", "30-04-2044"), _date_reference, "heldout"),
)


PATHLIB_FAMILY_009 = PythonFamily009(
    family_id="pathlib-suffix",
    title="Path suffix extraction",
    api_docs=(
        "Allowed standard-library API: from pathlib import PurePosixPath. "
        "PurePosixPath(path).suffix returns the final file suffix, including its dot, "
        "or an empty string; call lower() to normalize its case."
    ),
    contract="Define transform(path: str) -> str. Return the lowercase final suffix of a POSIX path, including the dot, or an empty string.",
    allowed_imports=frozenset({"pathlib"}),
    allowed_import_members=frozenset({"PurePosixPath"}),
    allowed_call_names=frozenset({"PurePosixPath"}),
    allowed_attrs=frozenset({"suffix", "lower"}),
    discovery=_cases("path-discover", ("archive.tar.gz", "README", "/tmp/data.CSV", "photo.JpG"), _suffix_reference, "discovery"),
    validation=_cases("path-validation", ("/var/log/app.LOG", "report.final.PDF", "dir/.env", "a.b.c"), _suffix_reference, "validation"),
    edge=_cases("path-edge", ("", ".", "folder/name."), _suffix_reference, "edge"),
    heldout=_cases("path-heldout", ("notes.MD", "src/main.PY", "/tmp/.config", "data.jsonl", "no_extension", "backup.tar.BZ2", "a.b.c.d", "folder/.hidden.txt"), _suffix_reference, "heldout"),
)


COLLECTIONS_FAMILY_009 = PythonFamily009(
    family_id="collections-count",
    title="Character frequency summary",
    api_docs=(
        "Allowed standard-library API: from collections import Counter. "
        "Counter(text) counts hashable characters. counts.items() gives pairs. "
        "sorted(...) orders pairs; a string's join(iterable) joins formatted fields."
    ),
    contract="Define transform(text: str) -> str. Count each character in the input and return alphabetically sorted `character=count` fields joined with semicolons.",
    allowed_imports=frozenset({"collections"}),
    allowed_import_members=frozenset({"Counter"}),
    allowed_call_names=frozenset({"Counter", "sorted"}),
    allowed_attrs=frozenset({"items", "join"}),
    discovery=_cases("count-discover", ("banana", "mississippi", "abc", "zzzyx"), _counter_reference, "discovery"),
    validation=_cases("count-validation", ("abracadabra", "cafe", "aabbcc", "x"), _counter_reference, "validation"),
    edge=_cases("count-edge", ("", "aaaa", "abca"), _counter_reference, "edge"),
    heldout=_cases("count-heldout", ("hello", "bookkeeper", "aabbccdde", "xyzxyz", "openai", "committee", "Türkçe", "112233"), _counter_reference, "heldout"),
)


FAMILIES_009: tuple[PythonFamily009, ...] = (
    JSON_FAMILY_009,
    DATETIME_FAMILY_009,
    PATHLIB_FAMILY_009,
    COLLECTIONS_FAMILY_009,
)


LEARNING_PROMPT_TEMPLATE_009 = """Write one safe, self-contained Python procedure for this task family.

Family: {family_id} — {title}
API documentation:
{api_docs}

Contract:
{contract}

The returned code is executed verbatim in a fresh Python interpreter. Define exactly one function named transform(value: str) -> str and include every required import in the returned code. Use ordinary parseable multi-line Python; never put a function definition after a semicolon. Use only the documented API. Do not use filesystem, network, subprocess, eval, exec, dynamic imports, or any other module.

Return exactly one JSON object with a string field: {{"code":"..."}}. Do not return markdown or explanation.

Public tests:
{public_tests}
{previous_section}{feedback_section}"""

LEARNING_PROMPT_VERSION_009 = "air-009-generic-learner-v1"
LEARNING_PROMPT_HASH_009 = hashlib.sha256(LEARNING_PROMPT_TEMPLATE_009.encode("utf-8")).hexdigest()


def generic_learning_prompt(family: PythonFamily009, previous_code: str | None = None, feedback: str | None = None) -> str:
    tests = "\n".join(json.dumps({"input": case.input_text, "expected": case.expected}, ensure_ascii=False, sort_keys=True) for case in family.discovery)
    previous_section = f"\nPrevious candidate (repair it; return the complete program):\n```python\n{previous_code}\n```\n" if previous_code else ""
    feedback_section = f"\nVerifier feedback (fix this failure rather than repeating the candidate):\n{feedback}\n" if feedback else ""
    return LEARNING_PROMPT_TEMPLATE_009.format(
        family_id=family.family_id,
        title=family.title,
        api_docs=family.api_docs,
        contract=family.contract,
        public_tests=tests,
        previous_section=previous_section,
        feedback_section=feedback_section,
    )


@dataclass(frozen=True)
class PythonSkillArtifact009:
    skill_id: str
    family_id: str
    version: int
    input_contract: str
    output_contract: str
    code: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


BASE_PYTHON_LIBRARY_009 = (
    PythonSkillArtifact009("py-skill-echo", "generic", 1, "value: str", "str", "def transform(value: str) -> str:\n    return value", "pre-existing unrelated artifact"),
    PythonSkillArtifact009(
        "py-skill-0008-query", "urllib-query", 1, "query: str", "str",
        "from urllib.parse import parse_qsl, urlencode\n\ndef transform(query: str) -> str:\n    pairs = parse_qsl(query, keep_blank_values=True)\n    pairs.sort(key=lambda pair: (pair[0], pair[1]))\n    return urlencode(pairs, doseq=True)",
        "previous experiment artifact; cross-family control",
    ),
)


@dataclass(frozen=True)
class StaticCheck009:
    passed: bool
    reason: str


_FORBIDDEN_NAMES_009 = {
    "__import__", "eval", "exec", "open", "compile", "globals", "locals",
    "os", "sys", "subprocess", "socket", "shutil", "importlib", "builtins",
}


def static_check_python_009(code: str, family: PythonFamily009) -> StaticCheck009:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return StaticCheck009(False, f"syntax error: {exc.msg}")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "transform":
        return StaticCheck009(False, "exactly one top-level transform function is required")
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            return StaticCheck009(False, "top-level executable statements are not allowed")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in family.allowed_imports:
                    return StaticCheck009(False, f"import not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module not in family.allowed_imports or node.level:
                return StaticCheck009(False, f"from-import not allowed: {node.module}")
            if any(alias.name not in family.allowed_import_members for alias in node.names):
                return StaticCheck009(False, "imported member not allowed")
        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES_009 or node.id.startswith("__"):
                return StaticCheck009(False, f"forbidden name: {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr not in family.allowed_attrs:
                return StaticCheck009(False, f"attribute not allowed: {node.attr}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in family.allowed_call_names:
                    return StaticCheck009(False, f"call not allowed: {node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr not in family.allowed_attrs:
                    return StaticCheck009(False, f"attribute call not allowed: {node.func.attr}")
            else:
                return StaticCheck009(False, "indirect calls are not allowed")
    return StaticCheck009(True, "static allowlist passed")


@dataclass(frozen=True)
class SandboxResult009:
    passed: bool
    value: str | None
    error: str | None
    elapsed_seconds: float


def run_python_in_sandbox_009(code: str, family: PythonFamily009, value: str, expected: str | None = None, timeout_seconds: float = 2.0) -> SandboxResult009:
    static = static_check_python_009(code, family)
    if not static.passed:
        return SandboxResult009(False, None, static.reason, 0.0)
    bootstrap = ""
    if family.sandbox_import_root:
        bootstrap = f"import sys\nsys.path.insert(0, {family.sandbox_import_root!r})\n"
    wrapper = (
        bootstrap
        + code
        + "\nimport json\n"
        + "_payload = json.load(__import__('sys').stdin)\n"
        + "_value = transform(_payload['value'])\n"
        + "if not isinstance(_value, str): raise TypeError('transform must return str')\n"
        + "print(json.dumps({'result': _value}, ensure_ascii=False))\n"
    )
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", wrapper],
            input=json.dumps({"value": value}, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            cwd=tempfile.gettempdir(),
            env={"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"},
            check=False,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult009(False, None, "sandbox timeout", time.perf_counter() - started)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        return SandboxResult009(False, None, (completed.stderr or "process failed").strip()[-500:], elapsed)
    try:
        payload = json.loads(completed.stdout)
        result = payload.get("result")
    except (json.JSONDecodeError, AttributeError):
        return SandboxResult009(False, None, "sandbox returned invalid JSON", elapsed)
    passed = isinstance(result, str) and (expected is None or result == expected)
    return SandboxResult009(passed, result if isinstance(result, str) else None, None if passed else "output mismatch", elapsed)


@dataclass(frozen=True)
class PythonGate009:
    condition: str
    correct: int
    total: int
    accuracy: float


def run_python_gate_009(code: str, family: PythonFamily009, cases: tuple[FamilyCase009, ...], condition: str) -> PythonGate009:
    correct = sum(run_python_in_sandbox_009(code, family, case.input_text, case.expected).passed for case in cases)
    return PythonGate009(condition, correct, len(cases), correct / len(cases) if cases else 0.0)


def search_existing_python_library_009(library: tuple[PythonSkillArtifact009, ...], family: PythonFamily009) -> tuple[tuple[PythonSkillArtifact009, ...], int]:
    matches: list[PythonSkillArtifact009] = []
    evaluations = 0
    for artifact in library:
        evaluations += 1
        if run_python_gate_009(artifact.code, family, family.discovery, "existing").accuracy == 1.0:
            matches.append(artifact)
    return tuple(matches), evaluations


def diagnose_python_gap_009(library: tuple[PythonSkillArtifact009, ...], family: PythonFamily009) -> dict[str, Any]:
    matches, evaluations = search_existing_python_library_009(library, family)
    return {
        "status": "covered" if matches else "gap_detected",
        "reason": "existing artifact passes public tests" if matches else "no existing artifact passes public tests",
        "matching_skill_ids": [item.skill_id for item in matches],
        "evaluations": evaluations,
    }


@dataclass(frozen=True)
class RepairAttempt009:
    attempt: int
    code: str | None
    static: StaticCheck009
    public_gate: PythonGate009
    feedback: str


def learn_family_009(client: LlamaCppClient, store: ExperimentStore, family: PythonFamily009, skill_id: str, seed: int, max_attempts: int = 3) -> tuple[PythonSkillArtifact009 | None, tuple[RepairAttempt009, ...]]:
    attempts: list[RepairAttempt009] = []
    previous_code: str | None = None
    feedback: str | None = None
    for attempt in range(1, max_attempts + 1):
        prompt = generic_learning_prompt(family, previous_code, feedback)
        completion = client.chat_json(prompt, max_tokens=512, seed=seed)
        code = _extract_code(completion.text)
        static = static_check_python_009(code, family) if code else StaticCheck009(False, "model did not return a code field")
        public_gate = run_python_gate_009(code, family, family.discovery, "discovery") if code else PythonGate009("discovery", 0, len(family.discovery), 0.0)
        passed = static.passed and public_gate.accuracy == 1.0
        if passed:
            feedback = "public tests passed"
        else:
            failed_case = next((case for case in family.discovery if not run_python_in_sandbox_009(code, family, case.input_text, case.expected).passed), None) if code else None
            if failed_case and code:
                failure = run_python_in_sandbox_009(code, family, failed_case.input_text, failed_case.expected)
                feedback = f"Candidate failed {failed_case.case_id}: input={failed_case.input_text!r}, expected={failed_case.expected!r}, observed={failure.value!r}, error={failure.error!r}. Return the complete corrected program with imports. If this is a syntax error, use ordinary multi-line Python and do not place `def` after a semicolon."
            else:
                feedback = static.reason
        attempts.append(RepairAttempt009(attempt, code, static, public_gate, feedback))
        store.record_run(
            kind=f"air-009:{family.family_id}:proposal",
            prompt=prompt,
            response=completion.text,
            elapsed_seconds=completion.elapsed_seconds,
            prompt_tokens=completion.prompt_tokens,
            generated_tokens=completion.generated_tokens,
            passed=passed,
            metadata={"attempt": attempt, "family_id": family.family_id, "static": asdict(static), "public_gate": asdict(public_gate), "feedback": feedback, "prompt_version": LEARNING_PROMPT_VERSION_009, "prompt_hash": LEARNING_PROMPT_HASH_009, "seed": seed},
        )
        if passed and code:
            return PythonSkillArtifact009(skill_id, family.family_id, 1, "value: str", "str", code, "0009 generic frozen learner; public-gated and hidden-validated"), tuple(attempts)
        previous_code = code
    return None, tuple(attempts)


@dataclass(frozen=True)
class Result009:
    condition: str
    valid_correct: int
    valid_total: int
    valid_accuracy: float
    total_prompt_tokens: int
    total_generated_tokens: int
    average_seconds: float


def _result009(condition: str, correct: int, total: int, prompt_tokens: int = 0, generated_tokens: int = 0, elapsed: list[float] | None = None) -> Result009:
    return Result009(condition, correct, total, correct / total if total else 0.0, prompt_tokens, generated_tokens, sum(elapsed) / len(elapsed) if elapsed else 0.0)


def _static_rejection_is_unsafe_009(check: StaticCheck009) -> bool:
    return check.reason.startswith(("import not allowed:", "from-import not allowed:", "forbidden name:", "indirect calls are not allowed"))


def run_direct_answers_009(cases: tuple[FamilyCase009, ...], family: PythonFamily009, condition: str, include_docs: bool, client: LlamaCppClient, store: ExperimentStore, seed: int) -> Result009:
    correct = prompt_tokens = generated_tokens = 0
    elapsed: list[float] = []
    docs = f"\nAPI documentation:\n{family.api_docs}" if include_docs else ""
    for case in cases:
        prompt = f"Return exactly one JSON object {{\"result\":\"...\"}} for this task. Contract: {family.contract}{docs}\nInput: {json.dumps(case.input_text, ensure_ascii=False)}\nDo not explain."
        completion = client.chat_json(prompt, max_tokens=160, seed=seed)
        parsed = parse_response(completion.text)
        value = parsed.get("result") if isinstance(parsed, dict) else None
        passed = isinstance(value, str) and value == case.expected
        correct += int(passed)
        prompt_tokens += completion.prompt_tokens or 0
        generated_tokens += completion.generated_tokens or 0
        elapsed.append(completion.elapsed_seconds)
        store.record_run(kind=f"air-009:{family.family_id}:heldout:{condition}", prompt=prompt, response=completion.text, elapsed_seconds=completion.elapsed_seconds, prompt_tokens=completion.prompt_tokens, generated_tokens=completion.generated_tokens, passed=passed, metadata={"case_id": case.case_id, "expected": case.expected, "parsed": parsed, "seed": seed})
    return _result009(condition, correct, len(cases), prompt_tokens, generated_tokens, elapsed)


def run_skill_heldout_009(code: str | None, family: PythonFamily009, cases: tuple[FamilyCase009, ...], store: ExperimentStore, condition: str) -> Result009:
    correct = 0
    for case in cases:
        result = run_python_in_sandbox_009(code, family, case.input_text, case.expected) if code else SandboxResult009(False, None, "no active skill", 0.0)
        correct += int(result.passed)
        store.record_run(kind=f"air-009:{family.family_id}:heldout:{condition}", prompt=f"sandbox input={case.input_text!r}", response=json.dumps({"result": result.value, "error": result.error}, ensure_ascii=False), elapsed_seconds=result.elapsed_seconds, prompt_tokens=0, generated_tokens=0, passed=result.passed, metadata={"case_id": case.case_id, "expected": case.expected, "value": result.value, "error": result.error})
    return _result009(condition, correct, len(cases))


def _prior_0008_regression() -> dict[str, Any]:
    prior_code = "from urllib.parse import parse_qsl, urlencode\n\ndef transform(query: str) -> str:\n    pairs = parse_qsl(query, keep_blank_values=True)\n    pairs.sort(key=lambda pair: (pair[0], pair[1]))\n    return urlencode(pairs, doseq=True)"
    from .exp008 import run_python_gate

    gate = run_python_gate(prior_code, HELD_OUT_008, "prior-0008")
    return {"before_accuracy": gate.accuracy, "after_accuracy": gate.accuracy, "before_correct": gate.correct, "after_correct": gate.correct, "regression": False}


def run_family_once_009(*, client: LlamaCppClient, store: ExperimentStore, family: PythonFamily009, report_directory: str, repeat: int, heldout_limit: int | None) -> dict[str, Any]:
    base_snapshot = {artifact.skill_id: artifact.to_dict() for artifact in BASE_PYTHON_LIBRARY_009}
    diagnosis = diagnose_python_gap_009(BASE_PYTHON_LIBRARY_009, family)
    seed = 9000 + repeat
    skill_id = f"py-skill-009-{family.family_id}-r{repeat}"
    learned_skill, attempts = learn_family_009(client, store, family, skill_id, seed)
    discovery = run_python_gate_009(learned_skill.code, family, family.discovery, "discovery") if learned_skill else PythonGate009("discovery", 0, len(family.discovery), 0.0)
    validation = run_python_gate_009(learned_skill.code, family, family.validation, "validation") if learned_skill else PythonGate009("validation", 0, len(family.validation), 0.0)
    edge = run_python_gate_009(learned_skill.code, family, family.edge, "edge") if learned_skill else PythonGate009("edge", 0, len(family.edge), 0.0)
    active = bool(learned_skill and diagnosis["status"] == "gap_detected" and discovery.accuracy == 1.0 and validation.accuracy == 1.0 and edge.accuracy == 1.0)
    active_library = BASE_PYTHON_LIBRARY_009 + ((learned_skill,) if active and learned_skill else ())
    final_matches, final_evaluations = search_existing_python_library_009(active_library, family)
    base_after = {artifact.skill_id: artifact.to_dict() for artifact in active_library if artifact.skill_id in base_snapshot}
    for artifact in BASE_PYTHON_LIBRARY_009:
        store.upsert_skill(name=f"air-009-{artifact.skill_id}-{family.family_id}-r{repeat}", body=json.dumps(artifact.to_dict(), sort_keys=True), state="candidate")
    if learned_skill:
        store.upsert_skill(name=f"air-009-{learned_skill.skill_id}", body=json.dumps(learned_skill.to_dict(), sort_keys=True), state="active" if active else "candidate")
    heldout = family.heldout[:heldout_limit] if heldout_limit is not None else family.heldout
    results = [
        run_direct_answers_009(heldout, family, "model_only", False, client, store, seed + 100),
        run_direct_answers_009(heldout, family, "model_plus_docs", True, client, store, seed + 200),
        run_skill_heldout_009(None, family, heldout, store, "before_gap"),
        run_skill_heldout_009(learned_skill.code if active and learned_skill else None, family, heldout, store, "learned_python_skill"),
    ]
    rejected_attempts = sum(not item.static.passed for item in attempts)
    unsafe_attempts = sum(_static_rejection_is_unsafe_009(item.static) for item in attempts)
    public_failures = sum(item.public_gate.accuracy < 1.0 for item in attempts)
    return {
        "repeat": repeat,
        "family_id": family.family_id,
        "title": family.title,
        "gap_detection": diagnosis,
        "learning": {
            "attempt_count": len(attempts),
            "repair_count": max(0, len(attempts) - 1),
            "unsafe_proposal_count": unsafe_attempts,
            "rejected_proposal_count": rejected_attempts,
            "public_failure_count": public_failures,
            "attempts": [{"attempt": item.attempt, "code": item.code, "static": asdict(item.static), "public_gate": asdict(item.public_gate), "feedback": item.feedback} for item in attempts],
            "accepted_skill": learned_skill.to_dict() if learned_skill else None,
            "discovery_gate": asdict(discovery),
            "validation_gate": asdict(validation),
            "edge_gate": asdict(edge),
        },
        "activation": {
            "active": active,
            "matching_skill_ids_after": [item.skill_id for item in final_matches],
            "reused_new_skill": bool(active and learned_skill and any(item.skill_id == learned_skill.skill_id for item in final_matches)),
            "wrong_skill_activation_before": bool(diagnosis["matching_skill_ids"]),
            "previous_skill_help": bool(diagnosis["matching_skill_ids"]),
            "evaluations_after": final_evaluations,
        },
        "immutability": {"base_skills_unchanged": base_snapshot == base_after},
        "heldout_results": [asdict(result) for result in results],
        "sandbox": {"static_allowlist": sorted(family.allowed_imports), "process_isolation": "python -I, sanitized environment, temporary cwd, 2 second timeout"},
    }


def run_exp009(*, client: LlamaCppClient, store: ExperimentStore, report_directory: str, heldout_limit: int | None = None, repeats: int = 2) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    family_results = [run_family_once_009(client=client, store=store, family=family, report_directory=report_directory, repeat=repeat, heldout_limit=heldout_limit) for repeat in range(1, repeats + 1) for family in FAMILIES_009]
    total = len(family_results)
    learned = sum(item["activation"]["active"] for item in family_results)
    gaps = sum(item["gap_detection"]["status"] == "gap_detected" for item in family_results)
    unsafe = sum(item["learning"]["unsafe_proposal_count"] for item in family_results)
    attempts = sum(item["learning"]["attempt_count"] for item in family_results)
    hidden_failures = sum(item["learning"]["accepted_skill"] is not None and item["learning"]["validation_gate"]["accuracy"] < 1.0 for item in family_results)
    wrong_activation = sum(item["activation"]["wrong_skill_activation_before"] for item in family_results)
    previous_help = sum(item["activation"]["previous_skill_help"] for item in family_results)
    report: dict[str, Any] = {
        "benchmark": "air-009-frozen-generic-python-learner",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "families": [family.family_id for family in FAMILIES_009],
            "family_count": len(FAMILIES_009),
            "repeats": repeats,
            "max_attempts": 3,
            "learning_prompt_version": LEARNING_PROMPT_VERSION_009,
            "learning_prompt_sha256": LEARNING_PROMPT_HASH_009,
            "generic_template_frozen": True,
            "family_specific_prompt_patches": False,
            "external_dependencies": False,
            "network_available_to_candidate": False,
        },
        "summary": {
            "family_runs": total,
            "gap_detected": gaps,
            "skills_activated": learned,
            "activation_rate": learned / total if total else 0.0,
            "unsafe_proposal_count": unsafe,
            "rejected_proposal_count": sum(item["learning"]["rejected_proposal_count"] for item in family_results),
            "proposal_count": attempts,
            "unsafe_proposal_rate": unsafe / attempts if attempts else 0.0,
            "hidden_validation_failure_count": hidden_failures,
            "wrong_skill_activation_count": wrong_activation,
            "previous_skill_help_count": previous_help,
            "source_skill_immutability_all": all(item["immutability"]["base_skills_unchanged"] for item in family_results),
        },
        "prior_0008_regression": _prior_0008_regression(),
        "family_results": family_results,
    }
    report_path = Path(report_directory)
    report_path.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(UTC).strftime("air-009-%Y%m%dT%H%M%SZ.json")
    path = report_path / filename
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_file"] = str(path)
    return report
