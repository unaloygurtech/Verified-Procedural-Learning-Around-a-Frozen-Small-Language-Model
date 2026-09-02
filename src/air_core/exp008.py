"""Experiment 0008: a small real Python/API learning loop.

0007 extended a hand-bounded DSL.  This experiment moves the learned artifact
into Python source: AIR detects that its existing Python library cannot solve a
query-canonicalization task, asks the local model for a procedure using a small
standard-library API document, executes proposals in a restricted subprocess,
repairs public-test failures, gates the result on hidden validation, and then
reuses the accepted code on new inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode

from .model_client import LlamaCppClient
from .neralis import parse_response
from .store import ExperimentStore


PYTHON_API_DOCS_008 = """Allowed standard-library API (no external packages):

from urllib.parse import parse_qsl, urlencode

parse_qsl(query, keep_blank_values=True) returns a list of (key, value) pairs,
decoding percent escapes and '+' spaces.  keep_blank_values=True retains keys
whose value is empty.  urlencode(pairs, doseq=True) converts pairs back to a
query string and encodes spaces as '+'.  A list of pairs can be sorted with
pair[0] as the key and pair[1] as the tie-breaker.
"""


@dataclass(frozen=True)
class PythonCase008:
    case_id: str
    query: str
    expected: str
    split: str


def _reference_canonical_query(query: str) -> str:
    pairs = parse_qsl(query, keep_blank_values=True)
    pairs.sort(key=lambda pair: (pair[0], pair[1]))
    return urlencode(pairs, doseq=True)


def _make_cases(prefix: str, queries: tuple[str, ...], split: str) -> tuple[PythonCase008, ...]:
    return tuple(PythonCase008(f"{prefix}-{index:02d}", query, _reference_canonical_query(query), split) for index, query in enumerate(queries, 1))


DISCOVERY_008 = _make_cases(
    "discover",
    (
        "b=2&a=1",
        "q=hello+world&empty=&a=1",
        "tag=z&tag=a&space=two+words",
        "utm_source=mail&x=1&utm_source=web",
    ),
    "discovery",
)

VALIDATION_008 = _make_cases(
    "validation",
    (
        "z=9&b=2&a=hello%20world",
        "empty=&a=&b=two%2Fwords",
        "k=2&k=10&k=1",
        "name=Alice+Smith&lang=en",
    ),
    "validation",
)

EDGE_008 = _make_cases(
    "edge",
    (
        "",
        "flag&b=2&a=1",
        "x=%2F&x=+&x=%20",
    ),
    "edge",
)

HELD_OUT_008 = _make_cases(
    "heldout",
    (
        "z=blue&name=Ann%20Lee&a=7",
        "city=New+York&empty=&b=2",
        "tag=delta&tag=alpha&tag=charlie",
        "path=%2Fapi%2Fv1&method=GET",
        "currency=EUR&amount=10&currency=USD",
        "q=one%2Btwo&b=last&a=first",
        "lang=tr&name=Example%20User&empty=",
        "x=2&x=02&x=20&x=",
        "filter=kind%3Abook&sort=desc&page=3",
        "emoji=%F0%9F%98%80&word=hello+there",
        "a=space%20here&b=slash%2Fvalue&c=plus%2Bsign",
        "z=1&z=0&aa=first&ab=second",
    ),
    "heldout",
)


@dataclass(frozen=True)
class PythonSkillArtifact008:
    skill_id: str
    version: int
    input_contract: str
    output_contract: str
    code: str
    source: str
    source_case_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "input_contract": self.input_contract,
            "output_contract": self.output_contract,
            "code": self.code,
            "source": self.source,
            "source_case_ids": self.source_case_ids,
        }


BASE_PYTHON_LIBRARY_008 = (
    PythonSkillArtifact008(
        "py-skill-echo",
        1,
        "query: str",
        "str",
        "def transform(query: str) -> str:\n    return query",
        "pre-existing unrelated Python artifact",
        (),
    ),
)


@dataclass(frozen=True)
class StaticCheck008:
    passed: bool
    reason: str


_ALLOWED_IMPORTS = {"urllib.parse"}
_ALLOWED_CALL_NAMES = {"parse_qsl", "urlencode", "sorted", "list", "tuple"}
_ALLOWED_ATTRS = {"parse_qsl", "urlencode", "sort"}
_FORBIDDEN_NAMES = {
    "__import__",
    "eval",
    "exec",
    "open",
    "compile",
    "globals",
    "locals",
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
}


def static_check_python(code: str) -> StaticCheck008:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return StaticCheck008(False, f"syntax error: {exc.msg}")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "transform":
        return StaticCheck008(False, "exactly one top-level transform function is required")
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            return StaticCheck008(False, "top-level executable statements are not allowed")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _ALLOWED_IMPORTS:
                    return StaticCheck008(False, f"import not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module not in _ALLOWED_IMPORTS or node.level:
                return StaticCheck008(False, f"from-import not allowed: {node.module}")
            if any(alias.name not in {"parse_qsl", "urlencode"} for alias in node.names):
                return StaticCheck008(False, "only parse_qsl and urlencode may be imported")
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            return StaticCheck008(False, f"forbidden name: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr not in _ALLOWED_ATTRS:
            return StaticCheck008(False, f"attribute not allowed: {node.attr}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _ALLOWED_CALL_NAMES:
                    return StaticCheck008(False, f"call not allowed: {node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr not in _ALLOWED_ATTRS:
                    return StaticCheck008(False, f"attribute call not allowed: {node.func.attr}")
            else:
                return StaticCheck008(False, "indirect calls are not allowed")
    return StaticCheck008(True, "static allowlist passed")


@dataclass(frozen=True)
class SandboxResult008:
    passed: bool
    value: str | None
    error: str | None
    elapsed_seconds: float


def run_python_in_sandbox(code: str, query: str, expected: str | None = None, timeout_seconds: float = 2.0) -> SandboxResult008:
    static = static_check_python(code)
    if not static.passed:
        return SandboxResult008(False, None, static.reason, 0.0)
    wrapper = (
        code
        + "\nimport json\n"
        + "_payload = json.load(__import__('sys').stdin)\n"
        + "_value = transform(_payload['query'])\n"
        + "if not isinstance(_value, str): raise TypeError('transform must return str')\n"
        + "print(json.dumps({'result': _value}, ensure_ascii=False))\n"
    )
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", wrapper],
            input=json.dumps({"query": query}, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            cwd=tempfile.gettempdir(),
            env={"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"},
            check=False,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult008(False, None, "sandbox timeout", time.perf_counter() - started)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        detail = (completed.stderr or "process failed").strip()[-500:]
        return SandboxResult008(False, None, detail, elapsed)
    try:
        payload = json.loads(completed.stdout)
        value = payload.get("result")
    except (json.JSONDecodeError, AttributeError):
        return SandboxResult008(False, None, "sandbox returned invalid JSON", elapsed)
    passed = isinstance(value, str) and (expected is None or value == expected)
    return SandboxResult008(passed, value if isinstance(value, str) else None, None if passed else "output mismatch", elapsed)


@dataclass(frozen=True)
class PythonGate008:
    condition: str
    correct: int
    total: int
    accuracy: float


def run_python_gate(code: str, cases: tuple[PythonCase008, ...], condition: str) -> PythonGate008:
    correct = sum(run_python_in_sandbox(code, case.query, case.expected).passed for case in cases)
    return PythonGate008(condition, correct, len(cases), correct / len(cases) if cases else 0.0)


@dataclass(frozen=True)
class GapDiagnosis008:
    status: str
    reason: str
    matching_skill_ids: tuple[str, ...]


def search_existing_python_library(library: tuple[PythonSkillArtifact008, ...], cases: tuple[PythonCase008, ...]) -> tuple[tuple[PythonSkillArtifact008, ...], int]:
    matches: list[PythonSkillArtifact008] = []
    evaluations = 0
    for artifact in library:
        evaluations += 1
        if run_python_gate(artifact.code, cases, "existing").accuracy == 1.0:
            matches.append(artifact)
    return tuple(matches), evaluations


def diagnose_python_gap(library: tuple[PythonSkillArtifact008, ...], cases: tuple[PythonCase008, ...]) -> GapDiagnosis008:
    matches, _ = search_existing_python_library(library, cases)
    if matches:
        return GapDiagnosis008("covered", "existing Python artifact already passes public tests", tuple(item.skill_id for item in matches))
    return GapDiagnosis008("gap_detected", "no existing Python artifact passes the public tests", ())


def _extract_code(response_text: str) -> str | None:
    parsed = parse_response(response_text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("code"), str):
        return None
    code = parsed["code"].strip()
    if code.startswith("```"):
        lines = code.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines).strip()
    return code or None


def python_proposal_prompt(public_cases: tuple[PythonCase008, ...], previous_code: str | None = None, feedback: str | None = None) -> str:
    tests = "\n".join(json.dumps({"query": case.query, "expected": case.expected}, ensure_ascii=False, sort_keys=True) for case in public_cases)
    sections = [
        "Write one safe Python procedure for the contract below.",
        PYTHON_API_DOCS_008,
        "Contract: define exactly transform(query: str) -> str. Parse the query, retain blank values, sort pairs by key then value, and return the canonical encoded query.",
        "The returned code is executed verbatim in a fresh Python interpreter. It must be self-contained: include `from urllib.parse import parse_qsl, urlencode` before transform and do not rely on names defined outside the returned code.",
        "Use only the documented urllib.parse functions. Do not use filesystem, network, subprocess, eval, exec, or other imports.",
        "Return exactly one JSON object: {\"code\":\"...\"}.",
        f"Public tests:\n{tests}",
    ]
    if previous_code:
        sections.append(f"Previous candidate:\n```python\n{previous_code}\n```")
    if feedback:
        sections.append(
            "Repair feedback (fix the candidate rather than repeating it):\n"
            f"{feedback}\n"
            "If the feedback reports an undefined name, add the required import explicitly. "
            "Return the complete corrected program, including imports, not a diff."
        )
    return "\n\n".join(sections)


@dataclass(frozen=True)
class RepairAttempt008:
    attempt: int
    code: str | None
    static: StaticCheck008
    public_gate: PythonGate008
    feedback: str


def learn_python_skill_008(client: LlamaCppClient, store: ExperimentStore, max_attempts: int = 3) -> tuple[PythonSkillArtifact008 | None, tuple[RepairAttempt008, ...]]:
    attempts: list[RepairAttempt008] = []
    previous_code: str | None = None
    feedback: str | None = None
    for attempt in range(1, max_attempts + 1):
        prompt = python_proposal_prompt(DISCOVERY_008, previous_code, feedback)
        completion = client.chat_json(prompt, max_tokens=512)
        code = _extract_code(completion.text)
        static = static_check_python(code) if code else StaticCheck008(False, "model did not return a code field")
        public_gate = run_python_gate(code, DISCOVERY_008, "discovery") if code else PythonGate008("discovery", 0, len(DISCOVERY_008), 0.0)
        if static.passed and public_gate.accuracy == 1.0:
            feedback = "public tests passed"
        else:
            failed_case = next((case for case in DISCOVERY_008 if not run_python_in_sandbox(code, case.query, case.expected).passed), None) if code else None
            if failed_case and code:
                failure = run_python_in_sandbox(code, failed_case.query, failed_case.expected)
                feedback = f"Candidate failed {failed_case.case_id}: query={failed_case.query!r}, expected={failed_case.expected!r}, observed={failure.value!r}, error={failure.error!r}."
            else:
                feedback = static.reason
        attempts.append(RepairAttempt008(attempt, code, static, public_gate, feedback))
        store.record_run(
            kind="air-008:learning:proposal",
            prompt=prompt,
            response=completion.text,
            elapsed_seconds=completion.elapsed_seconds,
            prompt_tokens=completion.prompt_tokens,
            generated_tokens=completion.generated_tokens,
            passed=static.passed and public_gate.accuracy == 1.0,
            metadata={"attempt": attempt, "static": asdict(static), "public_gate": asdict(public_gate), "feedback": feedback},
        )
        if static.passed and public_gate.accuracy == 1.0 and code:
            return PythonSkillArtifact008(
                "py-skill-1",
                1,
                "query: str",
                "str",
                code,
                "0008 model proposal repaired by public tests and gated on hidden validation",
                tuple(case.case_id for case in DISCOVERY_008),
            ), tuple(attempts)
        previous_code = code
    return None, tuple(attempts)


@dataclass(frozen=True)
class Result008:
    condition: str
    valid_correct: int
    valid_total: int
    valid_accuracy: float
    safe_rejections: int
    safe_rejection_total: int
    safe_rejection_rate: float
    total_prompt_tokens: int
    total_generated_tokens: int
    average_seconds: float


def _result008(condition: str, correct: int, total: int, safe: int, safe_total: int, prompt_tokens: int = 0, generated_tokens: int = 0, elapsed: list[float] | None = None) -> Result008:
    return Result008(condition, correct, total, correct / total if total else 0.0, safe, safe_total, safe / safe_total if safe_total else 0.0, prompt_tokens, generated_tokens, sum(elapsed) / len(elapsed) if elapsed else 0.0)


def run_skill_heldout(code: str | None, cases: tuple[PythonCase008, ...], store: ExperimentStore, condition: str, phase: str) -> Result008:
    correct = 0
    for case in cases:
        result = run_python_in_sandbox(code, case.query, case.expected) if code else SandboxResult008(False, None, "no active skill", 0.0)
        correct += int(result.passed)
        store.record_run(
            kind=f"air-008:{phase}:{condition}",
            prompt=f"sandbox input query={case.query!r}",
            response=json.dumps({"result": result.value, "error": result.error}, ensure_ascii=False),
            elapsed_seconds=result.elapsed_seconds,
            prompt_tokens=0,
            generated_tokens=0,
            passed=result.passed,
            metadata={"case_id": case.case_id, "expected": case.expected, "value": result.value, "error": result.error},
        )
    return _result008(condition, correct, len(cases), 0, 0)


def run_model_answers_008(cases: tuple[PythonCase008, ...], condition: str, context: str | None, client: LlamaCppClient, store: ExperimentStore) -> Result008:
    correct = 0
    prompt_tokens = generated_tokens = 0
    elapsed: list[float] = []
    prompt_suffix = f"\n\n{context}" if context else ""
    for case in cases:
        prompt = (
            "Return exactly one JSON object {\"result\":\"...\"} for the canonical query below. "
            "Do not explain.\n"
            + prompt_suffix
            + f"\n\nquery={json.dumps(case.query, ensure_ascii=False)}"
        )
        completion = client.chat_json(prompt, max_tokens=160)
        parsed = parse_response(completion.text)
        value = parsed.get("result") if isinstance(parsed, dict) else None
        passed = isinstance(value, str) and value == case.expected
        correct += int(passed)
        prompt_tokens += completion.prompt_tokens or 0
        generated_tokens += completion.generated_tokens or 0
        elapsed.append(completion.elapsed_seconds)
        store.record_run(
            kind=f"air-008:heldout:{condition}",
            prompt=prompt,
            response=completion.text,
            elapsed_seconds=completion.elapsed_seconds,
            prompt_tokens=completion.prompt_tokens,
            generated_tokens=completion.generated_tokens,
            passed=passed,
            metadata={"case_id": case.case_id, "expected": case.expected, "parsed": parsed},
        )
    return _result008(condition, correct, len(cases), 0, 0, prompt_tokens, generated_tokens, elapsed)


def _corrupted_code_008() -> str:
    return """from urllib.parse import parse_qsl, urlencode

def transform(query: str) -> str:
    pairs = parse_qsl(query, keep_blank_values=True)
    return urlencode(pairs, doseq=True)
"""


def run_exp008(*, client: LlamaCppClient, store: ExperimentStore, report_directory: str, heldout_limit: int | None = None) -> dict[str, object]:
    base_snapshot = {artifact.skill_id: artifact.to_dict() for artifact in BASE_PYTHON_LIBRARY_008}
    existing_matches, existing_evaluations = search_existing_python_library(BASE_PYTHON_LIBRARY_008, DISCOVERY_008)
    gap = diagnose_python_gap(BASE_PYTHON_LIBRARY_008, DISCOVERY_008)
    learned_skill, attempts = learn_python_skill_008(client, store)
    discovery_gate = run_python_gate(learned_skill.code, DISCOVERY_008, "discovery") if learned_skill else PythonGate008("discovery", 0, len(DISCOVERY_008), 0.0)
    validation_gate = run_python_gate(learned_skill.code, VALIDATION_008, "validation") if learned_skill else PythonGate008("validation", 0, len(VALIDATION_008), 0.0)
    edge_gate = run_python_gate(learned_skill.code, EDGE_008, "edge") if learned_skill else PythonGate008("edge", 0, len(EDGE_008), 0.0)
    corrupted_gate = run_python_gate(_corrupted_code_008(), VALIDATION_008, "corrupted")
    active = bool(
        learned_skill
        and gap.status == "gap_detected"
        and discovery_gate.accuracy == 1.0
        and validation_gate.accuracy == 1.0
        and edge_gate.accuracy == 1.0
        and corrupted_gate.accuracy < 0.9
    )
    extended_library = BASE_PYTHON_LIBRARY_008 + ((learned_skill,) if learned_skill else ())
    final_matches, final_evaluations = search_existing_python_library(extended_library, DISCOVERY_008)
    base_after = {artifact.skill_id: artifact.to_dict() for artifact in extended_library if artifact.skill_id in base_snapshot}
    for artifact in extended_library:
        store.upsert_skill(name=f"air-008-{artifact.skill_id}", body=json.dumps(artifact.to_dict(), sort_keys=True), state="active" if active and artifact.skill_id == "py-skill-1" else "candidate")
    heldout = HELD_OUT_008[:heldout_limit] if heldout_limit is not None else HELD_OUT_008
    results = [
        run_model_answers_008(heldout, "model_only", None, client, store),
        run_model_answers_008(heldout, "model_plus_docs", PYTHON_API_DOCS_008, client, store),
        run_skill_heldout(None, heldout, store, "before_gap", "heldout"),
        run_skill_heldout(learned_skill.code if active and learned_skill else None, heldout, store, "learned_python_skill", "heldout"),
    ]
    report = {
        "benchmark": "air-008-python-api-capability-gap",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "api": "urllib.parse.parse_qsl + urlencode",
            "discovery_cases": len(DISCOVERY_008),
            "validation_cases": len(VALIDATION_008),
            "edge_cases": len(EDGE_008),
            "heldout_cases": len(HELD_OUT_008),
            "heldout_is_new": True,
            "docs_are_limited": True,
            "external_dependencies": False,
            "network_available_to_candidate": False,
        },
        "gap_detection": {
            "diagnosis": asdict(gap),
            "existing_skill_ids": [artifact.skill_id for artifact in BASE_PYTHON_LIBRARY_008],
            "matching_skill_ids_before": [artifact.skill_id for artifact in existing_matches],
            "evaluations_before": existing_evaluations,
        },
        "learning": {
            "attempts": [
                {"attempt": item.attempt, "code": item.code, "static": asdict(item.static), "public_gate": asdict(item.public_gate), "feedback": item.feedback}
                for item in attempts
            ],
            "accepted_skill": learned_skill.to_dict() if learned_skill else None,
            "discovery_gate": asdict(discovery_gate),
            "validation_gate": asdict(validation_gate),
            "edge_gate": asdict(edge_gate),
            "corrupted_candidate": {"code": _corrupted_code_008(), "gate": asdict(corrupted_gate), "state": "rejected" if corrupted_gate.accuracy < 0.9 else "unsafe-active"},
        },
        "composition_after_learning": {
            "matching_skill_ids": [artifact.skill_id for artifact in final_matches],
            "evaluations": final_evaluations,
            "reused_new_skill": bool(learned_skill and any(artifact.skill_id == learned_skill.skill_id for artifact in final_matches)),
        },
        "sandbox": {
            "static_allowlist": "urllib.parse only; transform function only; dangerous names/calls rejected",
            "process_isolation": "python -I, sanitized environment, temporary cwd, 2 second timeout",
        },
        "immutability": {"base_skills_unchanged": base_snapshot == base_after},
        "baselines": {
            "model_only": {"source": "direct LLM output", "docs": False, "artifact_reuse": False},
            "model_plus_docs": {"source": "direct LLM output with limited API documentation", "docs": True, "artifact_reuse": False},
            "before_gap": {"source": "existing Python artifact library before learning", "docs": False, "artifact_reuse": True},
            "learned_python_skill": {"source": "accepted Python code artifact executed in sandbox", "docs": False, "artifact_reuse": True},
        },
        "heldout_results": [asdict(result) for result in results],
        "active": active,
    }
    report_path = Path(report_directory)
    report_path.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(UTC).strftime("air-008-%Y%m%dT%H%M%SZ.json")
    path = report_path / filename
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_file"] = str(path)
    return report
