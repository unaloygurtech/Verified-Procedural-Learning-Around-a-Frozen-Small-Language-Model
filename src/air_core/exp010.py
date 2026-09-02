"""Experiment 0010: learning a novel, synthetic Python API.

The package ``zorvik_010`` is created in this repository for this experiment;
its namespace and operation names are not an existing public API.  The same
frozen generic learner template from 0009 is used for four independent
operations.  A zero-knowledge baseline runs before any documentation is shown,
then AIR learns executable wrappers from documentation and public tests,
validates them in the restricted sandbox, and searches the immutable artifacts
for a held-out composition.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import zorvik_010

from .exp008 import HELD_OUT_008, run_python_gate
from .exp009 import (
    BASE_PYTHON_LIBRARY_009,
    FamilyCase009,
    LEARNING_PROMPT_HASH_009,
    LEARNING_PROMPT_VERSION_009,
    PythonFamily009,
    PythonGate009,
    PythonSkillArtifact009,
    RepairAttempt009,
    SandboxResult009,
    StaticCheck009,
    generic_learning_prompt,
    learn_family_009,
    run_python_gate_009,
    run_python_in_sandbox_009,
    run_skill_heldout_009,
    static_check_python_009,
)
from .model_client import LlamaCppClient
from .neralis import parse_response
from .store import ExperimentStore


MODEL_IDENTITY_010 = "SmolLM3-3B-GGUF-Q4 via llama.cpp"
SYNTHETIC_API_NAME_010 = "zorvik_010"
SYNTHETIC_IMPORT_ROOT_010 = str(Path(__file__).resolve().parents[1])


def _case(prefix: str, values: tuple[str, ...], operation: Callable[[str], str], split: str) -> tuple[FamilyCase009, ...]:
    return tuple(FamilyCase009(f"{prefix}-{index:02d}", value, operation(value), split) for index, value in enumerate(values, 1))


def _family(operation: str, title: str, docs: str, contract: str, reference: Callable[[str], str], public: tuple[str, ...], validation: tuple[str, ...], edge: tuple[str, ...], heldout: tuple[str, ...]) -> PythonFamily009:
    return PythonFamily009(
        family_id=f"zorvik-{operation}",
        title=title,
        api_docs=docs,
        contract=contract,
        allowed_imports=frozenset({SYNTHETIC_API_NAME_010}),
        allowed_import_members=frozenset({operation}),
        allowed_call_names=frozenset({operation}),
        allowed_attrs=frozenset({operation}),
        discovery=_case(f"{operation}-discover", public, reference, "discovery"),
        validation=_case(f"{operation}-validation", validation, reference, "validation"),
        edge=_case(f"{operation}-edge", edge, reference, "edge"),
        heldout=_case(f"{operation}-heldout", heldout, reference, "heldout"),
        sandbox_import_root=SYNTHETIC_IMPORT_ROOT_010,
    )


ZORVIK_FAMILIES_010: tuple[PythonFamily009, ...] = (
    _family(
        "kel",
        "Zorvik kel transformation",
        "Synthetic API only: `from zorvik_010 import kel`. Signature: `kel(value: str) -> str`. Split the input at every `~`, reverse the order of the resulting segments, reverse the characters inside each segment, then join with `~`. Empty segments are retained.",
        "Define transform(value: str) -> str and return zorvik_010.kel(value).",
        zorvik_010.kel,
        ("ab~cd", "one~two~3", "~lead", "a~~b"),
        ("red~blue", "123~xy~!", "left~"),
        ("", "~", "x~~"),
        ("north~east", "a1~b2~c3", "~middle~", "single", "aa~~bb", "zyx~wvu", "9~88", "foo~bar"),
    ),
    _family(
        "nam",
        "Zorvik nam transformation",
        "Synthetic API only: `from zorvik_010 import nam`. Signature: `nam(value: str) -> str`. Visit characters from left to right, swap each adjacent pair, and leave the final character unchanged when the length is odd.",
        "Define transform(value: str) -> str and return zorvik_010.nam(value).",
        zorvik_010.nam,
        ("abcd", "abcde", "12-345", "A"),
        ("abcdefg", "xy", "123456789"),
        ("", "a", "ab"),
        ("hello", "AIR", "987654", "oddlen", "pair-pair", "Türkçe", "x", "12345"),
    ),
    _family(
        "tesh",
        "Zorvik tesh transformation",
        "Synthetic API only: `from zorvik_010 import tesh`. Signature: `tesh(value: str) -> str`. Return the string rotated left by exactly two character positions; slicing naturally leaves strings shorter than two characters unchanged.",
        "Define transform(value: str) -> str and return zorvik_010.tesh(value).",
        zorvik_010.tesh,
        ("abcd", "abcde", "12-345", "A"),
        ("abcdefg", "rotate-me", "xy"),
        ("", "a", "ab"),
        ("hello", "AIR", "987654", "abcdefghi", "pair-pair", "Türkçe", "x", "12345"),
    ),
    _family(
        "vum",
        "Zorvik vum transformation",
        "Synthetic API only: `from zorvik_010 import vum`. Signature: `vum(value: str) -> str`. Return all characters at even zero-based indexes followed by all characters at odd zero-based indexes.",
        "Define transform(value: str) -> str and return zorvik_010.vum(value).",
        zorvik_010.vum,
        ("abcd", "abcde", "12-345", "A"),
        ("abcdefg", "rotate-me", "xy"),
        ("", "a", "ab"),
        ("hello", "AIR", "987654", "abcdefghi", "pair-pair", "Türkçe", "x", "12345"),
    ),
)


BASE_PYTHON_LIBRARY_010 = (
    PythonSkillArtifact009("py-skill-echo", "generic", 1, "value: str", "str", "def transform(value: str) -> str:\n    return value", "pre-existing unrelated artifact"),
    PythonSkillArtifact009("py-skill-0009-json", "json-canonical", 1, "value: str", "str", "import json\n\ndef transform(value: str) -> str:\n    return json.dumps(json.loads(value), sort_keys=True, separators=(',', ':'))", "previous experiment artifact; cross-family control"),
)


def _api_source_hash_010() -> str:
    source = Path(zorvik_010.__file__).read_bytes()
    return hashlib.sha256(source).hexdigest()


def _docs_hash_010(family: PythonFamily009) -> str:
    return hashlib.sha256(family.api_docs.encode("utf-8")).hexdigest()


def _provenance_010(artifact: PythonSkillArtifact009, family: PythonFamily009, attempts: tuple[RepairAttempt009, ...]) -> dict[str, Any]:
    return {
        "skill_id": artifact.skill_id,
        "version": artifact.version,
        "source_api": SYNTHETIC_API_NAME_010,
        "operation": family.family_id.removeprefix("zorvik-"),
        "documentation_sha256": _docs_hash_010(family),
        "public_case_ids": [case.case_id for case in family.discovery],
        "validation_case_ids": [case.case_id for case in family.validation],
        "edge_case_ids": [case.case_id for case in family.edge],
        "exact_python_source": artifact.code,
        "learner_prompt_version": LEARNING_PROMPT_VERSION_009,
        "learner_prompt_sha256": LEARNING_PROMPT_HASH_009,
        "model_runtime": MODEL_IDENTITY_010,
        "proposal_count": len(attempts),
        "repair_count": max(0, len(attempts) - 1),
        "activation_reason": "public, hidden-validation, and edge gates passed",
    }


def _zero_knowledge_prompt(family: PythonFamily009, value: str, include_names: bool = False) -> str:
    names = f" Available API name: {SYNTHETIC_API_NAME_010}.{family.family_id.removeprefix('zorvik-')}(str) but its semantics are unknown." if include_names else ""
    return f"Return exactly one JSON object {{\"result\":\"...\"}}. A new synthetic API exists.{names} No documentation or examples are available. Guess the result for input {json.dumps(value, ensure_ascii=False)}. Do not explain."


def _direct_prompt(family: PythonFamily009, case: FamilyCase009, mode: str) -> str:
    if mode == "model_only":
        return _zero_knowledge_prompt(family, case.input_text, False)
    if mode == "api_names_only":
        return _zero_knowledge_prompt(family, case.input_text, True)
    docs = f"\nDocumentation:\n{family.api_docs}"
    raw = ""
    if mode == "docs_plus_raw":
        raw = "\nPublic examples:\n" + "\n".join(json.dumps({"input": item.input_text, "expected": item.expected}, ensure_ascii=False) for item in family.discovery)
    return f"Return exactly one JSON object {{\"result\":\"...\"}}. Contract: {family.contract}{docs}{raw}\nInput: {json.dumps(case.input_text, ensure_ascii=False)}\nDo not explain."


def _run_direct_baseline_010(family: PythonFamily009, cases: tuple[FamilyCase009, ...], mode: str, client: LlamaCppClient, store: ExperimentStore, seed: int) -> dict[str, Any]:
    correct = prompt_tokens = generated_tokens = 0
    elapsed: list[float] = []
    for case in cases:
        prompt = _direct_prompt(family, case, mode)
        completion = client.chat_json(prompt, max_tokens=160, seed=seed)
        parsed = parse_response(completion.text)
        value = parsed.get("result") if isinstance(parsed, dict) else None
        passed = isinstance(value, str) and value == case.expected
        correct += int(passed)
        prompt_tokens += completion.prompt_tokens or 0
        generated_tokens += completion.generated_tokens or 0
        elapsed.append(completion.elapsed_seconds)
        store.record_run(kind=f"air-010:{family.family_id}:heldout:{mode}", prompt=prompt, response=completion.text, elapsed_seconds=completion.elapsed_seconds, prompt_tokens=completion.prompt_tokens, generated_tokens=completion.generated_tokens, passed=passed, metadata={"case_id": case.case_id, "expected": case.expected, "parsed": parsed, "mode": mode, "seed": seed})
    return {"condition": mode, "valid_correct": correct, "valid_total": len(cases), "valid_accuracy": correct / len(cases) if cases else 0.0, "total_prompt_tokens": prompt_tokens, "total_generated_tokens": generated_tokens, "average_seconds": sum(elapsed) / len(elapsed) if elapsed else 0.0}


def _compose_case_010(first: PythonSkillArtifact009, first_family: PythonFamily009, second: PythonSkillArtifact009, second_family: PythonFamily009, case: FamilyCase009) -> SandboxResult009:
    intermediate = run_python_in_sandbox_009(first.code, first_family, case.input_text)
    if not intermediate.passed and intermediate.value is None:
        return intermediate
    return run_python_in_sandbox_009(second.code, second_family, intermediate.value or "", case.expected)


def search_composition_010(artifacts: tuple[tuple[PythonSkillArtifact009, PythonFamily009], ...], cases: tuple[FamilyCase009, ...]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for first_index, (first, first_family) in enumerate(artifacts):
        for second_index, (second, second_family) in enumerate(artifacts):
            if first_index == second_index:
                continue
            if all(_compose_case_010(first, first_family, second, second_family, case).passed for case in cases):
                candidates.append({"first_skill_id": first.skill_id, "second_skill_id": second.skill_id, "first_family": first_family.family_id, "second_family": second_family.family_id})
    return candidates


def _compose_gate_010(plan: dict[str, str] | None, artifacts: tuple[tuple[PythonSkillArtifact009, PythonFamily009], ...], cases: tuple[FamilyCase009, ...], condition: str) -> dict[str, Any]:
    if not plan:
        return {"condition": condition, "correct": 0, "total": len(cases), "accuracy": 0.0}
    lookup = {artifact.skill_id: (artifact, family) for artifact, family in artifacts}
    first, first_family = lookup[plan["first_skill_id"]]
    second, second_family = lookup[plan["second_skill_id"]]
    correct = sum(_compose_case_010(first, first_family, second, second_family, case).passed for case in cases)
    return {"condition": condition, "correct": correct, "total": len(cases), "accuracy": correct / len(cases) if cases else 0.0}


def _composition_cases_010() -> tuple[tuple[FamilyCase009, ...], tuple[FamilyCase009, ...], tuple[FamilyCase009, ...]]:
    def reference(value: str) -> str:
        return zorvik_010.tesh(zorvik_010.kel(value))

    selection = _case("composition-select", ("ab~cd", "one~two", "x~~y"), reference, "selection")
    validation = _case("composition-validation", ("red~blue", "123~xy~!", "left~"), reference, "validation")
    heldout = _case("composition-heldout", ("north~east", "a1~b2~c3", "~middle~", "single", "aa~~bb", "zyx~wvu", "9~88", "foo~bar"), reference, "heldout")
    return selection, validation, heldout


def _unsafe_code_010() -> str:
    return "import os\ndef transform(value: str) -> str:\n    return os.getcwd()\n"


def _semantic_corruption_010() -> str:
    return "def transform(value: str) -> str:\n    return value\n"


def run_exp010(*, client: LlamaCppClient, store: ExperimentStore, report_directory: str, heldout_limit: int | None = None) -> dict[str, Any]:
    api_hash_before = _api_source_hash_010()
    zero_knowledge: list[dict[str, Any]] = []
    for family in ZORVIK_FAMILIES_010:
        public = family.discovery
        names = _run_direct_baseline_010(family, public, "model_only", client, store, 100)
        api_names = _run_direct_baseline_010(family, public, "api_names_only", client, store, 101)
        zero_knowledge.append({"family_id": family.family_id, "model_only": names, "api_names_only": api_names})
    family_results: list[dict[str, Any]] = []
    active_pairs: list[tuple[PythonSkillArtifact009, PythonFamily009]] = []
    for index, family in enumerate(ZORVIK_FAMILIES_010, 1):
        base_snapshot = {artifact.skill_id: artifact.to_dict() for artifact in BASE_PYTHON_LIBRARY_010}
        matches_before = []
        for artifact in BASE_PYTHON_LIBRARY_010:
            if run_python_gate_009(artifact.code, family, family.discovery, "existing").accuracy == 1.0:
                matches_before.append(artifact.skill_id)
        diagnosis = {"status": "covered" if matches_before else "gap_detected", "matching_skill_ids": matches_before, "evaluations": len(BASE_PYTHON_LIBRARY_010)}
        skill, attempts = learn_family_009(client, store, family, f"py-skill-010-{family.family_id}", 200 + index)
        discovery = run_python_gate_009(skill.code, family, family.discovery, "discovery") if skill else PythonGate009("discovery", 0, len(family.discovery), 0.0)
        validation = run_python_gate_009(skill.code, family, family.validation, "validation") if skill else PythonGate009("validation", 0, len(family.validation), 0.0)
        edge = run_python_gate_009(skill.code, family, family.edge, "edge") if skill else PythonGate009("edge", 0, len(family.edge), 0.0)
        active = bool(skill and diagnosis["status"] == "gap_detected" and discovery.accuracy == 1.0 and validation.accuracy == 1.0 and edge.accuracy == 1.0)
        if active and skill:
            active_pairs.append((skill, family))
        heldout = family.heldout[:heldout_limit] if heldout_limit is not None else family.heldout
        docs_result = _run_direct_baseline_010(family, heldout, "docs_only", client, store, 300 + index)
        raw_result = _run_direct_baseline_010(family, heldout, "docs_plus_raw", client, store, 400 + index)
        before_result = run_skill_heldout_009(None, family, heldout, store, "before_gap")
        learned_result = run_skill_heldout_009(skill.code if active and skill else None, family, heldout, store, "learned_synthetic_skill")
        family_results.append({
            "family_id": family.family_id,
            "title": family.title,
            "gap_detection": diagnosis,
            "learning": {"attempt_count": len(attempts), "repair_count": max(0, len(attempts) - 1), "syntax_shape_rejections": sum((not item.static.passed) and not item.static.reason.startswith(("import not allowed:", "from-import not allowed:", "forbidden name:", "indirect calls are not allowed")) for item in attempts), "safety_rejections": sum(item.static.reason.startswith(("import not allowed:", "from-import not allowed:", "forbidden name:", "indirect calls are not allowed")) for item in attempts), "public_failure_count": sum(item.public_gate.accuracy < 1.0 for item in attempts), "attempts": [{"attempt": item.attempt, "code": item.code, "static": asdict(item.static), "public_gate": asdict(item.public_gate), "feedback": item.feedback} for item in attempts], "accepted_skill": skill.to_dict() if skill else None, "provenance": _provenance_010(skill, family, attempts) if skill else None, "discovery_gate": asdict(discovery), "validation_gate": asdict(validation), "edge_gate": asdict(edge)},
            "activation": {"active": active, "matching_skill_ids_before": matches_before, "wrong_activation_before": bool(matches_before), "artifact_reuse": bool(active and skill), "previous_skill_help": bool(matches_before)},
            "heldout_results": [docs_result, raw_result, asdict(before_result), asdict(learned_result)],
            "base_skill_immutability": base_snapshot == {artifact.skill_id: artifact.to_dict() for artifact in BASE_PYTHON_LIBRARY_010},
        })
    selection, composition_validation, composition_heldout = _composition_cases_010()
    plans = search_composition_010(tuple(active_pairs), selection) if len(active_pairs) >= 2 else []
    plan = plans[0] if len(plans) == 1 else None
    composition = {"selection_candidates": plans, "unique_plan": plan, "validation_gate": _compose_gate_010(plan, tuple(active_pairs), composition_validation, "composition_validation"), "heldout_gate": _compose_gate_010(plan, tuple(active_pairs), composition_heldout, "composition_heldout"), "training_examples_in_primitive_prompts": False, "mapping_hard_coded": False, "status": "active" if plan and _compose_gate_010(plan, tuple(active_pairs), composition_validation, "composition_validation")["accuracy"] == 1.0 else "no_valid_composition"}
    semantic_corrupted = run_python_gate_009(_semantic_corruption_010(), ZORVIK_FAMILIES_010[0], ZORVIK_FAMILIES_010[0].validation, "semantic_corrupted")
    unsafe_static = static_check_python_009(_unsafe_code_010(), ZORVIK_FAMILIES_010[0])
    api_hash_after = _api_source_hash_010()
    total_runs = len(family_results)
    active_count = sum(item["activation"]["active"] for item in family_results)
    zero_knowledge_total = sum(item["model_only"]["valid_total"] for item in zero_knowledge)
    zero_knowledge_correct = sum(item["model_only"]["valid_correct"] for item in zero_knowledge)
    api_names_correct = sum(item["api_names_only"]["valid_correct"] for item in zero_knowledge)
    api_names_total = sum(item["api_names_only"]["valid_total"] for item in zero_knowledge)
    report: dict[str, Any] = {
        "benchmark": "air-010-novel-synthetic-api-learning",
        "created_at": datetime.now(UTC).isoformat(),
        "model_runtime": MODEL_IDENTITY_010,
        "synthetic_api": {"name": SYNTHETIC_API_NAME_010, "source_file": str(Path(zorvik_010.__file__)), "source_sha256_before": api_hash_before, "source_sha256_after": api_hash_after, "source_unchanged": api_hash_before == api_hash_after, "operations": [family.family_id.removeprefix("zorvik-") for family in ZORVIK_FAMILIES_010]},
        "protocol": {"family_count": len(ZORVIK_FAMILIES_010), "discovery_cases_per_family": 4, "validation_cases_per_family": 3, "edge_cases_per_family": 3, "heldout_cases_per_family": 8, "max_attempts": 3, "learner_prompt_version": LEARNING_PROMPT_VERSION_009, "learner_prompt_sha256": LEARNING_PROMPT_HASH_009, "generic_template_frozen": True, "family_specific_prompt_patches": False, "external_dependencies": False, "network_available_to_candidate": False},
        "zero_knowledge_baseline": zero_knowledge,
        "zero_knowledge_assessment": {"model_only_correct": zero_knowledge_correct, "model_only_total": zero_knowledge_total, "model_only_accuracy": zero_knowledge_correct / zero_knowledge_total if zero_knowledge_total else 0.0, "api_names_only_correct": api_names_correct, "api_names_only_total": api_names_total, "api_names_only_accuracy": api_names_correct / api_names_total if api_names_total else 0.0, "meaningful_accuracy_threshold": 0.5, "contamination_flag": api_names_correct / api_names_total >= 0.5 if api_names_total else False, "status": "possible_contamination" if api_names_correct / api_names_total >= 0.5 else "no_meaningful_zero_knowledge_signal"},
        "family_results": family_results,
        "composition": composition,
        "safety_controls": {"unsafe_candidate": {"static": asdict(unsafe_static), "activated": False}, "semantic_wrong_candidate": {"gate": asdict(semantic_corrupted), "state": "rejected" if semantic_corrupted.accuracy < 0.9 else "unsafe-active"}, "candidate_import_allowlist": [SYNTHETIC_API_NAME_010], "process_isolation": "python -I, sanitized env, temporary cwd, 2 second timeout"},
        "summary": {"family_runs": total_runs, "gap_detection_rate": sum(item["gap_detection"]["status"] == "gap_detected" for item in family_results) / total_runs if total_runs else 0.0, "skills_activated": active_count, "activation_rate": active_count / total_runs if total_runs else 0.0, "wrong_activation_count": sum(item["activation"]["wrong_activation_before"] for item in family_results), "previous_skill_help_count": sum(item["activation"]["previous_skill_help"] for item in family_results), "hidden_validation_failures": sum(item["learning"]["accepted_skill"] is not None and item["learning"]["validation_gate"]["accuracy"] < 1.0 for item in family_results), "base_skill_immutability_all": all(item["base_skill_immutability"] for item in family_results), "synthetic_package_immutability": api_hash_before == api_hash_after},
    }
    report_path = Path(report_directory)
    report_path.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(UTC).strftime("air-010-%Y%m%dT%H%M%SZ.json")
    path = report_path / filename
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_file"] = str(path)
    return report
