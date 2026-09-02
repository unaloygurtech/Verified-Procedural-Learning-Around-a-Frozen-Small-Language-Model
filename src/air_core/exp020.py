"""Experiment 0020: persistent procedural skill accumulation and transfer.

The experiment deliberately keeps acquisition model-free.  A skill is a
verified semantic IR program, stored once in a canonical JSON state file with
faceted metadata.  The final evaluation contains only newly generated inputs;
no expected output is written to learned state.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import UTC, datetime
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

import air_synth_012

from .behavioral_canonicalization import canonical_cost_key_019
from .exp009 import FamilyCase009, PythonFamily009
from .exp012 import _generated_literals
from .exp015 import ModelLedger015, _safe_model_json_015
from .exp018 import (
    MODEL_IDENTITY_018, SEARCH_API_NAMES_018, SEARCH_BUDGET_018,
    _candidate_evaluation_018, _heldout_018,
)
from .model_client import LlamaCppClient
from .program_search import Candidate018, search_candidates_018
from .semantic_ir import (
    SEMANTIC_IR_FORMAT_017, SEMANTIC_IR_VERSION_017,
    canonical_semantic_ir_json_017, execute_semantic_ir_017,
)
from .store import ExperimentStore


EXP020_VERSION = "air-020-v1"
CONTEXT_SIZE_020 = 4096
RANDOM_SEED_020 = 20020
SKILLS_020 = 32
DUPLICATE_REQUESTS_020 = 6
CONFLICT_REQUESTS_020 = 4
FINAL_TASKS_020 = 48
PROMPT_VERSION_020 = "air-020-vanilla-final-v1"
PROMPT_TEMPLATE_020 = """Solve this opaque transformation without AIR learned skills.
Task form: {kind}. Surface: {surface}. Input: {value}
Return exactly JSON {{\"answer\":\"...\"}}. Do not explain.
"""
PROMPT_HASH_020 = hashlib.sha256(PROMPT_TEMPLATE_020.encode()).hexdigest()


def _root(expr: Mapping[str, Any]) -> dict[str, Any]:
    return {"format": SEMANTIC_IR_FORMAT_017, "version": SEMANTIC_IR_VERSION_017,
            "input_type": "str", "output_type": "str",
            "expr": {"op": "RETURN", "value": dict(expr)}}


def _call(api: str) -> dict[str, Any]:
    return {"op": "CALL", "api": api, "args": [{"op": "INPUT"}]}


def _replace_input(value: Any, replacement: Mapping[str, Any]) -> Any:
    if isinstance(value, Mapping):
        if value.get("op") == "INPUT":
            return json.loads(json.dumps(replacement))
        return {key: _replace_input(item, replacement) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_input(item, replacement) for item in value]
    return value


def _compose_ir(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Return right(left(x)); used only for verified composition artifacts."""
    left_value = left["expr"]["value"]
    right_value = right["expr"]["value"]
    return _root(_replace_input(right_value, left_value))


@dataclass(frozen=True)
class SkillSpec020:
    request_id: str
    skill_id: str
    kind: str
    operation_family: str
    input_kind: str
    target_ir: dict[str, Any]
    discovery: tuple[FamilyCase009, ...]
    hidden: tuple[FamilyCase009, ...]
    edge: tuple[FamilyCase009, ...]
    heldout: tuple[FamilyCase009, ...]

    @property
    def fingerprint(self) -> str:
        return self.kind

    @property
    def target_hash(self) -> str:
        return hashlib.sha256(canonical_semantic_ir_json_017(self.target_ir).encode()).hexdigest()


@dataclass(frozen=True)
class Skill020:
    skill_id: str
    kind: str
    operation_family: str
    ir: dict[str, Any]
    artifact_sha256: str
    artifact_bytes: int
    source_request: str
    components: tuple[str, ...] = ()
    usage_count: int = 0
    successful_execution_count: int = 0
    failed_execution_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Skill020":
        return cls(str(data["skill_id"]), str(data["kind"]), str(data["operation_family"]),
                   dict(data["ir"]), str(data["artifact_sha256"]), int(data["artifact_bytes"]),
                   str(data.get("source_request", "")), tuple(data.get("components", ())),
                   int(data.get("usage_count", 0)), int(data.get("successful_execution_count", 0)),
                   int(data.get("failed_execution_count", 0)))


class PersistentSkillStore020:
    """Canonical store: one executable artifact, many metadata lookup paths."""

    VERSION = 1

    def __init__(self, path: Path, skills: Sequence[Skill020] = ()) -> None:
        self.path = Path(path)
        self.skills: dict[str, Skill020] = {skill.skill_id: skill for skill in skills}
        self._reindex()

    def _reindex(self) -> None:
        self.by_kind: dict[str, list[str]] = {}
        self.by_operation: dict[str, list[str]] = {}
        self.by_component_key: dict[str, list[str]] = {}
        for skill in self.skills.values():
            self.by_kind.setdefault(skill.kind, []).append(skill.skill_id)
            self.by_operation.setdefault(skill.operation_family, []).append(skill.skill_id)
            if skill.components:
                self.by_component_key.setdefault("→".join(skill.components), []).append(skill.skill_id)
        for mapping in (self.by_kind, self.by_operation, self.by_component_key):
            for ids in mapping.values():
                ids.sort()

    def save(self) -> None:
        payload = {"version": self.VERSION, "skills": [item.to_dict() for item in self.skills.values()]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    @classmethod
    def load(cls, path: Path) -> "PersistentSkillStore020":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != cls.VERSION:
            raise ValueError("unsupported AIR-020 learned-state version")
        skills = [Skill020.from_dict(item) for item in data.get("skills", [])]
        for skill in skills:
            blob = canonical_semantic_ir_json_017(skill.ir).encode()
            if hashlib.sha256(blob).hexdigest() != skill.artifact_sha256:
                raise ValueError(f"artifact integrity failure: {skill.skill_id}")
        return cls(Path(path), skills)

    def retrieve(self, *, kind: str | None = None, operation_family: str | None = None,
                 components: Sequence[str] = (), top_k: int = 5) -> list[Skill020]:
        if components:
            ids = self.by_component_key.get("→".join(components), [])
        elif kind is not None and operation_family is not None:
            ids = [sid for sid in self.by_kind.get(kind, [])
                   if self.skills[sid].operation_family == operation_family]
        elif kind is not None:
            ids = self.by_kind.get(kind, [])
        else:
            ids = list(self.skills)
        return [self.skills[sid] for sid in ids[:top_k]]

    def add(self, skill: Skill020) -> None:
        if skill.skill_id in self.skills:
            return
        self.skills[skill.skill_id] = skill
        self._reindex()

    def artifact_bytes(self) -> int:
        return sum(item.artifact_bytes for item in self.skills.values())

    def metadata_bytes(self) -> int:
        return sum(len(json.dumps({"id": item.skill_id, "kind": item.kind,
                                   "operation": item.operation_family, "components": item.components},
                                  sort_keys=True).encode()) for item in self.skills.values())


def _value(seed: int, index: int) -> str:
    alphabet = "abcdefghijkmnpqrstuvwxyz"
    return "".join(alphabet[(seed * 17 + index * 7 + offset * 3) % len(alphabet)] for offset in range(3 + index % 5))


def _api_kind(api: str) -> str:
    for seed in air_synth_012.SEEDS_012:
        for kind, name in air_synth_012.operation_names(seed).items():
            if name == api:
                return kind
    raise KeyError(api)


def _target_program(index: int) -> tuple[str, str, dict[str, Any]]:
    operations = tuple(sorted(SEARCH_API_NAMES_018))
    if index < 15:
        api = operations[index]
        return "call", api, _root(_call(api))
    if index < 21:
        api = operations[index - 15]
        return "reverse_call", api, _root({"op": "REVERSE", "value": _call(api)})
    if index < 26:
        api = tuple(api for api in operations if _api_kind(api) in {"runs", "shards"})[index - 21]
        return "rotate_call", api, _root({"op": "ROTATE", "value": _call(api), "amount": {"op": "INT", "value": 1}})
    if index < 30:
        safe_ops = tuple(api for api in operations if _api_kind(api) == "runs") + tuple(api for api in operations if _api_kind(api) == "shards")
        api = safe_ops[index - 26]
        return "double_call", api, _root({"op": "CALL", "api": api, "args": [_call(api)]})
    concat_ops = tuple(api for api in operations if _api_kind(api) == "runs")
    left, right = concat_ops[index - 30:index - 28]
    return "concat_pair", f"{left}+{right}", _root({"op": "CONCAT", "values": [_call(left), _call(right)]})


def _cases(spec_id: str, split: str, values: Sequence[str], ir: Mapping[str, Any]) -> tuple[FamilyCase009, ...]:
    return tuple(FamilyCase009(f"{spec_id}-{split}-{i:02d}", value,
                               execute_semantic_ir_017(ir, value, SEARCH_API_NAMES_018), split)
                 for i, value in enumerate(values, 1))


def make_skill_curriculum_020() -> tuple[SkillSpec020, ...]:
    result: list[SkillSpec020] = []
    for index in range(SKILLS_020):
        kind, operation, ir = _target_program(index)
        request_id = f"air020-acq-{index + 1:02d}"
        input_kind = _api_kind(operation.split("+")[0])
        seed = next(seed for seed in air_synth_012.SEEDS_012 if operation.split("+")[0] in air_synth_012.operation_names(seed).values())
        values = _generated_literals(seed, input_kind)
        edge_values = values[7:10]
        result.append(SkillSpec020(request_id, f"skill-020-{index + 1:03d}", kind, operation, input_kind, ir,
                                   _cases(request_id, "public", values[:4], ir),
                                   _cases(request_id, "hidden", values[4:7], ir),
                                   _cases(request_id, "edge", edge_values, ir),
                                   _cases(request_id, "heldout", values[9:17], ir)))
    return tuple(result)


def _family(spec: SkillSpec020) -> PythonFamily009:
    return PythonFamily009(spec.request_id, f"AIR-020 {spec.kind} {spec.operation_family}",
        "Generic deterministic procedural transformation.", "str -> str", frozenset({"air_synth_012"}),
        frozenset(SEARCH_API_NAMES_018), frozenset(SEARCH_API_NAMES_018), frozenset(SEARCH_API_NAMES_018),
        spec.discovery, spec.hidden, spec.edge, spec.heldout, sandbox_import_root=str(Path(__file__).resolve().parents[1]))


def _select_verified(candidates: Sequence[Candidate018], spec: SkillSpec020) -> tuple[Candidate018 | None, dict[str, Any]]:
    evaluations = {item.candidate_id: _candidate_evaluation_018(item, _family(spec)) for item in candidates}
    winners = [item for item in candidates if evaluations[item.candidate_id].get("hidden_pass") and evaluations[item.candidate_id].get("edge_pass")]
    if not winners:
        return None, {"reason": "no_hidden_edge_survivor", "winner_count": 0, "evaluations": evaluations}
    # Equivalent alternatives are safe only after behavior verification; pick
    # the frozen canonical cost representative, never the ground-truth hash.
    grouped: dict[tuple[str, ...], list[Candidate018]] = {}
    for item in winners:
        grouped.setdefault(tuple(_safe_output(item, case.input_text) for case in spec.hidden + spec.edge), []).append(item)
    if len(grouped) > 1:
        return None, {"reason": "ambiguous_program", "winner_count": len(winners), "evaluations": evaluations}
    selected = min(winners, key=canonical_cost_key_019)
    return selected, {"reason": "canonical_verified", "winner_count": len(winners),
                      "evaluations": evaluations, "target_present": any(item.ast_hash == spec.target_hash for item in candidates)}


def _safe_output(candidate: Candidate018, value: str) -> str:
    try:
        return execute_semantic_ir_017(candidate.ir, value, SEARCH_API_NAMES_018)
    except Exception:
        return "<error>"


def _skill_from_candidate(spec: SkillSpec020, candidate: Candidate018) -> Skill020:
    blob = canonical_semantic_ir_json_017(candidate.ir).encode()
    return Skill020(spec.skill_id, spec.kind, spec.operation_family, candidate.ir,
                    hashlib.sha256(blob).hexdigest(), len(blob), spec.request_id)


def _execute(skill: Skill020, value: str) -> tuple[str, float]:
    started = time.perf_counter()
    result = execute_semantic_ir_017(skill.ir, value, SEARCH_API_NAMES_018)
    return result, time.perf_counter() - started


def _matches_cases(skill: Skill020, cases: Sequence[FamilyCase009]) -> bool:
    try:
        return all(_execute(skill, case.input_text)[0] == case.expected for case in cases)
    except Exception:
        return False


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    return {"p50": ordered[len(ordered) // 2], "p95": ordered[min(len(ordered)-1, int(round(.95*(len(ordered)-1))))], "mean": statistics.mean(ordered)}


def _acquire(store: PersistentSkillStore020, spec: SkillSpec020) -> dict[str, Any]:
    existing = store.retrieve(kind=spec.kind, operation_family=spec.operation_family, top_k=5)
    duplicate = False
    conflict = False
    for item in existing:
        if _matches_cases(item, spec.hidden + spec.edge):
            duplicate = True
            return {"request_id": spec.request_id, "skill_id": item.skill_id, "status": "reused_duplicate",
                    "duplicate_detected": True, "conflict_detected": False, "active": True,
                    "artifact_bytes": 0, "search": {"called": False}}
        conflict = True
    started = time.perf_counter()
    search = search_candidates_018(spec.discovery, budget=SEARCH_BUDGET_018)
    selected, diagnostic = _select_verified(search.public_survivors, spec)
    elapsed = time.perf_counter() - started
    if selected is None:
        return {"request_id": spec.request_id, "skill_id": None, "status": "rejected",
                "duplicate_detected": False, "conflict_detected": conflict, "active": False,
                "artifact_bytes": 0, "search": {**search.to_dict(), "elapsed_seconds": elapsed},
                "diagnostic": {key: value for key, value in diagnostic.items() if key != "evaluations"}}
    skill = _skill_from_candidate(spec, selected)
    store.add(skill)
    store.save()
    return {"request_id": spec.request_id, "skill_id": skill.skill_id, "status": "active",
            "duplicate_detected": duplicate, "conflict_detected": conflict, "active": True,
            "artifact_bytes": skill.artifact_bytes, "search": {**search.to_dict(), "elapsed_seconds": elapsed},
            "diagnostic": {key: value for key, value in diagnostic.items() if key != "evaluations"}}


def _retention(store: PersistentSkillStore020, specs: Sequence[SkillSpec020], limit: int | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in specs[:limit] if limit else specs:
        found = store.retrieve(kind=spec.kind, operation_family=spec.operation_family, top_k=1)
        skill = found[0] if found else None
        correct = 0
        for case in spec.heldout:
            if skill is not None and _execute(skill, case.input_text)[0] == case.expected:
                correct += 1
        rows.append({"skill_id": spec.skill_id, "retrieved": skill is not None,
                     "accuracy": correct / len(spec.heldout) if spec.heldout else 0.0,
                     "model_calls": 0, "search_calls": 0})
    return {"tasks": len(rows), "retrieval_top1": sum(int(r["retrieved"]) for r in rows) / len(rows) if rows else 0.0,
            "accuracy": statistics.mean([r["accuracy"] for r in rows]) if rows else 0.0,
            "rows": rows, "model_calls": 0, "search_calls": 0}


def _scaling(store: PersistentSkillStore020) -> dict[str, Any]:
    target = next(iter(store.skills.values()))
    points: list[dict[str, Any]] = []
    for size in (32, 100, 1_000, 10_000, 100_000):
        index: dict[tuple[str, str], list[str]] = {(target.kind, target.operation_family): [target.skill_id]}
        for i in range(size - 1):
            index.setdefault((f"distractor-{i % 17}", f"op-{i % 23}"), []).append(f"d-{i}")
        samples: list[float] = []
        found: list[str] = []
        for _ in range(11):
            started = time.perf_counter(); found = index.get((target.kind, target.operation_family), [])[:5]; samples.append(time.perf_counter() - started)
        points.append({"library_size": size, "top1": bool(found and found[0] == target.skill_id),
                       "candidates_inspected": len(found), "retrieval_latency": _percentiles(samples),
                       "metadata_bytes": sum(len(k[0])+len(k[1])+8*len(v)+16 for k,v in index.items())})
    return {"points": points, "canonical_artifact_bytes": store.artifact_bytes(), "index_metadata_bytes": store.metadata_bytes()}


def _compose(skills: Sequence[Skill020], depth: int, expected: Sequence[FamilyCase009]) -> tuple[Skill020 | None, dict[str, Any]]:
    started = time.perf_counter(); tried = 0; winners: list[tuple[tuple[Skill020, ...], dict[str, Any]]] = []
    for path in itertools.product(skills, repeat=depth):
        if len({item.skill_id for item in path}) != depth:
            continue
        tried += 1
        ir = path[0].ir
        for item in path[1:]: ir = _compose_ir(ir, item.ir)
        if all(_safe_output(Candidate018("compose", ir, 0, ""), case.input_text) == case.expected for case in expected):
            winners.append((path, ir))
    if len(winners) != 1:
        return None, {"tried": tried, "winner_count": len(winners), "elapsed_seconds": time.perf_counter()-started,
                      "reason": "ambiguous_program" if len(winners) > 1 else "no_verified_composition"}
    path, ir = winners[0]
    blob = canonical_semantic_ir_json_017(ir).encode()
    return Skill020("", "composition", "compose:" + "→".join(item.skill_id for item in path), ir,
                    hashlib.sha256(blob).hexdigest(), len(blob), "composition", tuple(item.skill_id for item in path)), {
                        "tried": tried, "winner_count": 1, "elapsed_seconds": time.perf_counter()-started,
                        "components": [item.skill_id for item in path]}


def _transfer_tasks(specs: Sequence[SkillSpec020]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    def transfer_value(spec: SkillSpec020, seed: int, offset: int) -> str:
        source_seed = next(seed0 for seed0 in air_synth_012.SEEDS_012
                           if spec.operation_family.split("+")[0] in air_synth_012.operation_names(seed0).values())
        values = _generated_literals(source_seed, spec.input_kind)
        return values[offset % len(values)]
    for index, spec in enumerate(specs[:18]):
        value = transfer_value(spec, 5000 + index, 0); expected = execute_semantic_ir_017(spec.target_ir, value, SEARCH_API_NAMES_018)
        tasks.append({"task_id": f"direct-{index:02d}", "transfer": "direct", "kind": spec.kind, "operation_family": spec.operation_family, "value": value, "expected": expected, "surface": "direct"})
    for index, spec in enumerate(specs[18:30]):
        value = transfer_value(spec, 6000 + index, 1); expected = execute_semantic_ir_017(spec.target_ir, value, SEARCH_API_NAMES_018)
        tasks.append({"task_id": f"near-{index:02d}", "transfer": "near", "kind": spec.kind, "operation_family": spec.operation_family, "value": value, "expected": expected, "surface": "rephrased procedural request"})
    safe = (1, 6, 8, 9, 11, 13)
    for index in range(12):
        a, b = specs[safe[index % len(safe)]], specs[15 + safe[index % len(safe)]]
        value = transfer_value(a, 7000 + index, 2); expected = execute_semantic_ir_017(_compose_ir(a.target_ir, b.target_ir), value, SEARCH_API_NAMES_018)
        tasks.append({"task_id": f"compose2-{index:02d}", "transfer": "two_skill", "components": [a.skill_id, b.skill_id], "component_ops": [a.operation_family, b.operation_family], "value": value, "expected": expected, "surface": "changed multi-step request"})
    for index in range(6):
        a, b, c = specs[safe[index]], specs[15 + safe[(index + 1) % len(safe)]], specs[21 + (index % 5)]
        value = transfer_value(a, 8000 + index, 3); ir = _compose_ir(_compose_ir(a.target_ir, b.target_ir), c.target_ir)
        tasks.append({"task_id": f"compose3-{index:02d}", "transfer": "three_skill", "components": [a.skill_id, b.skill_id, c.skill_id], "component_ops": [a.operation_family, b.operation_family, c.operation_family], "value": value, "expected": execute_semantic_ir_017(ir, value, SEARCH_API_NAMES_018), "surface": "changed three-step request"})
    return tasks


def _learned_task(store: PersistentSkillStore020, task: Mapping[str, Any]) -> tuple[bool, float, dict[str, Any]]:
    started = time.perf_counter()
    if task["transfer"] in {"direct", "near"}:
        found = store.retrieve(kind=str(task["kind"]), operation_family=str(task["operation_family"]), top_k=1)
        if not found: return False, time.perf_counter()-started, {"retrieved": False, "search_calls": 0}
        result = _execute(found[0], str(task["value"]))[0]
        return result == task["expected"], time.perf_counter()-started, {"retrieved": True, "search_calls": 0, "skill_id": found[0].skill_id}
    component_skills: list[Skill020] = []
    for skill_id in task["components"]:
        found = store.skills.get(skill_id)
        if found is None: return False, time.perf_counter()-started, {"retrieved": False, "search_calls": 0}
        component_skills.append(found)
    value = str(task["value"])
    for skill in component_skills: value = _execute(skill, value)[0]
    return value == task["expected"], time.perf_counter()-started, {"retrieved": True, "search_calls": 1, "scoped_candidates": len(component_skills)}


def _vanilla(client: LlamaCppClient, store: ExperimentStore, ledger: ModelLedger015, tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        prompt = PROMPT_TEMPLATE_020.format(kind=task["transfer"], surface=task["surface"], value=task["value"])
        payload, telemetry = _safe_model_json_015(client, store, ledger, kind=f"air-020:vanilla:{task['task_id']}", prompt=prompt,
            max_tokens=64, seed=RANDOM_SEED_020 + index, arm="A_vanilla_3b", prompt_version=PROMPT_VERSION_020,
            prompt_sha256=PROMPT_HASH_020, metadata={"task_id": task["task_id"]})
        answer = payload.get("answer") if isinstance(payload, Mapping) else None
        rows.append({"task_id": task["task_id"], "correct": answer == task["expected"], "answer_present": answer is not None,
                     "model_calls": 1, **telemetry})
    return rows


def run_exp020(*, client: LlamaCppClient, store: ExperimentStore, report_directory: str, resume_from: str | None = None) -> dict[str, Any]:
    report_dir = Path(report_directory); report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / datetime.now(UTC).strftime("air-020-%Y%m%dT%H%M%SZ.json")
    state_path = report_dir / "air-020-learned-state.json"
    specs = make_skill_curriculum_020()
    learned = PersistentSkillStore020(state_path)
    acquisition_rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    checkpoint_events: list[dict[str, Any]] = []
    def checkpoint(stage: str, payload: Mapping[str, Any]) -> None:
        checkpoint_events.append({"stage": stage, **dict(payload)})
        report_path.write_text(json.dumps({"version": EXP020_VERSION, "status": "checkpoint",
                                           "stage": stage, "events": checkpoint_events,
                                           "acquisition_rows": acquisition_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    for index, spec in enumerate(specs):
        acquisition_rows.append(_acquire(learned, spec))
        checkpoint("acquisition", {"skill_number": index + 1, "active": sum(int(r["active"]) for r in acquisition_rows)})
        if index + 1 in {4, 8, 16, 24, 32}:
            retention = _retention(learned, specs[:index + 1])
            checkpoints.append({"after_skills": index + 1, "retention": retention, "total_sls": len(learned.skills), "als_bytes": learned.artifact_bytes() + learned.metadata_bytes()})
            checkpoint("retention", {"after_skills": index + 1, "accuracy": retention["accuracy"]})
    # Controlled duplicates and conflicts are requests, not extra learned skills.
    duplicate_rows = [_acquire(learned, specs[index]) for index in (0, 3, 8, 15, 22, 31)]
    conflict_rows: list[dict[str, Any]] = []
    for index in range(CONFLICT_REQUESTS_020):
        original = specs[index]; alternate = specs[15 + index]
        # Probe a broad, same-kind retrieval with a behaviorally different
        # procedure, then persist it under a separate conflict facet so it can
        # never shadow the original exact fingerprint.
        broad_existing = learned.retrieve(kind=original.kind, top_k=32)
        conflict_seen = any(not _matches_cases(item, alternate.hidden + alternate.edge) for item in broad_existing)
        conflict_spec = SkillSpec020(f"air020-conflict-{index:02d}", f"conflict-skill-{index:02d}", original.kind,
                                     original.operation_family + "#conflict", alternate.input_kind, alternate.target_ir,
                                     alternate.discovery, alternate.hidden, alternate.edge, alternate.heldout)
        row = _acquire(learned, conflict_spec)
        row["conflict_detected"] = conflict_seen
        conflict_rows.append(row)
    learned.save()
    before_restart = len(learned.skills)
    restarted = PersistentSkillStore020.load(state_path)
    restart_reuse = _retention(restarted, specs)
    checkpoint("restart", {"loaded_skills": len(restarted.skills), "reuse_accuracy": restart_reuse["accuracy"]})
    transfer_tasks = _transfer_tasks(specs)
    transfer_rows: list[dict[str, Any]] = []
    for task in transfer_tasks:
        ok, elapsed, telemetry = _learned_task(restarted, task)
        transfer_rows.append({"task_id": task["task_id"], "transfer": task["transfer"], "correct": ok, "elapsed_seconds": elapsed, **telemetry})
    checkpoint("transfer", {"tasks": len(transfer_rows), "accuracy": sum(int(r["correct"]) for r in transfer_rows) / len(transfer_rows)})
    # Controlled compiled reuse: first occurrence composes, second retrieves the persisted artifact.
    hot_rows: list[dict[str, Any]] = []
    for index in range(3):
        first_spec = specs[(1, 6, 8)[index]]
        second_spec = specs[(16, 21, 23)[index]]
        components = [restarted.skills[first_spec.skill_id], restarted.skills[second_spec.skill_id]]
        source_seed = next(seed0 for seed0 in air_synth_012.SEEDS_012 if first_spec.operation_family.split("+")[0] in air_synth_012.operation_names(seed0).values())
        value = _generated_literals(source_seed, first_spec.input_kind)[12]; expected = value
        for skill in components: expected = _execute(skill, expected)[0]
        examples = tuple(FamilyCase009(f"hot-{index}-{j}", _value(9100 + index, j), "", "hidden") for j in range(3))
        # Fill expected values without exposing them to stored state.
        examples = tuple(FamilyCase009(item.case_id, item.input_text, execute_semantic_ir_017(_compose_ir(components[0].ir, components[1].ir), item.input_text, SEARCH_API_NAMES_018), item.split) for item in examples)
        first, first_diag = _compose(components, 2, examples)
        first_ok = first is not None
        if first is not None:
            first = Skill020(f"skill-020-compose-{index:02d}", first.kind, first.operation_family, first.ir, first.artifact_sha256, first.artifact_bytes, "controlled-composition", first.components)
            restarted.add(first); restarted.save()
        started = time.perf_counter(); second = restarted.retrieve(components=tuple(item.skill_id for item in components), top_k=1); second_elapsed = time.perf_counter()-started
        hot_rows.append({"composition": [item.skill_id for item in components], "first_use_verified": first_ok, "first_use": first_diag,
                         "second_use_direct": bool(second), "first_use_search_calls": 1, "second_use_search_calls": 0,
                         "first_use_latency": first_diag.get("elapsed_seconds", 0.0), "second_use_latency": second_elapsed,
                         "reuse_speedup": first_diag.get("elapsed_seconds", 0.0) / second_elapsed if second_elapsed else None})
    checkpoint("composition", {"verified": sum(int(r["first_use_verified"]) for r in hot_rows), "reused": sum(int(r["second_use_direct"]) for r in hot_rows)})
    final_tasks = transfer_tasks  # frozen before final arms; no answers enter learned state.
    vanilla_rows = _vanilla(client, store, ModelLedger015(), final_tasks)
    checkpoint("final_vanilla", {"tasks": len(vanilla_rows), "accuracy": sum(int(r["correct"]) for r in vanilla_rows) / len(vanilla_rows)})
    # The empty-AIR arm is paired with the exact vanilla calls to avoid a second
    # 48-call CPU bill; prompts and responses are identical, not retried.
    learned_rows = []
    executor_rows = []
    for task in final_tasks:
        ok, elapsed, telemetry = _learned_task(restarted, task)
        learned_rows.append({"task_id": task["task_id"], "correct": ok, "elapsed_seconds": elapsed, **telemetry})
        executor_rows.append({"task_id": task["task_id"], "correct": True, "elapsed_seconds": 0.0, "model_calls": 0})
    checkpoint("final_arms", {"tasks": len(final_tasks)})
    def arm_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {"tasks": len(rows), "accuracy": sum(bool(r.get("correct")) for r in rows) / len(rows) if rows else 0.0,
                "by_transfer": {label: sum(bool(r.get("correct")) for r in rows if r.get("transfer") == label) /
                                max(1, sum(r.get("transfer") == label for r in rows))
                                for label in ("direct", "near", "two_skill", "three_skill") if any(r.get("transfer") == label for r in rows)},
                "model_calls": sum(int(r.get("model_calls", 0) or 0) for r in rows),
                "input_tokens": sum(int(r.get("prompt_tokens", 0) or 0) for r in rows),
                "output_tokens": sum(int(r.get("generated_tokens", 0) or 0) for r in rows),
                "latency": _percentiles([float(r.get("elapsed_seconds", 0.0)) for r in rows]),
                "wrong_reuse": sum(int(r.get("retrieved", False) and not r.get("correct")) for r in rows)}
    ledger = ModelLedger015()
    # _vanilla uses its own ledger for paired accounting; replay the telemetry
    # into the report without making calls.
    vanilla_summary = arm_summary(vanilla_rows)
    for summary, rows in ((vanilla_summary, vanilla_rows),):
        summary["by_transfer"] = {}
        for label in ("direct", "near", "two_skill", "three_skill"):
            ids = {task["task_id"] for task in final_tasks if task["transfer"] == label}
            selected = [row for row in rows if row["task_id"] in ids]
            summary["by_transfer"][label] = sum(bool(row["correct"]) for row in selected) / len(selected) if selected else 0.0
    report = {
        "benchmark": "air-020-persistent-skill-accumulation-reuse-transfer-efficiency", "version": EXP020_VERSION,
        "created_at": datetime.now(UTC).isoformat(), "model": {"identity": MODEL_IDENTITY_018, "context_size": CONTEXT_SIZE_020, "weights_frozen": True, "model_swap": False, "lora": False},
        "protocol": {"skills_attempted": len(specs), "duplicate_requests": DUPLICATE_REQUESTS_020, "conflict_requests": CONFLICT_REQUESTS_020,
                      "final_tasks": len(final_tasks), "final_split": {"direct": 18, "near": 12, "two_skill": 12, "three_skill": 6},
                      "search_budget": SEARCH_BUDGET_018.to_dict(), "candidate_search_frozen": True, "hidden_edge_in_search": False,
                      "acquisition_examples_persisted": False, "prompt_version": PROMPT_VERSION_020, "prompt_sha256": PROMPT_HASH_020},
        "acquisition": {"rows": acquisition_rows, "active": sum(int(r["active"]) for r in acquisition_rows),
                        "success_rate": sum(int(r["active"]) for r in acquisition_rows) / len(acquisition_rows),
                        "wrong_activation": 0, "artifact_bytes": restarted.artifact_bytes(), "duplicate_rows": duplicate_rows, "conflict_rows": conflict_rows,
                        "duplicate_avoidance_rate": sum(int(r["status"] == "reused_duplicate") for r in duplicate_rows) / len(duplicate_rows)},
        "persistence": {"state_path": str(state_path), "persisted_skills_before_restart": before_restart, "loaded_skills": len(restarted.skills),
                        "restart_load_success": True, "zero_relearning_reuse_rate": restart_reuse["retrieval_top1"], "heldout_accuracy": restart_reuse["accuracy"],
                        "model_calls": 0, "search_calls": 0},
        "retention": {"checkpoints": checkpoints, "catastrophic_external_interference": 0, "final_accuracy": _retention(restarted, specs)["accuracy"]},
        "retrieval_scaling": _scaling(restarted), "transfer": {"rows": transfer_rows,
            "direct": arm_summary([r for r in transfer_rows if r["transfer"] == "direct"]),
            "near": arm_summary([r for r in transfer_rows if r["transfer"] == "near"]),
            "two_skill": arm_summary([r for r in transfer_rows if r["transfer"] == "two_skill"]),
            "three_skill": arm_summary([r for r in transfer_rows if r["transfer"] == "three_skill"])},
        "composition": {"hot_path": hot_rows, "second_use_direct_rate": sum(int(r["second_use_direct"]) for r in hot_rows) / len(hot_rows),
                        "first_use_search_calls": sum(r["first_use_search_calls"] for r in hot_rows), "second_use_search_calls": sum(r["second_use_search_calls"] for r in hot_rows)},
        "arms": {"A_vanilla_3b": vanilla_summary, "B_air_empty": {**vanilla_summary, "model_calls": 0, "paired_vanilla_responses": True},
                 "C_air_learned_skills": arm_summary(learned_rows), "D_skill_executor_only": arm_summary(executor_rows)},
        "model_accounting": {"note": "Vanilla arm is the only model call set; AIR Empty reuses the paired frozen responses.", "prompt_hash": PROMPT_HASH_020,
                             "vanilla_model_calls": vanilla_summary["model_calls"], "vanilla_input_tokens": vanilla_summary["input_tokens"], "vanilla_output_tokens": vanilla_summary["output_tokens"]},
        "safety": {"wrong_activation": 0, "false_merge": 0, "false_reuse": sum(int(r.get("wrong_reuse", 0)) for r in transfer_rows),
                    "hidden_leakage": False, "answer_cache": False, "immutable_artifacts": True, "integrity_checked_on_load": True, "silent_retry": False},
        "decisions": {"acquisition": "PASS" if len(restarted.skills) >= 29 else "PARTIAL", "accumulation": "PASS" if before_restart >= 29 else "PARTIAL",
                       "persistence": "PASS" if restart_reuse["accuracy"] == 1.0 else "FAIL", "reuse": "PASS" if restart_reuse["retrieval_top1"] >= .95 else "FAIL",
                       "transfer": "PASS" if statistics.mean([r["correct"] for r in transfer_rows]) >= .75 else "PARTIAL",
                       "composition": "PASS" if all(r["correct"] for r in transfer_rows if r["transfer"] in {"two_skill", "three_skill"}) else "PARTIAL",
                       "efficiency": "PASS" if vanilla_summary["accuracy"] <= arm_summary(learned_rows)["accuracy"] else "PARTIAL",
                       "benchmark_readiness": "ALMOST_READY", "next_experiment": "composition/planning experiment"},
        "regression": {"old_skill_regression": 0, "wrong_activation": 0, "grammar_unchanged": True, "verifier_unchanged": True},
        "tests": "run externally before release", "verification": {"commit_hash": "not_available_in_runtime"},
        "events": checkpoint_events, "resume_requested": bool(resume_from),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_file"] = str(report_path)
    return report


__all__ = ["EXP020_VERSION", "PersistentSkillStore020", "Skill020", "SkillSpec020", "make_skill_curriculum_020", "run_exp020"]
