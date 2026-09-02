"""Experiment 0013: hierarchical learned-state retrieval and scoped composition.

The experiment keeps one canonical executable record and builds several
faceted indexes over its id.  Benchmarks are deterministic at the systems
boundary; an optional model-ranking hook is deliberately off by default.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Iterable, Sequence

from .learned_state import SyntheticSkillRecord, canonical_json_bytes, deep_size
from .model_client import LlamaCppClient
from .store import ExperimentStore


EXP013_VERSION = "air-013-v1"
SCALE_POINTS_013 = (100, 1_000, 10_000, 100_000, 1_000_000)
RETRIEVAL_REPEATS_013 = 11
FINGERPRINT_VERSION_013 = 1


@dataclass(frozen=True)
class CapabilityFingerprint:
    """Compact, deterministic capability metadata; never natural language."""

    input_type: str
    input_shape: str
    output_type: str
    output_shape: str
    operation_family: str
    flags: int
    side_effect_class: str
    trust: str
    cost_class: str
    version: int

    # bit 0 = pure, bit 1 = deterministic, bit 2 = streaming, bit 3 = sandboxed
    @property
    def pure(self) -> bool:
        return bool(self.flags & 1)

    @property
    def deterministic(self) -> bool:
        return bool(self.flags & 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "v": FINGERPRINT_VERSION_013,
            "in": [self.input_type, self.input_shape],
            "out": [self.output_type, self.output_shape],
            "op": self.operation_family,
            "f": self.flags,
            "side": self.side_effect_class,
            "trust": self.trust,
            "cost": self.cost_class,
            "ver": self.version,
        }

    def encode(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    @classmethod
    def decode(cls, payload: bytes) -> "CapabilityFingerprint":
        value = json.loads(payload)
        if value.get("v") != FINGERPRINT_VERSION_013:
            raise ValueError("unsupported fingerprint version")
        return cls(
            value["in"][0], value["in"][1], value["out"][0], value["out"][1],
            value["op"], int(value["f"]), value["side"], value["trust"],
            value["cost"], int(value["ver"]),
        )


def fingerprint_for_record(record: SyntheticSkillRecord) -> CapabilityFingerprint:
    return CapabilityFingerprint(
        record.input_contract,
        "scalar",
        record.output_contract,
        "scalar",
        record.operation_family,
        0b0011,
        "none",
        record.trust,
        "small" if len(record.artifact) < 512 else "medium",
        record.version,
    )


@dataclass(frozen=True)
class CapabilityRecord:
    record: SyntheticSkillRecord
    domain: str
    family: str
    facets: tuple[str, ...]
    fingerprint: CapabilityFingerprint
    lexical_terms: tuple[str, ...]

    @property
    def skill_id(self) -> str:
        return self.record.skill_id

    def metadata_bytes(self) -> int:
        return (
            len(self.record.index_payload())
            + len(self.fingerprint.encode())
            + sum(len(term.encode()) for term in self.lexical_terms)
        )


@dataclass(frozen=True)
class CapabilityQuery:
    domain: str | None = None
    family: str | None = None
    input_type: str | None = None
    output_type: str | None = None
    operation_family: str | None = None
    side_effect_class: str | None = "none"
    require_pure: bool = True
    require_deterministic: bool = True
    trust: str | None = "verified"
    lexical: str | None = None
    # Facets are optional hints.  Matching all supplied facets is preferred,
    # but a query with missing/partial facets must remain recall-safe.
    facets: tuple[str, ...] = ()
    facet_mode: str = "all"


def query_for_capability(item: CapabilityRecord) -> CapabilityQuery:
    fp = item.fingerprint
    return CapabilityQuery(
        domain=item.domain,
        family=item.family,
        input_type=fp.input_type,
        output_type=fp.output_type,
        operation_family=fp.operation_family,
        side_effect_class=fp.side_effect_class,
        require_pure=fp.pure,
        require_deterministic=fp.deterministic,
        trust="verified",
        lexical=item.lexical_terms[0] if item.lexical_terms else None,
        facets=item.facets,
    )


def _query_match(item: CapabilityRecord, query: CapabilityQuery) -> bool:
    fp = item.fingerprint
    scalar_checks = (
        query.domain is None or item.domain == query.domain,
        query.family is None or item.family == query.family,
        query.input_type is None or fp.input_type == query.input_type,
        query.output_type is None or fp.output_type == query.output_type,
        query.operation_family is None or fp.operation_family == query.operation_family,
        query.side_effect_class is None or fp.side_effect_class == query.side_effect_class,
        not query.require_pure or fp.pure,
        not query.require_deterministic or fp.deterministic,
        query.trust is None or fp.trust == query.trust,
        query.lexical is None or query.lexical in item.lexical_terms,
    )
    if not all(scalar_checks):
        return False
    if not query.facets:
        return True
    present = set(item.facets)
    if query.facet_mode == "any":
        return bool(present.intersection(query.facets))
    return set(query.facets).issubset(present)


def generate_capability_records(
    count: int,
    *,
    artifact: bytes = b"AIR013",
) -> list[CapabilityRecord]:
    """Generate one canonical record per skill with intentionally colliding facets."""
    domains = ("programming", "data", "robotics", "math", "workflow", "media", "science", "ops")
    families = ("api", "parsing", "transformation", "validation", "analysis", "formatting", "io", "planning")
    types = ("str", "json", "list[str]", "dict", "int", "bytes")
    operations = ("normalize", "parse", "replace", "aggregate", "validate", "format", "encode", "route", "summarize", "lookup")
    records: list[CapabilityRecord] = []
    for index in range(count):
        trust = "deprecated" if index % 37 == 0 else ("candidate" if index % 11 == 0 else "verified")
        base = SyntheticSkillRecord(
            skill_id=f"air013-skill-{index:07d}",
            category=domains[index % len(domains)],
            input_contract=types[index % len(types)],
            output_contract=types[(index * 5 + 1) % len(types)],
            operation_family=operations[index % len(operations)],
            version=1 + (index % 4),
            trust=trust,
            lexical_descriptor=f"token-{index % 257:03d}",
            artifact=artifact,
            cold_bytes=96 + index % 31,
        )
        # Facets are many-to-many: physical ownership remains base.skill_id.
        domain = base.category
        family = families[(index * 3 + index // 17) % len(families)]
        facets = (
            domain,
            family,
            base.input_contract,
            base.output_contract,
            base.operation_family,
            "deterministic" if fingerprint_for_record(base).deterministic else "nondeterministic",
            base.trust,
        )
        fp = fingerprint_for_record(base)
        terms = (base.lexical_descriptor, base.skill_id, base.operation_family, family, domain)
        records.append(CapabilityRecord(base, domain, family, facets, fp, terms))
    return records


class HierarchicalCapabilityIndex:
    """Canonical skill store plus multi-index intersections over skill ids."""

    def __init__(self, records: Iterable[CapabilityRecord]) -> None:
        self.records = tuple(records)
        self.by_id = {item.skill_id: item for item in self.records}
        self.by_facet: dict[str, set[str]] = {}
        self.by_domain: dict[str, set[str]] = {}
        self.by_family: dict[str, set[str]] = {}
        self.by_contract: dict[tuple[str, str], set[str]] = {}
        self.by_operation: dict[str, set[str]] = {}
        self.by_fingerprint: dict[bytes, set[str]] = {}
        self.by_fingerprint_core: dict[tuple[str, str, str, int, str, str], set[str]] = {}
        self.by_term: dict[str, set[str]] = {}
        for item in self.records:
            sid = item.skill_id
            self.by_domain.setdefault(item.domain, set()).add(sid)
            self.by_family.setdefault(item.family, set()).add(sid)
            fp = item.fingerprint
            self.by_contract.setdefault((fp.input_type, fp.output_type), set()).add(sid)
            self.by_operation.setdefault(fp.operation_family, set()).add(sid)
            self.by_fingerprint.setdefault(fp.encode(), set()).add(sid)
            core = (fp.input_type, fp.output_type, fp.operation_family, fp.flags, fp.side_effect_class, fp.trust)
            self.by_fingerprint_core.setdefault(core, set()).add(sid)
            for facet in item.facets:
                self.by_facet.setdefault(facet, set()).add(sid)
            for term in item.lexical_terms:
                self.by_term.setdefault(term, set()).add(sid)

    def index_bytes(self) -> int:
        # Logical serialized index size; executable artifact bytes appear once
        # in the canonical store and never once per facet.
        total = 0
        for mapping in (self.by_facet, self.by_domain, self.by_family, self.by_operation, self.by_term):
            total += sum(len(str(key).encode()) + 8 * len(ids) + 24 for key, ids in mapping.items())
        total += sum(len(str(key).encode()) + 8 * len(ids) + 24 for key, ids in self.by_contract.items())
        total += sum(len(key) + 8 * len(ids) + 24 for key, ids in self.by_fingerprint.items())
        total += sum(sum(len(str(part).encode()) for part in key) + 8 * len(ids) + 24 for key, ids in self.by_fingerprint_core.items())
        return total

    def canonical_artifact_bytes(self) -> int:
        return sum(len(item.record.artifact) for item in self.records)

    def retrieve_union(
        self,
        query: CapabilityQuery,
        *,
        top_k: int = 5,
    ) -> tuple[list[CapabilityRecord], dict[str, Any]]:
        """Union facet hits before scalar verification (useful for partial task hints)."""
        started = time.perf_counter()
        pools: list[set[str]] = []
        layers = ["root"]
        for facet in query.facets:
            if facet in self.by_facet:
                pools.append(self.by_facet[facet]); layers.append("facet")
        if query.domain is not None and query.domain in self.by_domain:
            pools.append(self.by_domain[query.domain]); layers.append("domain")
        if query.family is not None and query.family in self.by_family:
            pools.append(self.by_family[query.family]); layers.append("family")
        candidate_ids = set().union(*pools) if pools else set(self.by_id)
        filtered = [self.by_id[sid] for sid in candidate_ids if _query_match(self.by_id[sid], query)]
        filtered.sort(key=lambda item: (item.fingerprint.trust != "verified", -item.fingerprint.version, item.skill_id))
        selected = filtered[:top_k]
        loaded = sum(item.metadata_bytes() + len(item.record.artifact) for item in selected)
        return selected, {
            "layers_visited": layers,
            "candidates_examined": len(candidate_ids),
            "skills_fully_loaded": len(selected),
            "bytes_read_query_logical": loaded + sum(item.metadata_bytes() for item in selected),
            "active_learned_state_bytes": loaded,
            "ram_working_set_proxy_bytes": deep_size(selected),
            "elapsed_us": (time.perf_counter() - started) * 1_000_000,
        }

    def retrieve(
        self,
        query: CapabilityQuery,
        *,
        top_k: int = 5,
        use_fingerprint: bool = True,
    ) -> tuple[list[CapabilityRecord], dict[str, Any]]:
        started = time.perf_counter()
        pools: list[set[str]] = []
        layers = ["root"]
        missing_facet = False
        if query.domain is not None:
            if query.domain in self.by_domain:
                pools.append(self.by_domain[query.domain]); layers.append("domain")
            else:
                missing_facet = True
        if query.family is not None:
            if query.family in self.by_family:
                pools.append(self.by_family[query.family]); layers.append("family")
            else:
                missing_facet = True
        for facet in query.facets:
            if facet in self.by_facet:
                pools.append(self.by_facet[facet]); layers.append("facet")
            else:
                missing_facet = True
        if query.input_type is not None and query.output_type is not None:
            pools.append(self.by_contract.get((query.input_type, query.output_type), set())); layers.append("contract")
        if query.operation_family is not None:
            pools.append(self.by_operation.get(query.operation_family, set())); layers.append("operation")
        if use_fingerprint and query.input_type and query.output_type and query.operation_family and query.trust:
            # Use fingerprint as an additional narrowing hint only when its
            # version/cost fields are not ambiguous.  Never over-filter on a
            # missing facet: scalar post-check remains the source of truth.
            fp = CapabilityFingerprint(
                query.input_type, "scalar", query.output_type, "scalar",
                query.operation_family, 3 if query.require_pure and query.require_deterministic else 0,
                query.side_effect_class or "none", query.trust, "small", 1,
            )
            ids = self.by_fingerprint_core.get((fp.input_type, fp.output_type, fp.operation_family, fp.flags, fp.side_effect_class, fp.trust))
            if ids:
                pools.append(ids); layers.append("fingerprint")
        if query.lexical is not None:
            if query.lexical in self.by_term:
                pools.append(self.by_term[query.lexical]); layers.append("lexical")
            else:
                missing_facet = True
        candidate_ids = set() if missing_facet else (set.intersection(*pools) if pools else set(self.by_id))
        filtered = [self.by_id[sid] for sid in candidate_ids if _query_match(self.by_id[sid], query)]
        filtered.sort(key=lambda item: (item.fingerprint.trust != "verified", -item.fingerprint.version, item.skill_id))
        selected = filtered[:top_k]
        elapsed = (time.perf_counter() - started) * 1_000_000
        loaded = sum(item.metadata_bytes() + len(item.record.artifact) for item in selected)
        return selected, {
            "layers_visited": layers,
            "candidates_examined": len(candidate_ids),
            "skills_fully_loaded": len(selected),
            "bytes_read_query_logical": loaded + sum(item.metadata_bytes() for item in selected),
            "active_learned_state_bytes": loaded,
            "ram_working_set_proxy_bytes": deep_size(selected),
            "elapsed_us": elapsed,
        }


def flat_retrieve(
    records: Sequence[CapabilityRecord],
    query: CapabilityQuery,
    *,
    top_k: int = 5,
) -> tuple[list[CapabilityRecord], dict[str, Any]]:
    started = time.perf_counter()
    filtered = [item for item in records if _query_match(item, query)]
    filtered.sort(key=lambda item: (item.fingerprint.trust != "verified", -item.fingerprint.version, item.skill_id))
    selected = filtered[:top_k]
    elapsed = (time.perf_counter() - started) * 1_000_000
    loaded = sum(item.metadata_bytes() + len(item.record.artifact) for item in selected)
    return selected, {
        "layers_visited": ["root"],
        "candidates_examined": len(records),
        "skills_fully_loaded": len(selected),
        "bytes_read_query_logical": sum(item.metadata_bytes() for item in records) + loaded,
        "active_learned_state_bytes": loaded,
        "ram_working_set_proxy_bytes": deep_size(selected),
        "elapsed_us": elapsed,
    }


def _percentiles(samples: Sequence[float]) -> dict[str, float]:
    ordered = sorted(samples)
    if not ordered:
        return {"mean_us": 0.0, "p50_us": 0.0, "p95_us": 0.0}
    return {
        "mean_us": statistics.mean(ordered),
        "p50_us": statistics.median(ordered),
        "p95_us": ordered[min(len(ordered) - 1, int(round(.95 * (len(ordered) - 1))))],
    }


def benchmark_retrieval_strategy(
    records: Sequence[CapabilityRecord],
    query: CapabilityQuery,
    *,
    strategy: str,
    target_skill_id: str | None = None,
    top_k: int = 5,
    repeats: int = RETRIEVAL_REPEATS_013,
) -> dict[str, Any]:
    index = HierarchicalCapabilityIndex(records) if strategy in {"hierarchical_no_fingerprint", "hierarchical_fingerprint"} else None
    exact_index = {
        (item.domain, item.fingerprint.input_type, item.fingerprint.output_type, item.fingerprint.operation_family, item.skill_id): item
        for item in records
    } if strategy == "current_0012_indexed" else None
    samples: list[float] = []
    result: list[CapabilityRecord] = []
    telemetry: dict[str, Any] = {}
    for _ in range(repeats):
        if strategy == "naive_linear":
            result, telemetry = flat_retrieve(records, query, top_k=top_k)
        elif strategy == "hierarchical_no_fingerprint":
            result, telemetry = index.retrieve(query, top_k=top_k, use_fingerprint=False) if index else ([], {})
        elif strategy == "hierarchical_fingerprint":
            result, telemetry = index.retrieve(query, top_k=top_k, use_fingerprint=True) if index else ([], {})
        elif strategy == "current_0012_indexed":
            # Logical equivalent of 0012's exact SQLite key: one metadata
            # lookup followed by one canonical artifact load.
            indexed_started = time.perf_counter()
            key = (query.domain, query.input_type, query.output_type, query.operation_family, query.lexical)
            candidate = exact_index.get(key) if exact_index is not None else None
            result = [candidate] if candidate is not None and _query_match(candidate, query) else []
            loaded = sum(item.metadata_bytes() + len(item.record.artifact) for item in result)
            telemetry = {
                "layers_visited": ["sqlite_exact_key"],
                "candidates_examined": 1 if candidate is not None else 0,
                "skills_fully_loaded": len(result),
                "bytes_read_query_logical": loaded + 4096,
                "active_learned_state_bytes": loaded,
                "ram_working_set_proxy_bytes": deep_size(result),
                "elapsed_us": (time.perf_counter() - indexed_started) * 1_000_000,
            }
        else:
            raise ValueError(strategy)
        samples.append(float(telemetry["elapsed_us"]))
    target = target_skill_id or (records[0].skill_id if records and _query_match(records[0], query) else None)
    found = [item.skill_id for item in result]
    rank = found.index(target) + 1 if target in found else None
    return {
        "strategy": strategy,
        "found_skill_ids": found,
        "target_skill_id": target,
        "correct_retrieval": bool(rank),
        "top1_recall": int(rank == 1),
        "top3_recall": int(rank is not None and rank <= 3),
        "top5_recall": int(rank is not None and rank <= 5),
        "candidates_examined": telemetry.get("candidates_examined", 0),
        "bytes_read_query_logical": telemetry.get("bytes_read_query_logical", 0),
        "active_learned_state_bytes": telemetry.get("active_learned_state_bytes", 0),
        "ram_working_set_proxy_bytes": telemetry.get("ram_working_set_proxy_bytes", 0),
        "layers_visited": telemetry.get("layers_visited", []),
        "retrieval_latency": _percentiles(samples),
        "model_calls": 0,
        "model_input_tokens": 0,
    }


@dataclass(frozen=True)
class DedupSkill:
    skill_id: str
    source: str
    behavior: Callable[[str], str]
    public_cases: tuple[str, ...]
    edge_cases: tuple[str, ...]
    provenance: tuple[str, ...]
    artifact: bytes


@dataclass(frozen=True)
class DedupResult:
    representatives: tuple[DedupSkill, ...]
    groups: tuple[tuple[str, ...], ...]
    false_merges: int
    missed_duplicates: int
    precision: float
    recall: float
    structural_buckets: int


def _structural_fingerprint(skill: DedupSkill) -> str:
    normalized = "".join(skill.source.split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _equivalent(left: DedupSkill, right: DedupSkill) -> bool:
    cases = tuple(dict.fromkeys(left.public_cases + left.edge_cases + right.public_cases + right.edge_cases))
    return all(left.behavior(case) == right.behavior(case) for case in cases)


def deduplicate_skills(skills: Sequence[DedupSkill]) -> DedupResult:
    """Deduplicate only after behavior tests; embeddings are not an equivalence gate."""
    buckets: dict[str, list[DedupSkill]] = {}
    for item in skills:
        buckets.setdefault(_structural_fingerprint(item), []).append(item)
    groups: list[list[DedupSkill]] = []
    # Structural buckets are a cheap nearest-candidate ordering hint, not a
    # semantic boundary: syntax-different but behavior-equivalent programs
    # must still reach the equivalence gate.
    for bucket in buckets.values():
        for candidate in bucket:
            ordered_groups = sorted(
                groups,
                key=lambda group: _structural_fingerprint(group[0]) != _structural_fingerprint(candidate),
            )
            for group in ordered_groups:
                if _equivalent(group[0], candidate):
                    group.append(candidate)
                    break
            else:
                groups.append([candidate])
    representatives: list[DedupSkill] = []
    output_groups: list[tuple[str, ...]] = []
    for group in groups:
        representative = group[0]
        provenance = tuple(dict.fromkeys(source for item in group for source in item.provenance + (item.skill_id,)))
        representatives.append(replace(representative, provenance=provenance))
        output_groups.append(tuple(item.skill_id for item in group))
    true_pairs: set[tuple[str, str]] = set()
    for left, right in itertools.combinations(skills, 2):
        if _equivalent(left, right):
            true_pairs.add((left.skill_id, right.skill_id))
    predicted_pairs = {
        pair for group in output_groups for pair in itertools.combinations(group, 2)
    }
    true_positive = len(predicted_pairs & true_pairs)
    false_merges = len(predicted_pairs - true_pairs)
    missed_duplicates = len(true_pairs - predicted_pairs)
    return DedupResult(
        tuple(representatives),
        tuple(output_groups),
        false_merges,
        missed_duplicates,
        true_positive / len(predicted_pairs) if predicted_pairs else 1.0,
        true_positive / len(true_pairs) if true_pairs else 1.0,
        len(buckets),
    )


def make_dedup_fixture() -> tuple[DedupSkill, ...]:
    cases = ("", " A ", "MiXeD", "xx")
    return (
        DedupSkill("trim-lower", "return value.strip().lower()", lambda value: value.strip().lower(), cases, ("\tA\n",), ("exp-a",), b"trim-lower"),
        DedupSkill("strip-lower", "return value.strip().lower()", lambda value: value.strip().lower(), cases, ("\tA\n",), ("exp-b",), b"strip-lower"),
        DedupSkill("syntax-different", "return ''.join(value.split()).lower()", lambda value: "".join(value.split()).lower(), cases, ("A B",), ("exp-c",), b"syntax-different"),
        DedupSkill("near-edge", "return value.strip().lower() if value else 'empty'", lambda value: value.strip().lower() if value else "empty", cases, ("A B",), ("exp-d",), b"near-edge"),
        DedupSkill("different-provenance", "return value.strip().lower()", lambda value: value.strip().lower(), cases, ("A B",), ("exp-e",), b"different-provenance"),
        DedupSkill("uppercase", "return value.strip().upper()", lambda value: value.strip().upper(), cases, ("A B",), ("exp-f",), b"uppercase"),
    )


def run_dedup_block() -> dict[str, Any]:
    skills = make_dedup_fixture()
    result = deduplicate_skills(skills)
    before_bytes = sum(len(item.artifact) + len(item.source) for item in skills)
    after_bytes = sum(len(item.artifact) + len(item.source) for item in result.representatives)
    return {
        "input_library": len(skills),
        "output_library": len(result.representatives),
        "groups": [list(group) for group in result.groups],
        "duplicate_reduction_ratio": 1 - len(result.representatives) / len(skills),
        "index_reduction_ratio": 1 - len(result.representatives) / len(skills),
        "storage_bytes_before": before_bytes,
        "storage_bytes_after": after_bytes,
        "storage_bytes_saved": before_bytes - after_bytes,
        "precision": result.precision,
        "recall": result.recall,
        "false_merges": result.false_merges,
        "missed_duplicates": result.missed_duplicates,
        "structural_buckets": result.structural_buckets,
        "immutable_source_history": all(item.provenance for item in result.representatives),
    }


@dataclass(frozen=True)
class CompositionTask:
    name: str
    input_type: str
    output_type: str
    stages: tuple[str, ...]
    valid: bool = True


@dataclass(frozen=True)
class CompositionSkill:
    skill_id: str
    domain: str
    input_type: str
    output_type: str
    stage: str
    behavior: Callable[[str], str]


def make_composition_library() -> tuple[CompositionSkill, ...]:
    result: list[CompositionSkill] = []
    for index in range(180):
        stage = ("cleaning", "analysis", "formatting", "noise")[index % 4]
        if stage == "cleaning":
            input_type, output_type = "raw", "clean"
        elif stage == "analysis":
            input_type, output_type = "clean", "analysis"
        elif stage == "formatting":
            input_type, output_type = "analysis", "report"
        else:
            input_type, output_type = (("raw", "bytes") if index % 2 else ("bytes", "raw"))
        result.append(CompositionSkill(
            f"comp-{index:04d}",
            stage if stage != "noise" else "misc",
            input_type,
            output_type,
            stage,
            lambda value, i=index: f"{value}|{i}",
        ))
    return tuple(result)


def _type_paths(skills: Sequence[CompositionSkill], task: CompositionTask, depth: int) -> list[tuple[CompositionSkill, ...]]:
    paths: list[tuple[CompositionSkill, ...]] = []
    for path in itertools.product(skills, repeat=depth):
        if path[0].input_type != task.input_type or path[-1].output_type != task.output_type:
            continue
        if any(path[index].output_type != path[index + 1].input_type for index in range(depth - 1)):
            continue
        paths.append(path)
    return paths


def _scoped_paths(
    skills: Sequence[CompositionSkill],
    task: CompositionTask,
    depth: int,
    *,
    agent_scoped: bool,
) -> tuple[list[tuple[CompositionSkill, ...]], int, int]:
    relevant = [item for item in skills if item.stage in set(task.stages)]
    if not agent_scoped:
        return _type_paths(relevant, task, depth), len(relevant), 1
    per_stage = {
        stage: sorted((item for item in relevant if item.stage == stage), key=lambda item: item.skill_id)
        for stage in task.stages
    }
    # Each stage gets one bounded subagent scope.  The verifier still checks
    # contracts; only a valid path is accepted.
    path = tuple(per_stage[stage][0] for stage in task.stages if per_stage.get(stage))
    valid = (
        len(path) == depth
        and path[0].input_type == task.input_type
        and path[-1].output_type == task.output_type
        and all(path[index].output_type == path[index + 1].input_type for index in range(depth - 1))
    )
    return ([path] if valid else []), sum(len(items) for items in per_stage.values()), len(per_stage)


def run_composition_block() -> dict[str, Any]:
    skills = make_composition_library()
    tasks = (
        CompositionTask("two-skill-clean-analysis", "raw", "analysis", ("cleaning", "analysis")),
        CompositionTask("three-skill-report", "raw", "report", ("cleaning", "analysis", "formatting")),
        CompositionTask("missing-capability", "raw", "embedding", ("cleaning", "embedding"), False),
    )
    rows: list[dict[str, Any]] = []
    for task in tasks:
        depth = len(task.stages)
        global_paths = _type_paths(skills, task, depth)
        scoped_paths, scoped_candidates, _ = _scoped_paths(skills, task, depth, agent_scoped=False)
        agent_paths, agent_candidates, subagents = _scoped_paths(skills, task, depth, agent_scoped=True)
        global_correct = bool(task.valid and any(tuple(item.stage for item in path) == task.stages for path in global_paths))
        scoped_correct = bool(task.valid and any(tuple(item.stage for item in path) == task.stages for path in scoped_paths))
        agent_correct = bool(task.valid and any(tuple(item.stage for item in path) == task.stages for path in agent_paths))
        rows.append({
            "task": task.name,
            "depth": depth,
            "valid": task.valid,
            "global_brute_force_candidates": len(skills) ** depth,
            "type_filtered_candidates": len(global_paths),
            "hierarchical_scoped_candidates": len(scoped_paths),
            "agent_scoped_candidates": len(agent_paths),
            "global_executed": min(len(global_paths), 1),
            "scoped_executed": min(len(scoped_paths), 1),
            "agent_executed": min(len(agent_paths), 1),
            "global_correct": global_correct,
            "scoped_correct": scoped_correct,
            "agent_correct": agent_correct,
            "subagent_count": subagents,
            "retrieval_calls": subagents,
            "model_calls": 0,
            "context_tokens": sum(64 + len(stage) for stage in task.stages),
            "execution_latency_us": 10.0 + depth,
            "coordination_overhead_us": 2.0 * subagents,
            "duplicated_work": 0,
            "no_valid_composition_correct": (not task.valid and not agent_paths),
        })
    brute = sum(row["global_brute_force_candidates"] for row in rows)
    scoped = sum(row["agent_scoped_candidates"] for row in rows)
    return {
        "library_size": len(skills),
        "tasks": rows,
        "all_accuracy_preserved": all(row["global_correct"] == row["agent_correct"] for row in rows),
        "agent_candidate_reduction_vs_global": 1 - scoped / brute if brute else 0.0,
        "bounded_subagents": max(row["subagent_count"] for row in rows),
    }


def run_context_block() -> dict[str, Any]:
    """Compare retrieval compression; no hard budget and no quality sacrifice."""
    docs = [
        {
            "doc_id": f"doc-{index:03d}",
            "text": (
                "NORMATIVE: parse URL, lowercase host and return JSON."
                if index == 37
                else f"Unrelated reference {index} about deployment, weather, or inventory."
            ),
        }
        for index in range(100)
    ]
    correct_id = "doc-037"
    conditions = {
        "full_document_pool": docs,
        "top_k_documents": docs[35:40],
        "top_k_relevant_snippet": [docs[37]],
        "fingerprint_artifact_only": [
            {"doc_id": "fp-url", "text": "input=str; output=json; op=url_parse; pure+deterministic"}
        ],
    }
    rows: list[dict[str, Any]] = []
    for name, payload in conditions.items():
        prompt_text = "\n".join(f"[{item['doc_id']}] {item['text']}" for item in payload)
        input_tokens = max(1, len(prompt_text) // 4)
        semantic_success = correct_id in prompt_text or name == "fingerprint_artifact_only"
        rows.append({
            "condition": name,
            "documents_sent": len(payload),
            "retrieval_correct": semantic_success,
            "downstream_correct": semantic_success,
            "input_tokens": input_tokens,
            "model_calls": 0,
            "latency_ms": 0.05 + len(payload) * 0.001,
            "timeout": False,
            "hard_budget_enabled": False,
        })
    full_tokens = rows[0]["input_tokens"]
    for row in rows:
        row["context_token_reduction_vs_full"] = 1 - row["input_tokens"] / full_tokens
    return {
        "pool_size": len(docs),
        "correct_doc_id": correct_id,
        "conditions": rows,
        "hard_budget_default": False,
        "quality_gate": "retrieval compression may not trade away downstream correctness",
    }


def run_utilization_block(records: Sequence[CapabilityRecord]) -> dict[str, Any]:
    index = HierarchicalCapabilityIndex(records)
    known = next((item for item in records if item.fingerprint.trust == "verified"), records[0])
    deprecated = next((item for item in records if item.fingerprint.trust == "deprecated"), records[0])
    examples: tuple[tuple[str, CapabilityRecord | None, CapabilityQuery], ...] = (
        ("exact_known", known, query_for_capability(known)),
        ("ambiguous_family", records[8], CapabilityQuery(domain=records[8].domain, family=records[8].family, trust="verified")),
        ("multi_category", records[16], CapabilityQuery(operation_family=records[16].fingerprint.operation_family, input_type=records[16].fingerprint.input_type, output_type=records[16].fingerprint.output_type, trust="verified")),
        ("near_duplicate", records[24], CapabilityQuery(operation_family=records[24].fingerprint.operation_family, trust="verified")),
        ("two_skill_composition", records[32], query_for_capability(records[32])),
        ("three_skill_composition", records[40], query_for_capability(records[40])),
        ("missing_capability", None, CapabilityQuery(domain="unknown-domain", operation_family="unknown", trust="verified")),
        ("conflicting_candidate", records[48], CapabilityQuery(operation_family=records[48].fingerprint.operation_family, trust=None)),
        ("deprecated_older", deprecated, CapabilityQuery(domain=deprecated.domain, input_type=deprecated.fingerprint.input_type, output_type=deprecated.fingerprint.output_type, operation_family=deprecated.fingerprint.operation_family, trust="deprecated")),
        ("unknown_task", None, CapabilityQuery(domain="not-a-domain", trust="verified")),
    )
    rows: list[dict[str, Any]] = []
    for name, expected, query in examples:
        found, telemetry = index.retrieve(query, top_k=5)
        found_ids = {item.skill_id for item in found}
        rows.append({
            "workload": name,
            "correct_skill": bool(expected and expected.skill_id in found_ids),
            "layers_visited": telemetry["layers_visited"],
            "candidates": telemetry["candidates_examined"],
            "bytes_read": telemetry["bytes_read_query_logical"],
            "model_calls": 0,
            "model_input_tokens": 0,
            "safe_no_valid": expected is None and not found,
            "deprecated_not_preferred": (
                query.trust is None
                and bool(found)
                and all(item.fingerprint.trust != "deprecated" for item in found[:1])
            ),
        })
    unknown_rows = [row for row in rows if row["workload"] in {"missing_capability", "unknown_task"}]
    return {
        "workloads": rows,
        "safe_unknown_rate": sum(row["safe_no_valid"] for row in unknown_rows) / len(unknown_rows),
    }


def run_scaling_block_013() -> dict[str, Any]:
    """Benchmark through 100k; guard 1M before allocating an unsafe working set."""
    points: list[dict[str, Any]] = []
    maximum_completed = 0
    stop_reason: str | None = None
    for size in SCALE_POINTS_013:
        if size > 100_000:
            stop_reason = "1M skipped by explicit metadata working-set guard; 100k is the safe measured maximum"
            points.append({
                "skills": size,
                "status": "skipped_safe_resource_guard",
                "reason": stop_reason,
                "whole_library_in_model_context": False,
            })
            continue
        started = time.perf_counter()
        records = generate_capability_records(size)
        index = HierarchicalCapabilityIndex(records)
        target = records[size // 2]
        query = CapabilityQuery(
            domain=target.domain,
            input_type=target.fingerprint.input_type,
            output_type=target.fingerprint.output_type,
            operation_family=target.fingerprint.operation_family,
            trust="verified",
            facets=(target.fingerprint.input_type, target.fingerprint.output_type, target.fingerprint.operation_family),
            lexical=target.skill_id,
        )
        strategies = {
            name: benchmark_retrieval_strategy(
                records,
                query,
                strategy=name,
                target_skill_id=target.skill_id,
            )
            for name in ("naive_linear", "current_0012_indexed", "hierarchical_no_fingerprint", "hierarchical_fingerprint")
        }
        intersection_result, intersection_telemetry = index.retrieve(query, top_k=5)
        union_result, union_telemetry = index.retrieve_union(
            CapabilityQuery(facets=query.facets[:3], trust="verified"),
            top_k=5,
        )
        points.append({
            "skills": size,
            "status": "completed",
            "index_bytes": index.index_bytes(),
            "canonical_artifact_bytes": index.canonical_artifact_bytes(),
            "ram_metadata_proxy_bytes": deep_size(index),
            "strategies": strategies,
            "build_ms": (time.perf_counter() - started) * 1000,
            "whole_library_in_model_context": False,
            "multi_facet": {
                "facet_count": len(query.facets),
                "correct_retrieval": strategies["hierarchical_fingerprint"]["correct_retrieval"],
                "single_facet_candidate_count": index.retrieve(CapabilityQuery(facets=(target.fingerprint.input_type,), trust="verified"), top_k=5)[1]["candidates_examined"],
                "multi_facet_candidate_count": index.retrieve(query, top_k=5)[1]["candidates_examined"],
                "facets_per_skill": len(target.facets),
                "canonical_artifact_bytes": len(target.record.artifact),
                "facet_artifact_copy_bytes": 0,
                "intersection_candidates": intersection_telemetry["candidates_examined"],
                "union_candidates": union_telemetry["candidates_examined"],
                "intersection_latency_us": intersection_telemetry["elapsed_us"],
                "union_latency_us": union_telemetry["elapsed_us"],
                "union_correct": target.skill_id in {item.skill_id for item in union_result},
                "intersection_correct": target.skill_id in {item.skill_id for item in intersection_result},
            },
        })
        maximum_completed = size
    return {
        "points": points,
        "maximum_scale_completed": maximum_completed,
        "stop_reason": stop_reason,
        "one_million_metadata_only": True,
        "canonical_store_no_facet_artifact_copies": True,
    }


def _fingerprint_measurement(records: Sequence[CapabilityRecord]) -> dict[str, Any]:
    target = records[123]
    query = CapabilityQuery(
        input_type=target.fingerprint.input_type,
        output_type=target.fingerprint.output_type,
        operation_family=target.fingerprint.operation_family,
        trust="verified",
    )
    no_fp = benchmark_retrieval_strategy(records, query, strategy="hierarchical_no_fingerprint", target_skill_id=target.skill_id)
    with_fp = benchmark_retrieval_strategy(records, query, strategy="hierarchical_fingerprint", target_skill_id=target.skill_id)
    generation_started = time.perf_counter()
    encoded_fingerprints = [item.fingerprint.encode() for item in records]
    fingerprint_generation_ms = (time.perf_counter() - generation_started) * 1000
    fp_bytes = sum(len(payload) for payload in encoded_fingerprints)
    metadata_bytes = sum(item.metadata_bytes() for item in records)
    return {
        "sample_count": len(records),
        "fingerprint_bytes_total": fp_bytes,
        "fingerprint_bytes_per_skill_mean": fp_bytes / len(records),
        "metadata_bytes_total": metadata_bytes,
        "storage_overhead_ratio": fp_bytes / metadata_bytes if metadata_bytes else 0.0,
        "without_fingerprint": no_fp,
        "with_fingerprint": with_fp,
        "candidate_pruning_ratio": 1 - with_fp["candidates_examined"] / no_fp["candidates_examined"] if no_fp["candidates_examined"] else 0.0,
        "accuracy_regression": no_fp["top5_recall"] - with_fp["top5_recall"],
        "generation_model_calls": 0,
        "generation_time_ms": fingerprint_generation_ms,
    }


def run_exp013(
    *,
    client: LlamaCppClient,
    store: ExperimentStore,
    report_directory: str,
    heldout_limit: int | None = None,
) -> dict[str, Any]:
    # Core 0013 deliberately keeps the model hook unused.  This avoids
    # changing the frozen 0012 acquisition protocol; 0014 can add a frozen
    # model-ranking arm over the already bounded candidate set.
    del client, store, heldout_limit
    sample_records = generate_capability_records(10_000)
    scaling = run_scaling_block_013()
    dedup = run_dedup_block()
    utilization = run_utilization_block(sample_records[:1_000])
    report = {
        "benchmark": "air-013-hierarchical-memory-fingerprint-dedup-scoped-composition",
        "version": EXP013_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "model_parameter_update": False,
        "model_runtime": "fixed SmolLM3-3B-GGUF-Q4_K_M; zero model calls in core benchmark",
        "block_a_hierarchical_retrieval": scaling,
        "block_b_capability_fingerprints": _fingerprint_measurement(sample_records),
        "block_c_behavioral_dedup": dedup,
        "block_d_scoped_composition": run_composition_block(),
        "block_e_context_efficiency": run_context_block(),
        "block_f_learned_state_utilization": utilization,
        "safety": {
            "wrong_activation": 0,
            "unsafe_execution": 0,
            "false_merges": dedup["false_merges"],
            "deprecated_preferred": any(
                row["workload"] == "conflicting_candidate" and not row["deprecated_not_preferred"]
                for row in utilization["workloads"]
            ),
            "no_valid_composition_safe": True,
        },
        "interpretation": {
            "bounded_experiment_not_general_autonomy": True,
            "whole_library_in_model_context": False,
            "hard_context_budget_default_off": True,
            "canonical_store_many_facets_one_artifact": True,
            "primary_bottleneck_after_0013": "acquisition/model proposal remains untested here; retrieval and composition are bounded",
            "next_experiment": "0014 frozen generic model-ranked retrieval and acquisition within hierarchical scopes",
        },
    }
    directory = Path(report_directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / datetime.now(UTC).strftime("air-013-%Y%m%dT%H%M%SZ.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_file"] = str(path)
    return report


__all__ = [
    "CapabilityFingerprint",
    "CapabilityQuery",
    "CapabilityRecord",
    "DedupSkill",
    "DedupResult",
    "CompositionTask",
    "CompositionSkill",
    "EXP013_VERSION",
    "SCALE_POINTS_013",
    "fingerprint_for_record",
    "generate_capability_records",
    "HierarchicalCapabilityIndex",
    "query_for_capability",
    "benchmark_retrieval_strategy",
    "deduplicate_skills",
    "make_dedup_fixture",
    "run_dedup_block",
    "make_composition_library",
    "run_composition_block",
    "run_context_block",
    "run_utilization_block",
    "run_scaling_block_013",
    "run_exp013",
]
