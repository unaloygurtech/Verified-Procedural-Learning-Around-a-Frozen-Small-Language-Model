"""Learned-state storage primitives used by Experiment 0012.

The module keeps retrieval metadata separate from executable and cold audit
state.  SQLite is intentionally sufficient for v0; no vector database or new
service is required.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys
import time
from typing import Any, Iterable


STORAGE_VERSION = 1


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class HotState:
    skill_id: str
    input_contract: str
    output_contract: str
    category: str
    trust: str
    status: str
    version: int
    retrieval_descriptor: str
    artifact_format: str
    artifact_pointer: str


@dataclass(frozen=True)
class WarmState:
    lexical_terms: tuple[str, ...]
    usage_count: int
    success_count: int
    relations: tuple[str, ...]
    rank_bias: float
    operation_family: str


@dataclass(frozen=True)
class ColdState:
    raw_experiences: tuple[dict[str, str], ...]
    documentation: str
    provenance: dict[str, Any]
    validation_history: tuple[dict[str, Any], ...]
    edge_tests: tuple[dict[str, str], ...]
    source_blob: str


@dataclass(frozen=True)
class DerivedArtifactProvenance:
    source_skill_id: str
    source_version: int
    source_sha256: str
    ir_version: int
    compiler_version: str
    semantic_equivalence_test_ids: tuple[str, ...]
    created_at: str
    activation_status: str
    source_immutable: bool


@dataclass(frozen=True)
class LayeredSkillState:
    hot: HotState
    warm: WarmState
    cold: ColdState
    artifact: bytes

    def layer_bytes(self) -> dict[str, int]:
        return {
            "hot": len(canonical_json_bytes(asdict(self.hot))) + len(self.artifact),
            "warm": len(canonical_json_bytes(asdict(self.warm))),
            "cold": len(canonical_json_bytes(asdict(self.cold))),
        }


@dataclass(frozen=True)
class RetrievalQuery:
    category: str
    input_contract: str
    output_contract: str
    operation_family: str
    lexical_descriptor: str


@dataclass(frozen=True)
class SyntheticSkillRecord:
    skill_id: str
    category: str
    input_contract: str
    output_contract: str
    operation_family: str
    version: int
    trust: str
    lexical_descriptor: str
    artifact: bytes
    cold_bytes: int

    def index_payload(self) -> bytes:
        return canonical_json_bytes(
            {
                "skill_id": self.skill_id,
                "category": self.category,
                "input_contract": self.input_contract,
                "output_contract": self.output_contract,
                "operation_family": self.operation_family,
                "version": self.version,
                "trust": self.trust,
                "lexical_descriptor": self.lexical_descriptor,
            }
        )

    def hot_warm_bytes(self) -> int:
        return len(self.index_payload()) + len(self.artifact)

    def stored_bytes(self) -> int:
        return self.hot_warm_bytes() + self.cold_bytes


def deep_size(value: Any, seen: set[int] | None = None) -> int:
    """Deterministic Python-object working-set proxy, not process RSS."""

    visited = seen if seen is not None else set()
    object_id = id(value)
    if object_id in visited:
        return 0
    visited.add(object_id)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(deep_size(key, visited) + deep_size(item, visited) for key, item in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(deep_size(item, visited) for item in value)
    elif hasattr(value, "__dict__"):
        size += deep_size(vars(value), visited)
    return size


def generate_skill_records(count: int, artifact: bytes) -> list[SyntheticSkillRecord]:
    categories = tuple(f"domain-{index:02d}" for index in range(32))
    types = ("str", "int", "json", "list[int]")
    operations = tuple(f"operation-{index:03d}" for index in range(64))
    records: list[SyntheticSkillRecord] = []
    for index in range(count):
        category = categories[index % len(categories)]
        input_contract = types[index % len(types)]
        output_contract = types[(index * 3 + 1) % len(types)]
        operation = operations[index % len(operations)]
        records.append(
            SyntheticSkillRecord(
                skill_id=f"synthetic-skill-{index:06d}",
                category=category,
                input_contract=input_contract,
                output_contract=output_contract,
                operation_family=operation,
                version=1 + (index % 3),
                trust="verified" if index % 5 else "candidate",
                lexical_descriptor=f"opaque-token-{index:06d}",
                artifact=artifact,
                cold_bytes=96 + (index % 17),
            )
        )
    return records


def query_for_record(record: SyntheticSkillRecord) -> RetrievalQuery:
    return RetrievalQuery(record.category, record.input_contract, record.output_contract, record.operation_family, record.lexical_descriptor)


def _matches(record: SyntheticSkillRecord, query: RetrievalQuery) -> bool:
    return (
        record.category == query.category
        and record.input_contract == query.input_contract
        and record.output_contract == query.output_contract
        and record.operation_family == query.operation_family
        and record.lexical_descriptor == query.lexical_descriptor
    )


def _latency_summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p50 = statistics.median(ordered) if ordered else 0.0
    p95 = ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))] if ordered else 0.0
    return {"mean_us": statistics.mean(samples) * 1_000_000 if samples else 0.0, "p50_us": p50 * 1_000_000, "p95_us": p95 * 1_000_000}


def benchmark_naive_retrieval(records: list[SyntheticSkillRecord], query: RetrievalQuery, repeats: int = 7, top_k: int = 3) -> dict[str, Any]:
    samples: list[float] = []
    result: list[SyntheticSkillRecord] = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = [record for record in records if _matches(record, query)][:top_k]
        samples.append(time.perf_counter() - started)
    bytes_scanned = sum(len(record.index_payload()) for record in records)
    loaded_bytes = sum(record.hot_warm_bytes() for record in result)
    return {
        "strategy": "naive_linear_full_scan",
        "found_skill_ids": [record.skill_id for record in result],
        "candidates_examined": len(records),
        "skills_fully_loaded": len(result),
        "bytes_read_query_logical": bytes_scanned + loaded_bytes,
        "active_learned_state_bytes": loaded_bytes,
        "ram_working_set_proxy_bytes": deep_size(result),
        "retrieval_latency": _latency_summary(samples),
    }


class SQLiteSkillIndex:
    """Category/type/exact-aware metadata index with separately loaded blobs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE skill_meta (
                skill_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                input_contract TEXT NOT NULL,
                output_contract TEXT NOT NULL,
                operation_family TEXT NOT NULL,
                version INTEGER NOT NULL,
                trust TEXT NOT NULL,
                lexical_descriptor TEXT NOT NULL
            );
            CREATE TABLE skill_artifact (
                skill_id TEXT PRIMARY KEY,
                artifact BLOB NOT NULL,
                cold_bytes INTEGER NOT NULL
            );
            CREATE INDEX idx_skill_filter ON skill_meta(
                category, input_contract, output_contract, operation_family,
                lexical_descriptor, trust
            );
            """
        )

    def close(self) -> None:
        self.connection.close()

    def insert(self, records: Iterable[SyntheticSkillRecord]) -> None:
        rows = list(records)
        self.connection.executemany(
            "INSERT INTO skill_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(item.skill_id, item.category, item.input_contract, item.output_contract, item.operation_family, item.version, item.trust, item.lexical_descriptor) for item in rows],
        )
        self.connection.executemany(
            "INSERT INTO skill_artifact VALUES (?, ?, ?)",
            [(item.skill_id, item.artifact, item.cold_bytes) for item in rows],
        )
        self.connection.commit()

    def index_bytes(self) -> int:
        try:
            row = self.connection.execute(
                "SELECT COALESCE(SUM(pgsize), 0) AS total FROM dbstat WHERE name IN ('idx_skill_filter', 'sqlite_autoindex_skill_meta_1', 'sqlite_autoindex_skill_artifact_1')"
            ).fetchone()
            return int(row["total"])
        except sqlite3.OperationalError:
            page_count = int(self.connection.execute("PRAGMA page_count").fetchone()[0])
            page_size = int(self.connection.execute("PRAGMA page_size").fetchone()[0])
            return page_count * page_size

    def retrieve(self, query: RetrievalQuery, top_k: int = 3) -> list[SyntheticSkillRecord]:
        meta = self.connection.execute(
            """
            SELECT * FROM skill_meta
            WHERE category = ? AND input_contract = ? AND output_contract = ?
              AND operation_family = ? AND lexical_descriptor = ?
            ORDER BY CASE trust WHEN 'verified' THEN 0 ELSE 1 END, version DESC
            LIMIT ?
            """,
            (query.category, query.input_contract, query.output_contract, query.operation_family, query.lexical_descriptor, top_k),
        ).fetchall()
        result: list[SyntheticSkillRecord] = []
        for row in meta:
            artifact = self.connection.execute("SELECT artifact, cold_bytes FROM skill_artifact WHERE skill_id = ?", (row["skill_id"],)).fetchone()
            result.append(
                SyntheticSkillRecord(
                    row["skill_id"], row["category"], row["input_contract"], row["output_contract"],
                    row["operation_family"], int(row["version"]), row["trust"], row["lexical_descriptor"],
                    bytes(artifact["artifact"]), int(artifact["cold_bytes"]),
                )
            )
        return result

    def benchmark(self, query: RetrievalQuery, repeats: int = 7, top_k: int = 3) -> dict[str, Any]:
        samples: list[float] = []
        result: list[SyntheticSkillRecord] = []
        for _ in range(repeats):
            started = time.perf_counter()
            result = self.retrieve(query, top_k)
            samples.append(time.perf_counter() - started)
        loaded_bytes = sum(record.hot_warm_bytes() for record in result)
        page_size = int(self.connection.execute("PRAGMA page_size").fetchone()[0])
        total_rows = int(self.connection.execute("SELECT COUNT(*) FROM skill_meta").fetchone()[0])
        btree_depth_proxy = max(1, math.ceil(math.log(max(2, total_rows), 128)))
        return {
            "strategy": "sqlite_category_type_exact_index",
            "found_skill_ids": [record.skill_id for record in result],
            "candidates_examined": len(result),
            "skills_fully_loaded": len(result),
            "bytes_read_query_logical": loaded_bytes + btree_depth_proxy * page_size,
            "active_learned_state_bytes": loaded_bytes,
            "ram_working_set_proxy_bytes": deep_size(result),
            "index_page_traversal_proxy": btree_depth_proxy,
            "retrieval_latency": _latency_summary(samples),
        }


def composition_candidate_counts(records: Iterable[SyntheticSkillRecord]) -> dict[str, Any]:
    items = list(records)
    types = sorted({item.input_contract for item in items} | {item.output_contract for item in items})

    def matrix_for(group: Iterable[SyntheticSkillRecord]) -> dict[tuple[str, str], int]:
        matrix = {(left, right): 0 for left in types for right in types}
        for item in group:
            matrix[(item.input_contract, item.output_contract)] += 1
        return matrix

    def path_count(matrix: dict[tuple[str, str], int], depth: int) -> int:
        current = {(left, right): matrix[(left, right)] for left in types for right in types}
        if depth == 1:
            return sum(current.values())
        for _ in range(1, depth):
            following = {(left, right): 0 for left in types for right in types}
            for left in types:
                for middle in types:
                    for right in types:
                        following[(left, right)] += current[(left, middle)] * matrix[(middle, right)]
            current = following
        return sum(current.values())

    full = matrix_for(items)
    category_groups: dict[str, list[SyntheticSkillRecord]] = {}
    for item in items:
        category_groups.setdefault(item.category, []).append(item)
    n = len(items)
    by_depth: list[dict[str, int]] = []
    for depth in (1, 2, 3):
        brute_force = n ** depth
        type_filtered = path_count(full, depth)
        category_filtered = sum(path_count(matrix_for(group), depth) for group in category_groups.values())
        by_depth.append(
            {
                "max_depth": depth,
                "brute_force_upper_bound": brute_force,
                "after_type_filter": type_filtered,
                "after_category_and_type_filter": category_filtered,
            }
        )
    return {"artifact_reuse_allowed_in_count": True, "counts": by_depth}
