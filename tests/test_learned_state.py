import tempfile
import unittest
from pathlib import Path

from air_core.air_ir import build_rule_program, serialize_binary_ir
from air_core.learned_state import (
    SQLiteSkillIndex,
    benchmark_naive_retrieval,
    composition_candidate_counts,
    generate_skill_records,
    query_for_record,
)


class LearnedStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.artifact = serialize_binary_ir(build_rule_program({"A": 7, "B": 2}, 3, 2))

    def test_naive_and_indexed_retrieval_find_same_exact_skill(self) -> None:
        records = generate_skill_records(1_000, self.artifact)
        query = query_for_record(records[-1])
        naive = benchmark_naive_retrieval(records, query, repeats=2)
        with tempfile.TemporaryDirectory() as directory:
            index = SQLiteSkillIndex(Path(directory) / "skills.sqlite3")
            index.insert(records)
            indexed = index.benchmark(query, repeats=2)
            index.close()
        self.assertEqual(naive["found_skill_ids"], [records[-1].skill_id])
        self.assertEqual(indexed["found_skill_ids"], [records[-1].skill_id])
        self.assertEqual(naive["candidates_examined"], 1_000)
        self.assertEqual(indexed["candidates_examined"], 1)
        self.assertLess(indexed["active_learned_state_bytes"], sum(item.stored_bytes() for item in records))

    def test_type_and_category_filters_reduce_composition_upper_bound(self) -> None:
        counts = composition_candidate_counts(generate_skill_records(1_000, self.artifact))["counts"]
        for item in counts[1:]:
            self.assertLess(item["after_type_filter"], item["brute_force_upper_bound"])
            self.assertLess(item["after_category_and_type_filter"], item["after_type_filter"])


if __name__ == "__main__":
    unittest.main()
