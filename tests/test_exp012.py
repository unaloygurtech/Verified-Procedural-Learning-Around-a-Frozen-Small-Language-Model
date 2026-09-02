import tempfile
import unittest
from pathlib import Path

from air_core.air_ir import compile_python_rule_subset, execute_program
from air_core.exp012 import (
    POOL_SIZES_012,
    ROBUSTNESS_SEEDS_012,
    RULE_HELDOUT_012,
    RULE_SOURCE_012,
    _learn_family_resumable_012,
    make_document_pool_012,
    make_robustness_families_012,
    run_storage_block_012,
)
from air_core.model_client import ModelUnavailable
from air_core.store import ExperimentStore


class TimeoutClient:
    timeout_seconds = 0.01

    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, *args, **kwargs):
        self.calls += 1
        raise ModelUnavailable("model runtime unavailable: test timeout")


class Experiment012Tests(unittest.TestCase):
    def test_five_disjoint_families_exist_for_each_seed(self) -> None:
        all_case_ids: list[str] = []
        operation_names: set[str] = set()
        for seed in ROBUSTNESS_SEEDS_012:
            families = make_robustness_families_012(seed)
            self.assertEqual(len(families), 5)
            for family in families:
                self.assertEqual((len(family.discovery), len(family.validation), len(family.edge), len(family.heldout)), (4, 3, 3, 8))
                all_case_ids.extend(case.case_id for case in family.discovery + family.validation + family.edge + family.heldout)
                operation_names.add(family.family_id.rsplit("-", 1)[-1])
        self.assertEqual(len(operation_names), 15)
        self.assertEqual(len(all_case_ids), len(set(all_case_ids)))

    def test_document_pools_have_exact_size_and_required_distractors(self) -> None:
        family = make_robustness_families_012(ROBUSTNESS_SEEDS_012[0])[0]
        for size in POOL_SIZES_012:
            pool, correct_id = make_document_pool_012(family, size, ROBUSTNESS_SEEDS_012[0])
            self.assertEqual(len(pool), size)
            self.assertIn(correct_id, {item["doc_id"] for item in pool})
            kinds = {item["kind"] for item in pool}
            self.assertTrue({"correct", "related_wrong", "terminology_distractor", "unrelated"}.issubset(kinds))

    def test_storage_representations_are_equivalent_and_controls_reject(self) -> None:
        report = run_storage_block_012()
        self.assertTrue(report["semantic_equivalence"]["passed"])
        self.assertTrue(all(item["semantic_equivalence"]["passed"] for item in report["representations"]))
        self.assertTrue(all(item["rejected"] for item in report["safety_controls"].values()))
        sizes = {item["representation"]: item["serialized_bytes_per_skill"] for item in report["representations"]}
        self.assertLess(sizes["binary_air_ir"], sizes["json_typed_ast"])

    def test_rule_source_compiles_and_solves_heldout(self) -> None:
        program = compile_python_rule_subset(RULE_SOURCE_012)
        self.assertTrue(all(execute_program(program, case.input_text) == case.expected for case in RULE_HELDOUT_012))

    def test_runtime_timeout_is_checkpointed_and_not_retried(self) -> None:
        family = make_robustness_families_012(ROBUSTNESS_SEEDS_012[0])[0]
        client = TimeoutClient()
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(str(Path(directory) / "air.db"))
            skill, attempts = _learn_family_resumable_012(client, store, family, "test-skill", 999)
            self.assertIsNone(skill)
            self.assertEqual(len(attempts), 1)
            self.assertEqual(client.calls, 1)
            skill, attempts = _learn_family_resumable_012(client, store, family, "test-skill", 999)
            self.assertIsNone(skill)
            self.assertEqual(len(attempts), 1)
            self.assertEqual(client.calls, 1)


if __name__ == "__main__":
    unittest.main()
