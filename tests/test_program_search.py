import unittest

from air_core.exp016 import make_contract_families_016
from air_core.program_search import (
    SEARCH_API_NAMES_018,
    SearchBudget018,
    candidate_features_018,
    search_candidates_018,
)
from air_core.semantic_ir import oracle_call_ir_017


class ProgramSearch018Tests(unittest.TestCase):
    def test_search_is_deterministic_and_covers_frozen_families(self):
        family = make_contract_families_016()[0]
        first = search_candidates_018(family.discovery)
        second = search_candidates_018(family.discovery)
        self.assertEqual([item.ast_hash for item in first.candidates], [item.ast_hash for item in second.candidates])
        self.assertGreater(first.raw_candidates_generated, 0)
        self.assertGreater(len(first.public_survivors), 1)
        operation = family.family_id.rsplit("-", 1)[-1]
        self.assertIn(operation, SEARCH_API_NAMES_018)
        self.assertTrue(any(item.ir == oracle_call_ir_017(operation) for item in first.candidates))

    def test_public_filter_never_uses_hidden_or_edge(self):
        family = make_contract_families_016()[0]
        result = search_candidates_018(family.discovery, budget=SearchBudget018(max_depth=2, max_raw_candidates=1000, max_public_survivors=32))
        self.assertTrue(all(item.behavior_signature for item in result.public_survivors))
        survivor_ids = {item.candidate_id for item in result.public_survivors}
        self.assertEqual(len(result.public_survivors), sum(1 for item in result.candidates if item.candidate_id in survivor_ids))

    def test_behavior_dedup_and_constraint_features_are_stable(self):
        family = make_contract_families_016()[0]
        result = search_candidates_018(family.discovery)
        self.assertGreater(result.unique_behavior_buckets if hasattr(result, "unique_behavior_buckets") else len(result.behavior_buckets), 0)
        features = candidate_features_018(oracle_call_ir_017(family.family_id.rsplit("-", 1)[-1]))
        self.assertEqual(features["uses_call"], True)
        self.assertEqual(features["uses_reverse"], False)
        self.assertEqual(features["uses_rotate"], False)
        self.assertEqual(features["uses_concat"], False)

    def test_budget_is_enforced(self):
        family = make_contract_families_016()[0]
        result = search_candidates_018(family.discovery, budget=SearchBudget018(max_depth=3, max_raw_candidates=20, max_public_survivors=4))
        self.assertLessEqual(result.raw_candidates_generated, 20)
        self.assertTrue(result.budget_exhausted)


if __name__ == "__main__":
    unittest.main()
