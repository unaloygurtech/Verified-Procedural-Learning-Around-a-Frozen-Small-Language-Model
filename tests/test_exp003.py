import unittest

from air_core.exp003 import (
    HELD_OUT_003,
    TRAIN_003,
    VALIDATION_003,
    MANUAL_SKILL_003,
    generate_skill_003,
    raw_experiences_003,
    routed_context_003,
)


class Experiment003Tests(unittest.TestCase):
    def test_partitions_are_disjoint(self) -> None:
        ids = [case.case_id for case in TRAIN_003 + VALIDATION_003 + HELD_OUT_003]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(TRAIN_003), 16)
        self.assertEqual(len(VALIDATION_003), 8)
        self.assertEqual(len(HELD_OUT_003), 12)

    def test_family_labels_are_not_in_raw_prompt(self) -> None:
        text = raw_experiences_003()
        self.assertNotIn("signal-normalization", text)
        self.assertNotIn("ticket-triage", text)
        self.assertNotIn("inventory-pricing", text)
        self.assertNotIn("message-routing", text)
        self.assertIn('"verified_output"', text)

    def test_symbolic_consolidator_discovers_four_schemas(self) -> None:
        candidate = generate_skill_003()
        self.assertEqual(candidate.method, "schema-cluster-and-bounded-dsl")
        self.assertEqual(candidate.discovered_families, ("family-1", "family-2", "family-3", "family-4"))
        self.assertIn("Route by exact field schema", candidate.body)
        self.assertIn("discount lookup none=>0, save5=>5, save10=>10", candidate.body)

    def test_router_selects_one_rule_without_exposing_family_name(self) -> None:
        candidate = generate_skill_003()
        case = next(item for item in VALIDATION_003 if item.family == "inventory-pricing")
        routed = routed_context_003(case, candidate.body, "generated_skill")
        self.assertIn("sku_ref", routed)
        self.assertNotIn("family-", routed)
        self.assertNotIn("ticket", routed)

    def test_manual_skill_covers_all_families(self) -> None:
        for field in ("code", "title", "unit_price", "body"):
            self.assertIn(field, MANUAL_SKILL_003)


if __name__ == "__main__":
    unittest.main()
