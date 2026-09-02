import unittest

from air_core.exp004 import (
    HELD_OUT_004,
    TRAIN_004,
    VALIDATION_004,
    generate_skill_004,
    discover_rules_004,
    raw_experiences_004,
    routed_context_004,
)


class Experiment004Tests(unittest.TestCase):
    def test_partitions_are_disjoint(self) -> None:
        ids = [case.case_id for case in TRAIN_004 + VALIDATION_004 + HELD_OUT_004]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(TRAIN_004), 8)
        self.assertEqual(len(VALIDATION_004), 8)
        self.assertEqual(len(HELD_OUT_004), 12)

    def test_all_examples_share_schema_and_raw_hides_family(self) -> None:
        self.assertEqual({tuple(sorted(case.payload())) for case in TRAIN_004}, {("recipe", "x", "y")})
        self.assertEqual({tuple(sorted(case.output)) for case in TRAIN_004}, {("label", "value")})
        text = raw_experiences_004()
        for hidden in ("arithmetic-total", "arithmetic-gap", "text-forward", "text-reverse"):
            self.assertNotIn(hidden, text)

    def test_consolidator_infers_content_rules(self) -> None:
        rules = discover_rules_004()
        self.assertEqual([rule.recipe for rule in rules], ["difference", "join", "reverse", "sum"])
        candidate = generate_skill_004()
        self.assertEqual(candidate.method, "content-discriminator-and-bounded-dsl")
        self.assertIn("recipe=difference", candidate.body)
        self.assertIn("trim(lower(x))", candidate.body)

    def test_ambiguous_recipe_is_rejected(self) -> None:
        conflict = TRAIN_004 + (TRAIN_004[0].__class__("conflict", "bad", "sum", "7", "5", {"value": 99, "label": "total"}),)
        with self.assertRaises(ValueError):
            discover_rules_004(conflict)

    def test_router_selects_only_recipe_rule(self) -> None:
        candidate = generate_skill_004()
        case = VALIDATION_004[0]
        routed = routed_context_004(case, candidate.body, "generated_skill")
        self.assertIn("recipe=sum", routed)
        self.assertNotIn("recipe=join", routed)


if __name__ == "__main__":
    unittest.main()
