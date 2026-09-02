import unittest

from air_core.exp006 import CompositionPlan, execute_composition, learn_primitive_library
from air_core.exp007 import (
    GAP_HELD_OUT_007,
    GAP_VALIDATION_007,
    MISSING_SKILL_EDGE_007,
    MISSING_SKILL_TRAINING_007,
    MISSING_SKILL_VALIDATION_007,
    _corrupted_gap_plans,
    _skill_gate,
    diagnose_capability_gaps,
    raw_gap_context_007,
    raw_missing_experiences_007,
    search_gap_compositions,
    synthesize_missing_skill_007,
)


class Experiment007Tests(unittest.TestCase):
    def test_partitions_and_raw_context_have_no_composed_examples(self) -> None:
        ids = [case.case_id for case in GAP_VALIDATION_007 + GAP_HELD_OUT_007]
        self.assertEqual(len(ids), len(set(ids)))
        raw = raw_missing_experiences_007()
        self.assertNotIn("g1", raw)
        self.assertNotIn("g2", raw)
        self.assertNotIn("g3", raw)
        self.assertNotIn("skill-6", raw)
        self.assertNotIn("g1", raw_gap_context_007())

    def test_existing_library_reports_positive_capability_gaps(self) -> None:
        base, _ = learn_primitive_library()
        plans, no_valid, stats = search_gap_compositions(base, GAP_VALIDATION_007)
        diagnoses = diagnose_capability_gaps(GAP_VALIDATION_007, plans, no_valid)
        self.assertEqual(plans, {})
        self.assertEqual({item.task_token for item in diagnoses if item.status == "gap_detected"}, {"g1", "g2", "g3"})
        self.assertEqual(no_valid["gx"], "no valid composition")
        self.assertGreater(stats.type_invalid_composition_count, 0)

    def test_missing_skill_is_synthesized_and_gated(self) -> None:
        skill, stats = synthesize_missing_skill_007()
        self.assertEqual(skill.skill_id, "skill-6")
        self.assertEqual(skill.source_set_id, "missing-set-007")
        self.assertEqual(skill.expression.op, "REPLACE")
        self.assertEqual(_skill_gate("training", skill, MISSING_SKILL_TRAINING_007).accuracy, 1.0)
        self.assertEqual(_skill_gate("validation", skill, MISSING_SKILL_VALIDATION_007).accuracy, 1.0)
        self.assertEqual(_skill_gate("edge", skill, MISSING_SKILL_EDGE_007).accuracy, 1.0)
        self.assertGreater(stats.semantic_training_rejected_count, 0)
        self.assertGreater(stats.type_invalid_candidate_count, 0)

    def test_extended_search_finds_ordered_compositions(self) -> None:
        base, _ = learn_primitive_library()
        missing, _ = synthesize_missing_skill_007()
        skills = base + (missing,)
        plans, no_valid, stats = search_gap_compositions(skills, GAP_VALIDATION_007)
        self.assertEqual(set(plans), {"g1", "g2", "g3"})
        self.assertEqual(no_valid, {"gx": "no valid composition"})
        self.assertEqual(plans["g1"].depth(), 3)
        self.assertEqual(plans["g2"].depth(), 5)
        self.assertEqual(plans["g3"].depth(), 5)
        self.assertNotEqual(plans["g2"].binary_skill, plans["g3"].binary_skill)
        self.assertGreater(stats.candidate_evaluation_count, stats.candidate_composition_count)
        for case in GAP_VALIDATION_007[:3] + GAP_HELD_OUT_007[:9]:
            self.assertEqual(execute_composition(plans[case.task_token], skills, case), case.expected)

    def test_corrupted_composition_is_typed_but_rejected(self) -> None:
        base, _ = learn_primitive_library()
        missing, _ = synthesize_missing_skill_007()
        skills = base + (missing,)
        plans, _, _ = search_gap_compositions(skills, GAP_VALIDATION_007)
        corrupted = _corrupted_gap_plans(plans, skills)
        failures = 0
        for case in GAP_VALIDATION_007[:3]:
            result = execute_composition(corrupted[case.task_token], skills, case)
            failures += int(result != case.expected)
        self.assertGreaterEqual(failures, 2)


if __name__ == "__main__":
    unittest.main()
