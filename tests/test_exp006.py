import unittest

from air_core.exp005 import Expr, TypeCheckError, binary, field
from air_core.exp006 import (
    EDGE_006,
    HELD_OUT_006,
    IMPOSSIBLE_VALIDATION_006,
    PRIMITIVE_SETS_006,
    VALID_COMPOSITION_VALIDATION_006,
    VALIDATION_006,
    CompositionPlan,
    _corrupted_plan,
    execute_composition,
    iter_composition_candidates,
    learn_primitive_library,
    raw_primitive_experiences_006,
    search_compositions,
)


class Experiment006Tests(unittest.TestCase):
    def test_partitions_are_disjoint_and_composed_examples_are_absent(self) -> None:
        ids = [case.case_id for case in VALIDATION_006 + EDGE_006 + HELD_OUT_006]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(PRIMITIVE_SETS_006), 5)
        self.assertEqual(len(HELD_OUT_006), 12)
        raw = raw_primitive_experiences_006()
        self.assertNotIn("m2", raw)
        self.assertNotIn("m3", raw)
        self.assertNotIn("m4", raw)
        self.assertNotIn("skill-", raw)

    def test_library_learns_independent_typed_artifacts(self) -> None:
        skills, stats = learn_primitive_library()
        self.assertEqual([skill.skill_id for skill in skills], ["skill-1", "skill-2", "skill-3", "skill-4", "skill-5"])
        self.assertGreater(stats.candidate_search_count, 0)
        self.assertGreater(stats.type_invalid_candidate_count, 0)
        self.assertEqual({skill.source_set_id for skill in skills}, {item.set_id for item in PRIMITIVE_SETS_006})
        self.assertEqual({skill.version for skill in skills}, {1})

    def test_type_invalid_composition_is_rejected_before_execution(self) -> None:
        skills, _ = learn_primitive_library()
        invalid = CompositionPlan(left_chain=("skill-4",))
        with self.assertRaises(TypeCheckError):
            execute_composition(invalid, skills, VALID_COMPOSITION_VALIDATION_006[0])

    def test_search_discovers_depth_two_and_three_without_task_mapping(self) -> None:
        skills, _ = learn_primitive_library()
        plans, no_valid, stats = search_compositions(skills, VALIDATION_006)
        self.assertEqual(set(plans), {"m2", "m3", "m4"})
        self.assertEqual(no_valid, {"mx": "no valid composition"})
        self.assertEqual(plans["m2"].depth(), 2)
        self.assertEqual(plans["m3"].depth(), 3)
        self.assertEqual(plans["m4"].depth(), 3)
        self.assertGreater(stats.candidate_composition_count, 0)
        self.assertGreater(stats.type_invalid_composition_count, 0)
        self.assertGreater(stats.semantic_validation_rejected_count, 0)
        self.assertEqual(stats.ambiguous_composition_count, 0)
        self.assertNotEqual(plans["m3"].binary_skill, plans["m4"].binary_skill)
        self.assertGreater(stats.candidate_evaluation_count, stats.candidate_composition_count)

    def test_edge_cases_and_impossible_task_execute_safely(self) -> None:
        skills, _ = learn_primitive_library()
        plans, _, _ = search_compositions(skills, VALIDATION_006)
        for case in VALID_COMPOSITION_VALIDATION_006 + EDGE_006:
            if case.expected and case.expected.get("status") == "ok":
                self.assertEqual(execute_composition(plans[case.task_token], skills, case), case.expected)
        for case in IMPOSSIBLE_VALIDATION_006:
            self.assertNotIn(case.task_token, plans)

    def test_corrupted_type_valid_compositions_fail_semantic_validation(self) -> None:
        skills, _ = learn_primitive_library()
        plans, _, _ = search_compositions(skills, VALIDATION_006)
        corrupted = _corrupted_plan(plans, skills)
        for token, plan in corrupted.items():
            # The corruption is still statically well typed.
            for case in VALID_COMPOSITION_VALIDATION_006:
                if case.task_token == token:
                    execute_composition(plan, skills, case)
        failures = sum(
            execute_composition(corrupted[case.task_token], skills, case) != case.expected
            for case in VALID_COMPOSITION_VALIDATION_006
            if case.task_token in corrupted
        )
        self.assertGreaterEqual(failures, 4)

    def test_candidate_space_is_artifact_driven(self) -> None:
        skills, _ = learn_primitive_library()
        candidates = tuple(iter_composition_candidates(skills))
        self.assertGreater(len(candidates), 100)
        self.assertTrue(any(plan.left_chain == ("skill-2", "skill-3") for plan in candidates))
        self.assertTrue(any(plan.binary_skill == "skill-5" for plan in candidates))


if __name__ == "__main__":
    unittest.main()
