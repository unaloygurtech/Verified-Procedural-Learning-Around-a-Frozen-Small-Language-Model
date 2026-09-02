import unittest

from air_core.exp005 import (
    EDGE_005,
    HELD_OUT_005,
    TRAIN_005,
    VALIDATION_005,
    Expr,
    Program,
    TypeCheckError,
    corrupted_programs_005,
    execute_expr,
    execute_program,
    field,
    literal,
    synthesize_005,
    type_of,
)


class Experiment005Tests(unittest.TestCase):
    def test_partitions_are_disjoint_and_heldout_is_new(self) -> None:
        ids = [case.case_id for case in TRAIN_005 + VALIDATION_005 + EDGE_005 + HELD_OUT_005]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(TRAIN_005), 8)
        self.assertEqual(len(VALIDATION_005), 8)
        self.assertEqual(len(HELD_OUT_005), 12)

    def test_type_contracts_reject_invalid_program(self) -> None:
        invalid = Program(Expr("ADD", args=(Expr("FIELD", value="x"), Expr("FIELD", value="y"))), Expr("LITERAL", value="bad"))
        with self.assertRaises(TypeCheckError):
            execute_program(invalid, {"recipe": "sum", "x": "1", "y": "2"})

    def test_synthesis_discovers_typed_programs(self) -> None:
        candidate = synthesize_005()
        self.assertEqual(set(candidate.programs), {"sum", "difference", "join", "reverse"})
        self.assertGreater(candidate.stats.candidate_search_count, 0)
        self.assertGreater(candidate.stats.type_invalid_candidate_count, 0)
        for program in candidate.programs.values():
            self.assertIn("value", program.to_dict())
            self.assertIn("label", program.to_dict())

    def test_executable_candidate_passes_validation_and_edges(self) -> None:
        candidate = synthesize_005()
        for case in VALIDATION_005 + EDGE_005:
            self.assertEqual(execute_program(candidate.programs[case.recipe], case.payload()), case.output)

    def test_corruption_is_typed_but_semantically_wrong(self) -> None:
        corrupted = corrupted_programs_005()
        for recipe, program in corrupted.items():
            # Corrupted candidates remain syntactically and type valid.
            execute_program(program, {"recipe": recipe, "x": "2", "y": "1"})
        failures = sum(
            execute_program(corrupted[case.recipe], case.payload()) != case.output
            for case in VALIDATION_005
        )
        self.assertGreaterEqual(failures, 4)

    def test_replace_extension_is_typed_and_deterministic(self) -> None:
        expression = Expr("REPLACE", args=(field("input"), literal(" "), literal("_")))
        self.assertEqual(type_of(expression, {"input": "String"}), "String")
        self.assertEqual(execute_expr(expression, {"input": "A B"}), "A_B")


if __name__ == "__main__":
    unittest.main()
