import unittest

from air_core.cli import build_parser


class CliTests(unittest.TestCase):
    def test_auto_skill_strategy_is_parsed(self) -> None:
        args = build_parser().parse_args(
            ["auto-skill", "--strategy", "decomposed", "--heldout-limit", "2"]
        )
        self.assertEqual(args.command, "auto-skill")
        self.assertEqual(args.strategy, "decomposed")
        self.assertEqual(args.heldout_limit, 2)

    def test_manual_benchmark_has_no_strategy(self) -> None:
        args = build_parser().parse_args(["benchmark"])
        self.assertEqual(args.command, "benchmark")
        self.assertFalse(hasattr(args, "strategy"))

    def test_symbolic_strategy_is_available(self) -> None:
        args = build_parser().parse_args(["auto-skill", "--strategy", "symbolic"])
        self.assertEqual(args.strategy, "symbolic")

    def test_multi_family_experiment_is_available(self) -> None:
        args = build_parser().parse_args(["experiment-003", "--heldout-limit", "2"])
        self.assertEqual(args.command, "experiment-003")
        self.assertEqual(args.heldout_limit, 2)

    def test_same_schema_experiment_is_available(self) -> None:
        args = build_parser().parse_args(["experiment-004", "--heldout-limit", "2"])
        self.assertEqual(args.command, "experiment-004")
        self.assertEqual(args.heldout_limit, 2)

    def test_typed_experiment_is_available(self) -> None:
        args = build_parser().parse_args(["experiment-005", "--heldout-limit", "2"])
        self.assertEqual(args.command, "experiment-005")
        self.assertEqual(args.heldout_limit, 2)

    def test_compositional_experiment_is_available(self) -> None:
        args = build_parser().parse_args(["experiment-006", "--heldout-limit", "2"])
        self.assertEqual(args.command, "experiment-006")
        self.assertEqual(args.heldout_limit, 2)

    def test_capability_gap_experiment_is_available(self) -> None:
        args = build_parser().parse_args(["experiment-007", "--heldout-limit", "2"])
        self.assertEqual(args.command, "experiment-007")
        self.assertEqual(args.heldout_limit, 2)

    def test_python_gap_experiment_is_available(self) -> None:
        args = build_parser().parse_args(["experiment-008", "--heldout-limit", "2"])
        self.assertEqual(args.command, "experiment-008")
        self.assertEqual(args.heldout_limit, 2)

    def test_frozen_generic_experiment_is_available(self) -> None:
        args = build_parser().parse_args(["experiment-009", "--repeats", "3", "--heldout-limit", "2"])
        self.assertEqual(args.command, "experiment-009")
        self.assertEqual(args.repeats, 3)
        self.assertEqual(args.heldout_limit, 2)

    def test_novel_synthetic_api_experiment_is_available(self) -> None:
        args = build_parser().parse_args(["experiment-010", "--heldout-limit", "2"])
        self.assertEqual(args.command, "experiment-010")
        self.assertEqual(args.heldout_limit, 2)

    def test_retrieval_efficiency_experiment_is_available(self) -> None:
        args = build_parser().parse_args(["experiment-011", "--heldout-limit", "2"])
        self.assertEqual(args.command, "experiment-011")
        self.assertEqual(args.heldout_limit, 2)

    def test_storage_scaling_experiment_is_available(self) -> None:
        args = build_parser().parse_args(["experiment-012", "--heldout-limit", "2"])
        self.assertEqual(args.command, "experiment-012")
        self.assertEqual(args.heldout_limit, 2)

    def test_model_utilization_experiment_is_available(self) -> None:
        args = build_parser().parse_args(["experiment-014", "--heldout-limit", "1"])
        self.assertEqual(args.command, "experiment-014")
        self.assertEqual(args.heldout_limit, 1)

    def test_structured_acquisition_experiment_is_available(self) -> None:
        args = build_parser().parse_args(["experiment-015", "--resume", "checkpoint.json"])
        self.assertEqual(args.command, "experiment-015")
        self.assertEqual(args.resume_from, "checkpoint.json")

    def test_contract_induction_experiment_is_available(self) -> None:
        args = build_parser().parse_args(["experiment-016", "--resume", "checkpoint.json"])
        self.assertEqual(args.command, "experiment-016")
        self.assertEqual(args.resume_from, "checkpoint.json")

    def test_semantic_boundary_experiment_is_available(self) -> None:
        args = build_parser().parse_args(["experiment-017", "--resume", "checkpoint.json"])
        self.assertEqual(args.command, "experiment-017")
        self.assertEqual(args.resume_from, "checkpoint.json")

    def test_verified_candidate_search_experiment_is_available(self) -> None:
        args = build_parser().parse_args(["experiment-018", "--resume", "checkpoint.json"])
        self.assertEqual(args.command, "experiment-018")
        self.assertEqual(args.resume_from, "checkpoint.json")

    def test_canonicalization_and_grounding_experiment_is_available(self) -> None:
        args = build_parser().parse_args(["experiment-019", "--resume", "checkpoint.json"])
        self.assertEqual(args.command, "experiment-019")
        self.assertEqual(args.resume_from, "checkpoint.json")


    def test_hierarchical_memory_experiment_is_available(self) -> None:
        args = build_parser().parse_args(["experiment-013"])
        self.assertEqual(args.command, "experiment-013")

    def test_persistent_accumulation_experiment_is_available(self) -> None:
        args = build_parser().parse_args(["experiment-020", "--resume", "checkpoint.json"])
        self.assertEqual(args.command, "experiment-020")
        self.assertEqual(args.resume_from, "checkpoint.json")

if __name__ == "__main__":
    unittest.main()
