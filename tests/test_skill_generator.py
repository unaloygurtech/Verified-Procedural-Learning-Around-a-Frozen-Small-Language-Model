import unittest

from air_core.model_client import Completion
from air_core.skill_generator import (
    _candidate_from_completion,
    _signal_examples,
    consolidation_prompt,
    generate_skill_symbolic,
)


class SkillGeneratorTests(unittest.TestCase):
    def test_builds_candidate_from_json(self) -> None:
        completion = Completion(
            text='{"name":"neralis-3-generated","body":"' + ("rule " * 20) + '"}',
            elapsed_seconds=1.5,
            prompt_tokens=100,
            generated_tokens=40,
            raw={},
        )
        candidate = _candidate_from_completion(completion)
        self.assertEqual(candidate.name, "neralis-3-generated")
        self.assertEqual(candidate.prompt_tokens, 100)

    def test_prompt_has_training_but_not_heldout_data(self) -> None:
        prompt = consolidation_prompt()
        self.assertIn("zaf", prompt)
        self.assertNotIn("navik", prompt)

    def test_rejects_unexpected_name(self) -> None:
        completion = Completion(
            text='{"name":"wrong","body":"' + ("rule " * 20) + '"}',
            elapsed_seconds=0.1,
            prompt_tokens=1,
            generated_tokens=1,
            raw={},
        )
        with self.assertRaises(ValueError):
            _candidate_from_completion(completion)

    def test_signal_examples_are_scoped(self) -> None:
        amber = _signal_examples("amber")
        self.assertIn("input value=4 -> verified score=9", amber)
        self.assertNotIn("verified score=26", amber)

    def test_symbolic_inducer_finds_training_rules(self) -> None:
        candidate = generate_skill_symbolic()
        self.assertEqual(candidate.method, "symbolic-hypothesis-search")
        self.assertIn("join `code`, a colon (`:`), and `tag`", candidate.body)
        self.assertIn("amber: `value + 5`", candidate.body)
        self.assertIn("cobalt: `value * 2`", candidate.body)
        self.assertIn("ivory: `value - 3`", candidate.body)


if __name__ == "__main__":
    unittest.main()
