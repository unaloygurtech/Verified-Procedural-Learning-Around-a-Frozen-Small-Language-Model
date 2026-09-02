import unittest

from air_core.neralis import (
    HELD_OUT_CASES,
    TRAIN_CASES,
    VALIDATION_CASES,
    parse_response,
    raw_experiences,
)


class NeralisTests(unittest.TestCase):
    def test_expected_rules(self) -> None:
        self.assertEqual(
            TRAIN_CASES[0].expected(),
            {"key": "zaf:mori", "score": 9, "label": "north"},
        )
        self.assertEqual(
            TRAIN_CASES[5].expected(),
            {"key": "bex:nora", "score": 26, "label": "west"},
        )
        self.assertEqual(
            TRAIN_CASES[9].expected(),
            {"key": "sulon:kir", "score": 12, "label": "east"},
        )

    def test_partitions_are_disjoint(self) -> None:
        ids = [case.case_id for case in TRAIN_CASES + VALIDATION_CASES + HELD_OUT_CASES]
        self.assertEqual(len(ids), len(set(ids)))

    def test_raw_experiences_contain_only_training_cases(self) -> None:
        text = raw_experiences()
        self.assertIn("zaf", text)
        self.assertNotIn("navik", text)

    def test_parse_response_requires_json_object(self) -> None:
        self.assertEqual(parse_response('{"key":"x"}'), {"key": "x"})
        self.assertEqual(parse_response("```json\n{}\n```"), {})
        self.assertIsNone(parse_response("[]"))
        self.assertEqual(
            parse_response('<think>{"draft":1}</think>\n{"key":"final"}'),
            {"key": "final"},
        )


if __name__ == "__main__":
    unittest.main()
