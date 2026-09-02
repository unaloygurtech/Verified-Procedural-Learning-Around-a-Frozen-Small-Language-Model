import unittest

from air_core.exp009 import (
    BASE_PYTHON_LIBRARY_009,
    COLLECTIONS_FAMILY_009,
    DATETIME_FAMILY_009,
    FAMILIES_009,
    JSON_FAMILY_009,
    LEARNING_PROMPT_HASH_009,
    LEARNING_PROMPT_TEMPLATE_009,
    PATHLIB_FAMILY_009,
    diagnose_python_gap_009,
    generic_learning_prompt,
    run_python_gate_009,
    static_check_python_009,
)


GOOD_CODE = {
    "json-canonical": '''import json

def transform(value: str) -> str:
    return json.dumps(json.loads(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
''',
    "datetime-date": '''from datetime import datetime

def transform(value: str) -> str:
    return datetime.strptime(value, "%d-%m-%Y").strftime("%Y-%m-%d")
''',
    "pathlib-suffix": '''from pathlib import PurePosixPath

def transform(value: str) -> str:
    return PurePosixPath(value).suffix.lower()
''',
    "collections-count": '''from collections import Counter

def transform(value: str) -> str:
    counts = Counter(value)
    return ";".join(f"{key}={count}" for key, count in sorted(counts.items()))
''',
}


class Experiment009Tests(unittest.TestCase):
    def test_four_families_share_one_frozen_template(self) -> None:
        self.assertEqual(len(FAMILIES_009), 4)
        self.assertEqual(len({family.family_id for family in FAMILIES_009}), 4)
        self.assertEqual(len(LEARNING_PROMPT_HASH_009), 64)
        self.assertIn("{family_id}", LEARNING_PROMPT_TEMPLATE_009)
        for family in FAMILIES_009:
            prompt = generic_learning_prompt(family)
            self.assertIn(family.family_id, prompt)
            self.assertIn(family.api_docs, prompt)

    def test_prior_library_does_not_wrongly_activate(self) -> None:
        for family in FAMILIES_009:
            diagnosis = diagnose_python_gap_009(BASE_PYTHON_LIBRARY_009, family)
            self.assertEqual(diagnosis["status"], "gap_detected")
            self.assertEqual(diagnosis["matching_skill_ids"], [])

    def test_known_procedures_pass_all_family_gates(self) -> None:
        for family in FAMILIES_009:
            code = GOOD_CODE[family.family_id]
            self.assertTrue(static_check_python_009(code, family).passed)
            self.assertEqual(run_python_gate_009(code, family, family.discovery, "discovery").accuracy, 1.0)
            self.assertEqual(run_python_gate_009(code, family, family.validation, "validation").accuracy, 1.0)
            self.assertEqual(run_python_gate_009(code, family, family.edge, "edge").accuracy, 1.0)
            self.assertEqual(run_python_gate_009(code, family, family.heldout, "heldout").accuracy, 1.0)

    def test_static_allowlist_rejects_cross_family_and_dangerous_code(self) -> None:
        dangerous = "import os\ndef transform(value: str) -> str:\n    return os.getcwd()\n"
        self.assertFalse(static_check_python_009(dangerous, JSON_FAMILY_009).passed)
        cross_family = GOOD_CODE["json-canonical"]
        self.assertFalse(static_check_python_009(cross_family, DATETIME_FAMILY_009).passed)

    def test_family_contracts_cover_requested_api_families(self) -> None:
        self.assertIn("json", JSON_FAMILY_009.api_docs)
        self.assertIn("datetime", DATETIME_FAMILY_009.api_docs)
        self.assertIn("PurePosixPath", PATHLIB_FAMILY_009.api_docs)
        self.assertIn("Counter", COLLECTIONS_FAMILY_009.api_docs)


if __name__ == "__main__":
    unittest.main()
