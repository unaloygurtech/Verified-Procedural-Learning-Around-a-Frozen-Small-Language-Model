import unittest

from air_core.exp008 import (
    BASE_PYTHON_LIBRARY_008,
    DISCOVERY_008,
    EDGE_008,
    HELD_OUT_008,
    PYTHON_API_DOCS_008,
    VALIDATION_008,
    _corrupted_code_008,
    diagnose_python_gap,
    run_python_in_sandbox,
    run_python_gate,
    search_existing_python_library,
    static_check_python,
)


GOOD_CODE = """from urllib.parse import parse_qsl, urlencode

def transform(query: str) -> str:
    pairs = parse_qsl(query, keep_blank_values=True)
    pairs.sort(key=lambda pair: (pair[0], pair[1]))
    return urlencode(pairs, doseq=True)
"""


class Experiment008Tests(unittest.TestCase):
    def test_partitions_are_new_and_docs_are_limited(self) -> None:
        ids = [case.case_id for case in DISCOVERY_008 + VALIDATION_008 + EDGE_008 + HELD_OUT_008]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("parse_qsl", PYTHON_API_DOCS_008)
        self.assertNotIn("heldout", PYTHON_API_DOCS_008.lower())

    def test_existing_library_detects_python_capability_gap(self) -> None:
        matches, evaluations = search_existing_python_library(BASE_PYTHON_LIBRARY_008, DISCOVERY_008)
        self.assertEqual(matches, ())
        self.assertEqual(evaluations, 1)
        diagnosis = diagnose_python_gap(BASE_PYTHON_LIBRARY_008, DISCOVERY_008)
        self.assertEqual(diagnosis.status, "gap_detected")

    def test_static_allowlist_rejects_dangerous_code(self) -> None:
        dangerous = "import os\ndef transform(query: str) -> str:\n    return os.getcwd()\n"
        check = static_check_python(dangerous)
        self.assertFalse(check.passed)
        result = run_python_in_sandbox(dangerous, "a=1", "")
        self.assertFalse(result.passed)

    def test_python_api_procedure_passes_all_deterministic_gates(self) -> None:
        self.assertTrue(static_check_python(GOOD_CODE).passed)
        self.assertEqual(run_python_gate(GOOD_CODE, DISCOVERY_008, "discovery").accuracy, 1.0)
        self.assertEqual(run_python_gate(GOOD_CODE, VALIDATION_008, "validation").accuracy, 1.0)
        self.assertEqual(run_python_gate(GOOD_CODE, EDGE_008, "edge").accuracy, 1.0)
        self.assertEqual(run_python_gate(GOOD_CODE, HELD_OUT_008, "heldout").accuracy, 1.0)

    def test_corrupted_api_procedure_is_runtime_valid_but_semantically_rejected(self) -> None:
        corrupted = _corrupted_code_008()
        self.assertTrue(static_check_python(corrupted).passed)
        result = run_python_gate(corrupted, VALIDATION_008, "corrupted")
        self.assertLess(result.accuracy, 0.9)


if __name__ == "__main__":
    unittest.main()
