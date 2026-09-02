import unittest

from air_core.exp009 import PythonSkillArtifact009, static_check_python_009
from air_core.exp010 import (
    BASE_PYTHON_LIBRARY_010,
    SYNTHETIC_API_NAME_010,
    ZORVIK_FAMILIES_010,
    _semantic_corruption_010,
    _unsafe_code_010,
    _api_source_hash_010,
    _composition_cases_010,
    search_composition_010,
)


GOOD_CODE = {
    "zorvik-kel": "from zorvik_010 import kel\n\ndef transform(value: str) -> str:\n    return kel(value)\n",
    "zorvik-nam": "from zorvik_010 import nam\n\ndef transform(value: str) -> str:\n    return nam(value)\n",
    "zorvik-tesh": "from zorvik_010 import tesh\n\ndef transform(value: str) -> str:\n    return tesh(value)\n",
    "zorvik-vum": "from zorvik_010 import vum\n\ndef transform(value: str) -> str:\n    return vum(value)\n",
}


class Experiment010Tests(unittest.TestCase):
    def test_synthetic_api_has_four_disjoint_families(self) -> None:
        self.assertEqual(SYNTHETIC_API_NAME_010, "zorvik_010")
        ids = []
        for family in ZORVIK_FAMILIES_010:
            ids.extend(case.case_id for case in family.discovery + family.validation + family.edge + family.heldout)
            self.assertEqual(len(family.discovery), 4)
            self.assertEqual(len(family.validation), 3)
            self.assertEqual(len(family.edge), 3)
            self.assertEqual(len(family.heldout), 8)
        self.assertEqual(len(ids), len(set(ids)))

    def test_documented_wrappers_pass_public_hidden_edge_and_heldout(self) -> None:
        for family in ZORVIK_FAMILIES_010:
            code = GOOD_CODE[family.family_id]
            self.assertTrue(static_check_python_009(code, family).passed)
            from air_core.exp009 import run_python_gate_009

            self.assertEqual(run_python_gate_009(code, family, family.discovery, "discovery").accuracy, 1.0)
            self.assertEqual(run_python_gate_009(code, family, family.validation, "validation").accuracy, 1.0)
            self.assertEqual(run_python_gate_009(code, family, family.edge, "edge").accuracy, 1.0)
            self.assertEqual(run_python_gate_009(code, family, family.heldout, "heldout").accuracy, 1.0)

    def test_zero_knowledge_controls_are_rejected_and_package_is_immutable(self) -> None:
        self.assertFalse(static_check_python_009(_unsafe_code_010(), ZORVIK_FAMILIES_010[0]).passed)
        corrupted = _semantic_corruption_010()
        self.assertTrue(static_check_python_009(corrupted, ZORVIK_FAMILIES_010[0]).passed)
        from air_core.exp009 import run_python_gate_009

        self.assertLess(run_python_gate_009(corrupted, ZORVIK_FAMILIES_010[0], ZORVIK_FAMILIES_010[0].validation, "corrupted").accuracy, 0.9)
        self.assertEqual(_api_source_hash_010(), _api_source_hash_010())

    def test_composition_search_finds_only_the_hidden_pair(self) -> None:
        first_family, second_family = ZORVIK_FAMILIES_010[0], ZORVIK_FAMILIES_010[2]
        artifacts = (
            (PythonSkillArtifact009("skill-kel", first_family.family_id, 1, "value: str", "str", GOOD_CODE[first_family.family_id], "test"), first_family),
            (PythonSkillArtifact009("skill-tesh", second_family.family_id, 1, "value: str", "str", GOOD_CODE[second_family.family_id], "test"), second_family),
        )
        selection, _, _ = _composition_cases_010()
        plans = search_composition_010(artifacts, selection)
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0]["first_skill_id"], "skill-kel")
        self.assertEqual(plans[0]["second_skill_id"], "skill-tesh")

    def test_base_library_is_unrelated(self) -> None:
        self.assertEqual(len(BASE_PYTHON_LIBRARY_010), 2)
        self.assertNotIn("zorvik", BASE_PYTHON_LIBRARY_010[0].code)


if __name__ == "__main__":
    unittest.main()
