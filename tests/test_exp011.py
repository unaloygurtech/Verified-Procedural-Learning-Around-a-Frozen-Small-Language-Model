import unittest

from air_core.exp009 import run_python_gate_009, static_check_python_009
from air_core.exp010 import ZORVIK_FAMILIES_010
from air_core.exp011 import (
    DOC_POOL_011,
    RULE_DOC_011,
    RULE_FAMILY_011,
    RULE_HASH_011,
    RULE_SPEC_011,
    RETRIEVAL_PROMPT_HASH_011,
    RETRIEVAL_PROMPT_VERSION_011,
    _family_with_retrieved_doc,
    retrieval_prompt_011,
)


RULE_CODE = """def transform(value: str) -> str:
    symbol = value[0]
    number = int(value[1:])
    mapping = {mapping}
    transformed = number + {even_add} if number % 2 == 0 else number * {odd_multiply}
    return str(mapping[symbol] * transformed)
""".format(mapping=RULE_SPEC_011["symbols"], even_add=RULE_SPEC_011["even_add"], odd_multiply=RULE_SPEC_011["odd_multiply"])


class Experiment011Tests(unittest.TestCase):
    def test_document_pool_and_retriever_protocol_are_frozen(self) -> None:
        self.assertEqual(len(DOC_POOL_011), 8)
        self.assertEqual(len({record["doc_id"] for record in DOC_POOL_011}), 8)
        self.assertEqual(len(RETRIEVAL_PROMPT_HASH_011), 64)
        self.assertTrue(RETRIEVAL_PROMPT_VERSION_011.startswith("air-011-"))
        for family in ZORVIK_FAMILIES_010:
            prompt = retrieval_prompt_011(family)
            self.assertIn(family.family_id.removeprefix("zorvik-"), prompt)
            self.assertNotIn("Public examples:", prompt)

    def test_only_the_selected_document_is_injected_into_learner(self) -> None:
        selected = _family_with_retrieved_doc(ZORVIK_FAMILIES_010[0], "manual-amber")
        self.assertIn("kel", selected.api_docs)
        unrelated = _family_with_retrieved_doc(ZORVIK_FAMILIES_010[0], "note-charcoal")
        self.assertNotIn("reverse the order", unrelated.api_docs)

    def test_rule_hash_and_document_are_deterministic(self) -> None:
        self.assertEqual(len(RULE_HASH_011), 64)
        self.assertIn(RULE_SPEC_011["namespace"], RULE_DOC_011)
        self.assertIn(str(RULE_SPEC_011["even_add"]), RULE_DOC_011)

    def test_rule_artifact_passes_gates_and_unknown_is_not_valid(self) -> None:
        self.assertTrue(static_check_python_009(RULE_CODE, RULE_FAMILY_011).passed)
        self.assertEqual(run_python_gate_009(RULE_CODE, RULE_FAMILY_011, RULE_FAMILY_011.discovery, "discovery").accuracy, 1.0)
        self.assertEqual(run_python_gate_009(RULE_CODE, RULE_FAMILY_011, RULE_FAMILY_011.validation, "validation").accuracy, 1.0)
        self.assertEqual(run_python_gate_009(RULE_CODE, RULE_FAMILY_011, RULE_FAMILY_011.edge, "edge").accuracy, 1.0)
        from air_core.exp009 import run_python_in_sandbox_009

        unknown = run_python_in_sandbox_009(RULE_CODE, RULE_FAMILY_011, "Z7")
        self.assertFalse(unknown.passed)

    def test_semantic_wrong_rule_is_rejected(self) -> None:
        wrong = "def transform(value: str) -> str:\n    return str(int(value[1:]))\n"
        self.assertTrue(static_check_python_009(wrong, RULE_FAMILY_011).passed)
        self.assertLess(run_python_gate_009(wrong, RULE_FAMILY_011, RULE_FAMILY_011.validation, "wrong").accuracy, 0.9)


if __name__ == "__main__":
    unittest.main()
