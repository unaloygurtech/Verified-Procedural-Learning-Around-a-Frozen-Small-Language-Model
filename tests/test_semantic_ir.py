import unittest

from air_core.exp012 import make_robustness_families_012, ROBUSTNESS_SEEDS_012
from air_core.semantic_ir import (
    SemanticIRValidationError,
    compile_semantic_ir_python_017,
    execute_semantic_ir_017,
    oracle_call_ir_017,
    validate_semantic_ir_017,
)


class SemanticIRTests(unittest.TestCase):
    def setUp(self):
        self.family = make_robustness_families_012(ROBUSTNESS_SEEDS_012[0])[0]
        self.operation = next(iter(self.family.allowed_call_names))
        self.program = oracle_call_ir_017(self.operation)

    def test_oracle_ir_validates_executes_and_compiles(self):
        validate_semantic_ir_017(self.program, self.family.allowed_call_names)
        case = self.family.discovery[0]
        self.assertEqual(execute_semantic_ir_017(self.program, case.input_text, self.family.allowed_call_names), case.expected)
        source = compile_semantic_ir_python_017(self.program, self.family.allowed_call_names)
        self.assertIn(f"from air_synth_012 import {self.operation}", source)
        self.assertIn("def transform(value: str) -> str:", source)

    def test_unknown_opcode_and_wrong_types_are_rejected(self):
        unknown = {**self.program, "expr": {"op": "RETURN", "value": {"op": "NOPE"}}}
        with self.assertRaises(SemanticIRValidationError):
            validate_semantic_ir_017(unknown, self.family.allowed_call_names)
        wrong = {**self.program, "expr": {"op": "RETURN", "value": {"op": "CALL", "api": self.operation, "args": [{"op": "INT", "value": 2}]}}}
        with self.assertRaises(SemanticIRValidationError):
            validate_semantic_ir_017(wrong, self.family.allowed_call_names)

    def test_unknown_api_is_rejected(self):
        wrong = oracle_call_ir_017("op_not_allowed")
        with self.assertRaises(SemanticIRValidationError):
            validate_semantic_ir_017(wrong, self.family.allowed_call_names)


if __name__ == "__main__":
    unittest.main()
