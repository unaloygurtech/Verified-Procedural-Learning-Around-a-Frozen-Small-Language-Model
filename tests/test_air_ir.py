import unittest

from air_core.air_ir import (
    AIRProgram,
    IR_VERSION,
    IRExecutionError,
    IRValidationError,
    Instruction,
    build_rule_program,
    compile_python_rule_subset,
    deserialize_binary_ir,
    deserialize_compact_ir,
    deserialize_json_ast,
    execute_program,
    serialize_binary_ir,
    serialize_compact_ir,
    serialize_json_ast,
    slot,
    validate_program,
)


SOURCE = """def transform(value: str) -> str:
    factors = {'A': 7, 'B': 2, 'C': 9}
    symbol = value[0]
    number = int(value[1:])
    factor = factors[symbol]
    adjusted = number + 3 if number % 2 == 0 else number * 2
    return str(factor * adjusted)
"""


class AIRIRTests(unittest.TestCase):
    def test_all_serializations_round_trip_deterministically(self) -> None:
        program = compile_python_rule_subset(SOURCE)
        formats = (
            (serialize_json_ast, deserialize_json_ast),
            (serialize_compact_ir, deserialize_compact_ir),
            (serialize_binary_ir, deserialize_binary_ir),
        )
        for serialize, deserialize in formats:
            payload = serialize(program)
            restored = deserialize(payload)
            self.assertEqual(payload, serialize(restored))
            self.assertEqual(execute_program(restored, "A4"), "49")
            self.assertEqual(execute_program(restored, "B3"), "12")

    def test_unknown_symbol_and_corrupt_binary_fail_closed(self) -> None:
        binary = serialize_binary_ir(compile_python_rule_subset(SOURCE))
        with self.assertRaises(IRExecutionError):
            execute_program(deserialize_binary_ir(binary), "Z4")
        with self.assertRaises(IRValidationError):
            deserialize_binary_ir(binary[:-2])
        unknown = bytearray(binary)
        unknown[7] = 255
        with self.assertRaises(IRValidationError):
            deserialize_binary_ir(bytes(unknown))
        wrong_version = bytearray(binary)
        wrong_version[4] = 99
        with self.assertRaises(IRValidationError):
            deserialize_binary_ir(bytes(wrong_version))

    def test_type_invalid_and_semantic_wrong_are_distinct(self) -> None:
        invalid = AIRProgram(
            IR_VERSION,
            (
                Instruction("PARSE_TOKEN", (slot("symbol"), slot("number"))),
                Instruction("MUL_INT", (slot("product"), slot("symbol"), slot("number"))),
                Instruction("TO_STR", (slot("result"), slot("product"))),
                Instruction("RETURN", (slot("result"),)),
            ),
        )
        with self.assertRaises(IRValidationError):
            validate_program(invalid)
        wrong = build_rule_program({"A": 8, "B": 2, "C": 9}, 3, 2)
        self.assertNotEqual(execute_program(wrong, "A4"), "49")

    def test_compiler_rejects_unsupported_python(self) -> None:
        with self.assertRaises(IRValidationError):
            compile_python_rule_subset("import os\ndef transform(value):\n    return os.getcwd()\n")


if __name__ == "__main__":
    unittest.main()
