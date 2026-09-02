"""Small, typed, model-independent AIR intermediate representation.

AIR IR v1 deliberately supports one verified rule-program subset.  It is not
Python bytecode and it is not a general-purpose language.  Unsupported Python
artifacts remain readable Python artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import json
import struct
from typing import Any, Callable, Iterable


IR_VERSION = 1
COMPILER_VERSION = "air-ir-rule-compiler-v0"
_MAGIC = b"AIR\x00"

OPCODE_IDS = {
    "PARSE_TOKEN": 1,
    "LOOKUP_INT": 2,
    "PARITY_INT": 3,
    "MUL_INT": 4,
    "TO_STR": 5,
    "RETURN": 6,
}
ID_OPCODES = {value: key for key, value in OPCODE_IDS.items()}


class IRValidationError(ValueError):
    """The representation is malformed, unsafe, unsupported, or type-invalid."""


class IRExecutionError(RuntimeError):
    """A valid program rejected or could not process a runtime input."""


class UnsupportedPythonSubset(IRValidationError):
    """A Python artifact is outside the intentionally small compiler subset."""


@dataclass(frozen=True)
class Operand:
    kind: str
    value: Any

    def verbose(self) -> dict[str, Any]:
        return {"type": self.kind, "value": self.value}

    def compact(self) -> list[Any]:
        tags = {"slot": "s", "int": "i", "text": "t", "map": "m"}
        if self.kind not in tags:
            raise IRValidationError(f"unknown operand type: {self.kind}")
        value = self.value
        if self.kind == "map":
            value = [[key, value[key]] for key in sorted(value)]
        return [tags[self.kind], value]


@dataclass(frozen=True)
class Instruction:
    opcode: str
    operands: tuple[Operand, ...]

    def verbose(self) -> dict[str, Any]:
        return {"opcode": self.opcode, "operands": [item.verbose() for item in self.operands]}

    def compact(self) -> list[Any]:
        if self.opcode not in OPCODE_IDS:
            raise IRValidationError(f"unknown opcode: {self.opcode}")
        return [OPCODE_IDS[self.opcode], [item.compact() for item in self.operands]]


@dataclass(frozen=True)
class AIRProgram:
    version: int
    instructions: tuple[Instruction, ...]


def slot(name: str) -> Operand:
    return Operand("slot", name)


def integer(value: int) -> Operand:
    return Operand("int", value)


def mapping(value: dict[str, int]) -> Operand:
    return Operand("map", dict(value))


_SCHEMAS: dict[str, tuple[str, ...]] = {
    "PARSE_TOKEN": ("slot", "slot"),
    "LOOKUP_INT": ("slot", "slot", "map"),
    "PARITY_INT": ("slot", "slot", "int", "int"),
    "MUL_INT": ("slot", "slot", "slot"),
    "TO_STR": ("slot", "slot"),
    "RETURN": ("slot",),
}


def _valid_slot_name(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value.replace("_", "a").isalnum() and not value[0].isdigit()


def validate_program(program: AIRProgram) -> None:
    if program.version != IR_VERSION:
        raise IRValidationError(f"unsupported AIR IR version: {program.version}")
    if not program.instructions or len(program.instructions) > 64:
        raise IRValidationError("instruction count must be between 1 and 64")
    slots: dict[str, str] = {}
    returned = False
    for index, instruction in enumerate(program.instructions):
        schema = _SCHEMAS.get(instruction.opcode)
        if schema is None:
            raise IRValidationError(f"unknown opcode: {instruction.opcode}")
        if tuple(item.kind for item in instruction.operands) != schema:
            raise IRValidationError(f"type-invalid operands for {instruction.opcode}")
        for operand in instruction.operands:
            if operand.kind == "slot" and not _valid_slot_name(operand.value):
                raise IRValidationError("invalid slot name")
            if operand.kind == "int" and (not isinstance(operand.value, int) or isinstance(operand.value, bool)):
                raise IRValidationError("integer operand required")
            if operand.kind == "map":
                if not isinstance(operand.value, dict) or not operand.value:
                    raise IRValidationError("non-empty map operand required")
                if any(not isinstance(key, str) or len(key) != 1 or not isinstance(value, int) or isinstance(value, bool) for key, value in operand.value.items()):
                    raise IRValidationError("map must contain one-character string keys and integer values")
        values = [item.value for item in instruction.operands]
        if instruction.opcode == "PARSE_TOKEN":
            slots[values[0]], slots[values[1]] = "str", "int"
        elif instruction.opcode == "LOOKUP_INT":
            if slots.get(values[1]) != "str":
                raise IRValidationError("LOOKUP_INT key slot must be str")
            slots[values[0]] = "int"
        elif instruction.opcode == "PARITY_INT":
            if slots.get(values[1]) != "int":
                raise IRValidationError("PARITY_INT source slot must be int")
            slots[values[0]] = "int"
        elif instruction.opcode == "MUL_INT":
            if slots.get(values[1]) != "int" or slots.get(values[2]) != "int":
                raise IRValidationError("MUL_INT sources must be int")
            slots[values[0]] = "int"
        elif instruction.opcode == "TO_STR":
            if slots.get(values[1]) != "int":
                raise IRValidationError("TO_STR source must be int")
            slots[values[0]] = "str"
        elif instruction.opcode == "RETURN":
            if slots.get(values[0]) != "str":
                raise IRValidationError("RETURN source must be str")
            if index != len(program.instructions) - 1:
                raise IRValidationError("RETURN must be the final instruction")
            returned = True
    if not returned:
        raise IRValidationError("program must end with RETURN")


def build_rule_program(symbols: dict[str, int], even_add: int, odd_multiply: int) -> AIRProgram:
    program = AIRProgram(
        IR_VERSION,
        (
            Instruction("PARSE_TOKEN", (slot("symbol"), slot("number"))),
            Instruction("LOOKUP_INT", (slot("factor"), slot("symbol"), mapping(symbols))),
            Instruction("PARITY_INT", (slot("adjusted"), slot("number"), integer(even_add), integer(odd_multiply))),
            Instruction("MUL_INT", (slot("product"), slot("factor"), slot("adjusted"))),
            Instruction("TO_STR", (slot("result"), slot("product"))),
            Instruction("RETURN", (slot("result"),)),
        ),
    )
    validate_program(program)
    return program


def execute_program(program: AIRProgram, value: str) -> str:
    validate_program(program)
    state: dict[str, Any] = {"input": value}
    for instruction in program.instructions:
        values = [item.value for item in instruction.operands]
        if instruction.opcode == "PARSE_TOKEN":
            if not isinstance(value, str) or len(value) < 2:
                raise IRExecutionError("input must be <symbol><integer>")
            try:
                number = int(value[1:])
            except ValueError as exc:
                raise IRExecutionError("integer suffix required") from exc
            state[values[0]], state[values[1]] = value[0], number
        elif instruction.opcode == "LOOKUP_INT":
            table = instruction.operands[2].value
            key = state[values[1]]
            if key not in table:
                raise IRExecutionError("unknown symbol")
            state[values[0]] = table[key]
        elif instruction.opcode == "PARITY_INT":
            number = state[values[1]]
            state[values[0]] = number + values[2] if number % 2 == 0 else number * values[3]
        elif instruction.opcode == "MUL_INT":
            state[values[0]] = state[values[1]] * state[values[2]]
        elif instruction.opcode == "TO_STR":
            state[values[0]] = str(state[values[1]])
        elif instruction.opcode == "RETURN":
            return state[values[0]]
        else:  # validate_program makes this unreachable; keep fail-closed.
            raise IRExecutionError(f"unknown opcode: {instruction.opcode}")
    raise IRExecutionError("program returned no value")


def serialize_json_ast(program: AIRProgram) -> bytes:
    validate_program(program)
    payload = {"format": "AIR-IR", "version": program.version, "instructions": [item.verbose() for item in program.instructions]}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _operand_from_verbose(payload: Any) -> Operand:
    if not isinstance(payload, dict) or set(payload) != {"type", "value"}:
        raise IRValidationError("malformed JSON AST operand")
    return Operand(payload["type"], payload["value"])


def deserialize_json_ast(data: bytes) -> AIRProgram:
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise IRValidationError("malformed JSON AST") from exc
    if not isinstance(payload, dict) or payload.get("format") != "AIR-IR" or not isinstance(payload.get("instructions"), list):
        raise IRValidationError("malformed JSON AST root")
    instructions: list[Instruction] = []
    for item in payload["instructions"]:
        if not isinstance(item, dict) or set(item) != {"opcode", "operands"} or not isinstance(item["operands"], list):
            raise IRValidationError("malformed JSON AST instruction")
        instructions.append(Instruction(item["opcode"], tuple(_operand_from_verbose(value) for value in item["operands"])))
    program = AIRProgram(payload.get("version"), tuple(instructions))
    validate_program(program)
    return program


def serialize_compact_ir(program: AIRProgram) -> bytes:
    validate_program(program)
    payload = [program.version, [instruction.compact() for instruction in program.instructions]]
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _operand_from_compact(payload: Any) -> Operand:
    if not isinstance(payload, list) or len(payload) != 2:
        raise IRValidationError("malformed compact operand")
    kinds = {"s": "slot", "i": "int", "t": "text", "m": "map"}
    kind = kinds.get(payload[0])
    if kind is None:
        raise IRValidationError("unknown compact operand tag")
    value = payload[1]
    if kind == "map":
        if not isinstance(value, list):
            raise IRValidationError("malformed compact map")
        try:
            value = {item[0]: item[1] for item in value if isinstance(item, list) and len(item) == 2}
        except (TypeError, IndexError) as exc:
            raise IRValidationError("malformed compact map") from exc
    return Operand(kind, value)


def deserialize_compact_ir(data: bytes) -> AIRProgram:
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise IRValidationError("malformed compact AIR IR") from exc
    if not isinstance(payload, list) or len(payload) != 2 or not isinstance(payload[1], list):
        raise IRValidationError("malformed compact AIR IR root")
    instructions: list[Instruction] = []
    for item in payload[1]:
        if not isinstance(item, list) or len(item) != 2 or not isinstance(item[1], list):
            raise IRValidationError("malformed compact instruction")
        opcode = ID_OPCODES.get(item[0])
        if opcode is None:
            raise IRValidationError(f"unknown opcode id: {item[0]}")
        instructions.append(Instruction(opcode, tuple(_operand_from_compact(value) for value in item[1])))
    program = AIRProgram(payload[0], tuple(instructions))
    validate_program(program)
    return program


def _pack_bytes(value: bytes, length_format: str) -> bytes:
    maximum = 255 if length_format == "B" else 65535
    if len(value) > maximum:
        raise IRValidationError("binary field too large")
    return struct.pack(">" + length_format, len(value)) + value


def serialize_binary_ir(program: AIRProgram) -> bytes:
    validate_program(program)
    output = bytearray(_MAGIC + struct.pack(">BH", program.version, len(program.instructions)))
    tags = {"slot": 1, "int": 2, "map": 3, "text": 4}
    for instruction in program.instructions:
        output.extend(struct.pack(">BB", OPCODE_IDS[instruction.opcode], len(instruction.operands)))
        for operand in instruction.operands:
            output.append(tags[operand.kind])
            if operand.kind == "slot":
                output.extend(_pack_bytes(operand.value.encode("utf-8"), "B"))
            elif operand.kind == "int":
                output.extend(struct.pack(">q", operand.value))
            elif operand.kind == "text":
                output.extend(_pack_bytes(operand.value.encode("utf-8"), "H"))
            elif operand.kind == "map":
                output.extend(struct.pack(">H", len(operand.value)))
                for key in sorted(operand.value):
                    output.extend(_pack_bytes(key.encode("utf-8"), "B"))
                    output.extend(struct.pack(">q", operand.value[key]))
    return bytes(output)


class _BinaryReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def take(self, count: int) -> bytes:
        if count < 0 or self.offset + count > len(self.data):
            raise IRValidationError("truncated binary AIR IR")
        value = self.data[self.offset:self.offset + count]
        self.offset += count
        return value

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        size = struct.calcsize(">" + fmt)
        return struct.unpack(">" + fmt, self.take(size))

    def sized(self, fmt: str) -> bytes:
        (length,) = self.unpack(fmt)
        return self.take(length)


def deserialize_binary_ir(data: bytes) -> AIRProgram:
    reader = _BinaryReader(data)
    if reader.take(len(_MAGIC)) != _MAGIC:
        raise IRValidationError("invalid binary AIR IR magic")
    version, count = reader.unpack("BH")
    instructions: list[Instruction] = []
    for _ in range(count):
        opcode_id, operand_count = reader.unpack("BB")
        opcode = ID_OPCODES.get(opcode_id)
        if opcode is None:
            raise IRValidationError(f"unknown opcode id: {opcode_id}")
        operands: list[Operand] = []
        for _ in range(operand_count):
            (tag,) = reader.unpack("B")
            if tag == 1:
                operands.append(Operand("slot", reader.sized("B").decode("utf-8")))
            elif tag == 2:
                operands.append(Operand("int", reader.unpack("q")[0]))
            elif tag == 3:
                (entries,) = reader.unpack("H")
                table: dict[str, int] = {}
                for _ in range(entries):
                    key = reader.sized("B").decode("utf-8")
                    table[key] = reader.unpack("q")[0]
                operands.append(Operand("map", table))
            elif tag == 4:
                operands.append(Operand("text", reader.sized("H").decode("utf-8")))
            else:
                raise IRValidationError(f"unknown binary operand tag: {tag}")
        instructions.append(Instruction(opcode, tuple(operands)))
    if reader.offset != len(data):
        raise IRValidationError("trailing bytes in binary AIR IR")
    program = AIRProgram(version, tuple(instructions))
    validate_program(program)
    return program


_ALLOWED_PYTHON_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Assign, ast.AnnAssign,
    ast.If, ast.Return, ast.Name, ast.Load, ast.Store, ast.Constant, ast.Dict,
    ast.Subscript, ast.Slice, ast.BinOp, ast.Add, ast.Mult, ast.Mod, ast.IfExp,
    ast.Compare, ast.Eq, ast.In, ast.NotIn, ast.Call, ast.UnaryOp, ast.USub,
)


def _int_constant(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _int_constant(node.operand)
        return -value if value is not None else None
    return None


def _has_parity_test(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
            continue
        if _int_constant(node.comparators[0]) != 0:
            continue
        left = node.left
        if isinstance(left, ast.BinOp) and isinstance(left.op, ast.Mod) and _int_constant(left.right) == 2:
            return True
    return False


def _has_string_product_return(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Call):
            continue
        if not isinstance(node.value.func, ast.Name) or node.value.func.id != "str" or len(node.value.args) != 1:
            continue
        if isinstance(node.value.args[0], ast.BinOp) and isinstance(node.value.args[0].op, ast.Mult):
            return True
    return False


def compile_python_rule_subset(source: str) -> AIRProgram:
    """Compile the recognized safe rule subset or fail closed.

    The compiler extracts only a literal symbol map and the constants from an
    explicit parity branch.  It rejects imports, attributes, loops, arbitrary
    calls, and every node outside the allowlist.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise UnsupportedPythonSubset("Python source is not parseable") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != "transform":
        raise UnsupportedPythonSubset("exactly one transform function is required")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_PYTHON_NODES):
            raise UnsupportedPythonSubset(f"unsupported Python node: {type(node).__name__}")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in {"int", "str"}):
            raise UnsupportedPythonSubset("only int and str calls are supported")
    symbol_map: dict[str, int] | None = None
    add_constants: list[int] = []
    multiply_constants: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            candidate: dict[str, int] = {}
            valid = len(node.keys) >= 2
            for key_node, value_node in zip(node.keys, node.values):
                key = key_node.value if isinstance(key_node, ast.Constant) else None
                value = _int_constant(value_node)
                if not isinstance(key, str) or len(key) != 1 or value is None:
                    valid = False
                    break
                candidate[key] = value
            if valid:
                if symbol_map is not None:
                    raise UnsupportedPythonSubset("multiple symbol maps are unsupported")
                symbol_map = candidate
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            constant = _int_constant(node.right)
            if constant is None:
                constant = _int_constant(node.left)
            if constant is not None:
                add_constants.append(constant)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            constant = _int_constant(node.right)
            if constant is None:
                constant = _int_constant(node.left)
            if constant is not None:
                multiply_constants.append(constant)
    if symbol_map is None or len(add_constants) != 1 or len(multiply_constants) != 1:
        raise UnsupportedPythonSubset("one literal map, add constant, and multiply constant are required")
    if not _has_parity_test(tree) or not _has_string_product_return(tree):
        raise UnsupportedPythonSubset("explicit parity test and string product return are required")
    return build_rule_program(symbol_map, add_constants[0], multiply_constants[0])


def semantic_equivalence(program: AIRProgram, reference: Callable[[str], str], cases: Iterable[tuple[str, str]]) -> tuple[bool, list[str]]:
    passed_ids: list[str] = []
    for case_id, value in cases:
        try:
            observed = execute_program(program, value)
            expected = reference(value)
        except (IRExecutionError, ValueError, KeyError):
            return False, passed_ids
        if observed != expected:
            return False, passed_ids
        passed_ids.append(case_id)
    return True, passed_ids


def source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()
