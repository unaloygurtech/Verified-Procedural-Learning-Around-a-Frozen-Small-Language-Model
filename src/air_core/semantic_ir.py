"""Minimal typed semantic IR used only by Experiment 0017.

This representation deliberately contains no Python syntax, imports, wrappers,
or sandbox/provenance state.  It is a small expression language for calling one
opaque deterministic API and applying bounded string combinators.  The
experiment freezes this grammar before any model call.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import air_synth_012


SEMANTIC_IR_VERSION_017 = 1
SEMANTIC_IR_FORMAT_017 = "AIR-SEMANTIC-IR"
SEMANTIC_IR_OPCODES_017 = (
    "INPUT", "INT", "CALL", "REVERSE", "ROTATE", "CONCAT", "RETURN",
)
MAX_IR_DEPTH_017 = 12
MAX_CONCAT_ARGS_017 = 8


class SemanticIRValidationError(ValueError):
    """Malformed, unsafe, unknown, or type-invalid semantic IR."""


class SemanticIRExecutionError(RuntimeError):
    """A valid semantic IR could not execute on an input."""


def canonical_semantic_ir_json_017(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _expect_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise SemanticIRValidationError(f"{label} keys must be {sorted(keys)}")


def _validate_expr_017(expr: Any, allowed_operations: set[str], depth: int = 0) -> str:
    if depth > MAX_IR_DEPTH_017:
        raise SemanticIRValidationError("semantic IR nesting limit exceeded")
    if not isinstance(expr, Mapping) or not isinstance(expr.get("op"), str):
        raise SemanticIRValidationError("expression must contain an opcode")
    opcode = expr["op"]
    if opcode not in SEMANTIC_IR_OPCODES_017:
        raise SemanticIRValidationError(f"unknown semantic opcode: {opcode}")
    if opcode == "INPUT":
        _expect_keys(expr, {"op"}, "INPUT")
        return "str"
    if opcode == "INT":
        _expect_keys(expr, {"op", "value"}, "INT")
        if not isinstance(expr["value"], int) or isinstance(expr["value"], bool):
            raise SemanticIRValidationError("INT value must be an integer")
        if abs(expr["value"]) > 64:
            raise SemanticIRValidationError("INT value out of bounded range")
        return "int"
    if opcode == "CALL":
        _expect_keys(expr, {"op", "api", "args"}, "CALL")
        api = expr["api"]
        args = expr["args"]
        if not isinstance(api, str) or api not in allowed_operations:
            raise SemanticIRValidationError("CALL API is not allowed for this family")
        if not isinstance(args, list) or len(args) != 1 or _validate_expr_017(args[0], allowed_operations, depth + 1) != "str":
            raise SemanticIRValidationError("CALL expects exactly one string argument")
        return "str"
    if opcode == "REVERSE":
        _expect_keys(expr, {"op", "value"}, "REVERSE")
        if _validate_expr_017(expr["value"], allowed_operations, depth + 1) != "str":
            raise SemanticIRValidationError("REVERSE expects a string")
        return "str"
    if opcode == "ROTATE":
        _expect_keys(expr, {"op", "value", "amount"}, "ROTATE")
        if _validate_expr_017(expr["value"], allowed_operations, depth + 1) != "str":
            raise SemanticIRValidationError("ROTATE expects a string")
        if _validate_expr_017(expr["amount"], allowed_operations, depth + 1) != "int":
            raise SemanticIRValidationError("ROTATE amount must be an integer")
        return "str"
    if opcode == "CONCAT":
        _expect_keys(expr, {"op", "values"}, "CONCAT")
        values = expr["values"]
        if not isinstance(values, list) or not 2 <= len(values) <= MAX_CONCAT_ARGS_017:
            raise SemanticIRValidationError("CONCAT requires a bounded list of at least two values")
        if any(_validate_expr_017(item, allowed_operations, depth + 1) != "str" for item in values):
            raise SemanticIRValidationError("CONCAT values must be strings")
        return "str"
    if opcode == "RETURN":
        _expect_keys(expr, {"op", "value"}, "RETURN")
        if _validate_expr_017(expr["value"], allowed_operations, depth + 1) != "str":
            raise SemanticIRValidationError("RETURN expects a string")
        return "str"
    raise SemanticIRValidationError(f"unsupported semantic opcode: {opcode}")


def validate_semantic_ir_017(program: Any, allowed_operations: set[str] | frozenset[str]) -> None:
    if not isinstance(program, Mapping):
        raise SemanticIRValidationError("semantic IR root must be an object")
    _expect_keys(program, {"format", "version", "input_type", "output_type", "expr"}, "root")
    if program["format"] != SEMANTIC_IR_FORMAT_017 or program["version"] != SEMANTIC_IR_VERSION_017:
        raise SemanticIRValidationError("unsupported semantic IR format or version")
    if program["input_type"] != "str" or program["output_type"] != "str":
        raise SemanticIRValidationError("semantic IR must be str -> str")
    if _validate_expr_017(program["expr"], set(allowed_operations)) != "str":
        raise SemanticIRValidationError("semantic IR must return str")
    if program["expr"].get("op") != "RETURN":
        raise SemanticIRValidationError("root expression must be RETURN")


def _execute_expr_017(expr: Mapping[str, Any], value: str, allowed_operations: set[str]) -> Any:
    opcode = expr["op"]
    if opcode == "INPUT":
        return value
    if opcode == "INT":
        return expr["value"]
    if opcode == "CALL":
        api = expr["api"]
        if api not in allowed_operations:
            raise SemanticIRExecutionError("CALL API is not allowed")
        argument = _execute_expr_017(expr["args"][0], value, allowed_operations)
        try:
            return getattr(air_synth_012, api)(argument)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise SemanticIRExecutionError(f"API call failed: {api}") from exc
    if opcode == "REVERSE":
        return _execute_expr_017(expr["value"], value, allowed_operations)[::-1]
    if opcode == "ROTATE":
        text = _execute_expr_017(expr["value"], value, allowed_operations)
        amount = _execute_expr_017(expr["amount"], value, allowed_operations)
        if not text:
            return text
        amount %= len(text)
        return text[amount:] + text[:amount]
    if opcode == "CONCAT":
        return "".join(_execute_expr_017(item, value, allowed_operations) for item in expr["values"])
    if opcode == "RETURN":
        result = _execute_expr_017(expr["value"], value, allowed_operations)
        if not isinstance(result, str):
            raise SemanticIRExecutionError("semantic IR returned a non-string")
        return result
    raise SemanticIRExecutionError(f"unknown semantic opcode: {opcode}")


def execute_semantic_ir_017(program: Mapping[str, Any], value: str, allowed_operations: set[str] | frozenset[str]) -> str:
    validate_semantic_ir_017(program, allowed_operations)
    if not isinstance(value, str):
        raise SemanticIRExecutionError("input must be a string")
    return _execute_expr_017(program["expr"], value, set(allowed_operations))


def _compile_expr_017(expr: Mapping[str, Any]) -> str:
    opcode = expr["op"]
    if opcode == "INPUT":
        return "value"
    if opcode == "INT":
        return repr(expr["value"])
    if opcode == "CALL":
        return f"{expr['api']}({_compile_expr_017(expr['args'][0])})"
    if opcode == "REVERSE":
        return f"({_compile_expr_017(expr['value'])})[::-1]"
    if opcode == "ROTATE":
        source = _compile_expr_017(expr["value"])
        amount = _compile_expr_017(expr["amount"])
        return f"(({source})[{amount}:] + ({source})[:{amount}])"
    if opcode == "CONCAT":
        return "(" + " + ".join(_compile_expr_017(item) for item in expr["values"]) + ")"
    if opcode == "RETURN":
        return _compile_expr_017(expr["value"])
    raise SemanticIRValidationError(f"unknown semantic opcode: {opcode}")


def compile_semantic_ir_python_017(program: Mapping[str, Any], allowed_operations: set[str] | frozenset[str]) -> str:
    validate_semantic_ir_017(program, allowed_operations)
    api_names = sorted({
        expr["api"] for expr in _walk_expr_017(program["expr"]) if expr.get("op") == "CALL"
    })
    imports = f"from air_synth_012 import {', '.join(api_names)}\n\n" if api_names else ""
    expression = _compile_expr_017(program["expr"])
    return imports + "def transform(value: str) -> str:\n    return " + expression + "\n"


def _walk_expr_017(expr: Mapping[str, Any]):
    yield expr
    for key in ("value", "amount"):
        child = expr.get(key)
        if isinstance(child, Mapping):
            yield from _walk_expr_017(child)
    children = expr.get("args") or expr.get("values") or []
    if isinstance(children, list):
        for child in children:
            if isinstance(child, Mapping):
                yield from _walk_expr_017(child)


def oracle_call_ir_017(operation: str) -> dict[str, Any]:
    """Canonical minimum IR for the opaque API family used by the benchmark."""
    return {
        "format": SEMANTIC_IR_FORMAT_017,
        "version": SEMANTIC_IR_VERSION_017,
        "input_type": "str",
        "output_type": "str",
        "expr": {"op": "RETURN", "value": {"op": "CALL", "api": operation, "args": [{"op": "INPUT"}]}},
    }


__all__ = [
    "SEMANTIC_IR_VERSION_017", "SEMANTIC_IR_FORMAT_017", "SEMANTIC_IR_OPCODES_017",
    "SemanticIRValidationError", "SemanticIRExecutionError", "canonical_semantic_ir_json_017",
    "validate_semantic_ir_017", "execute_semantic_ir_017", "compile_semantic_ir_python_017",
    "oracle_call_ir_017",
]
