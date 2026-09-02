"""Experiment-local opaque APIs for AIR Experiment 0012.

The names are generated deterministically and exported through ``__getattr__``
so each robustness seed exposes a distinct operation namespace.  This package
did not exist before the experiment and is not a public API.
"""

from __future__ import annotations

import json
import random
from typing import Callable


SEEDS_012 = (1201, 1202, 1203)
FAMILY_KINDS_012 = ("shards", "numbers", "object", "mixed", "runs")


def operation_names(seed: int) -> dict[str, str]:
    if seed not in SEEDS_012:
        raise KeyError(f"unsupported experiment seed: {seed}")
    rng = random.Random(seed * 7919 + 12)
    names: dict[str, str] = {}
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for kind in FAMILY_KINDS_012:
        while True:
            candidate = "op_" + "".join(rng.choice(alphabet) for _ in range(7))
            if candidate not in names.values():
                names[kind] = candidate
                break
    return names


def apply_operation(seed: int, kind: str, value: str) -> str:
    if seed not in SEEDS_012 or kind not in FAMILY_KINDS_012:
        raise KeyError("unknown synthetic operation")
    variant = SEEDS_012.index(seed)
    if kind == "shards":
        delimiter = ("|", "~", "^")[variant]
        parts = value.split(delimiter)
        return delimiter.join(part[::-1].upper() for part in reversed(parts))
    if kind == "numbers":
        factor = (3, 5, 7)[variant]
        offset = (1, 2, 4)[variant]
        numbers = sorted({int(item.strip()) * factor + offset for item in value.split(",")}, reverse=True)
        return ";".join(str(item) for item in numbers)
    if kind == "object":
        payload = json.loads(value)
        keys = sorted(payload, reverse=bool(variant % 2))
        return "|".join(f"{key}={payload[key]}" for key in keys)
    if kind == "mixed":
        text, raw_number = value.rsplit("#", 1)
        number = int(raw_number)
        transformed = number * (variant + 2) + (variant + 3)
        rotated = text[variant + 1:] + text[:variant + 1]
        return f"{rotated[::-1].upper()}:{transformed}"
    if not value:
        return ""
    separator = ("/", ":", ".")[variant]
    groups: list[str] = []
    current = value[0]
    count = 1
    for character in value[1:]:
        if character == current:
            count += 1
        else:
            groups.append(f"{current}{count}")
            current, count = character, 1
    groups.append(f"{current}{count}")
    return separator.join(groups)


def _registry() -> dict[str, Callable[[str], str]]:
    registry: dict[str, Callable[[str], str]] = {}
    for seed in SEEDS_012:
        for kind, name in operation_names(seed).items():
            registry[name] = lambda value, selected_seed=seed, selected_kind=kind: apply_operation(selected_seed, selected_kind, value)
    return registry


_OPERATIONS = _registry()


def __getattr__(name: str) -> Callable[[str], str]:
    try:
        return _OPERATIONS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc


__all__ = sorted(_OPERATIONS)
