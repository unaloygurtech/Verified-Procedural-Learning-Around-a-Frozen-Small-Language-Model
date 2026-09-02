"""Synthetic API created for AIR Experiment 0010.

The namespace and operation names are experiment-local.  The behavior is
deterministic and intentionally not documented here for the model; the
model-facing documentation is assembled separately in exp010.py.
"""

from __future__ import annotations


def kel(value: str) -> str:
    """Reverse segment order and characters, using ``~`` as separator."""
    segments = value.split("~")
    return "~".join(segment[::-1] for segment in reversed(segments))


def nam(value: str) -> str:
    """Swap adjacent characters, leaving a final odd character in place."""
    pairs = [value[index + 1] + value[index] for index in range(0, len(value) - 1, 2)]
    if len(value) % 2:
        pairs.append(value[-1])
    return "".join(pairs)


def tesh(value: str) -> str:
    """Rotate a string left by two positions."""
    return value[2:] + value[:2]


def vum(value: str) -> str:
    """Move even-indexed characters before odd-indexed characters."""
    return value[::2] + value[1::2]


__all__ = ["kel", "nam", "tesh", "vum"]
