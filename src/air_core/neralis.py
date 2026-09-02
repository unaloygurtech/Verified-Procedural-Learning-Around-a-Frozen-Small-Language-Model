from __future__ import annotations

from dataclasses import dataclass
import json


@dataclass(frozen=True)
class NeralisCase:
    case_id: str
    code: str
    value: int
    signal: str
    tag: str

    def payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "value": self.value,
            "signal": self.signal,
            "tag": self.tag,
        }

    def expected(self) -> dict[str, object]:
        if self.signal == "amber":
            score = self.value + 5
            label = "north"
        elif self.signal == "cobalt":
            score = self.value * 2
            label = "west"
        elif self.signal == "ivory":
            score = self.value - 3
            label = "east"
        else:
            raise ValueError(f"unknown signal: {self.signal}")

        return {
            "key": f"{self.code}:{self.tag}",
            "score": score,
            "label": label,
        }


TRAIN_CASES = (
    NeralisCase("train-01", "zaf", 4, "amber", "mori"),
    NeralisCase("train-02", "pel", 7, "amber", "nex"),
    NeralisCase("train-03", "dorin", 12, "amber", "uv"),
    NeralisCase("train-04", "kiv", 3, "amber", "salo"),
    NeralisCase("train-05", "ruma", 8, "cobalt", "tiv"),
    NeralisCase("train-06", "bex", 13, "cobalt", "nora"),
    NeralisCase("train-07", "falin", 6, "cobalt", "zep"),
    NeralisCase("train-08", "wok", 11, "cobalt", "lume"),
    NeralisCase("train-09", "qer", 10, "ivory", "pavo"),
    NeralisCase("train-10", "sulon", 15, "ivory", "kir"),
    NeralisCase("train-11", "jex", 2, "ivory", "vani"),
    NeralisCase("train-12", "tora", 19, "ivory", "bem"),
)

VALIDATION_CASES = (
    NeralisCase("validation-01", "havor", 16, "amber", "ceti"),
    NeralisCase("validation-02", "lum", 9, "amber", "rako"),
    NeralisCase("validation-03", "prax", 20, "cobalt", "dumi"),
    NeralisCase("validation-04", "serin", 5, "cobalt", "avo"),
    NeralisCase("validation-05", "golan", 14, "ivory", "mepi"),
)

HELD_OUT_CASES = (
    NeralisCase("heldout-01", "navik", 18, "amber", "zoru"),
    NeralisCase("heldout-02", "cem", 5, "amber", "pila"),
    NeralisCase("heldout-03", "ulor", 21, "amber", "dex"),
    NeralisCase("heldout-04", "brin", 24, "cobalt", "savu"),
    NeralisCase("heldout-05", "mekal", 7, "cobalt", "oti"),
    NeralisCase("heldout-06", "yox", 18, "cobalt", "faren"),
    NeralisCase("heldout-07", "caldor", 23, "ivory", "nemi"),
    NeralisCase("heldout-08", "vuren", 4, "ivory", "qast"),
)


SKILL_TEXT = """# Neralis-3 normalization skill

Input fields are `code` (text), `value` (integer), `signal`, and `tag`.

1. `key`: join `code`, a colon (`:`), and `tag`, preserving their characters.
2. `score`:
   - amber: `value + 5`
   - cobalt: `value * 2`
   - ivory: `value - 3`
3. `label`: amber maps to `north`, cobalt maps to `west`, and ivory maps to
   `east`.
4. Return exactly one JSON object with keys `key`, `score`, and `label`.
"""


def raw_experiences() -> str:
    lines = ["Verified past Neralis-3 experiences:"]
    for case in TRAIN_CASES:
        lines.append(
            f"input={json.dumps(case.payload(), sort_keys=True)} "
            f"verified_output={json.dumps(case.expected(), sort_keys=True)}"
        )
    return "\n".join(lines)


def task_prompt(case: NeralisCase, context: str | None = None) -> str:
    sections = []
    if context:
        sections.append(context)
    sections.extend(
        [
            "Apply Neralis-3 normalization to the new input below.",
            f"input={json.dumps(case.payload(), sort_keys=True)}",
            "Return only a JSON object with keys key, score, and label.",
        ]
    )
    return "\n\n".join(sections)


def parse_response(text: str) -> dict[str, object] | None:
    stripped = text.strip()
    decoder = json.JSONDecoder()
    last_object: dict[str, object] | None = None
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            last_object = value
    return last_object
