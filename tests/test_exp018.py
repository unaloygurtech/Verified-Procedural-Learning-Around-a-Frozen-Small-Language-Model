import json
import re
import tempfile
import unittest
from pathlib import Path

from air_core.exp018 import (
    ARMS_018,
    EXP018_VERSION,
    make_candidate_families_018,
    run_exp018,
)
from air_core.model_client import Completion
from air_core.store import ExperimentStore


class Scripted018Client:
    timeout_seconds = 0.01

    @staticmethod
    def _operation(prompt: str) -> str:
        match = re.search(r"from air_synth_012 import ([A-Za-z_][A-Za-z0-9_]*)", prompt)
        if match:
            return match.group(1)
        match = re.search(r'"api":\s*"(op_[a-z]+)"', prompt)
        return match.group(1) if match else ""

    @staticmethod
    def _find_correct(prompt: str, operation: str) -> str:
        for line in prompt.splitlines():
            if not line.startswith("candidate_") or ": " not in line:
                continue
            candidate_id, raw = line.split(": ", 1)
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            expr = item.get("expr", {}).get("value", {})
            if expr.get("op") == "CALL" and expr.get("api") == operation and expr.get("args") == [{"op": "INPUT"}]:
                return candidate_id
        return "candidate_1"

    def chat_json(self, prompt: str, **kwargs):
        operation = self._operation(prompt)
        if "Predict only the structural constraints" in prompt:
            payload = {"uses_call": True, "uses_reverse": False, "uses_rotate": False, "uses_concat": False, "max_depth": 3}
        elif "Select one executable candidate" in prompt:
            payload = {"candidate_id": self._find_correct(prompt, operation)}
        else:
            payload = {
                "format": "AIR-SEMANTIC-IR", "version": 1,
                "input_type": "str", "output_type": "str",
                "expr": {"op": "RETURN", "value": {"op": "CALL", "api": operation, "args": [{"op": "INPUT"}]}}
            }
        text = json.dumps(payload)
        return Completion(text, 0.001, max(1, len(prompt) // 4), max(1, len(text) // 4), {})


class Experiment018Tests(unittest.TestCase):
    def test_family_count_and_new_seed(self):
        families = make_candidate_families_018()
        self.assertEqual(len(families), 12)
        self.assertEqual(sum("-1203-" in family.family_id for family in families), 4)

    def test_scripted_controls_and_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_exp018(
                client=Scripted018Client(),
                store=ExperimentStore(str(Path(directory) / "air.db")),
                report_directory=directory,
            )
            self.assertEqual(report["version"], EXP018_VERSION)
            self.assertEqual(set(report["arms"]), set(ARMS_018))
            self.assertEqual(report["funnel"]["correct_candidate_generated"], 12)
            self.assertEqual(report["arms"]["E_search_oracle"]["active"], 12)
            self.assertEqual(report["comparison"]["constraint_correct_program_retention"], 1.0)
            self.assertGreater(report["arms"]["G_constraint_search"]["active"], 0)
            resumed = run_exp018(
                client=Scripted018Client(),
                store=ExperimentStore(str(Path(directory) / "air-resume.db")),
                report_directory=directory,
                resume_from=report["report_file"],
            )
            self.assertEqual(resumed["ladder"]["families_attempted"], 12)
            self.assertEqual(resumed["funnel"]["correct_candidate_generated"], 12)

    def test_invalid_candidate_id_is_not_accepted(self):
        class InvalidClient(Scripted018Client):
            def chat_json(self, prompt: str, **kwargs):
                if "Select one executable candidate" in prompt:
                    payload = {"candidate_id": "not-in-set"}
                    text = json.dumps(payload)
                    return Completion(text, 0.001, 2, 2, {})
                return super().chat_json(prompt, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            report = run_exp018(
                client=InvalidClient(),
                store=ExperimentStore(str(Path(directory) / "air.db")),
                report_directory=directory,
            )
        self.assertGreater(report["failure_counts"]["invalid_candidate_id"], 0)
        self.assertLess(report["arms"]["C_search_smollm"]["active"], 12)


if __name__ == "__main__":
    unittest.main()
