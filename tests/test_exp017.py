import json
import re
import tempfile
import unittest
from pathlib import Path

from air_core.exp012 import make_robustness_families_012
from air_core.exp017 import (
    ARMS_017,
    EXP017_VERSION,
    candidate_set_017,
    oracle_semantic_plan_017,
    run_exp017,
    validate_semantic_plan_017,
)
from air_core.model_client import Completion
from air_core.store import ExperimentStore


class Scripted017Client:
    timeout_seconds = 0.01

    def _operation(self, prompt):
        match = re.search(r"from air_synth_012 import ([A-Za-z_][A-Za-z0-9_]*)", prompt)
        if match:
            return match.group(1)
        match = re.search(r'"operation":\s*"(op_[a-z]+)"', prompt)
        return match.group(1) if match else ""

    def chat_json(self, prompt: str, **kwargs):
        operation = self._operation(prompt)
        if "normative documentation record" in prompt:
            match = re.search(r"\[(record-[0-9a-f]+)\]\nNormative manual", prompt)
            payload = {"doc_id": match.group(1) if match else None}
        elif "semantic body for transform" in prompt:
            payload = {"semantic_body": f"return {operation}(value)"}
        elif '"candidate_id"' in prompt:
            # The correct candidate is identified from the frozen candidate JSON.
            candidates = re.findall(r"(candidate_\d+): (\{.*?\})(?=\n|$)", prompt)
            correct = next((cid for cid, raw in candidates if (
                (lambda item: item.get("expr", {}).get("value", {}).get("op") == "CALL" and
                 item.get("expr", {}).get("value", {}).get("api") == operation and
                 item.get("expr", {}).get("value", {}).get("args") == [{"op": "INPUT"}])(json.loads(raw))
            )), candidates[0][0])
            payload = {"candidate_id": correct}
        elif "candidate semantic IR" in prompt or "minimal typed semantic IR" in prompt or "Compile this minimum" in prompt:
            payload = {"format": "AIR-SEMANTIC-IR", "version": 1, "input_type": "str", "output_type": "str", "expr": {"op": "RETURN", "value": {"op": "CALL", "api": operation, "args": [{"op": "INPUT"}]}}}
        elif "minimum executable semantic plan" in prompt:
            payload = {"operation": operation, "arguments": ["INPUT"], "ordering": ["single_call"], "return": "str"}
        else:
            payload = {}
        text = json.dumps(payload)
        return Completion(text, 0.001, max(1, len(prompt) // 4), max(1, len(text) // 4), {})


class Experiment017Tests(unittest.TestCase):
    def test_minimum_plan_policy_is_deterministic(self):
        family = make_robustness_families_012(1201)[0]
        self.assertEqual(validate_semantic_plan_017(oracle_semantic_plan_017(family), family), (True, None))
        invalid = {"operation": "wrong", "arguments": ["INPUT"]}
        self.assertFalse(validate_semantic_plan_017(invalid, family)[0])

    def test_candidate_fixture_has_no_duplicates(self):
        family = make_robustness_families_012(1201)[0]
        candidates = candidate_set_017(family)
        self.assertEqual(len(candidates), 5)
        self.assertEqual(len({json.dumps(item["ir"], sort_keys=True) for item in candidates}), 5)

    def test_scripted_ladder_reaches_infrastructure_upper_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_exp017(client=Scripted017Client(), store=ExperimentStore(str(Path(directory) / "air.db")), report_directory=directory)
            self.assertTrue(Path(report["report_file"]).exists())
        self.assertEqual(report["version"], EXP017_VERSION)
        self.assertEqual(report["ladder"]["families_attempted"], 8)
        self.assertEqual(set(report["arms"]), set(ARMS_017))
        self.assertEqual(report["arms"]["G_oracle_compiler"]["active"], 8)
        self.assertEqual(report["arms"]["F_candidate_selection"]["candidate_selection_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
