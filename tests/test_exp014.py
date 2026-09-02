import json
import re
import tempfile
import unittest
from pathlib import Path

from air_core.exp014 import (
    CONTEXT_PROMPT_HASH_014,
    DECOMPOSITION_PROMPT_HASH_014,
    EXP014_VERSION,
    ModelLedger014,
    RANKING_PROMPT_HASH_014,
    make_ranking_cases_014,
    run_context_block_014,
    run_decomposition_block_014,
    run_exp014,
    run_ranking_block_014,
)
from air_core.model_client import Completion, ModelUnavailable
from air_core.store import ExperimentStore


class ScriptedClient:
    timeout_seconds = 0.01

    def chat_json(self, prompt: str, **kwargs):
        if "required_capabilities" in prompt:
            if "analytical value" in prompt:
                value = ["cap-alpha", "cap-beta"]
            elif "report-ready" in prompt:
                value = ["cap-alpha", "cap-beta", "cap-gamma"]
            else:
                value = ["cap-alpha", "cap-unknown"]
            payload = {"required_capabilities": value}
        elif "Select the capability artifact" in prompt:
            match = re.search(r'"skill_id": "([^"]+)"', prompt)
            payload = {"skill_id": match.group(1) if match else None}
        elif "indistinguishable normalization" in prompt:
            payload = {"skill_id": None}
        elif "skill_id" in prompt:
            match = re.search(r'"skill_id": "([^"]+)"', prompt)
            payload = {"skill_id": match.group(1) if match else None}
        else:
            payload = {"result": "AIR_CONTEXT_OK_014"}
        text = json.dumps(payload)
        return Completion(text, 0.001, len(prompt) // 4, len(text) // 4, {})


class TimeoutClient:
    timeout_seconds = 0.01

    def chat_json(self, *args, **kwargs):
        raise ModelUnavailable("model runtime unavailable: test timeout")


class Experiment014Tests(unittest.TestCase):
    def test_fixture_has_controlled_distractors_and_ambiguous_case(self) -> None:
        cases = make_ranking_cases_014()
        self.assertEqual([len(case.candidates) for case in cases], [5, 3, 2, 4])
        self.assertTrue(cases[2].intentionally_ambiguous)
        self.assertIsNone(cases[2].target_skill_id)
        self.assertEqual(len({item.skill_id for item in cases[0].candidates}), 5)

    def test_frozen_model_blocks_record_hashes_and_selection(self) -> None:
        client = ScriptedClient()
        ledger = ModelLedger014()
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(str(Path(directory) / "air.db"))
            ranking = run_ranking_block_014(client, store, ledger)
            context = run_context_block_014(client, store, ledger)
            decomposition = run_decomposition_block_014(client, store, ledger)
        self.assertEqual(ranking["prompt_sha256"], RANKING_PROMPT_HASH_014)
        self.assertEqual(context["prompt_sha256"], CONTEXT_PROMPT_HASH_014)
        self.assertEqual(decomposition["decomposition_prompt_sha256"], DECOMPOSITION_PROMPT_HASH_014)
        self.assertEqual(ranking["top3_ranking_accuracy"], 1.0)
        self.assertEqual(ranking["safe_abstention_count"], 1)
        self.assertGreater(context["conditions"][0]["input_tokens"], context["conditions"][-1]["input_tokens"])
        self.assertEqual(decomposition["decomposition_failure_count"], 1)
        self.assertEqual(ledger.calls, 4 + 8 + 8)

    def test_timeout_is_a_recorded_negative_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(str(Path(directory) / "air.db"))
            report = run_exp014(client=TimeoutClient(), store=store, report_directory=directory, heldout_limit=1)
        self.assertEqual(report["version"], EXP014_VERSION)
        self.assertGreater(report["model_accounting"]["timeout_count"], 0)
        self.assertEqual(report["model_accounting"]["total_model_calls"], report["model_accounting"]["timeout_count"])
        self.assertTrue(report["report_file"].endswith(".json"))
        self.assertIn("timeout", report["failure_taxonomy"])
        self.assertTrue(report["regression"]["source_skill_immutability"])


if __name__ == "__main__":
    unittest.main()
