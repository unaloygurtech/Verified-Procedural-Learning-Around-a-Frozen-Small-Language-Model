import json
import re
import tempfile
import unittest
from pathlib import Path

from air_core.exp012 import ROBUSTNESS_SEEDS_012, make_robustness_families_012
from air_core.exp015 import (
    ARMS_015,
    CONTRACT_PROMPT_HASH_015,
    DIAGNOSTIC_REPAIR_PROMPT_HASH_015,
    EXP015_VERSION,
    MAX_SEMANTIC_REPAIRS_015,
    ModelLedger015,
    STRUCTURED_SYNTHESIS_PROMPT_HASH_015,
    build_structured_candidate_015,
    classify_failure_015,
    deterministic_skeleton_015,
    diagnostic_repair_prompt_015,
    expected_contract_015,
    load_checkpoint_015,
    normalized_ast_hash_015,
    run_acquisition_block_015,
    run_exp015,
    source_sha256_015,
    write_checkpoint_015,
)
from air_core.exp009 import StaticCheck009
from air_core.model_client import Completion, ModelUnavailable
from air_core.store import ExperimentStore


class ScriptedClient:
    timeout_seconds = 0.01

    def chat_json(self, prompt: str, **kwargs):
        if "Extract a machine-checkable contract" in prompt:
            match = re.search(r"from air_synth_012 import ([A-Za-z_][A-Za-z0-9_]*)", prompt)
            operation = match.group(1) if match else ""
            payload = {
                "input_type": "str", "output_type": "str", "callable": "transform",
                "allowed_imports": ["air_synth_012"], "allowed_import_members": [operation],
                "allowed_call_names": [operation], "allowed_attrs": [operation],
                "side_effect_policy": "pure; no filesystem, network, subprocess, or mutation",
                "deterministic": True, "return_requirements": "return a str",
                "known_invariants": ["exactly one top-level transform(value: str) -> str"],
            }
        elif "Return exactly one JSON object: {\"semantic_body\"" in prompt:
            match = re.search(r"from air_synth_012 import ([A-Za-z_][A-Za-z0-9_]*)", prompt)
            operation = match.group(1) if match else ""
            payload = {"semantic_body": f"return {operation}(value)"}
        elif "Return exactly one JSON object with a string field" in prompt:
            # Baseline full-program response.
            match = re.search(r"from air_synth_012 import ([A-Za-z_][A-Za-z0-9_]*)", prompt)
            operation = match.group(1) if match else "air_operation_missing"
            payload = {"code": f"from air_synth_012 import {operation}\n\ndef transform(value: str) -> str:\n    return {operation}(value)"}
        elif "Select the normative" in prompt or "normative" in prompt.lower() and "doc_id" in prompt:
            match = re.search(r"\[(record-[0-9a-f]+)\]\nNormative manual", prompt)
            payload = {"doc_id": match.group(1) if match else None}
        else:
            payload = {"result": "not-tested"}
        text = json.dumps(payload)
        return Completion(text, 0.001, max(1, len(prompt) // 4), max(1, len(text) // 4), {})


class TimeoutClient:
    timeout_seconds = 0.01

    def chat_json(self, *args, **kwargs):
        raise ModelUnavailable("model runtime unavailable: test timeout")


class Experiment015Tests(unittest.TestCase):
    def test_prompt_hashes_are_frozen(self):
        self.assertEqual(len(CONTRACT_PROMPT_HASH_015), 64)
        self.assertEqual(len(STRUCTURED_SYNTHESIS_PROMPT_HASH_015), 64)
        self.assertEqual(len(DIAGNOSTIC_REPAIR_PROMPT_HASH_015), 64)

    def test_skeleton_and_body_preserve_contract(self):
        family = make_robustness_families_012(ROBUSTNESS_SEEDS_012[0])[0]
        skeleton = deterministic_skeleton_015(family)
        candidate = build_structured_candidate_015(family, "return air_synth_012.bad_name(value)")
        self.assertIn("def transform(value: str) -> str:", skeleton)
        self.assertIn("from air_synth_012 import", candidate)
        self.assertNotIn("bad_name", skeleton)
        self.assertIn("bad_name", candidate)

    def test_wrapper_code_is_not_accepted_as_semantic_body(self):
        family = make_robustness_families_012(ROBUSTNESS_SEEDS_012[0])[0]
        with self.assertRaises(ValueError):
            build_structured_candidate_015(family, "def transform(value):\n    return value")
        with self.assertRaises(ValueError):
            build_structured_candidate_015(family, "from os import getcwd")

    def test_ast_hash_ignores_formatting_and_source_hash_does_not(self):
        first = "def transform(value: str) -> str:\n    return value\n"
        second = "def transform(value: str)->str:\n return value\n"
        self.assertEqual(normalized_ast_hash_015(first), normalized_ast_hash_015(second))
        self.assertNotEqual(source_sha256_015(first), source_sha256_015(second))
        self.assertIsNone(normalized_ast_hash_015("def transform(:"))

    def test_failure_classification_and_targeted_prompt(self):
        self.assertEqual(classify_failure_015(static=StaticCheck009(False, "syntax error: invalid syntax")), "syntax_error")
        self.assertEqual(classify_failure_015(expected="a", actual="b"), "semantic_mismatch")
        family = make_robustness_families_012(ROBUSTNESS_SEEDS_012[0])[0]
        prompt = diagnostic_repair_prompt_015(family, "semantic_mismatch", "expected='x' actual='y'", "return value")
        self.assertIn("Failure class: semantic_mismatch", prompt)
        self.assertNotIn("try again", prompt.lower())

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            write_checkpoint_015(path, {"version": EXP015_VERSION, "results": [{"family_id": "x"}]})
            self.assertEqual(load_checkpoint_015(path)["results"][0]["family_id"], "x")

    def test_scripted_acquisition_activates_all_arms(self):
        client = ScriptedClient()
        ledger = ModelLedger015()
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(str(Path(directory) / "air.db"))
            result = run_acquisition_block_015(client, store, ledger, heldout_limit=1)
        self.assertEqual(result["families_attempted"], 5)
        self.assertEqual(result["A_full_program"]["activated"], 5)
        self.assertEqual(result["B_structured"]["activated"], 5)
        self.assertEqual(result["C_diagnostic"]["activated"], 5)
        self.assertEqual(result["wrong_activation_count"], 0)

    def test_timeout_is_negative_and_checkpointed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(str(Path(directory) / "air.db"))
            report = run_exp015(client=TimeoutClient(), store=store, report_directory=directory, heldout_limit=1)
            self.assertEqual(report["version"], EXP015_VERSION)
            self.assertTrue(Path(report["report_file"]).exists())
            self.assertGreater(report["model_accounting"]["timeout_count"], 0)
            self.assertIn("timeout", report["failure_taxonomy"])
            self.assertTrue(report["regression"]["base_library_immutable"])

    def test_repair_budget_is_frozen(self):
        self.assertEqual(MAX_SEMANTIC_REPAIRS_015, 3)


if __name__ == "__main__":
    unittest.main()
