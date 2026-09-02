import json
import re
import tempfile
import unittest
from pathlib import Path

from air_core.exp012 import make_robustness_families_012
from air_core.exp016 import (
    ARMS_016,
    EXP016_VERSION,
    SEMANTIC_FIELDS_016,
    make_contract_families_016,
    run_exp016,
    semantic_ground_truth_016,
    structural_metadata_016,
)
from air_core.model_client import Completion
from air_core.store import ExperimentStore


class ScriptedContractClient:
    timeout_seconds = 0.01

    @staticmethod
    def _ground(prompt: str):
        families = make_contract_families_016()
        for family in families:
            operation = family.family_id.rsplit("-", 1)[-1]
            if operation in prompt:
                return semantic_ground_truth_016(family)
        return semantic_ground_truth_016(families[0])

    def chat_json(self, prompt: str, **kwargs):
        operation_match = re.search(r"from air_synth_012 import ([A-Za-z_][A-Za-z0-9_]*)", prompt)
        operation = operation_match.group(1) if operation_match else ""
        if "bind every claim to evidence" in prompt:
            field = re.search(r"Requested field: ([a-z_]+)", prompt).group(1)
            ground = self._ground(prompt)
            doc_id = re.search(r"Document id: (record-[0-9a-f]+)", prompt).group(1)
            payload = {"field": field, "value": ground[field], "evidence": {"doc_id": doc_id, "quote_or_span": "Normative manual"}}
        elif "normative documentation record" in prompt:
            match = re.search(r"\[(record-[0-9a-f]+)\]\nNormative manual", prompt)
            payload = {"doc_id": match.group(1) if match else None}
        elif "Extract a machine-checkable contract" in prompt:
            payload = {
                "input_type": "str", "output_type": "str", "callable": "transform",
                "allowed_imports": ["air_synth_012"], "allowed_import_members": [operation],
                "allowed_call_names": [operation], "allowed_attrs": [operation],
                "side_effect_policy": "pure; no filesystem, network, subprocess, or mutation",
                "deterministic": True, "return_requirements": "return a str",
                "known_invariants": ["exactly one top-level transform(value: str) -> str"],
            }
        elif "Extract exactly one semantic contract field" in prompt:
            field = re.search(r"Requested field: ([a-z_]+)", prompt).group(1)
            ground = self._ground(prompt)
            payload = {"field": field, "value": ground[field]}
        elif "Extract only the semantic part" in prompt:
            # The unit test exercises protocol shape; the real model is measured
            # against family-specific semantics in the runtime benchmark.
            ground = self._ground(prompt)
            payload = ground
        elif "Fill only the semantic body" in prompt:
            payload = {"semantic_body": f"return {operation}(value)"}
        else:
            payload = {}
        text = json.dumps(payload)
        return Completion(text, 0.001, max(1, len(prompt) // 4), max(1, len(text) // 4), {})


class Experiment016Tests(unittest.TestCase):
    def test_families_and_structural_metadata_are_deterministic(self):
        families = make_contract_families_016()
        self.assertEqual(len(families), 8)
        self.assertEqual(len({family.family_id for family in families}), 8)
        for family in families:
            metadata = structural_metadata_016(family)
            self.assertEqual(metadata.module, "air_synth_012")
            self.assertEqual(metadata.callable, "transform")
            self.assertEqual(metadata.input_type, "str")
            self.assertEqual(metadata.output_type, "str")
            self.assertEqual(len(metadata.allowed_calls), 1)

    def test_scripted_run_keeps_baseline_and_completes_structured_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_exp016(
                client=ScriptedContractClient(),
                store=ExperimentStore(str(Path(directory) / "air.db")),
                report_directory=directory,
            )
            self.assertTrue(Path(report["report_file"]).exists())
        self.assertEqual(report["version"], EXP016_VERSION)
        self.assertEqual(report["acquisition"]["families_attempted"], 8)
        self.assertEqual(set(report["arms"]), set(ARMS_016))
        self.assertEqual(report["arms"]["B_structured_one_shot"]["complete_contracts"], 8)
        self.assertEqual(report["arms"]["C_field_by_field"]["complete_contracts"], 8)
        self.assertEqual(report["arms"]["D_evidence_grounded"]["complete_contracts"], 8)
        self.assertEqual(report["arms"]["D_evidence_grounded"]["evidence_supported_claim_rate"], 1.0)
        self.assertEqual(len(SEMANTIC_FIELDS_016), 7)


if __name__ == "__main__":
    unittest.main()
