import json
import re
import tempfile
import unittest
from pathlib import Path

from air_core.exp015 import ModelLedger015
from air_core.exp019 import (
    EXP019_VERSION,
    PROMPT_HASHES_019,
    _ordered_candidates,
    _rank_019,
    _ranking_pool_019,
    make_document_ranking_families_019,
    run_exp019,
)
from air_core.model_client import Completion
from air_core.store import ExperimentStore


class FirstValid019Client:
    timeout_seconds = 0.01

    def chat_json(self, prompt: str, **kwargs):
        match = re.search(r'"candidate_id":"(choice_[0-9a-f]+)"', prompt)
        payload = {"candidate_id": match.group(1) if match else None}
        text = json.dumps(payload)
        return Completion(text, 0.001, max(1, len(prompt) // 4), max(1, len(text) // 4), {})


class Invalid019Client(FirstValid019Client):
    def chat_json(self, prompt: str, **kwargs):
        text = json.dumps({"candidate_id": "outside_the_frozen_set"})
        return Completion(text, 0.001, 2, 2, {})


class Experiment019Tests(unittest.TestCase):
    def test_twenty_families_have_public_ambiguity(self):
        families = make_document_ranking_families_019()
        self.assertEqual(len(families), 20)
        self.assertEqual(len({item.data_seed for item in families}), 20)
        for item in families:
            _, pool = _ranking_pool_019(item)
            self.assertGreaterEqual(len(pool), 2)
            self.assertLessEqual(len(pool), 4)

    def test_prompt_set_is_frozen_and_hashed(self):
        self.assertEqual(set(PROMPT_HASHES_019), {
            "no_doc", "correct_doc", "wrong_doc", "distractor_doc",
            "counterfactual_doc", "hybrid",
        })
        self.assertTrue(all(len(value) == 64 for value in PROMPT_HASHES_019.values()))

    def test_candidate_permutation_is_deterministic(self):
        item = make_document_ranking_families_019()[0]
        _, pool = _ranking_pool_019(item)
        first = _ordered_candidates(pool, item.data_seed, 0)
        repeated = _ordered_candidates(pool, item.data_seed, 0)
        second = _ordered_candidates(pool, item.data_seed, 1)
        self.assertEqual([x.ast_hash for x in first], [x.ast_hash for x in repeated])
        self.assertNotEqual([x.ast_hash for x in first], [x.ast_hash for x in second])

    def test_invalid_candidate_id_is_rejected(self):
        item = make_document_ranking_families_019()[0]
        _, pool = _ranking_pool_019(item)
        with tempfile.TemporaryDirectory() as directory:
            result = _rank_019(
                Invalid019Client(), ExperimentStore(str(Path(directory) / "air.db")),
                ModelLedger015(), item, pool, prompt_key="wrong_doc",
                documentation=item.wrong_docs, representation="compact_ir",
                ordering=0, seed=1,
            )
        self.assertFalse(result["selection_valid"])
        self.assertEqual(result["selection_reason"], "invalid_candidate_id")

    def test_full_scripted_run_and_resume_preserve_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_exp019(
                client=FirstValid019Client(),
                store=ExperimentStore(str(Path(directory) / "air.db")),
                report_directory=directory,
            )
            self.assertEqual(report["version"], EXP019_VERSION)
            self.assertEqual(report["part_a"]["summary"]["families"], 12)
            self.assertEqual(report["part_a"]["summary"]["canonical_activation"], 12)
            self.assertEqual(report["part_b"]["summary"]["families"], 20)
            self.assertEqual(report["part_c"]["model_avoidance_rate"], 1.0)
            self.assertEqual(report["model_accounting"]["total_model_calls"], 148)
            resumed = run_exp019(
                client=FirstValid019Client(),
                store=ExperimentStore(str(Path(directory) / "resume.db")),
                report_directory=directory,
                resume_from=report["report_file"],
            )
        self.assertEqual(resumed["model_accounting"]["total_model_calls"], 148)
        self.assertEqual(resumed["part_c"]["arms"]["hybrid"]["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
