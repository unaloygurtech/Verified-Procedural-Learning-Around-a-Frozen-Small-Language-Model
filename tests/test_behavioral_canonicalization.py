import hashlib
import json
import unittest

from air_core.behavioral_canonicalization import (
    canonical_cost_key_019,
    canonicalize_verified_019,
    normalized_ast_hash_019,
)
from air_core.exp018 import make_candidate_families_018
from air_core.exp019 import false_equivalence_controls_019
from air_core.program_search import Candidate018, search_candidates_018
from air_core.semantic_ir import oracle_call_ir_017


def candidate(candidate_id, ir):
    raw = json.dumps(ir, sort_keys=True, separators=(",", ":"))
    return Candidate018(candidate_id, ir, 3, hashlib.sha256(raw.encode()).hexdigest())


class BehavioralCanonicalizationTests(unittest.TestCase):
    def test_rotate_zero_normalizes_to_direct_call(self):
        family = make_candidate_families_018()[0]
        operation = family.family_id.rsplit("-", 1)[-1]
        direct_ir = oracle_call_ir_017(operation)
        wrapped_ir = {**direct_ir, "expr": {"op": "RETURN", "value": {
            "op": "ROTATE", "value": direct_ir["expr"]["value"],
            "amount": {"op": "INT", "value": 0},
        }}}
        direct = candidate("direct", direct_ir)
        wrapped = candidate("wrapped", wrapped_ir)
        self.assertEqual(normalized_ast_hash_019(direct), normalized_ast_hash_019(wrapped))
        self.assertLess(canonical_cost_key_019(direct), canonical_cost_key_019(wrapped))

    def test_canonicalization_is_deterministic_and_activates(self):
        family = make_candidate_families_018()[0]
        search = search_candidates_018(family.discovery)
        forward = canonicalize_verified_019(search.public_survivors, family.discovery, family.validation, family.edge)
        reverse = canonicalize_verified_019(tuple(reversed(search.public_survivors)), family.discovery, family.validation, family.edge)
        self.assertIsNotNone(forward.selected)
        self.assertEqual(forward.selected.ast_hash, reverse.selected.ast_hash)
        self.assertTrue(forward.stable)
        self.assertEqual(len(forward.equivalence_classes), 1)

    def test_six_false_equivalence_controls_do_not_merge(self):
        controls = false_equivalence_controls_019()
        self.assertEqual(len(controls), 6)
        self.assertTrue(all(item["public_equal"] for item in controls))
        self.assertTrue(all(item["class_count"] == 2 for item in controls))
        self.assertFalse(any(item["false_merge"] for item in controls))

    def test_hidden_and_edge_witnesses_are_separate(self):
        controls = {item["control"]: item for item in false_equivalence_controls_019()}
        self.assertFalse(controls["same_public_different_hidden"]["hidden_equal"])
        self.assertTrue(controls["same_public_hidden_different_edge"]["hidden_equal"])
        self.assertFalse(controls["same_public_hidden_different_edge"]["edge_equal"])


if __name__ == "__main__":
    unittest.main()
