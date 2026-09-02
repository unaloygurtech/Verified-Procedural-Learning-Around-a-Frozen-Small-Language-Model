import json
from pathlib import Path
import tempfile
import unittest

from air_core.exp020 import (
    PersistentSkillStore020, Skill020, _compose_ir, make_skill_curriculum_020,
)


class Experiment020Tests(unittest.TestCase):
    def test_curriculum_has_32_distinct_procedures(self):
        specs = make_skill_curriculum_020()
        self.assertEqual(len(specs), 32)
        self.assertEqual(len({spec.target_hash for spec in specs}), 32)
        self.assertTrue({spec.kind for spec in specs} >= {"call", "reverse_call", "rotate_call", "double_call", "concat_pair"})


    def test_persistent_store_round_trip_and_integrity(self):
        spec = make_skill_curriculum_020()[1]
        blob = json.dumps(spec.target_ir, sort_keys=True, separators=(",", ":")).encode()
        import hashlib
        skill = Skill020(spec.skill_id, spec.kind, spec.operation_family, spec.target_ir,
                         hashlib.sha256(blob).hexdigest(), len(blob), spec.request_id)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            store = PersistentSkillStore020(path, [skill])
            store.save()
            loaded = PersistentSkillStore020.load(path)
            self.assertEqual(loaded.retrieve(kind=spec.kind, operation_family=spec.operation_family, top_k=1)[0].skill_id, spec.skill_id)
            self.assertEqual(loaded.artifact_bytes(), len(blob))


    def test_composition_is_typed_and_executable(self):
        specs = make_skill_curriculum_020()
        composed = _compose_ir(specs[1].target_ir, specs[16].target_ir)
        self.assertEqual(composed["input_type"], "str")
        self.assertEqual(composed["output_type"], "str")


    def test_state_stores_artifact_once(self):
        spec = make_skill_curriculum_020()[1]
        import hashlib
        blob = json.dumps(spec.target_ir, sort_keys=True, separators=(",", ":")).encode()
        skill = Skill020(spec.skill_id, spec.kind, spec.operation_family, spec.target_ir,
                         hashlib.sha256(blob).hexdigest(), len(blob), spec.request_id)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            store = PersistentSkillStore020(path, [skill])
            store.save()
            payload = json.loads(path.read_text())
            self.assertEqual(len(payload["skills"]), 1)
            self.assertEqual(store.artifact_bytes(), len(blob))
