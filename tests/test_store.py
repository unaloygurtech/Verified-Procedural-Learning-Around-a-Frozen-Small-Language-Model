from pathlib import Path
import tempfile
import unittest

from air_core.store import ExperimentStore


class StoreTests(unittest.TestCase):
    def test_records_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(str(Path(directory) / "air.db"))
            run_id = store.record_run(
                kind="test",
                prompt="p",
                response="r",
                elapsed_seconds=0.1,
                prompt_tokens=1,
                generated_tokens=1,
                passed=True,
            )
            self.assertEqual(run_id, 1)
            with store.connect() as connection:
                row = connection.execute("SELECT passed FROM runs WHERE id = 1").fetchone()
            self.assertEqual(row["passed"], 1)

    def test_skill_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(str(Path(directory) / "air.db"))
            skill_id = store.upsert_skill(name="demo", body="candidate body")
            self.assertEqual(skill_id, 1)
            store.set_skill_state(name="demo", state="active")
            with store.connect() as connection:
                row = connection.execute(
                    "SELECT body, state FROM skills WHERE name = 'demo'"
                ).fetchone()
            self.assertEqual(dict(row), {"body": "candidate body", "state": "active"})

    def test_rejects_unknown_skill_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(str(Path(directory) / "air.db"))
            with self.assertRaises(ValueError):
                store.upsert_skill(name="demo", body="body", state="unsafe")


if __name__ == "__main__":
    unittest.main()
