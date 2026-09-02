import os
import unittest
from unittest.mock import patch

from air_core.config import Settings


class SettingsTests(unittest.TestCase):
    def test_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.context_size, 4096)
        self.assertEqual(settings.model_url, "http://model-runtime:8080")

    def test_rejects_tiny_context(self) -> None:
        with patch.dict(os.environ, {"AIR_CONTEXT_SIZE": "128"}, clear=True):
            with self.assertRaises(ValueError):
                Settings.from_env()


if __name__ == "__main__":
    unittest.main()

