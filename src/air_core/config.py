from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    model_url: str
    db_path: str
    context_size: int

    @classmethod
    def from_env(cls) -> "Settings":
        context_size = int(os.getenv("AIR_CONTEXT_SIZE", "4096"))
        if context_size < 512:
            raise ValueError("AIR_CONTEXT_SIZE must be at least 512")
        return cls(
            model_url=os.getenv("AIR_MODEL_URL", "http://model-runtime:8080").rstrip("/"),
            db_path=os.getenv("AIR_DB_PATH", "/workspace/data/air.db"),
            context_size=context_size,
        )

