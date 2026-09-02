from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ModelUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Completion:
    text: str
    elapsed_seconds: float
    prompt_tokens: int | None
    generated_tokens: int | None
    raw: dict[str, Any]


class LlamaCppClient:
    def __init__(self, base_url: str, timeout_seconds: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _json(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            raise ModelUnavailable(f"model runtime unavailable: {exc}") from exc

    def health(self) -> dict[str, Any]:
        return self._json("/health")

    def complete(self, prompt: str, *, max_tokens: int = 128, temperature: float = 0.0) -> Completion:
        started = time.perf_counter()
        result = self._json(
            "/completion",
            {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "seed": 42,
                "cache_prompt": True,
            },
        )
        elapsed = time.perf_counter() - started
        timings = result.get("timings", {})
        return Completion(
            text=result.get("content", ""),
            elapsed_seconds=elapsed,
            prompt_tokens=timings.get("prompt_n"),
            generated_tokens=timings.get("predicted_n"),
            raw=result,
        )

    def chat_json(self, prompt: str, *, max_tokens: int = 128, seed: int = 42) -> Completion:
        started = time.perf_counter()
        result = self._json(
            "/v1/chat/completions",
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "Return only one valid JSON object. Do not add markdown or commentary.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "seed": seed,
                "response_format": {"type": "json_object"},
            },
        )
        elapsed = time.perf_counter() - started
        usage = result.get("usage", {})
        choices = result.get("choices", [])
        text = "" if not choices else choices[0].get("message", {}).get("content", "")
        return Completion(
            text=text,
            elapsed_seconds=elapsed,
            prompt_tokens=usage.get("prompt_tokens"),
            generated_tokens=usage.get("completion_tokens"),
            raw=result,
        )

    def chat(self, prompt: str, *, max_tokens: int = 512, thinking: bool = False) -> Completion:
        started = time.perf_counter()
        mode = "/think" if thinking else "/no_think"
        result = self._json(
            "/v1/chat/completions",
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"{mode}\nAnalyze the evidence carefully. End with exactly one "
                            "JSON object containing the requested fields."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2 if thinking else 0.0,
                "seed": 42,
            },
        )
        elapsed = time.perf_counter() - started
        usage = result.get("usage", {})
        choices = result.get("choices", [])
        text = "" if not choices else choices[0].get("message", {}).get("content", "")
        return Completion(
            text=text,
            elapsed_seconds=elapsed,
            prompt_tokens=usage.get("prompt_tokens"),
            generated_tokens=usage.get("completion_tokens"),
            raw=result,
        )
