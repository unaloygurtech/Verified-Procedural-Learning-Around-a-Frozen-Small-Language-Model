from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import mean

from .model_client import LlamaCppClient
from .neralis import (
    HELD_OUT_CASES,
    SKILL_TEXT,
    VALIDATION_CASES,
    NeralisCase,
    parse_response,
    raw_experiences,
    task_prompt,
)
from .store import ExperimentStore
from .skill_generator import generate_skill, generate_skill_decomposed, generate_skill_symbolic


@dataclass(frozen=True)
class ConditionResult:
    condition: str
    correct: int
    total: int
    accuracy: float
    total_prompt_tokens: int
    total_generated_tokens: int
    average_seconds: float


def _context_for(condition: str) -> str | None:
    if condition == "model":
        return None
    if condition == "raw":
        return raw_experiences()
    if condition in {"skill", "manual_skill"}:
        return SKILL_TEXT
    raise ValueError(f"unknown condition: {condition}")


def run_cases(
    *,
    client: LlamaCppClient,
    store: ExperimentStore,
    condition: str,
    cases: tuple[NeralisCase, ...],
    phase: str,
    context_override: str | None = None,
) -> ConditionResult:
    prompt_tokens: list[int] = []
    generated_tokens: list[int] = []
    elapsed: list[float] = []
    correct = 0
    context = context_override if context_override is not None else _context_for(condition)

    for case in cases:
        prompt = task_prompt(case, context)
        completion = client.chat_json(prompt, max_tokens=96)
        parsed = parse_response(completion.text)
        expected = case.expected()
        passed = parsed == expected
        correct += int(passed)
        prompt_tokens.append(completion.prompt_tokens or 0)
        generated_tokens.append(completion.generated_tokens or 0)
        elapsed.append(completion.elapsed_seconds)
        store.record_run(
            kind=f"neralis:{phase}:{condition}",
            prompt=prompt,
            response=completion.text,
            elapsed_seconds=completion.elapsed_seconds,
            prompt_tokens=completion.prompt_tokens,
            generated_tokens=completion.generated_tokens,
            passed=passed,
            metadata={
                "case_id": case.case_id,
                "condition": condition,
                "phase": phase,
                "expected": expected,
                "parsed": parsed,
            },
        )

    total = len(cases)
    return ConditionResult(
        condition=condition,
        correct=correct,
        total=total,
        accuracy=correct / total if total else 0.0,
        total_prompt_tokens=sum(prompt_tokens),
        total_generated_tokens=sum(generated_tokens),
        average_seconds=mean(elapsed) if elapsed else 0.0,
    )


def run_neralis_benchmark(
    *,
    client: LlamaCppClient,
    store: ExperimentStore,
    report_directory: str,
    heldout_limit: int | None = None,
    skill_threshold: float = 0.8,
) -> dict[str, object]:
    validation = run_cases(
        client=client,
        store=store,
        condition="skill",
        cases=VALIDATION_CASES,
        phase="validation",
    )
    skill_active = validation.accuracy >= skill_threshold

    heldout_cases = HELD_OUT_CASES[:heldout_limit]
    conditions = ("model", "raw", "skill") if skill_active else ("model", "raw")
    results = [
        run_cases(
            client=client,
            store=store,
            condition=condition,
            cases=heldout_cases,
            phase="heldout",
        )
        for condition in conditions
    ]

    report: dict[str, object] = {
        "benchmark": "neralis-3",
        "created_at": datetime.now(UTC).isoformat(),
        "skill": {
            "origin": "manually distilled from verified training experiences",
            "threshold": skill_threshold,
            "validation": asdict(validation),
            "state": "active" if skill_active else "rejected",
        },
        "heldout_results": [asdict(result) for result in results],
    }
    report_path = Path(report_directory)
    report_path.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(UTC).strftime("neralis-%Y%m%dT%H%M%SZ.json")
    (report_path / filename).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report["report_file"] = str(report_path / filename)
    return report


def run_auto_skill_benchmark(
    *,
    client: LlamaCppClient,
    store: ExperimentStore,
    report_directory: str,
    heldout_limit: int | None = None,
    skill_threshold: float = 0.8,
    strategy: str = "one-shot",
) -> dict[str, object]:
    if strategy == "one-shot":
        candidate = generate_skill(client)
    elif strategy == "decomposed":
        candidate = generate_skill_decomposed(client)
    elif strategy == "symbolic":
        candidate = generate_skill_symbolic()
    else:
        raise ValueError(f"unknown generation strategy: {strategy}")
    store.upsert_skill(name=candidate.name, body=candidate.body, state="candidate")
    store.record_run(
        kind="neralis:consolidation:generated_skill",
        prompt="raw verified experiences -> candidate skill",
        response=candidate.body,
        elapsed_seconds=candidate.elapsed_seconds,
        prompt_tokens=candidate.prompt_tokens,
        generated_tokens=candidate.generated_tokens,
        passed=None,
        metadata={"skill_name": candidate.name},
    )

    validation = run_cases(
        client=client,
        store=store,
        condition="generated_skill",
        cases=VALIDATION_CASES,
        phase="validation",
        context_override=candidate.body,
    )
    skill_active = validation.accuracy >= skill_threshold
    state = "active" if skill_active else "rejected"
    store.set_skill_state(name=candidate.name, state=state)

    heldout_cases = HELD_OUT_CASES[:heldout_limit]
    results = [
        run_cases(
            client=client,
            store=store,
            condition=condition,
            cases=heldout_cases,
            phase="heldout-auto",
        )
        for condition in ("model", "raw", "manual_skill")
    ]
    if skill_active:
        results.append(
            run_cases(
                client=client,
                store=store,
                condition="generated_skill",
                cases=heldout_cases,
                phase="heldout-auto",
                context_override=candidate.body,
            )
        )

    report: dict[str, object] = {
        "benchmark": "neralis-3-auto-skill",
        "created_at": datetime.now(UTC).isoformat(),
        "candidate": {
            "name": candidate.name,
            "method": candidate.method,
            "body": candidate.body,
            "generation_prompt_tokens": candidate.prompt_tokens,
            "generation_tokens": candidate.generated_tokens,
            "generation_seconds": candidate.elapsed_seconds,
            "threshold": skill_threshold,
            "validation": asdict(validation),
            "state": state,
        },
        "heldout_results": [asdict(result) for result in results],
    }
    report_path = Path(report_directory)
    report_path.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(UTC).strftime("neralis-auto-%Y%m%dT%H%M%SZ.json")
    (report_path / filename).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report["report_file"] = str(report_path / filename)
    return report
