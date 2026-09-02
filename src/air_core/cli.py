from __future__ import annotations

import argparse
import json
import os
import sys

from .benchmark import run_auto_skill_benchmark, run_neralis_benchmark
from .config import Settings
from .exp003 import run_exp003
from .exp004 import run_exp004
from .exp005 import run_exp005
from .exp006 import run_exp006
from .exp007 import run_exp007
from .exp008 import run_exp008
from .exp009 import run_exp009
from .exp010 import run_exp010
from .exp011 import run_exp011
from .exp012 import run_exp012
from .exp013 import run_exp013
from .exp014 import run_exp014
from .exp015 import run_exp015
from .exp016 import run_exp016
from .exp017 import run_exp017
from .exp018 import run_exp018
from .exp019 import run_exp019
from .exp020 import run_exp020
from .model_client import LlamaCppClient, ModelUnavailable
from .store import ExperimentStore


SMOKE_PROMPT = """Return exactly this object:
{\"air\":\"ready\",\"sum\":17}
The sum is 9 + 8.
"""


def doctor(settings: Settings) -> int:
    print(f"model_url={settings.model_url}")
    print(f"db_path={settings.db_path}")
    print(f"context_size={settings.context_size}")
    store = ExperimentStore(settings.db_path)
    with store.connect():
        print("database=ok")
    try:
        health = LlamaCppClient(settings.model_url, timeout_seconds=10).health()
    except ModelUnavailable as exc:
        print(f"model=unavailable ({exc})")
        return 2
    print(f"model={health.get('status', 'unknown')}")
    return 0


def smoke(settings: Settings) -> int:
    client = LlamaCppClient(settings.model_url)
    try:
        completion = client.chat_json(SMOKE_PROMPT, max_tokens=64)
    except ModelUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2

    raw_text = completion.text.strip()
    passed = False
    try:
        payload = json.loads(raw_text)
        passed = payload == {"air": "ready", "sum": 17}
    except json.JSONDecodeError:
        payload = {"raw": raw_text}

    run_id = ExperimentStore(settings.db_path).record_run(
        kind="smoke",
        prompt=SMOKE_PROMPT,
        response=raw_text,
        elapsed_seconds=completion.elapsed_seconds,
        prompt_tokens=completion.prompt_tokens,
        generated_tokens=completion.generated_tokens,
        passed=passed,
        metadata={"parsed": payload},
    )
    print(json.dumps({
        "run_id": run_id,
        "passed": passed,
        "elapsed_seconds": round(completion.elapsed_seconds, 3),
        "prompt_tokens": completion.prompt_tokens,
        "generated_tokens": completion.generated_tokens,
        "response": raw_text,
    }, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="airctl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="check configuration, database, and model runtime")
    subparsers.add_parser("smoke", help="run and record a deterministic model smoke test")
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="run the Neralis-3 model/raw/skill experiment"
    )
    benchmark_parser.add_argument(
        "--heldout-limit", type=int, default=None, help="limit held-out cases for a quick run"
    )
    auto_parser = subparsers.add_parser(
        "auto-skill", help="generate, gate, and benchmark a skill from raw experiences"
    )
    auto_parser.add_argument(
        "--heldout-limit", type=int, default=None, help="limit held-out cases for a quick run"
    )
    auto_parser.add_argument(
        "--strategy", choices=("one-shot", "decomposed", "symbolic"), default="one-shot"
    )
    multi_parser = subparsers.add_parser(
        "experiment-003", help="run the multi-family discovery and routing experiment"
    )
    multi_parser.add_argument(
        "--heldout-limit", type=int, default=None, help="limit held-out cases for a quick run"
    )
    collision_parser = subparsers.add_parser(
        "experiment-004", help="run same-schema content-discriminator experiment"
    )
    collision_parser.add_argument(
        "--heldout-limit", type=int, default=None, help="limit held-out cases for a quick run"
    )
    typed_parser = subparsers.add_parser(
        "experiment-005", help="run typed executable skill synthesis experiment"
    )
    typed_parser.add_argument(
        "--heldout-limit", type=int, default=None, help="limit held-out cases for a quick run"
    )
    composition_parser = subparsers.add_parser(
        "experiment-006", help="run compositional skill reuse experiment"
    )
    composition_parser.add_argument(
        "--heldout-limit", type=int, default=None, help="limit held-out cases for a quick run"
    )
    gap_parser = subparsers.add_parser(
        "experiment-007", help="run capability-gap and missing-skill synthesis experiment"
    )
    gap_parser.add_argument(
        "--heldout-limit", type=int, default=None, help="limit held-out cases for a quick run"
    )
    python_parser = subparsers.add_parser(
        "experiment-008", help="run Python/API capability-gap experiment"
    )
    python_parser.add_argument(
        "--heldout-limit", type=int, default=None, help="limit held-out cases for a quick run"
    )
    generic_parser = subparsers.add_parser(
        "experiment-009", help="run frozen generic multi-family Python learner experiment"
    )
    generic_parser.add_argument(
        "--heldout-limit", type=int, default=None, help="limit held-out cases for a quick run"
    )
    generic_parser.add_argument(
        "--repeats", type=int, default=2, help="number of independent learner runs"
    )
    synthetic_parser = subparsers.add_parser(
        "experiment-010", help="run novel synthetic API learning experiment"
    )
    synthetic_parser.add_argument(
        "--heldout-limit", type=int, default=None, help="limit held-out cases for a quick run"
    )
    retrieval_parser = subparsers.add_parser(
        "experiment-011", help="run documentation retrieval and rule efficiency experiment"
    )
    retrieval_parser.add_argument(
        "--heldout-limit", type=int, default=None, help="limit held-out cases for a quick run"
    )
    scaling_parser = subparsers.add_parser(
        "experiment-012", help="run robust acquisition, learned-state storage, and scaling experiment"
    )
    scaling_parser.add_argument(
        "--heldout-limit", type=int, default=None, help="limit held-out cases for a quick run"
    )
    hierarchy_parser = subparsers.add_parser(
        "experiment-013", help="run hierarchical retrieval, fingerprint, dedup, and scoped composition experiment"
    )
    hierarchy_parser.add_argument(
        "--heldout-limit", type=int, default=None, help="reserved compatibility option; core 0013 is model-independent"
    )
    utilization_parser = subparsers.add_parser(
        "experiment-014", help="run frozen-model ranking, context, decomposition, acquisition, and reuse experiment"
    )
    utilization_parser.add_argument(
        "--heldout-limit", type=int, default=2, help="bounded held-out cases per acquisition family (default: 2)"
    )
    acquisition_parser = subparsers.add_parser(
        "experiment-015", help="run structured synthesis and diagnostic repair acquisition experiment"
    )
    acquisition_parser.add_argument(
        "--heldout-limit", type=int, default=None, help="limit held-out cases per active artifact (default: all 8)"
    )
    acquisition_parser.add_argument(
        "--resume", dest="resume_from", default=None, help="resume completed family checkpoints from a runtime JSON"
    )
    contract_parser = subparsers.add_parser(
        "experiment-016", help="run reliable contract induction and downstream probe experiment"
    )
    contract_parser.add_argument(
        "--resume", dest="resume_from", default=None, help="resume completed family checkpoints from a runtime JSON"
    )
    boundary_parser = subparsers.add_parser(
        "experiment-017", help="measure the frozen SmolLM3-3B semantic capability boundary"
    )
    boundary_parser.add_argument(
        "--resume", dest="resume_from", default=None, help="resume completed family checkpoints from a runtime JSON"
    )
    search_parser = subparsers.add_parser(
        "experiment-018", help="run verified candidate search and program induction experiment"
    )
    search_parser.add_argument(
        "--resume", dest="resume_from", default=None, help="resume completed family checkpoints from a runtime JSON"
    )
    grounding_parser = subparsers.add_parser(
        "experiment-019", help="run behavioral canonicalization and documentation-grounded ranking experiment"
    )
    grounding_parser.add_argument(
        "--resume", dest="resume_from", default=None, help="resume completed family checkpoints from a runtime JSON"
    )
    accumulation_parser = subparsers.add_parser(
        "experiment-020", help="run persistent skill accumulation, reuse, transfer, and efficiency experiment"
    )
    accumulation_parser.add_argument(
        "--resume", dest="resume_from", default=None, help="resume completed checkpoints from a runtime JSON"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    if args.command == "doctor":
        return doctor(settings)
    if args.command == "smoke":
        return smoke(settings)
    if args.command == "benchmark":
        report = run_neralis_benchmark(
            client=LlamaCppClient(settings.model_url),
            store=ExperimentStore(settings.db_path),
            report_directory=os.getenv("AIR_REPORT_DIR", "/workspace/data/runs"),
            heldout_limit=args.heldout_limit,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "auto-skill":
        report = run_auto_skill_benchmark(
            client=LlamaCppClient(settings.model_url),
            store=ExperimentStore(settings.db_path),
            report_directory=os.getenv("AIR_REPORT_DIR", "/workspace/data/runs"),
            heldout_limit=args.heldout_limit,
            strategy=args.strategy,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "experiment-003":
        report = run_exp003(
            client=LlamaCppClient(settings.model_url),
            store=ExperimentStore(settings.db_path),
            report_directory=os.getenv("AIR_REPORT_DIR", "/workspace/data/runs"),
            heldout_limit=args.heldout_limit,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "experiment-004":
        report = run_exp004(
            client=LlamaCppClient(settings.model_url),
            store=ExperimentStore(settings.db_path),
            report_directory=os.getenv("AIR_REPORT_DIR", "/workspace/data/runs"),
            heldout_limit=args.heldout_limit,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "experiment-005":
        report = run_exp005(
            client=LlamaCppClient(settings.model_url),
            store=ExperimentStore(settings.db_path),
            report_directory=os.getenv("AIR_REPORT_DIR", "/workspace/data/runs"),
            heldout_limit=args.heldout_limit,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "experiment-006":
        report = run_exp006(
            client=LlamaCppClient(settings.model_url),
            store=ExperimentStore(settings.db_path),
            report_directory=os.getenv("AIR_REPORT_DIR", "/workspace/data/runs"),
            heldout_limit=args.heldout_limit,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "experiment-007":
        report = run_exp007(
            client=LlamaCppClient(settings.model_url),
            store=ExperimentStore(settings.db_path),
            report_directory=os.getenv("AIR_REPORT_DIR", "/workspace/data/runs"),
            heldout_limit=args.heldout_limit,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "experiment-008":
        report = run_exp008(
            client=LlamaCppClient(settings.model_url),
            store=ExperimentStore(settings.db_path),
            report_directory=os.getenv("AIR_REPORT_DIR", "/workspace/data/runs"),
            heldout_limit=args.heldout_limit,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "experiment-009":
        report = run_exp009(
            client=LlamaCppClient(settings.model_url),
            store=ExperimentStore(settings.db_path),
            report_directory=os.getenv("AIR_REPORT_DIR", "/workspace/data/runs"),
            heldout_limit=args.heldout_limit,
            repeats=args.repeats,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "experiment-010":
        report = run_exp010(
            client=LlamaCppClient(settings.model_url),
            store=ExperimentStore(settings.db_path),
            report_directory=os.getenv("AIR_REPORT_DIR", "/workspace/data/runs"),
            heldout_limit=args.heldout_limit,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "experiment-011":
        report = run_exp011(
            client=LlamaCppClient(settings.model_url),
            store=ExperimentStore(settings.db_path),
            report_directory=os.getenv("AIR_REPORT_DIR", "/workspace/data/runs"),
            heldout_limit=args.heldout_limit,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "experiment-012":
        report = run_exp012(
            client=LlamaCppClient(settings.model_url),
            store=ExperimentStore(settings.db_path),
            report_directory=os.getenv("AIR_REPORT_DIR", "/workspace/data/runs"),
            heldout_limit=args.heldout_limit,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "experiment-013":
        report = run_exp013(
            client=LlamaCppClient(settings.model_url),
            store=ExperimentStore(settings.db_path),
            report_directory=os.getenv("AIR_REPORT_DIR", "/workspace/data/runs"),
            heldout_limit=args.heldout_limit,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "experiment-014":
        report = run_exp014(
            client=LlamaCppClient(settings.model_url),
            store=ExperimentStore(settings.db_path),
            report_directory=os.getenv("AIR_REPORT_DIR", "/workspace/data/runs"),
            heldout_limit=args.heldout_limit,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "experiment-015":
        report = run_exp015(
            client=LlamaCppClient(settings.model_url),
            store=ExperimentStore(settings.db_path),
            report_directory=os.getenv("AIR_REPORT_DIR", "/workspace/data/runs"),
            heldout_limit=args.heldout_limit,
            resume_from=args.resume_from,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "experiment-016":
        report = run_exp016(
            client=LlamaCppClient(settings.model_url),
            store=ExperimentStore(settings.db_path),
            report_directory=os.getenv("AIR_REPORT_DIR", "/workspace/data/runs"),
            resume_from=args.resume_from,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "experiment-017":
        report = run_exp017(
            client=LlamaCppClient(settings.model_url),
            store=ExperimentStore(settings.db_path),
            report_directory=os.getenv("AIR_REPORT_DIR", "/workspace/data/runs"),
            resume_from=args.resume_from,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "experiment-018":
        report = run_exp018(
            client=LlamaCppClient(settings.model_url),
            store=ExperimentStore(settings.db_path),
            report_directory=os.getenv("AIR_REPORT_DIR", "/workspace/data/runs"),
            resume_from=args.resume_from,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "experiment-019":
        report = run_exp019(
            client=LlamaCppClient(settings.model_url),
            store=ExperimentStore(settings.db_path),
            report_directory=os.getenv("AIR_REPORT_DIR", "/workspace/data/runs"),
            resume_from=args.resume_from,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "experiment-020":
        report = run_exp020(
            client=LlamaCppClient(settings.model_url),
            store=ExperimentStore(settings.db_path),
            report_directory=os.getenv("AIR_REPORT_DIR", "/workspace/data/runs"),
            resume_from=args.resume_from,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
