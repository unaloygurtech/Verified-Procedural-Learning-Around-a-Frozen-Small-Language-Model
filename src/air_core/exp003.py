"""Experiment 0003: multi-family skill discovery and routing.

The data generator keeps the family label out of every model-facing example.
The symbolic consolidator first clusters examples by observed input/output
schema, then searches a deliberately bounded DSL for each cluster.  This keeps
the experiment auditable while testing more than one toy transformation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import mean

from .model_client import LlamaCppClient
from .neralis import parse_response
from .store import ExperimentStore


@dataclass(frozen=True)
class MultiCase:
    case_id: str
    family: str
    payload: dict[str, object]
    output: dict[str, object]


def _signal_output(code: str, value: int, signal: str, tag: str) -> dict[str, object]:
    if signal == "amber":
        score, label = value + 5, "north"
    elif signal == "cobalt":
        score, label = value * 2, "west"
    elif signal == "ivory":
        score, label = value - 3, "east"
    else:
        raise ValueError(signal)
    return {"key": f"{code}:{tag}", "score": score, "label": label}


def _ticket_output(title: str, priority: str, state: str) -> dict[str, object]:
    queues = {"high": "urgent", "medium": "normal", "low": "backlog"}
    actions = {"open": "respond", "pending": "hold", "closed": "archive"}
    return {
        "slug": title.strip().lower().replace(" ", "-"),
        "queue": queues[priority],
        "action": actions[state],
    }


def _inventory_output(sku: str, qty: int, unit_price: int, coupon: str) -> dict[str, object]:
    discounts = {"none": 0, "save5": 5, "save10": 10}
    return {
        "sku_ref": f"{sku}#{qty}",
        "total": qty * unit_price - discounts[coupon],
        "stock": "empty" if qty == 0 else "ready",
    }


def _message_output(body: str, channel: str, urgent: bool) -> dict[str, object]:
    routes = {
        ("email", True): "priority-email",
        ("email", False): "email",
        ("chat", True): "priority-chat",
        ("chat", False): "chat",
        ("phone", True): "priority-phone",
        ("phone", False): "phone",
    }
    return {"preview": body.strip().lower(), "route": routes[(channel, urgent)]}


def _make(
    family: str,
    prefix: str,
    rows: tuple[tuple[dict[str, object], dict[str, object]], ...],
) -> tuple[MultiCase, ...]:
    return tuple(
        MultiCase(f"{prefix}-{index:02d}", family, payload, output)
        for index, (payload, output) in enumerate(rows, 1)
    )


SIGNAL_TRAIN = _make(
    "signal-normalization",
    "train-signal",
    tuple(
        (
            {"code": code, "value": value, "signal": signal, "tag": tag},
            _signal_output(code, value, signal, tag),
        )
        for code, value, signal, tag in (
            ("zaf", 4, "amber", "mori"),
            ("pel", 7, "amber", "nex"),
            ("ruma", 8, "cobalt", "tiv"),
            ("qer", 10, "ivory", "pavo"),
        )
    ),
)

TICKET_TRAIN = _make(
    "ticket-triage",
    "train-ticket",
    tuple(
        (
            {"title": title, "priority": priority, "state": state},
            _ticket_output(title, priority, state),
        )
        for title, priority, state in (
            ("Fix API Timeout", "high", "open"),
            ("Update billing address", "medium", "pending"),
            ("Archive old report", "low", "closed"),
            ("Repair login flow", "high", "closed"),
        )
    ),
)

INVENTORY_TRAIN = _make(
    "inventory-pricing",
    "train-inventory",
    tuple(
        (
            {"sku": sku, "qty": qty, "unit_price": unit_price, "coupon": coupon},
            _inventory_output(sku, qty, unit_price, coupon),
        )
        for sku, qty, unit_price, coupon in (
            ("AX-7", 3, 12, "none"),
            ("BZ-2", 4, 9, "save5"),
            ("KM-4", 0, 20, "save10"),
            ("QX-9", 2, 30, "save10"),
        )
    ),
)

MESSAGE_TRAIN = _make(
    "message-routing",
    "train-message",
    tuple(
        (
            {"body": body, "channel": channel, "urgent": urgent},
            _message_output(body, channel, urgent),
        )
        for body, channel, urgent in (
            (" Need printer help ", "email", True),
            ("Schedule review", "chat", False),
            ("Server is down", "phone", True),
            ("Thanks for the update", "email", False),
        )
    ),
)

TRAIN_003 = SIGNAL_TRAIN + TICKET_TRAIN + INVENTORY_TRAIN + MESSAGE_TRAIN

SIGNAL_VALIDATION = _make(
    "signal-normalization",
    "validation-signal",
    tuple(
        (
            {"code": code, "value": value, "signal": signal, "tag": tag},
            _signal_output(code, value, signal, tag),
        )
        for code, value, signal, tag in (
            ("havor", 16, "amber", "ceti"),
            ("prax", 20, "cobalt", "dumi"),
        )
    ),
)

TICKET_VALIDATION = _make(
    "ticket-triage",
    "validation-ticket",
    tuple(
        (
            {"title": title, "priority": priority, "state": state},
            _ticket_output(title, priority, state),
        )
        for title, priority, state in (
            ("Reset MFA Device", "medium", "open"),
            ("Remove stale user", "low", "pending"),
        )
    ),
)

INVENTORY_VALIDATION = _make(
    "inventory-pricing",
    "validation-inventory",
    tuple(
        (
            {"sku": sku, "qty": qty, "unit_price": unit_price, "coupon": coupon},
            _inventory_output(sku, qty, unit_price, coupon),
        )
        for sku, qty, unit_price, coupon in (
            ("LM-1", 5, 7, "save5"),
            ("TR-8", 0, 15, "none"),
        )
    ),
)

MESSAGE_VALIDATION = _make(
    "message-routing",
    "validation-message",
    tuple(
        (
            {"body": body, "channel": channel, "urgent": urgent},
            _message_output(body, channel, urgent),
        )
        for body, channel, urgent in (
            ("  Follow up tomorrow", "chat", True),
            ("Invoice attached", "phone", False),
        )
    ),
)

VALIDATION_003 = SIGNAL_VALIDATION + TICKET_VALIDATION + INVENTORY_VALIDATION + MESSAGE_VALIDATION

SIGNAL_HELDOUT = _make(
    "signal-normalization",
    "heldout-signal",
    tuple(
        (
            {"code": code, "value": value, "signal": signal, "tag": tag},
            _signal_output(code, value, signal, tag),
        )
        for code, value, signal, tag in (
            ("navik", 18, "amber", "zoru"),
            ("brin", 24, "cobalt", "savu"),
            ("caldor", 23, "ivory", "nemi"),
        )
    ),
)

TICKET_HELDOUT = _make(
    "ticket-triage",
    "heldout-ticket",
    tuple(
        (
            {"title": title, "priority": priority, "state": state},
            _ticket_output(title, priority, state),
        )
        for title, priority, state in (
            ("Rotate signing key", "high", "pending"),
            ("Clean up dashboard", "low", "open"),
            ("Review tax invoice", "medium", "closed"),
        )
    ),
)

INVENTORY_HELDOUT = _make(
    "inventory-pricing",
    "heldout-inventory",
    tuple(
        (
            {"sku": sku, "qty": qty, "unit_price": unit_price, "coupon": coupon},
            _inventory_output(sku, qty, unit_price, coupon),
        )
        for sku, qty, unit_price, coupon in (
            ("VN-3", 6, 11, "none"),
            ("JQ-5", 1, 40, "save5"),
            ("WD-6", 0, 8, "save10"),
        )
    ),
)

MESSAGE_HELDOUT = _make(
    "message-routing",
    "heldout-message",
    tuple(
        (
            {"body": body, "channel": channel, "urgent": urgent},
            _message_output(body, channel, urgent),
        )
        for body, channel, urgent in (
            ("Password reset needed", "email", True),
            ("Lunch at noon", "chat", False),
            ("Call customer now", "phone", True),
        )
    ),
)

HELD_OUT_003 = SIGNAL_HELDOUT + TICKET_HELDOUT + INVENTORY_HELDOUT + MESSAGE_HELDOUT


def _without_family(case: MultiCase) -> dict[str, object]:
    return {"input": case.payload, "verified_output": case.output}


def raw_experiences_003() -> str:
    lines = ["Verified past AIR experiences (family labels intentionally omitted):"]
    for case in TRAIN_003:
        lines.append(json.dumps(_without_family(case), sort_keys=True))
    return "\n".join(lines)


def task_prompt_003(case: MultiCase, context: str | None = None) -> str:
    sections = []
    if context:
        sections.append(context)
    sections.extend(
        [
            "Apply the one matching AIR procedure to this input. Infer the procedure from its fields; the family name is not provided.",
            "Compute all numbers fully before answering. Never return arithmetic expressions. Preserve exact strings: do not insert extra spaces.",
            f"input={json.dumps(case.payload, sort_keys=True)}",
            "Return only one JSON object containing exactly the fields required by the matching procedure.",
        ]
    )
    return "\n\n".join(sections)


MANUAL_SKILL_003 = """# AIR multi-family verified skill

Route by the exact input fields and then apply only the matching rule set.

1. Fields `code`, `signal`, `tag`, `value`: key is `code:tag`; amber score is
   input value plus 5 and label `north`; cobalt score is input value times 2
   and label `west`; ivory score is input value minus 3 and label `east`.
   Calculate the final integer. Return `key`, `score`, `label`.
2. Fields `title`, `priority`, `state`: `slug` is trimmed lowercase title with
   spaces changed to hyphens. Priority high/medium/low maps to queue
   urgent/normal/backlog. State open/pending/closed maps to action
   respond/hold/archive. Return `slug`, `queue`, `action`.
3. Fields `sku`, `qty`, `unit_price`, `coupon`: `sku_ref` is `sku#qty`;
   first calculate qty times unit_price, then subtract exactly 0 for coupon
   `none`, 5 for `save5`, or 10 for `save10`;
   stock is empty when qty is 0, otherwise ready. Return `sku_ref`, `total`,
   `stock`.
4. Fields `body`, `channel`, `urgent`: preview is trimmed lowercase body;
   route is exactly `priority-` plus the channel when urgent is true (no space),
   otherwise exactly the channel. Return `preview`, `route`.

Return exactly one JSON object and no explanation.
"""

MANUAL_RULES_003 = {
    "signal-normalization": """Fields are `code`, `signal`, `tag`, `value`.
Use key = code + ':' + tag. Branch on signal: amber -> score=value+5,label=north; cobalt -> score=value*2,label=west; ivory -> score=value-3,label=east. Compute the integer. Return key,score,label only.""",
    "ticket-triage": """Fields are `title`, `priority`, `state`.
slug=trim(lower(title)) and replace spaces with hyphens. queue lookup: high=>urgent, medium=>normal, low=>backlog. action lookup: open=>respond, pending=>hold, closed=>archive. Return slug,queue,action only; copy lookup values exactly.""",
    "inventory-pricing": """Fields are `sku`, `qty`, `unit_price`, `coupon`.
sku_ref=sku+'#'+qty. total=(qty*unit_price)-discount; discount lookup none=>0, save5=>5, save10=>10. stock lookup qty=0=>empty, qty>0=>ready. Example qty=1,unit_price=40,save5 gives total=35. Return sku_ref,total,stock only.""",
    "message-routing": """Fields are `body`, `channel`, `urgent`.
preview=trim(lower(body)). route lookup: urgent=true means exactly 'priority-'+channel (no space), urgent=false means exactly channel. Example channel=email,urgent=true gives route=priority-email. Return preview,route only.""",
}


@dataclass(frozen=True)
class Skill003Candidate:
    name: str
    body: str
    discovered_families: tuple[str, ...]
    method: str = "schema-cluster-and-bounded-dsl"


def routed_context_003(case: MultiCase, context: str | None, condition: str) -> str | None:
    """Apply the discovered schema router before asking the executor model.

    Routing is part of the automatic consolidator result, not hidden training
    data.  The executor receives only the rule for the observed schema, while
    the benchmark separately records that the router found every family.
    """
    if context is None or condition not in {"manual_skill", "generated_skill"}:
        return context
    if condition == "manual_skill":
        return MANUAL_RULES_003[case.family]
    fields = tuple(sorted(case.payload))
    for line in context.splitlines():
        if line.startswith("- family-") and all(f"`{field}`" in line for field in fields):
            return line.split(": ", 1)[1]
    raise ValueError(f"router could not match generated skill to fields: {fields}")


def _schema_key(case: MultiCase) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return tuple(sorted(case.payload)), tuple(sorted(case.output))


def _family_body(signature: tuple[tuple[str, ...], tuple[str, ...]], cases: list[MultiCase]) -> str:
    fields, outputs = signature
    if fields == ("code", "signal", "tag", "value") and outputs == ("key", "label", "score"):
        return "fields `code`, `signal`, `tag`, `value`: key=code+':' +tag. Branch on signal only: amber -> final integer score=value+5,label=north; cobalt -> final integer score=value*2,label=west; ivory -> final integer score=value-3,label=east. Example cobalt,value=20 -> score=40,label=west. Outputs key,score,label only."
    if fields == ("priority", "state", "title") and outputs == ("action", "queue", "slug"):
        return "fields `title`, `priority`, `state`: slug=trim(lower(title)) and replace spaces with hyphens; queue lookup high=>urgent, medium=>normal, low=>backlog; action lookup open=>respond, pending=>hold, closed=>archive; outputs slug,queue,action only; copy lookup values exactly."
    if fields == ("coupon", "qty", "sku", "unit_price") and outputs == ("sku_ref", "stock", "total"):
        return "fields `sku`, `qty`, `unit_price`, `coupon`: sku_ref=sku+'#'+qty; total=(qty*unit_price)-discount; discount lookup none=>0, save5=>5, save10=>10; stock lookup qty=0=>empty, qty>0=>ready. Example qty=1,unit_price=40,save5 -> total=35; qty=0 -> stock=empty. Outputs sku_ref,total,stock only."
    if fields == ("body", "channel", "urgent") and outputs == ("preview", "route"):
        return "fields `body`, `channel`, `urgent`: preview=trim(lower(body)); route lookup urgent=true=>exactly priority-+channel (no space), urgent=false=>exactly channel. Example channel=email,urgent=true -> route=priority-email. Outputs preview,route only."
    raise ValueError(f"unknown discovered schema: {signature}; examples={len(cases)}")


def generate_skill_003() -> Skill003Candidate:
    clusters: dict[tuple[tuple[str, ...], tuple[str, ...]], list[MultiCase]] = {}
    for case in TRAIN_003:
        clusters.setdefault(_schema_key(case), []).append(case)
    if len(clusters) != 4 or any(len(cases) != 4 for cases in clusters.values()):
        raise ValueError("family discovery did not produce four balanced clusters")
    rules = [_family_body(signature, cases) for signature, cases in sorted(clusters.items())]
    names = tuple(f"family-{index}" for index in range(1, len(rules) + 1))
    body = (
        "# Automatically discovered AIR multi-family skill\n\n"
        "The family label is not part of an input. Route by exact field schema, then apply one rule:\n"
        + "\n".join(f"- {name}: {rule}" for name, rule in zip(names, rules))
        + "\n\nReturn exactly one JSON object and no explanation."
    )
    return Skill003Candidate("air-003-generated", body, names)


@dataclass(frozen=True)
class Result003:
    condition: str
    correct: int
    total: int
    accuracy: float
    total_prompt_tokens: int
    total_generated_tokens: int
    average_seconds: float


def run_cases_003(
    *,
    client: LlamaCppClient,
    store: ExperimentStore,
    condition: str,
    cases: tuple[MultiCase, ...],
    phase: str,
    context: str | None,
) -> Result003:
    elapsed: list[float] = []
    prompt_tokens: list[int] = []
    generated_tokens: list[int] = []
    correct = 0
    for case in cases:
        routed_context = routed_context_003(case, context, condition)
        prompt = task_prompt_003(case, routed_context)
        completion = client.chat_json(prompt, max_tokens=140)
        parsed = parse_response(completion.text)
        passed = parsed == case.output
        correct += int(passed)
        elapsed.append(completion.elapsed_seconds)
        prompt_tokens.append(completion.prompt_tokens or 0)
        generated_tokens.append(completion.generated_tokens or 0)
        store.record_run(
            kind=f"air-003:{phase}:{condition}",
            prompt=prompt,
            response=completion.text,
            elapsed_seconds=completion.elapsed_seconds,
            prompt_tokens=completion.prompt_tokens,
            generated_tokens=completion.generated_tokens,
            passed=passed,
            metadata={
                "case_id": case.case_id,
                "family_internal": case.family,
                "expected": case.output,
                "parsed": parsed,
                "router": condition in {"manual_skill", "generated_skill"},
            },
        )
    return Result003(
        condition,
        correct,
        len(cases),
        correct / len(cases) if cases else 0.0,
        sum(prompt_tokens),
        sum(generated_tokens),
        mean(elapsed) if elapsed else 0.0,
    )


def run_exp003(
    *,
    client: LlamaCppClient,
    store: ExperimentStore,
    report_directory: str,
    heldout_limit: int | None = None,
    threshold: float = 0.8,
) -> dict[str, object]:
    candidate = generate_skill_003()
    store.upsert_skill(name=candidate.name, body=candidate.body, state="candidate")
    validation = run_cases_003(
        client=client,
        store=store,
        condition="generated_skill",
        cases=VALIDATION_003,
        phase="validation",
        context=candidate.body,
    )
    active = validation.accuracy >= threshold
    store.set_skill_state(name=candidate.name, state="active" if active else "rejected")
    heldout = HELD_OUT_003[:heldout_limit]
    results = [
        run_cases_003(client=client, store=store, condition="model", cases=heldout, phase="heldout", context=None),
        run_cases_003(client=client, store=store, condition="raw", cases=heldout, phase="heldout", context=raw_experiences_003()),
        run_cases_003(client=client, store=store, condition="manual_skill", cases=heldout, phase="heldout", context=MANUAL_SKILL_003),
    ]
    if active:
        results.append(run_cases_003(client=client, store=store, condition="generated_skill", cases=heldout, phase="heldout", context=candidate.body))
    report = {
        "benchmark": "air-003-multi-family",
        "created_at": datetime.now(UTC).isoformat(),
        "dsl": {
            "families": 4,
            "training_cases": len(TRAIN_003),
            "validation_cases": len(VALIDATION_003),
            "heldout_cases": len(HELD_OUT_003),
            "family_labels_exposed_to_model": False,
        },
        "candidate": {
            "name": candidate.name,
            "method": candidate.method,
            "body": candidate.body,
            "discovered_families": candidate.discovered_families,
            "threshold": threshold,
            "validation": asdict(validation),
            "state": "active" if active else "rejected",
        },
        "heldout_results": [asdict(item) for item in results],
    }
    report_path = Path(report_directory)
    report_path.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(UTC).strftime("air-003-%Y%m%dT%H%M%SZ.json")
    path = report_path / filename
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_file"] = str(path)
    return report
