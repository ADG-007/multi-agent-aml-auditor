"""Grade a 50-account run against the answer key (mvp_accounts.json). Evaluation only.

Metrics:
- exact: the verdict equals the answer (CLEAR / MONITOR / SUSPICIOUS-ESCALATE).
- detection recall / precision: "flagged" means MONITOR or ESCALATE. Recall = bad accounts
  flagged / bad accounts. Precision = flagged accounts that are bad / flagged accounts.
- escalate recall / precision: the same for SUSPICIOUS-ESCALATE alone.
- trap false alarms: traps flagged / traps.
A BLOCKED_BY_ADMIN_REVIEW verdict counts as flagged (the admin stopped it).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from src.schemas import Disposition, InvestigationReport


SHORT = {
    Disposition.CLEAR.value: "CLEAR",
    Disposition.MONITOR.value: "MONITOR",
    Disposition.SUSPICIOUS_ESCALATE.value: "ESCALATE",
    Disposition.BLOCKED_BY_ADMIN_REVIEW.value: "BLOCKED",
}
FIELDS = {
    "TP": "transaction_pattern_findings",
    "KYC": "kyc_findings",
    "AM": "adverse_media_findings",
    "G": "graph_findings",
}


@dataclass(frozen=True)
class Row:
    account_id: str
    kind: str
    group: str
    expected: str
    got: str
    confidence: float
    stages: dict[str, str]
    forgery: str
    case_id: str

    @property
    def right(self) -> bool:
        return self.expected == self.got


def rows(subset: dict, latest: dict[str, InvestigationReport]) -> list[Row]:
    out = []
    for item in subset["accounts"]:
        r = latest[item["account_id"]]
        stages = {}
        for label, name in FIELDS.items():
            f = getattr(r, name)
            status = r.stage_status.get(_stage_key(label), "NOT_ROUTED")
            stages[label] = f"{f.confidence:.2f}" if f is not None else _dash(status)
        vision = r.vision_findings
        out.append(
            Row(
                account_id=item["account_id"],
                kind=item["kind"],
                group=item["group"],
                expected=SHORT[item["expected"]],
                got=SHORT[r.disposition.value],
                confidence=r.confidence,
                stages=stages,
                forgery=f"{vision.forgery_confidence:.2f}" if vision else "-",
                case_id=r.case_id,
            )
        )
    return out


def metrics(table: list[Row]) -> dict[str, float | int | str]:
    bad = [r for r in table if r.kind == "bad"]
    flagged = [r for r in table if r.got != "CLEAR"]
    esc_true = [r for r in table if r.expected == "ESCALATE"]
    esc_pred = [r for r in table if r.got == "ESCALATE"]
    traps = [r for r in table if r.kind == "trap"]
    clean = [r for r in table if r.kind == "clean"]

    def ratio(a: int, b: int) -> str:
        return f"{a}/{b} = {a / b:.2f}" if b else f"{a}/0"

    return {
        "exact": ratio(sum(r.right for r in table), len(table)),
        "detection_recall": ratio(sum(r.got != "CLEAR" for r in bad), len(bad)),
        "detection_precision": ratio(sum(r.kind == "bad" for r in flagged), len(flagged)),
        "escalate_recall": ratio(sum(r.got == "ESCALATE" for r in esc_true), len(esc_true)),
        "escalate_precision": ratio(sum(r.expected == "ESCALATE" for r in esc_pred), len(esc_pred)),
        "trap_false_alarms": ratio(sum(r.got != "CLEAR" for r in traps), len(traps)),
        "clean_false_alarms": ratio(sum(r.got != "CLEAR" for r in clean), len(clean)),
    }


def confusion(table: list[Row]) -> Counter[tuple[str, str]]:
    return Counter((r.expected, r.got) for r in table)


def format_table(table: list[Row]) -> str:
    head = (
        f"{'Account':<9} {'Kind':<5} {'Group':<26} {'Expected':<8} {'Got':<8} {'':<2} "
        f"{'Score':>6}  {'TP':>4} {'KYC':>4} {'AM':>4} {'G':>4} {'ID':>4}  Case"
    )
    lines = [head, "-" * len(head)]
    for r in table:
        s = r.stages
        lines.append(
            f"{r.account_id:<9} {r.kind:<5} {r.group[:26]:<26} {r.expected:<8} {r.got:<8} "
            f"{'ok' if r.right else 'XX':<2} {r.confidence:>6.3f}  {s['TP']:>4} {s['KYC']:>4} "
            f"{s['AM']:>4} {s['G']:>4} {r.forgery:>4}  {r.case_id}"
        )
    return "\n".join(lines)


def format_summary(table: list[Row]) -> str:
    m = metrics(table)
    c = confusion(table)
    labels = ["CLEAR", "MONITOR", "ESCALATE", "BLOCKED"]
    lines = ["Confusion (rows = expected, columns = got)"]
    lines.append(f"{'':<10}" + "".join(f"{lab:>10}" for lab in labels))
    for e in labels[:3]:
        lines.append(f"{e:<10}" + "".join(f"{c[(e, g)]:>10}" for g in labels))
    lines.append("")
    lines += [f"{k.replace('_', ' '):<22} {v}" for k, v in m.items()]
    return "\n".join(lines)


def _stage_key(label: str) -> str:
    return {"TP": "TransactionPattern", "KYC": "KYC", "AM": "AdverseMedia", "G": "Graph"}[label]


def _dash(status: str) -> str:
    return {"NOT_ROUTED": "-", "FAILED": "FAIL", "TIMED_OUT": "TIME"}.get(status, "?")
