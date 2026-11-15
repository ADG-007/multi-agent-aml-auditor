"""Grades a run against the answer key.

Two labels per account:

- the key label (EvidenceLedgerEntry.expected_disposition);
- the design label: the same rule without the forged-document channel, because the ID check
  only pauses a case and never scores it. The fake-ID accounts are graded against both.

Reported apart: the cash-business traps, structurer recall by style, KYC recall, and the ID
pause on the fake-ID accounts. Reads data/_ground_truth/, so pipeline code never imports it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from src.evaluation.stage2_subset import load_ledger
from src.schemas import CORROBORATION_STAGE_MIN, Disposition, EvidenceLedgerEntry, SeverityTier


CLEAR = Disposition.CLEAR.value
MONITOR = Disposition.MONITOR.value
ESCALATE = Disposition.SUSPICIOUS_ESCALATE.value
LABELS = (CLEAR, MONITOR, ESCALATE)
STYLES = ("naive", "small", "medium", "mixed")


@dataclass(frozen=True)
class Outcome:
    """What a run decided for one account. Stage confidences are None when a stage did not run."""

    account_id: str
    disposition: str
    paused: bool = False
    transaction_pattern: float | None = None
    kyc: float | None = None
    adverse_media: float | None = None
    graph: float | None = None


def design_label(entry: EvidenceLedgerEntry) -> str:
    """The key's rule with the forged-document channel left out."""
    channels = entry.real_channel_count - int(entry.forged_document_planted)
    if channels >= 2 or entry.adverse_media_hit == SeverityTier.CONFIRMED_SANCTIONS:
        return ESCALATE
    return MONITOR if channels == 1 else CLEAR


@dataclass
class Grade:
    accounts: int = 0
    exact_key: int = 0
    exact_design: int = 0
    bad: int = 0
    caught: int = 0
    clean: int = 0
    false_alarms: int = 0
    confusion: Counter = field(default_factory=Counter)  # (key label, predicted)
    kyc_planted: int = 0
    kyc_found: int = 0
    kyc_false_high: int = 0
    kyc_honest: int = 0
    cash_business: int = 0
    cash_business_alarms: int = 0
    structurers: Counter = field(default_factory=Counter)  # style -> accounts
    structurers_caught: Counter = field(default_factory=Counter)
    structurers_scored: Counter = field(default_factory=Counter)
    fake_ids: int = 0
    fake_ids_paused: int = 0
    fake_ids_exact_key: int = 0
    fake_ids_exact_design: int = 0
    false_pauses: int = 0
    misses: list[str] = field(default_factory=list)

    @property
    def exact_match(self) -> float:
        return _share(self.exact_key, self.accounts)

    @property
    def catch_rate(self) -> float:
        return _share(self.caught, self.bad)

    @property
    def false_alarm_rate(self) -> float:
        return _share(self.false_alarms, self.clean)

    @property
    def kyc_recall(self) -> float:
        return _share(self.kyc_found, self.kyc_planted)


def grade(
    outcomes: Iterable[Outcome], ledger: Mapping[str, EvidenceLedgerEntry] | None = None
) -> Grade:
    ledger = ledger or load_ledger()
    g = Grade()
    for o in sorted(outcomes, key=lambda o: o.account_id):
        e = ledger[o.account_id]
        key, design = e.expected_disposition.value, design_label(e)
        flagged = o.disposition != CLEAR
        g.accounts += 1
        g.exact_key += o.disposition == key
        g.exact_design += o.disposition == design
        g.confusion[(key, o.disposition)] += 1
        if o.disposition != key:
            g.misses.append(f"{o.account_id}: key {key}, design {design}, got {o.disposition}")

        if key == CLEAR:
            g.clean += 1
            g.false_alarms += flagged
        else:
            g.bad += 1
            g.caught += flagged

        kyc_high = o.kyc is not None and o.kyc >= CORROBORATION_STAGE_MIN
        if e.kyc_mismatch_planted:
            g.kyc_planted += 1
            g.kyc_found += kyc_high
        else:
            g.kyc_honest += 1
            g.kyc_false_high += kyc_high

        if "cash_business" in e.decoy_types:
            g.cash_business += 1
            g.cash_business_alarms += flagged

        if e.structuring_planted:
            style = e.structuring_style or "unknown"
            g.structurers[style] += 1
            g.structurers_caught[style] += flagged
            scored = o.transaction_pattern is not None
            g.structurers_scored[style] += (
                scored and o.transaction_pattern >= CORROBORATION_STAGE_MIN
            )

        if e.forged_document_planted:
            g.fake_ids += 1
            g.fake_ids_paused += o.paused
            g.fake_ids_exact_key += o.disposition == key
            g.fake_ids_exact_design += o.disposition == design
        else:
            g.false_pauses += o.paused
    return g


def render(g: Grade, title: str) -> str:
    lines = [
        title,
        f"  accounts                 {g.accounts}",
        f"  exact match (key)        {g.exact_key}/{g.accounts} = {g.exact_match:.3f}",
        f"  exact match (design)     {g.exact_design}/{g.accounts} = "
        f"{_share(g.exact_design, g.accounts):.3f}",
        f"  catch rate               {g.caught}/{g.bad} = {g.catch_rate:.3f}",
        f"  false alarm rate         {g.false_alarms}/{g.clean} = {g.false_alarm_rate:.3f}",
        f"  KYC recall (score>=0.40) {g.kyc_found}/{g.kyc_planted} = {g.kyc_recall:.3f}",
        f"  KYC false HIGH           {g.kyc_false_high}/{g.kyc_honest}",
        f"  cash-business alarms     {g.cash_business_alarms}/{g.cash_business} (reported apart)",
        f"  fake IDs paused          {g.fake_ids_paused}/{g.fake_ids}; exact vs key "
        f"{g.fake_ids_exact_key}, vs design {g.fake_ids_exact_design}; "
        f"false pauses {g.false_pauses}",
        "  structurers by style     "
        + ", ".join(
            f"{s} caught {g.structurers_caught[s]}/{g.structurers[s]} "
            f"(scored {g.structurers_scored[s]})"
            for s in STYLES
            if g.structurers[s]
        ),
        "  confusion (key -> got)   "
        + "; ".join(
            f"{k} -> {p}: {g.confusion[(k, p)]}"
            for k in LABELS
            for p in LABELS
            if g.confusion[(k, p)]
        ),
    ]
    return "\n".join(lines)


def _share(part: int, whole: int) -> float:
    return part / whole if whole else 0.0
