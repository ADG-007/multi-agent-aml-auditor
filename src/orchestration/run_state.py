"""What one run remembers between cases. An object, not globals, so each run starts clean.

- Known-bad list: only an ESCALATE adds its devices, IPs and address. Nothing is removed.
- Flagged = latest case MONITOR or ESCALATE.
- Flagged on own evidence = still flagged with Stage 5 left out. Only these count as links.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bank import BankData
from src.schemas import (
    AUTO_VETO_THRESHOLD,
    STAGE_WEIGHTS,
    Disposition,
    InvestigationReport,
    SeverityTier,
    StageName,
    StageStatus,
)
from src.scoring import StageResult, score_case


FLAGGED_DISPOSITIONS = frozenset({Disposition.MONITOR, Disposition.SUSPICIOUS_ESCALATE})


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    disposition: Disposition
    stages_ran: frozenset[StageName]
    triggering_transaction_id: str | None
    #: Escalated on a confirmed sanctions match (Stage 4 veto). Stage 5 reads it.
    sanctions_escalation: bool = False
    #: Still flagged with Stage 5 left out of the score. Only these count as links.
    flagged_on_own: bool = True


class RunState:
    def __init__(self) -> None:
        # identifier -> the escalated account it was listed from
        self.known_bad_devices: dict[str, str] = {}
        self.known_bad_ips: dict[str, str] = {}
        self.known_bad_addresses: dict[str, str] = {}
        self._history: dict[str, list[CaseRecord]] = {}

    # ----------------------------------------------------------------------------------
    # Case history
    # ----------------------------------------------------------------------------------

    def next_case_id(self, account_id: str) -> str:
        return f"CASE-{account_id}-{len(self._history.get(account_id, ())) + 1}"

    def history(self, account_id: str) -> tuple[CaseRecord, ...]:
        return tuple(self._history.get(account_id, ()))

    def standing(self, account_id: str) -> Disposition | None:
        """The disposition of the account's latest case, or None if it has none."""
        cases = self._history.get(account_id)
        return cases[-1].disposition if cases else None

    def is_flagged(self, account_id: str) -> bool:
        return self.standing(account_id) in FLAGGED_DISPOSITIONS

    def is_flagged_on_own(self, account_id: str) -> bool:
        """Flagged, and not only because of its links (the accounts Stage 5 counts)."""
        latest = self.latest(account_id)
        return self.is_flagged(account_id) and latest is not None and latest.flagged_on_own

    def latest(self, account_id: str) -> CaseRecord | None:
        cases = self._history.get(account_id)
        return cases[-1] if cases else None

    def record_case(self, report: InvestigationReport, bank: BankData) -> CaseRecord:
        """Append a finalised case; an escalation also feeds the known-bad list."""
        account_id = report.target_account
        expected = self.next_case_id(account_id)
        if report.case_id != expected:
            raise ValueError(f"case id {report.case_id} out of sequence; expected {expected}")

        record = CaseRecord(
            case_id=report.case_id,
            disposition=report.disposition,
            stages_ran=frozenset(
                StageName(stage)
                for stage, status in report.stage_status.items()
                if status == StageStatus.RAN
            ),
            triggering_transaction_id=report.triggering_transaction_id,
            sanctions_escalation=_sanctions_escalation(report),
            flagged_on_own=_flagged_on_own(report),
        )
        self._history.setdefault(account_id, []).append(record)

        if report.disposition == Disposition.SUSPICIOUS_ESCALATE:
            self._list_identifiers(account_id, bank)
        return record

    # ----------------------------------------------------------------------------------
    # Known-bad list
    # ----------------------------------------------------------------------------------

    def _list_identifiers(self, account_id: str, bank: BankData) -> None:
        for t in bank.transactions(account_id):
            if t.device_id:
                self.known_bad_devices.setdefault(t.device_id, account_id)
            if t.ip_address:
                self.known_bad_ips.setdefault(t.ip_address, account_id)
        self.known_bad_addresses.setdefault(bank.customer_for(account_id).address, account_id)


_FINDINGS = {
    StageName.TRANSACTION_PATTERN: "transaction_pattern_findings",
    StageName.KYC: "kyc_findings",
    StageName.ADVERSE_MEDIA: "adverse_media_findings",
}


def _flagged_on_own(report: InvestigationReport) -> bool:
    """The same judge on the stages that RAN, Stage 5 left out: still MONITOR or ESCALATE?"""
    if report.disposition not in FLAGGED_DISPOSITIONS:
        return False
    results = {}
    for stage, name in _FINDINGS.items():
        findings = getattr(report, name)
        if report.stage_status.get(stage.value) == StageStatus.RAN and findings is not None:
            results[stage] = StageResult(StageStatus.RAN, findings.confidence)
    assert StageName.GRAPH in STAGE_WEIGHTS and set(_FINDINGS) | {StageName.GRAPH} == set(
        STAGE_WEIGHTS
    )
    return score_case(results).disposition in FLAGGED_DISPOSITIONS


def _sanctions_escalation(report: InvestigationReport) -> bool:
    am = report.adverse_media_findings
    return (
        report.disposition == Disposition.SUSPICIOUS_ESCALATE
        and am is not None
        and am.severity_tier == SeverityTier.CONFIRMED_SANCTIONS
        and am.confidence >= AUTO_VETO_THRESHOLD
    )
