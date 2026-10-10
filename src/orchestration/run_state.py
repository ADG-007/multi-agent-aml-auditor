"""What one run remembers between cases: the known-bad list and each account's case history.

An explicit object passed to Stage 0 and updated when a case is finalised, never module-level
globals, so every test and every run starts clean (phase_2_execution.md D5). In memory only;
Phase 10 can add persistence without changing Stage 0.

Two rules from Phase 2's kickoff live here:
- only a SUSPICIOUS-ESCALATE case adds identifiers to the known-bad list - the device IDs and
  IPs on the account's transactions plus its customer's address - and nothing is ever
  removed during a run;
- an account is "flagged" when its latest case is MONITOR or SUSPICIOUS-ESCALATE, so a
  re-investigation that ends CLEAR takes it off monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bank import BankData
from src.schemas import Disposition, InvestigationReport, StageName, StageStatus


FLAGGED_DISPOSITIONS = frozenset({Disposition.MONITOR, Disposition.SUSPICIOUS_ESCALATE})


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    disposition: Disposition
    stages_ran: frozenset[StageName]
    triggering_transaction_id: str | None


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
