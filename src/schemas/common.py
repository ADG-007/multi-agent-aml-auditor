"""Shared enums, judge constants and ReasoningStep. One definition each, used everywhere."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------------------
# Domain vocabulary
# --------------------------------------------------------------------------------------


class Gender(str, Enum):
    """Gender as printed on the ID (Aadhaar in full, passport as one letter)."""

    MALE = "MALE"
    FEMALE = "FEMALE"


class ResidencyStatus(str, Enum):
    """Whether the customer banks from inside India or is a Non-Resident Indian."""

    DOMESTIC = "DOMESTIC"
    NRI = "NRI"


class IdDocumentType(str, Enum):
    """Identity document types accepted as an Officially Valid Document (OVD)."""

    AADHAAR = "AADHAAR"
    PAN = "PAN"
    PASSPORT = "PASSPORT"


class AccountType(str, Enum):
    """SAVINGS and CURRENT are resident accounts. Every NRI holds an NRE account (FEMA)."""

    SAVINGS = "SAVINGS"
    CURRENT = "CURRENT"
    NRE = "NRE"


class TransactionType(str, Enum):
    """Indian payment modes. SWIFT is the only cross-border rail."""

    NEFT = "NEFT"
    RTGS = "RTGS"
    IMPS = "IMPS"
    UPI = "UPI"
    CASH_DEPOSIT = "CASH_DEPOSIT"
    CASH_WITHDRAWAL = "CASH_WITHDRAWAL"
    CHEQUE = "CHEQUE"
    ATM = "ATM"
    POS = "POS"
    SWIFT = "SWIFT"


class TransactionDirection(str, Enum):
    """Which way the money moved, from the account holder's side."""

    CREDIT = "CREDIT"
    DEBIT = "DEBIT"


class SeverityTier(str, Enum):
    """Blacklist tiers. NAME_SIMILARITY_ONLY is a decoy: a match, but no evidence."""

    CONFIRMED_SANCTIONS = "CONFIRMED_SANCTIONS"
    ADVERSE_NEWS = "ADVERSE_NEWS"
    NAME_SIMILARITY_ONLY = "NAME_SIMILARITY_ONLY"


class RingType(str, Enum):
    """MULE: rented accounts run from one phone, forwarding money to a shared collector."""

    LOCAL = "LOCAL"
    INTERNATIONAL = "INTERNATIONAL"
    ROUND_TRIP = "ROUND_TRIP"
    MULE = "MULE"


class RulebookCategory(str, Enum):
    """The eight clause categories the rulebook spans."""

    STRUCTURING = "STRUCTURING"
    KYC_CDD = "KYC_CDD"
    SANCTIONS = "SANCTIONS"
    NETWORK = "NETWORK"
    DOCUMENT_FORGERY = "DOCUMENT_FORGERY"
    CROSS_BORDER = "CROSS_BORDER"
    MONEY_MULE = "MONEY_MULE"
    #: The STR duty on every rail, incl. non-cash rails with no amount-based report.
    SUSPICIOUS_TRANSACTION = "SUSPICIOUS_TRANSACTION"


# --------------------------------------------------------------------------------------
# Pipeline vocabulary
# --------------------------------------------------------------------------------------


class StageStatus(str, Enum):
    """Why a stage has findings or not. Only FAILED / TIMED_OUT floor a case at MONITOR."""

    RAN = "RAN"
    NOT_ROUTED = "NOT_ROUTED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"


class PipelineStatus(str, Enum):
    COMPLETED = "COMPLETED"
    PAUSED_PENDING_ADMIN_REVIEW = "PAUSED_PENDING_ADMIN_REVIEW"
    BLOCKED_BY_ADMIN = "BLOCKED_BY_ADMIN"


class Disposition(str, Enum):
    CLEAR = "CLEAR"
    MONITOR = "MONITOR"
    SUSPICIOUS_ESCALATE = "SUSPICIOUS-ESCALATE"
    BLOCKED_BY_ADMIN_REVIEW = "BLOCKED_BY_ADMIN_REVIEW"


class StageName(str, Enum):
    """Canonical stage keys used in InvestigationReport.stage_status."""

    TRIAGE = "Triage"
    VISION = "Vision"
    TRANSACTION_PATTERN = "TransactionPattern"
    KYC = "KYC"
    ADVERSE_MEDIA = "AdverseMedia"
    GRAPH = "Graph"
    RULEBOOK = "Rulebook"
    AUDITOR = "Auditor"


class TriageEventKind(str, Enum):
    """What Stage 0 evaluates: one new transaction, a new KYC profile, or a whole account."""

    TRANSACTION = "TRANSACTION"
    KYC_SUBMITTED = "KYC_SUBMITTED"
    PERIODIC_REVIEW = "PERIODIC_REVIEW"


class TriageRule(str, Enum):
    """The rows of Stage 0's routing table."""

    CASH_THRESHOLD = "CASH_THRESHOLD"
    CROSS_BORDER = "CROSS_BORDER"
    KYC_EVENT = "KYC_EVENT"
    KNOWN_BAD_IDENTIFIER = "KNOWN_BAD_IDENTIFIER"
    NAME_MATCH = "NAME_MATCH"
    PASS_THROUGH = "PASS_THROUGH"
    REINVESTIGATION = "REINVESTIGATION"


#: The four scored stages. Vision, Triage, Rulebook and Auditor never score.
STAGE_WEIGHTS: dict[StageName, float] = {
    StageName.TRANSACTION_PATTERN: 0.25,
    StageName.KYC: 0.20,
    StageName.ADVERSE_MEDIA: 0.25,
    StageName.GRAPH: 0.30,
}

#: Stages 4 and 5 may escalate a case on their own at or above this confidence.
AUTO_VETO_THRESHOLD: float = 0.90

#: A weighted score below this is CLEAR; from here up to ESCALATE_FROM it is MONITOR.
CLEAR_BELOW: float = 0.35

#: A weighted score at or above this escalates, if the corroboration gate also passes.
ESCALATE_FROM: float = 0.60

#: Gate, branch 1: this many stages at or above CORROBORATION_STAGE_MIN.
CORROBORATION_MIN_STAGES: int = 2
CORROBORATION_STAGE_MIN: float = 0.40

#: Gate, branch 2: one stage this strong escalates alone...
SOLO_SIGNAL_MIN: float = 0.75

#: ...but only when at least this many stages found something (> 0).
SOLO_SIGNAL_MIN_RAN: int = 2

#: Any stage at or above this lifts a CLEAR to MONITOR.
SIGNAL_FLOOR_MIN: float = 0.40

#: Round before comparing, so float error (0.34999…) cannot flip a boundary.
SCORE_DECIMALS: int = 4

#: Forgery confidence at or above this pauses the case for an admin.
FORGERY_PAUSE_THRESHOLD: float = 0.90

#: From here up to the pause threshold: a notice on the report, no pause.
FORGERY_NOTICE_THRESHOLD: float = 0.60


# --------------------------------------------------------------------------------------
# The reasoning primitive
# --------------------------------------------------------------------------------------


class ReasoningStep(BaseModel):
    """One auditable step. Every stage output carries a trail of these. Keep them short."""

    step: str = Field(description="What was examined, in a few words.")
    finding: str = Field(description="What was actually observed.")
    impact: str = Field(description="Why that observation matters to the assessment.")
