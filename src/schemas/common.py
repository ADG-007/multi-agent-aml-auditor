"""Shared enums, constants and the ReasoningStep primitive.

Every inter-stage contract in the pipeline is built from these. Kept in one module so a
value like the set of weighted stages or the disposition vocabulary has exactly one
definition — MASTER_PLAN.md Section 3 invariant 6 ("zero hardcoding") is much easier to
honour when the vocabulary itself is not duplicated across modules.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------------------
# Domain vocabulary
# --------------------------------------------------------------------------------------


class Gender(str, Enum):
    """Gender as printed on the identity documents. The Aadhaar card prints it in full, the
    passport data page as a single letter under "Sex"."""

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
    """SAVINGS and CURRENT are resident accounts. An NRI may not hold a resident savings
    account under FEMA, so every NRI holds an NRE (Non-Resident External) account instead."""

    SAVINGS = "SAVINGS"
    CURRENT = "CURRENT"
    NRE = "NRE"


class TransactionType(str, Enum):
    """Real Indian payment-mode categories, replacing PaySim-style generic labels.

    NEFT/RTGS/IMPS/UPI are the four interbank rails; CASH/CHEQUE/ATM/POS are branch and card
    channels. SWIFT is the only cross-border rail: money entering or leaving India arrives as
    a SWIFT wire, never over UPI, NEFT or a card.
    """

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
    """Adverse-media/sanctions corpus tiers.

    NAME_SIMILARITY_ONLY is the deliberate false-positive decoy tier: it is a real corpus
    entry that a name matcher will hit, but it is not evidence of anything.
    """

    CONFIRMED_SANCTIONS = "CONFIRMED_SANCTIONS"
    ADVERSE_NEWS = "ADVERSE_NEWS"
    NAME_SIMILARITY_ONLY = "NAME_SIMILARITY_ONLY"


class RingType(str, Enum):
    """MULE is a money-mule network: accounts rented out and run from a handler's phone,
    forwarding scam victims' money to a shared collector."""

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
    #: The suspicious-transaction obligation on every payment rail, and the non-cash domestic
    #: rails that carry no amount-based report (Amendment 3).
    SUSPICIOUS_TRANSACTION = "SUSPICIOUS_TRANSACTION"


# --------------------------------------------------------------------------------------
# Pipeline vocabulary
# --------------------------------------------------------------------------------------


class StageStatus(str, Enum):
    """Why a stage does or does not have findings.

    NOT_ROUTED (Stage 0 chose not to run it) is deliberately distinct from FAILED/TIMED_OUT
    (it should have run and did not) — only the latter two floor a case at MONITOR.
    """

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
    """What Stage 0 is being asked to evaluate.

    TRANSACTION is a single new transaction (live mode). KYC_SUBMITTED is a new or updated
    KYC profile. PERIODIC_REVIEW is Phase 11's review of a whole account: every transaction
    in its history is screened and the profile on file is reviewed.
    """

    TRANSACTION = "TRANSACTION"
    KYC_SUBMITTED = "KYC_SUBMITTED"
    PERIODIC_REVIEW = "PERIODIC_REVIEW"


class TriageRule(str, Enum):
    """The rows of Stage 0's routing table (MASTER_PLAN.md Phase 2), one value per row."""

    CASH_THRESHOLD = "CASH_THRESHOLD"
    CROSS_BORDER = "CROSS_BORDER"
    KYC_EVENT = "KYC_EVENT"
    KNOWN_BAD_IDENTIFIER = "KNOWN_BAD_IDENTIFIER"
    NAME_MATCH = "NAME_MATCH"
    PASS_THROUGH = "PASS_THROUGH"
    REINVESTIGATION = "REINVESTIGATION"


#: The four stages that feed the renormalized weighted average (MASTER_PLAN.md 2.2).
#: Vision is excluded by invariant 2; Triage, Rulebook and Auditor never score.
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

#: Corroboration gate, first branch: at least this many weighted stages, each at or above
#: CORROBORATION_STAGE_MIN, agree.
CORROBORATION_MIN_STAGES: int = 2
CORROBORATION_STAGE_MIN: float = 0.40

#: Corroboration gate, second branch: one weighted stage at or above this is strong enough
#: to escalate on its own.
SOLO_SIGNAL_MIN: float = 0.75

#: Every confidence and score is rounded to this many decimals before any comparison, so
#: floating-point error (0.3499999999999999 for two stages at 0.35) cannot flip a boundary.
SCORE_DECIMALS: int = 4

#: Forgery confidence at or above this pauses the pipeline for a human admin decision.
FORGERY_PAUSE_THRESHOLD: float = 0.90

#: Forgery confidence in [FORGERY_NOTICE_THRESHOLD, FORGERY_PAUSE_THRESHOLD) is a
#: non-blocking notice on the report.
FORGERY_NOTICE_THRESHOLD: float = 0.60


# --------------------------------------------------------------------------------------
# The reasoning primitive
# --------------------------------------------------------------------------------------


class ReasoningStep(BaseModel):
    """One auditable step in a stage's reasoning.

    MASTER_PLAN.md invariant 5: no agent output is complete without a trail of these.
    Strings are kept short on purpose — Stage 7 concatenates every stage's trail into a
    single prompt, and the models this project targets have a tight context budget.
    """

    step: str = Field(description="What was examined, in a few words.")
    finding: str = Field(description="What was actually observed.")
    impact: str = Field(description="Why that observation matters to the assessment.")
