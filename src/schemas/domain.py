"""Domain models — the simulated bank's data.

These are what the Phase 1 generators produce and what every later stage reads. They are
deliberately not pipeline outputs: nothing here carries a confidence or a disposition.

Field sets follow phase_1_execution.md's "format-plausible, not algorithmically exact"
calibration: a real Aadhaar/PAN/passport's actual field set and number pattern, but no
attempt to reproduce Aadhaar's Verhoeff checksum or ICAO 9303 MRZ construction.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field, computed_field

from src.schemas.common import (
    AccountType,
    Disposition,
    Gender,
    IdDocumentType,
    ResidencyStatus,
    RingType,
    RulebookCategory,
    SeverityTier,
    TransactionDirection,
    TransactionType,
)


class Customer(BaseModel):
    """An account holder of the simulated bank."""

    customer_id: str
    name: str
    dob: date
    gender: Gender = Field(description="Printed on the Aadhaar card and the passport data page.")
    occupation: str
    stated_income: int = Field(description="Declared annual income in INR (3,00,000-50,00,000).")
    address: str
    residency_status: ResidencyStatus
    country: str = Field(description="Country of residence; 'India' for DOMESTIC customers.")
    id_document_type: IdDocumentType = Field(description="Primary OVD on file.")
    id_number: str = Field(description="Formatted to match id_document_type; no real checksum.")
    id_expiration_date: date | None = Field(
        default=None,
        description="Populated only for PASSPORT — Aadhaar and PAN do not expire.",
    )


class Account(BaseModel):
    account_id: str
    customer_id: str
    opened_date: date
    account_type: AccountType


class Transaction(BaseModel):
    """A single posted transaction.

    No balance_after and no narration: both were considered during Phase 1 planning and
    rejected as real generation cost for cosmetic-only benefit.
    """

    transaction_id: str
    account_id: str
    amount: float = Field(description="INR, always positive. Which way it moved is `direction`.")
    direction: TransactionDirection
    type: TransactionType
    timestamp: datetime
    device_id: str | None = Field(
        default=None,
        description=(
            "The customer's phone or computer, recorded only when the customer initiated the "
            "transaction digitally (an outgoing UPI/IMPS/NEFT/RTGS transfer or SWIFT wire). "
            "Empty on money coming in, which the sender initiated, and on cash, cheque, ATM "
            "and card transactions, which no customer device takes part in."
        ),
    )
    ip_address: str | None = Field(
        default=None,
        description="The connection used, recorded under the same rule as device_id.",
    )
    counterparty_id: str
    counterparty_name: str
    purpose_code: str | None = Field(
        default=None,
        description=(
            "RBI purpose code, carried only by SWIFT wires: an S-code on money leaving India, "
            "a P-code on money arriving. Declared by the customer, so it can be checked "
            "against what the money actually did."
        ),
    )

    @computed_field
    @property
    def is_cross_border(self) -> bool:
        """The national/international split. Derived from the rail rather than stored
        independently, so the two can never disagree."""
        return self.type == TransactionType.SWIFT


class Counterparty(BaseModel):
    """The other side of a transaction, and a node in the Stage 5 graph.

    is_foreign / country is what makes the cross-border clause category reachable without
    adding a new pipeline stage — Stage 5 reads it off the graph node.
    """

    counterparty_id: str
    name: str
    country: str
    is_foreign: bool
    registered_address: str | None = Field(
        default=None,
        description=(
            "Foreign entities only. Becomes an address node in the graph, which is how two "
            "companies registered at the same premises become visibly linked."
        ),
    )


class KYCProfile(BaseModel):
    """The free-text profile Stage 3 reasons over.

    Deliberately unstructured: the whole point of Stage 3 is judging whether a human-written
    account of someone's life is plausible against what their account actually does.
    """

    customer_id: str
    narrative: str


class AdverseMediaEntry(BaseModel):
    """One entry in the adverse-media / sanctions corpus.

    Modelled on the UN Security Council Consolidated List (1267/1989) as given effect in
    India through UAPA Section 51A notifications. aliases and date_of_birth are functional,
    not decorative — alias matching and DOB disambiguation are exactly what real sanctions
    screening uses to separate a true hit from a name collision.
    """

    entry_id: str
    entity_name: str
    severity_tier: SeverityTier
    description: str
    linked_actor_id: str | None = Field(
        default=None,
        description="Ground-truth link to a planted customer. Never read by a pipeline stage.",
    )
    aliases: list[str] = Field(default_factory=list)
    date_of_birth: str | None = None
    listing_date: date | None = None


class RulebookClause(BaseModel):
    """One policy clause, chunked 1:1 into ChromaDB.

    metadata externalizes machine-readable numbers so Stage 0's routing check and Stage 2's
    formula read the same values from one source — STRUCT-001 carries threshold_inr and
    near_threshold_pct. Injecting a replacement clause at demo time must move both together.
    """

    clause_id: str
    title: str
    text: str
    category: RulebookCategory
    sources: list[str] = Field(description="Real published regulatory sources this clause paraphrases.")
    metadata: dict = Field(default_factory=dict)


class FraudRing(BaseModel):
    """A deliberately planted cluster of accounts sharing identifiers with no legitimate reason.

    Never handed to Stage 5 — the graph gets the shared identifiers, not the ring label. Stage 5
    has to rediscover the cluster from topology alone.
    """

    ring_id: str
    ring_type: RingType
    member_account_ids: list[str]
    shared_device_id: str | None = None
    shared_ip_address: str | None = None
    shared_address: str | None = None
    foreign_counterparty_ids: list[str] = Field(
        default_factory=list,
        description=(
            "INTERNATIONAL rings funnel to this shared foreign-counterparty cluster; a ROUND_TRIP "
            "ring's pair is [outbound entity, return entity]."
        ),
    )
    collector_counterparty_id: str | None = Field(
        default=None,
        description="MULE rings only: the account every member sweeps victims' money into.",
    )


class EvidenceLedgerEntry(BaseModel):
    """Per-account record of what was planted, and at what severity.

    Generation-time only. MASTER_PLAN.md Phase 1 is explicit that this is never visible to
    any pipeline stage at inference time; it exists solely as the input to Phase 11's
    label-derivation rule, which is written independently of the scoring formula.
    """

    account_id: str
    customer_id: str
    structuring_planted: bool = False
    kyc_mismatch_planted: bool = False
    adverse_media_hit: SeverityTier | None = None
    graph_ring_membership: str | None = None
    forged_document_planted: bool = False
    pass_through_planted: bool = Field(
        default=False,
        description=(
            "Money-mule pass-through: victims' money arriving and forwarded to a collector "
            "within hours. Network mules, solo mules and complicit merchants."
        ),
    )
    decoy_types: list[str] = Field(
        default_factory=list,
        description=(
            "Deliberate false-positive bait: name_similarity, graph_coincidence, round_amount, "
            "cross_border_legitimate, merchant_lookalike, life_event."
        ),
    )

    @property
    def real_channel_count(self) -> int:
        """Channels that count as real evidence.

        A NAME_SIMILARITY_ONLY adverse-media hit is explicitly not a real channel — it is
        the decoy, and counting it would make the false-positive test meaningless.
        """
        real_media = self.adverse_media_hit is not None and (
            self.adverse_media_hit != SeverityTier.NAME_SIMILARITY_ONLY
        )
        return sum(
            [
                self.structuring_planted,
                self.kyc_mismatch_planted,
                real_media,
                self.graph_ring_membership is not None,
                self.forged_document_planted,
                self.pass_through_planted,
            ]
        )

    @property
    def expected_disposition(self) -> Disposition:
        """The answer-key label Phase 11 grades against. The one home of the labelling rule.

        Two or more real channels -> SUSPICIOUS-ESCALATE, one -> MONITOR, none -> CLEAR. One
        exception (Amendment 3, A4): a confirmed sanctions hit escalates on its own, because
        the sanctions veto must escalate it and the law requires a report on it.
        """
        if self.real_channel_count >= 2 or (
            self.adverse_media_hit == SeverityTier.CONFIRMED_SANCTIONS
        ):
            return Disposition.SUSPICIOUS_ESCALATE
        if self.real_channel_count == 1:
            return Disposition.MONITOR
        return Disposition.CLEAR
