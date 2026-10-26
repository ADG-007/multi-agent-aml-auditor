"""The planting plan and the answer-key ledger (data/_ground_truth/).

`build_planting_plan` decides once which accounts get which channels. Every generator reads
it, so the data and the ledger always agree. `write` saves the ledger, which no stage reads.
About 10% of accounts carry 2+ channels, 20% one, 70% none (many of those carry a decoy).
Each block plants from its own stream, so adding a block moves no existing label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.generators import config as cfg
from src.generators import rings as rings_module
from src.generators.streams import Streams
from src.schemas import Account, EvidenceLedgerEntry, FraudRing, RingType, SeverityTier

#: Cross-border cohorts that are legitimate look-alikes rather than planted typologies.
CROSS_BORDER_DECOY_COHORTS = frozenset({"family_inward", "education_outward", "lrs_outward"})

#: Mule-block cohorts that carry the pass-through channel.
PASS_THROUGH_COHORTS = frozenset({"network_mule", "solo_mule", "complicit_merchant"})

#: Legitimate one-off spikes for the life_event cohort, dealt in this order:
#:   moneylender_payoff  relatives chip in over a day or two; the lot goes to one lender
#:   medical_emergency   relatives send money; it goes to a hospital within hours
#:   windfall            one large receipt (a share of a land sale) spread to new payees
LIFE_EVENT_KINDS = ("moneylender_payoff", "medical_emergency", "windfall")


@dataclass
class PlantingPlan:
    """Which accounts carry which planted evidence, decided before any data is generated."""

    structuring: set[str] = field(default_factory=set)
    kyc_mismatch: set[str] = field(default_factory=set)
    adverse_media: dict[str, SeverityTier] = field(default_factory=dict)
    forged_document: set[str] = field(default_factory=set)
    ring_membership: dict[str, str] = field(default_factory=dict)
    pass_through: set[str] = field(default_factory=set)

    name_similarity_decoys: set[str] = field(default_factory=set)
    round_amount_decoys: set[str] = field(default_factory=set)
    coincidence_clusters: list[dict] = field(default_factory=list)

    #: Every cohort block's accounts: account -> cohort (config.CROSS_BORDER_COHORTS,
    #: config.MULE_BLOCK_COHORTS).
    cohorts: dict[str, str] = field(default_factory=dict)
    #: Round-trip ring members who complete the loop (money out, money back). The ring's
    #: fourth member only sends money out: the one who places funds into the scheme.
    round_trip_core: set[str] = field(default_factory=set)
    round_trip_solo: set[str] = field(default_factory=set)

    #: Where each pass-through account sweeps victims' money: its network's collector, its
    #: own collector for a solo mule, or a network's collector for a complicit merchant.
    mule_collector: dict[str, str] = field(default_factory=dict)
    #: Which legitimate life event each life_event account goes through.
    life_events: dict[str, str] = field(default_factory=dict)

    def cohort(self, account_id: str) -> str | None:
        return self.cohorts.get(account_id)

    @property
    def cross_border_decoys(self) -> set[str]:
        return {a for a, c in self.cohorts.items() if c in CROSS_BORDER_DECOY_COHORTS}

    @property
    def merchant_lookalike_decoys(self) -> set[str]:
        return {a for a, c in self.cohorts.items() if c == "merchant_lookalike"}

    @property
    def life_event_decoys(self) -> set[str]:
        return set(self.life_events)

    @property
    def cross_border_mismatch(self) -> set[str]:
        """KYC mismatches carried by foreign money (domestic activity stays clean)."""
        unexplained = {a for a, c in self.cohorts.items() if c == "unexplained_inward"}
        return self.round_trip_core | unexplained

    @property
    def graph_coincidence_decoys(self) -> set[str]:
        return {
            account_id
            for cluster in self.coincidence_clusters
            for account_id in cluster["member_account_ids"]
        }

    def real_channels_for(self, account_id: str) -> int:
        """Real channels on this account. A look-alike name hit is a decoy, not counted."""
        tier = self.adverse_media.get(account_id)
        return sum(
            [
                account_id in self.structuring,
                account_id in self.kyc_mismatch,
                tier is not None and tier != SeverityTier.NAME_SIMILARITY_ONLY,
                account_id in self.ring_membership,
                account_id in self.forged_document,
                account_id in self.pass_through,
            ]
        )


def _take(pool: list[str], count: int) -> list[str]:
    """Pop `count` accounts off a shuffled pool."""
    taken = [pool.pop() for _ in range(count)]
    return sorted(taken)


def build_planting_plan(
    streams: Streams,
    accounts: list[Account],
    fraud_rings: list[FraudRing],
    cohorts: dict[str, str],
) -> PlantingPlan:
    """Decide the full layout. Per ring: 3 members get a second channel (ESCALATE), the
    fourth has the ring alone (MONITOR, for Stage 5 to catch unaided)."""
    plan = PlantingPlan(cohorts=dict(cohorts))
    rng = streams.rng("planting", "base")

    for ring in fraud_rings:
        for account_id in ring.member_account_ids:
            plan.ring_membership[account_id] = ring.ring_id

    # ---------------------------------------------------------------------------------
    # Round-trip rings: 3 close the loop + KYC mismatch (ESCALATE), the 4th sends out only.
    # ---------------------------------------------------------------------------------
    round_trip_rings = [r for r in fraud_rings if r.ring_type == RingType.ROUND_TRIP]
    for ring in round_trip_rings:
        members = list(ring.member_account_ids)
        streams.rng("planting", ring.ring_id).shuffle(members)
        for account_id in members[: cfg.RING_CORE_MEMBERS]:
            plan.round_trip_core.add(account_id)
            plan.kyc_mismatch.add(account_id)
        plan.round_trip_solo.update(members[cfg.RING_CORE_MEMBERS :])

    # Unexplained foreign money: a KYC mismatch on its own, so monitor-worthy.
    for account_id, cohort in cohorts.items():
        if cohort == "unexplained_inward":
            plan.kyc_mismatch.add(account_id)

    # Round-trip rings are planned above and mule networks by _plant_mule_block below; only
    # the local and international rings take their second channels from here.
    base_rings = [r for r in fraud_rings if r.ring_type in (RingType.LOCAL, RingType.INTERNATIONAL)]

    # ---------------------------------------------------------------------------------
    # Ring members: 3 core (escalate-worthy) + 1 solo (monitor-worthy) per ring.
    # ---------------------------------------------------------------------------------
    core_members: list[str] = []
    for ring in base_rings:
        members = list(ring.member_account_ids)
        rng.shuffle(members)
        core = sorted(members[: cfg.RING_CORE_MEMBERS])
        core_members.extend(core)

        # One core member per ring gets a media hit; the others are reachable only by Stage 5.
        plan.adverse_media[core[0]] = (
            SeverityTier.CONFIRMED_SANCTIONS if rng.random() < 0.35 else SeverityTier.ADVERSE_NEWS
        )

    # The other two split structuring (most common), KYC and document channels.
    remaining_core = [m for m in core_members if m not in plan.adverse_media]
    rng.shuffle(remaining_core)
    for account_id in _take(remaining_core, 8):
        plan.structuring.add(account_id)
    for account_id in _take(remaining_core, 4):
        plan.kyc_mismatch.add(account_id)
    for account_id in _take(remaining_core, 4):
        plan.forged_document.add(account_id)

    ring_member_ids = set(plan.ring_membership)

    # ---------------------------------------------------------------------------------
    # Non-ring escalate-worthy accounts: two non-graph channels each.
    # ---------------------------------------------------------------------------------
    # The cohort blocks are fully planned on their own and never drawn into other channels.
    unplanted = sorted({a.account_id for a in accounts} - ring_member_ids - set(cohorts))
    rng.shuffle(unplanted)

    for account_id in _take(unplanted, 2):
        plan.structuring.add(account_id)
        plan.kyc_mismatch.add(account_id)
    for account_id in _take(unplanted, 2):
        plan.structuring.add(account_id)
        plan.adverse_media[account_id] = SeverityTier.ADVERSE_NEWS
    for account_id in _take(unplanted, 2):
        plan.kyc_mismatch.add(account_id)
        plan.forged_document.add(account_id)

    # ---------------------------------------------------------------------------------
    # Monitor-worthy accounts: exactly one channel each. The eight ring-solo members are
    # already accounted for, so 52 more are drawn here.
    # ---------------------------------------------------------------------------------
    for account_id in _take(unplanted, 18):
        plan.structuring.add(account_id)
    for account_id in _take(unplanted, 14):
        plan.kyc_mismatch.add(account_id)
    for account_id in _take(unplanted, 12):
        plan.adverse_media[account_id] = (
            SeverityTier.CONFIRMED_SANCTIONS if rng.random() < 0.25 else SeverityTier.ADVERSE_NEWS
        )
    for account_id in _take(unplanted, 8):
        plan.forged_document.add(account_id)

    # ---------------------------------------------------------------------------------
    # Decoys, drawn only from accounts carrying no real channel at all.
    # ---------------------------------------------------------------------------------
    clean_pool = list(unplanted)
    rng.shuffle(clean_pool)

    for account_id in _take(clean_pool, cfg.NUM_NAME_SIMILARITY_DECOYS):
        plan.name_similarity_decoys.add(account_id)
        plan.adverse_media[account_id] = SeverityTier.NAME_SIMILARITY_ONLY

    for account_id in _take(clean_pool, cfg.NUM_ROUND_AMOUNT_DECOYS):
        plan.round_amount_decoys.add(account_id)

    still_clean = [a for a in accounts if a.account_id in set(clean_pool)]
    plan.coincidence_clusters = rings_module.generate_coincidence_clusters(
        streams, still_clean, ring_member_ids
    )

    _plant_mule_block(plan, fraud_rings, cohorts)
    return plan


def _plant_mule_block(plan: PlantingPlan, fraud_rings: list[FraudRing], cohorts: dict[str, str]) -> None:
    """Mule-block planting, fixed by cohort: network mules ring + pass-through (ESCALATE),
    solo mules and complicit merchants pass-through only (MONITOR), the rest are decoys."""
    mule_rings = sorted(
        (r for r in fraud_rings if r.ring_type == RingType.MULE), key=lambda r: r.ring_id
    )
    for ring in mule_rings:
        for account_id in ring.member_account_ids:
            plan.mule_collector[account_id] = ring.collector_counterparty_id

    solo = sorted(a for a, c in cohorts.items() if c == "solo_mule")
    for account_id, collector_id in zip(solo, rings_module.SOLO_MULE_COLLECTOR_IDS):
        plan.mule_collector[account_id] = collector_id

    # Each complicit merchant feeds one network's collector, so both collectors are shared by
    # accounts that have no phone in common - the link Stage 5 has to notice.
    complicit = sorted(a for a, c in cohorts.items() if c == "complicit_merchant")
    for position, account_id in enumerate(complicit):
        if mule_rings:
            plan.mule_collector[account_id] = mule_rings[position % len(mule_rings)].collector_counterparty_id

    for account_id, cohort in cohorts.items():
        if cohort in PASS_THROUGH_COHORTS:
            plan.pass_through.add(account_id)

    life = sorted(a for a, c in cohorts.items() if c == "life_event")
    for position, account_id in enumerate(life):
        plan.life_events[account_id] = LIFE_EVENT_KINDS[position % len(LIFE_EVENT_KINDS)]


def build_ledger(accounts: list[Account], plan: PlantingPlan) -> list[EvidenceLedgerEntry]:
    """Turn the plan into one ledger row per account."""
    coincidence_accounts = plan.graph_coincidence_decoys

    ledger: list[EvidenceLedgerEntry] = []
    for account in accounts:
        account_id = account.account_id
        decoys: list[str] = []
        if account_id in plan.name_similarity_decoys:
            decoys.append("name_similarity")
        if account_id in coincidence_accounts:
            decoys.append("graph_coincidence")
        if account_id in plan.round_amount_decoys:
            decoys.append("round_amount")
        if account_id in plan.cross_border_decoys:
            decoys.append("cross_border_legitimate")
        if account_id in plan.merchant_lookalike_decoys:
            decoys.append("merchant_lookalike")
        if account_id in plan.life_event_decoys:
            decoys.append("life_event")

        ledger.append(
            EvidenceLedgerEntry(
                account_id=account_id,
                customer_id=account.customer_id,
                structuring_planted=account_id in plan.structuring,
                kyc_mismatch_planted=account_id in plan.kyc_mismatch,
                adverse_media_hit=plan.adverse_media.get(account_id),
                graph_ring_membership=plan.ring_membership.get(account_id),
                forged_document_planted=account_id in plan.forged_document,
                pass_through_planted=account_id in plan.pass_through,
                decoy_types=decoys,
            )
        )
    return ledger


def write(
    accounts: list[Account],
    plan: PlantingPlan,
    fraud_rings: list[FraudRing],
) -> list[EvidenceLedgerEntry]:
    """Write the ledger and the ring definitions to data/_ground_truth/."""
    cfg.GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

    ledger = build_ledger(accounts, plan)
    cfg.EVIDENCE_LEDGER_FILE.write_text(
        json.dumps([entry.model_dump(mode="json") for entry in ledger], indent=2),
        encoding="utf-8",
    )
    cfg.FRAUD_RINGS_FILE.write_text(
        json.dumps([ring.model_dump(mode="json") for ring in fraud_rings], indent=2),
        encoding="utf-8",
    )
    return ledger
