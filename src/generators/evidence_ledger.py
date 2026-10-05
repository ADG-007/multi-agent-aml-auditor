"""Generator 9 of 9 - the hidden ground-truth record, and the planting plan behind it.

Two responsibilities that belong together:

`build_planting_plan` decides, once and deterministically, which accounts get which evidence
channels. Every other generator reads that decision rather than making its own, which is
what keeps the six channels consistent - an account marked as a structuring case gets the
sub-threshold deposits in transactions.py, and the same account is what the ledger records.
Letting each generator plant independently would eventually drift, and the drift would show
up as unexplainable Phase 11 metrics rather than as an obvious bug.

`write` then emits the ledger itself, into data/_ground_truth/. Nothing under that path may
be read by a pipeline stage at inference time. It exists solely as the input to Phase 11's
label-derivation rule, which is written independently of the scoring formula so that
evaluation is not simply the scoring engine grading its own homework.

Planting density follows phase_1_execution.md: about 10% of accounts carry two or more real
channels (escalate-worthy), about 20% carry exactly one (monitor-worthy), and the remaining
70% carry none - though a good number of those still carry a deliberate decoy, which is what
makes the false-positive rate measurable.

The base population is planted from a stream of its own and from pools that never contain a
later block, and each later block plants its own cases. That is what lets a block be added at
any phase without moving a single existing label (streams.py).
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

#: The legitimate life events, dealt out across the life_event cohort in this order. Each is
#: one sudden spike an ordinary customer can have for a perfectly good reason, and each looks
#: like pass-through for a few days:
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
        """KYC mismatches whose contradiction is foreign money, not inflated domestic turnover.

        transactions.py keeps these accounts' ordinary activity at a clean level and lets the
        foreign flows carry the mismatch, so the contradiction is the one the typology is
        actually about.
        """
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
        """How many genuine evidence channels this account carries.

        A NAME_SIMILARITY_ONLY hit is excluded on purpose - it is the decoy, and counting it
        would quietly relabel every false-positive bait case as monitor-worthy.
        """
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
    """Decide the full planting layout.

    Ring members are handled first because ring membership is already fixed by rings.py.
    Within each ring, three members become escalate-worthy by picking up a second channel
    alongside graph membership, and the fourth carries ring membership alone. That fourth
    account is deliberate: it is monitor-worthy on a single circumstantial signal, and it is
    exactly the case Stage 5 has to catch without any corroboration from the other stages.
    """
    plan = PlantingPlan(cohorts=dict(cohorts))
    rng = streams.rng("planting", "base")

    for ring in fraud_rings:
        for account_id in ring.member_account_ids:
            plan.ring_membership[account_id] = ring.ring_id

    # ---------------------------------------------------------------------------------
    # Round-trip rings: three members complete the loop and also carry a KYC mismatch (the
    # money that comes back is foreign income their profile cannot explain), so they are
    # escalate-worthy. The fourth only sends money out and carries ring membership alone.
    # Handled separately from the other rings, whose second channels are drawn below.
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

        # Exactly one core member per ring picks up an adverse-media hit. That leaves two
        # planted actors per ring with no media footprint at all, which is the condition
        # MASTER_PLAN.md calls out: those accounts are only reachable through Stage 5.
        plan.adverse_media[core[0]] = (
            SeverityTier.CONFIRMED_SANCTIONS if rng.random() < 0.35 else SeverityTier.ADVERSE_NEWS
        )

    # The remaining two core members per ring split the other three channels. Structuring is
    # the most common second channel because a ring moving money is the typical pattern; the
    # KYC and document channels are rarer, matching how often each is actually the giveaway.
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
    """The mule block's planting, fixed entirely by cohort - no random draws needed.

    Network mules carry ring membership (from rings.py) plus pass-through, so they are
    escalate-worthy. Solo mules and complicit merchants carry pass-through alone and are
    monitor-worthy: a solo mule is the case Stage 2 must catch on its own, a complicit
    merchant the case where the illicit layer hides inside real trade and the collector link
    is what gives it away. The look-alikes and life events carry nothing and are decoys.
    """
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
