"""Generator 9 of 9 - the hidden ground-truth record, and the planting plan behind it.

Two responsibilities that belong together:

`build_planting_plan` decides, once and deterministically, which accounts get which evidence
channels. Every other generator reads that decision rather than making its own, which is
what keeps the five channels consistent - an account marked as a structuring case gets the
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
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from src.generators import config as cfg
from src.generators import rings as rings_module
from src.schemas import Account, EvidenceLedgerEntry, FraudRing, RingType, SeverityTier

#: Cross-border cohorts that are legitimate look-alikes rather than planted typologies.
CROSS_BORDER_DECOY_COHORTS = frozenset({"family_inward", "education_outward", "lrs_outward"})


@dataclass
class PlantingPlan:
    """Which accounts carry which planted evidence, decided before any data is generated."""

    structuring: set[str] = field(default_factory=set)
    kyc_mismatch: set[str] = field(default_factory=set)
    adverse_media: dict[str, SeverityTier] = field(default_factory=dict)
    forged_document: set[str] = field(default_factory=set)
    ring_membership: dict[str, str] = field(default_factory=dict)

    name_similarity_decoys: set[str] = field(default_factory=set)
    round_amount_decoys: set[str] = field(default_factory=set)
    coincidence_clusters: list[dict] = field(default_factory=list)

    #: The cross-border block: account -> cohort (see config.CROSS_BORDER_COHORTS).
    cross_border_cohorts: dict[str, str] = field(default_factory=dict)
    #: Round-trip ring members who complete the loop (money out, money back). The ring's
    #: fourth member only sends money out: the one who places funds into the scheme.
    round_trip_core: set[str] = field(default_factory=set)
    round_trip_solo: set[str] = field(default_factory=set)

    def cohort(self, account_id: str) -> str | None:
        return self.cross_border_cohorts.get(account_id)

    @property
    def cross_border_decoys(self) -> set[str]:
        return {a for a, c in self.cross_border_cohorts.items() if c in CROSS_BORDER_DECOY_COHORTS}

    @property
    def cross_border_mismatch(self) -> set[str]:
        """KYC mismatches whose contradiction is foreign money, not inflated domestic turnover.

        transactions.py keeps these accounts' ordinary activity at a clean level and lets the
        foreign flows carry the mismatch, so the contradiction is the one the typology is
        actually about.
        """
        unexplained = {a for a, c in self.cross_border_cohorts.items() if c == "unexplained_inward"}
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
            ]
        )


def _take(pool: list[str], count: int) -> list[str]:
    """Pop `count` accounts off a shuffled pool."""
    taken = [pool.pop() for _ in range(count)]
    return sorted(taken)


def build_planting_plan(
    rng: random.Random,
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
    plan = PlantingPlan(cross_border_cohorts=dict(cohorts))

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
        rng.shuffle(members)
        for account_id in members[: cfg.RING_CORE_MEMBERS]:
            plan.round_trip_core.add(account_id)
            plan.kyc_mismatch.add(account_id)
        plan.round_trip_solo.update(members[cfg.RING_CORE_MEMBERS :])

    # Unexplained foreign money: a KYC mismatch on its own, so monitor-worthy.
    for account_id, cohort in cohorts.items():
        if cohort == "unexplained_inward":
            plan.kyc_mismatch.add(account_id)

    fraud_rings = [r for r in fraud_rings if r.ring_type != RingType.ROUND_TRIP]

    # ---------------------------------------------------------------------------------
    # Ring members: 3 core (escalate-worthy) + 1 solo (monitor-worthy) per ring.
    # ---------------------------------------------------------------------------------
    core_members: list[str] = []
    for ring in fraud_rings:
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
    # The cross-border block is fully planned above and never drawn into other channels.
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
        rng, still_clean, ring_member_ids
    )

    return plan


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

        ledger.append(
            EvidenceLedgerEntry(
                account_id=account_id,
                customer_id=account.customer_id,
                structuring_planted=account_id in plan.structuring,
                kyc_mismatch_planted=account_id in plan.kyc_mismatch,
                adverse_media_hit=plan.adverse_media.get(account_id),
                graph_ring_membership=plan.ring_membership.get(account_id),
                forged_document_planted=account_id in plan.forged_document,
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
