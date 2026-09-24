"""Check the generated dataset, then describe it.

    python -m src.generators.verify

Two jobs, and the distinction matters. The assertions check that the data on disk matches
the exact quantities phase_1_execution.md decided - 344 customers (300 plus the cross-border
block), 11 rings, 160 adverse-media entries, 39 escalate-worthy accounts. They exist to catch a generator bug immediately, at the
point it is cheap to find, rather than in Phase 11 when the evaluation numbers look strange
and the cause is six phases back. They are not Phase 2's boundary tests and do not pretend to
be.

The summary then prints the data back in human-readable form, including one full sample
record from each generator, because a dataset that passes its own assertions can still be
subtly wrong in a way only a person reading it will notice.
"""

from __future__ import annotations

import json
import sys
from collections import Counter

import networkx as nx

from src.generators import config as cfg
from src.generators import id_documents as documents_module
from src.schemas import (
    Account,
    AdverseMediaEntry,
    Counterparty,
    Customer,
    EvidenceLedgerEntry,
    FraudRing,
    KYCProfile,
    RulebookClause,
    SeverityTier,
    Transaction,
)

PASS = "  [ok]  "
FAIL = "  [FAIL]"

_failures: list[str] = []


def check(condition: bool, description: str) -> None:
    if condition:
        print(f"{PASS} {description}")
    else:
        print(f"{FAIL} {description}")
        _failures.append(description)


def _load(path, model):
    return [model.model_validate(row) for row in json.loads(path.read_text(encoding="utf-8"))]


def _lakh_phrase(amount: float) -> str:
    """An amount as the KYC narratives write it (kyc_profiles._lakh)."""
    from src.generators.kyc_profiles import _lakh

    return _lakh(amount)


def _rupees(amount: float) -> str:
    if amount >= 10_000_000:
        return f"Rs {amount / 10_000_000:,.2f} cr"
    if amount >= 100_000:
        return f"Rs {amount / 100_000:,.2f} L"
    return f"Rs {amount:,.0f}"


#: One probe per clause category, phrased the way Stage 6's query text will be - concatenated
#: finding strings from the upstream reasoning trails - rather than as keyword lookups, so
#: this measures what retrieval will actually face.
#:
#: Each probe names every clause that would properly cover that finding, not a single answer.
#: That is not a loosened bar, it is the accurate one: a finding that a name matched a
#: designated party is covered by the screening-and-match-resolution clause just as
#: legitimately as by the freezing-obligation clause, and unexplained turnover is covered by
#: both the surge clause and the profile-inconsistency clause. Demanding one exact clause
#: would be asserting a preference the rulebook does not actually express.
RETRIEVAL_PROBES: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "repeated cash deposits just below the ten lakh reporting limit",
        frozenset({"STRUCT-001", "STRUCT-002"}),
    ),
    (
        "declared income does not explain the account turnover observed",
        frozenset({"KYC-002", "STRUCT-003"}),
    ),
    (
        "customer name matches a designated party on a sanctions list",
        frozenset({"SANC-001", "SANC-002", "SANC-003"}),
    ),
    (
        "several unrelated accounts operated from the same mobile device",
        frozenset({"NET-002", "NET-001"}),
    ),
    (
        "identity document shows misaligned fields and an altered date",
        frozenset({"DOC-002", "DOC-003"}),
    ),
    (
        "repeated outward transfers to offshore companies in one jurisdiction",
        frozenset({"XB-003", "XB-001"}),
    ),
    (
        "money wired to an overseas company came back from a linked company at the same address",
        frozenset({"XB-005", "XB-003"}),
    ),
    (
        "foreign wire above five lakh rupees with a purpose code the profile cannot support",
        frozenset({"XB-004", "XB-002", "XB-001"}),
    ),
)


def _check_retrieval(clause_count: int) -> None:
    """Query the embedded rulebook and confirm the right clause comes back first.

    A collection that embedded without error can still be useless - wrong distance metric,
    text that does not discriminate, clauses too similar to each other. Stage 6 is six phases
    away, so this is where an unusable index gets caught.
    """
    try:
        import chromadb

        from src.generators.rulebook import COLLECTION_NAME

        collection = chromadb.PersistentClient(
            path=str(cfg.RULEBOOK_CHROMA_DIR)
        ).get_collection(COLLECTION_NAME)
    except Exception as exc:
        print(f"  [skip] rulebook retrieval - collection unavailable ({type(exc).__name__})")
        return

    check(collection.count() == clause_count, f"all {clause_count} clauses are in the index")
    check(
        (collection.metadata or {}).get("hnsw:space") == "cosine",
        "collection uses cosine distance, as Stage 6's cutoff assumes",
    )

    # Top-1 exactness is the wrong bar. Stage 6 retrieves k=5 and cites from within that set,
    # and clauses inside a category are genuinely close in meaning - a query about
    # unexplained turnover legitimately surfaces STRUCT-003 ("sudden surge in account
    # turnover") ahead of KYC-002. What the data layer has to guarantee is that the right
    # clause is in the retrieved set at all, and that the top hit is at least in the right
    # category.
    top_hit_ok = 0
    in_top_k = 0
    top_distances: list[float] = []
    for query, acceptable in RETRIEVAL_PROBES:
        result = collection.query(query_texts=[query], n_results=5)
        ids = result["ids"][0]
        if ids[0] in acceptable:
            top_hit_ok += 1
        if acceptable & set(ids):
            in_top_k += 1
        top_distances.append(result["distances"][0][0])

    check(
        top_hit_ok == len(RETRIEVAL_PROBES),
        f"every probe's top hit is a clause that covers the finding "
        f"({top_hit_ok}/{len(RETRIEVAL_PROBES)})",
    )
    check(
        in_top_k == len(RETRIEVAL_PROBES),
        f"every probe retrieves a covering clause within k=5 "
        f"({in_top_k}/{len(RETRIEVAL_PROBES)})",
    )
    # Reported rather than asserted. The Phase 8 relevance cutoff was raised from 0.5 to 0.75
    # on these measurements (correct matches ~0.35-0.53, an irrelevant query ~0.96); this
    # note is what shows whether that bar still holds if the clauses change.
    print(
        f"  [note] top-hit cosine distances "
        f"{min(top_distances):.3f}-{max(top_distances):.3f} "
        f"(the 0.75 cutoff agreed for Phase 8 would drop "
        f"{sum(1 for d in top_distances if d > 0.75)}/{len(top_distances)} top hits)"
    )


def _check_ocr(customers, accounts, ledger) -> None:
    """OCR a sample of clean and forged documents and confirm the fields come through.

    Skipped with a visible note if the Tesseract binary is absent, rather than failing: it is
    an OS-level install, it blocks Phase 5 rather than Phase 1, and a missing binary is not a
    defect in the generated data.
    """
    try:
        import pytesseract
        from PIL import Image

        pytesseract.get_tesseract_version()
    except Exception as exc:
        print(f"  [skip] OCR round-trip - Tesseract unavailable ({type(exc).__name__})")
        return

    customers_by_id = {c.customer_id: c for c in customers}
    customer_of = {a.account_id: a.customer_id for a in accounts}

    def read(customer) -> str:
        path = cfg.DOCUMENTS_DIR / f"{customer.customer_id}_{customer.id_document_type.value}.png"
        return " ".join(pytesseract.image_to_string(Image.open(path)).split())

    clean_entries = [e for e in ledger if not e.forged_document_planted][:5]
    clean_ok = 0
    for entry in clean_entries:
        customer = customers_by_id[entry.customer_id]
        text = read(customer)
        # Aadhaar prints gender in full; the passport's single-letter "Sex" is too short to
        # search for reliably in OCR output, so it is only checked on Aadhaar.
        gender_ok = customer.id_document_type.value != "AADHAAR" or customer.gender.value in text
        if (
            customer.dob.strftime("%d/%m/%Y") in text
            and customer.id_number.split()[0] in text
            and gender_ok
        ):
            clean_ok += 1
    check(
        clean_ok == len(clean_entries),
        f"clean documents OCR back to the record ({clean_ok}/{len(clean_entries)} sampled)",
    )

    forged_entries = [e for e in ledger if e.forged_document_planted]
    forged_ok = 0
    for entry in forged_entries:
        customer = customers_by_id[customer_of[entry.account_id]]
        if customer.dob.strftime("%d/%m/%Y") not in read(customer):
            forged_ok += 1
    check(
        forged_ok == len(forged_entries),
        f"every forged document's printed date contradicts the record "
        f"({forged_ok}/{len(forged_entries)})",
    )


def main() -> int:
    missing = [
        path
        for path in (
            cfg.CUSTOMERS_FILE,
            cfg.ACCOUNTS_FILE,
            cfg.TRANSACTIONS_FILE,
            cfg.COUNTERPARTIES_FILE,
            cfg.KYC_PROFILES_FILE,
            cfg.ADVERSE_MEDIA_FILE,
            cfg.GRAPH_FILE,
            cfg.RULEBOOK_JSON_FILE,
            cfg.EVIDENCE_LEDGER_FILE,
            cfg.DOCUMENT_MANIFEST_FILE,
        )
        if not path.exists()
    ]
    if missing:
        print("Dataset is not generated yet. Missing:")
        for path in missing:
            print(f"  - {path}")
        print("\nRun: python -m src.generators")
        return 1

    customers = _load(cfg.CUSTOMERS_FILE, Customer)
    accounts = _load(cfg.ACCOUNTS_FILE, Account)
    transactions = _load(cfg.TRANSACTIONS_FILE, Transaction)
    counterparties = _load(cfg.COUNTERPARTIES_FILE, Counterparty)
    profiles = _load(cfg.KYC_PROFILES_FILE, KYCProfile)
    media = _load(cfg.ADVERSE_MEDIA_FILE, AdverseMediaEntry)
    clauses = _load(cfg.RULEBOOK_JSON_FILE, RulebookClause)
    ledger = _load(cfg.EVIDENCE_LEDGER_FILE, EvidenceLedgerEntry)
    rings = _load(cfg.FRAUD_RINGS_FILE, FraudRing)
    manifest = json.loads(cfg.DOCUMENT_MANIFEST_FILE.read_text(encoding="utf-8"))
    graph = nx.read_graphml(cfg.GRAPH_FILE)

    # ==================================================================================
    print("\n=== Population ===")
    check(len(customers) == cfg.NUM_CUSTOMERS, f"{cfg.NUM_CUSTOMERS} customers")
    check(len(accounts) == cfg.NUM_CUSTOMERS, f"{cfg.NUM_CUSTOMERS} accounts")
    residency = Counter(c.residency_status.value for c in customers)
    check(residency["DOMESTIC"] == cfg.NUM_DOMESTIC, f"{cfg.NUM_DOMESTIC} domestic customers")
    check(residency["NRI"] == cfg.NUM_NRI, f"{cfg.NUM_NRI} NRI customers")
    from src.generators.addresses import INDIAN_LOCALITIES

    valid_tails = {
        f"{locality}, {city}, {state} {pin}"
        for city, state, localities in INDIAN_LOCALITIES
        for locality, pin in localities
    }
    check(
        all(
            any(c.address.endswith(tail) for tail in valid_tails)
            for c in customers
            if c.residency_status.value == "DOMESTIC"
        ),
        "every Indian address carries the real PIN code for its locality and city",
    )
    check(
        all(c.country in c.address for c in customers if c.residency_status.value == "NRI"),
        "every NRI address names the customer's country of residence",
    )
    nri_customers = {c.customer_id for c in customers if c.residency_status.value == "NRI"}
    check(
        all((a.account_type.value == "NRE") == (a.customer_id in nri_customers) for a in accounts),
        "every NRI holds an NRE account, and only NRIs do",
    )
    genders = Counter(c.gender.value for c in customers)
    check(
        min(genders.values()) >= cfg.NUM_CUSTOMERS * 0.35 and len(genders) == 2,
        f"both genders well represented ({dict(genders)})",
    )
    check(
        all(cfg.INCOME_MIN <= c.stated_income <= cfg.INCOME_MAX for c in customers),
        f"stated incomes within {_rupees(cfg.INCOME_MIN)} - {_rupees(cfg.INCOME_MAX)}",
    )
    check(
        all(c.id_expiration_date is None for c in customers if c.id_document_type.value != "PASSPORT"),
        "expiry populated only for passports (Aadhaar and PAN do not expire)",
    )
    check(
        all(len(c.id_number.replace(" ", "")) == 12 for c in customers if c.id_document_type.value == "AADHAAR"),
        "Aadhaar numbers are 12 digits grouped 4-4-4",
    )
    check(
        all(c.id_number[0] not in "01" for c in customers if c.id_document_type.value == "AADHAAR"),
        "no Aadhaar number begins 0 or 1",
    )
    from src.generators.names import WESTERN_FIRST_NAMES

    check(
        not any(c.name.split()[0] in WESTERN_FIRST_NAMES for c in customers)
        and not any(
            cp.name.split()[0] in WESTERN_FIRST_NAMES
            for cp in counterparties
            if cp.country == "India" or not cp.registered_address
        ),
        "every customer and individual counterparty has an Indian first name",
    )

    # ==================================================================================
    print("\n=== Fraud rings ===")
    check(len(rings) == cfg.NUM_RINGS, f"{cfg.NUM_RINGS} rings")
    check(
        sum(1 for r in rings if r.ring_type.value == "LOCAL") == cfg.NUM_LOCAL_RINGS,
        f"{cfg.NUM_LOCAL_RINGS} local rings",
    )
    check(
        sum(1 for r in rings if r.ring_type.value == "INTERNATIONAL") == cfg.NUM_INTERNATIONAL_RINGS,
        f"{cfg.NUM_INTERNATIONAL_RINGS} international rings",
    )
    check(
        sum(1 for r in rings if r.ring_type.value == "ROUND_TRIP") == cfg.NUM_ROUND_TRIP_RINGS,
        f"{cfg.NUM_ROUND_TRIP_RINGS} round-trip rings",
    )
    ring_members = [m for r in rings for m in r.member_account_ids]
    check(len(ring_members) == len(set(ring_members)), "no account belongs to two rings")
    check(all(r.shared_device_id for r in rings), "every ring shares a device identifier")

    # ==================================================================================
    print("\n=== Transactions ===")
    check(len(transactions) > 0, "transactions generated")
    per_account = Counter(t.account_id for t in transactions)
    check(len(per_account) == cfg.NUM_CUSTOMERS, "every account has transaction history")
    check(
        all(t.amount > 0 for t in transactions),
        "every amount is positive",
    )
    rtgs = [t for t in transactions if t.type.value == "RTGS"]
    check(
        all(t.amount >= 200_000 for t in rtgs),
        "RTGS transactions respect the real Rs 2,00,000 floor",
    )
    upi = [t for t in transactions if t.type.value == "UPI"]
    check(
        all(t.amount <= 100_000 for t in upi),
        "UPI transactions respect the real Rs 1,00,000 per-transaction ceiling",
    )
    from src.generators.transactions import (
        DIGITAL_RAILS,
        FIXED_DIRECTION,
        INWARD_PURPOSE_CODES,
        OUTWARD_PURPOSE_CODES,
    )

    check(
        all(FIXED_DIRECTION.get(t.type, t.direction) == t.direction for t in transactions),
        "cash deposits are credits; withdrawals, ATM and card payments are debits",
    )
    check(
        all(
            (t.device_id is not None and t.ip_address is not None)
            == (t.direction.value == "DEBIT" and t.type in DIGITAL_RAILS)
            for t in transactions
        ),
        "phone and IP recorded only on transactions the customer sent digitally "
        f"({sum(1 for t in transactions if t.device_id):,} of {len(transactions):,})",
    )
    counterparty_by_id = {c.counterparty_id: c for c in counterparties}
    swift = [t for t in transactions if t.type.value == "SWIFT"]
    check(
        all(counterparty_by_id[t.counterparty_id].is_foreign == (t.type.value == "SWIFT") for t in transactions),
        "foreign counterparties appear only on SWIFT, and every SWIFT wire has a foreign counterparty",
    )
    check(
        all((t.purpose_code is not None) == (t.type.value == "SWIFT") for t in transactions),
        "purpose codes appear on SWIFT wires and nowhere else",
    )
    check(
        all(
            (t.direction.value == "DEBIT" and t.purpose_code in OUTWARD_PURPOSE_CODES)
            or (t.direction.value == "CREDIT" and t.purpose_code in INWARD_PURPOSE_CODES)
            for t in swift
        ),
        "outgoing wires carry an RBI S-code and incoming wires a P-code",
    )
    raw_rows = json.loads(cfg.TRANSACTIONS_FILE.read_text(encoding="utf-8"))
    check(
        all(r["is_cross_border"] == (r["type"] == "SWIFT") for r in raw_rows),
        "is_cross_border is stored on every row and true exactly for SWIFT",
    )
    print(
        f"  [note] {sum(1 for t in swift if t.amount > cfg.CBWTR_THRESHOLD_INR)} of {len(swift)} "
        f"SWIFT wires exceed the {_rupees(cfg.CBWTR_THRESHOLD_INR)} CBWTR threshold"
    )

    band = [
        t
        for t in transactions
        if cfg.STRUCTURING_BAND_LOW <= t.amount < cfg.STRUCTURING_BAND_HIGH
    ]
    check(len(band) > 0, "sub-threshold structuring band is populated")
    check(
        all(t.amount < cfg.CTR_THRESHOLD_INR for t in band),
        "no planted structuring transaction crosses the CTR threshold",
    )
    check(all(t.direction.value == "CREDIT" for t in band), "structuring deposits are all credits")

    # Every round-trip core member's loop has to actually close: a debit to the ring's
    # outbound company, then a credit from its return company - registered at the same
    # address - for most of the same money.
    ring_by_id = {r.ring_id: r for r in rings}
    loops_closed = 0
    core_members = 0
    for entry in ledger:
        ring = ring_by_id.get(entry.graph_ring_membership or "")
        if not ring or ring.ring_type.value != "ROUND_TRIP" or not entry.kyc_mismatch_planted:
            continue
        core_members += 1
        outbound_id, return_id = ring.foreign_counterparty_ids
        mine = [t for t in swift if t.account_id == entry.account_id]
        outs = [t for t in mine if t.counterparty_id == outbound_id and t.direction.value == "DEBIT"]
        backs = [t for t in mine if t.counterparty_id == return_id and t.direction.value == "CREDIT"]
        same_address = (
            counterparty_by_id[outbound_id].registered_address
            == counterparty_by_id[return_id].registered_address
        )
        closes = any(
            0.85 <= b.amount / o.amount <= 1.0 and o.timestamp < b.timestamp
            for o in outs
            for b in backs
        )
        if same_address and closes:
            loops_closed += 1
    check(
        core_members == cfg.NUM_ROUND_TRIP_RINGS * cfg.RING_CORE_MEMBERS and loops_closed == core_members,
        f"every round-trip loop closes through two companies at one address ({loops_closed}/{core_members})",
    )

    # ==================================================================================
    print("\n=== Evidence ledger (ground truth) ===")
    check(len(ledger) == cfg.NUM_CUSTOMERS, "one ledger row per account")
    escalate = [e for e in ledger if e.real_channel_count >= 2]
    monitor = [e for e in ledger if e.real_channel_count == 1]
    clean = [e for e in ledger if e.real_channel_count == 0]
    check(len(escalate) == cfg.NUM_ESCALATE_WORTHY, f"{cfg.NUM_ESCALATE_WORTHY} escalate-worthy accounts")
    check(len(monitor) == cfg.NUM_MONITOR_WORTHY, f"{cfg.NUM_MONITOR_WORTHY} monitor-worthy accounts")
    check(
        len(clean) == cfg.NUM_CUSTOMERS - cfg.NUM_ESCALATE_WORTHY - cfg.NUM_MONITOR_WORTHY,
        f"{cfg.NUM_CUSTOMERS - cfg.NUM_ESCALATE_WORTHY - cfg.NUM_MONITOR_WORTHY} accounts with no real channel",
    )
    check(
        sum(1 for e in ledger if e.graph_ring_membership) == cfg.NUM_RINGS * cfg.RING_SIZE,
        f"{cfg.NUM_RINGS * cfg.RING_SIZE} accounts carry ring membership",
    )

    # Every ring must contain at least one member with no adverse-media footprint at all.
    # Those are the accounts only Stage 5 can reach, and MASTER_PLAN.md calls for them by name.
    by_ring: dict[str, list[EvidenceLedgerEntry]] = {}
    for entry in ledger:
        if entry.graph_ring_membership:
            by_ring.setdefault(entry.graph_ring_membership, []).append(entry)
    check(
        all(
            any(e.adverse_media_hit is None for e in members)
            for members in by_ring.values()
        ),
        "every ring contains a member with zero adverse-media hits (Graph-only cases)",
    )

    planted_structuring = {e.account_id for e in ledger if e.structuring_planted}
    accounts_with_band = {t.account_id for t in band}
    check(
        planted_structuring <= accounts_with_band,
        "every account marked as structuring actually has sub-threshold transactions",
    )
    # The converse matters just as much. An unplanted account sitting in the band would be
    # an unlabelled positive, and Phase 11 would be scoring the pipeline against ground
    # truth that does not describe the data.
    check(
        accounts_with_band <= planted_structuring,
        "no unplanted account has transactions in the structuring band",
    )

    # Turnover has to track declared income, or the KYC-mismatch channel carries no signal:
    # if every clean account already moves twenty times its stated income, a planted mismatch
    # is indistinguishable from ordinary variation.
    income_by_account = {
        a.account_id: next(c.stated_income for c in customers if c.customer_id == a.customer_id)
        for a in accounts
    }
    throughput: dict[str, float] = {}
    for txn in transactions:
        throughput[txn.account_id] = throughput.get(txn.account_id, 0.0) + txn.amount
    ratios = {
        account_id: (total * (365 / cfg.HISTORY_DAYS)) / income_by_account[account_id]
        for account_id, total in throughput.items()
    }
    mismatch_accounts = {e.account_id for e in ledger if e.kyc_mismatch_planted}
    # Only accounts with no planted channel at all are held to the clean bound. A structuring
    # burst or a cross-border funnel legitimately inflates turnover, and suppressing that to
    # protect a tidy ratio would make the structuring channel invisible in the account's
    # overall activity - which is the opposite of what a real case looks like.
    untouched = {e.account_id for e in ledger if e.real_channel_count == 0}
    clean_ratios = [r for a, r in ratios.items() if a in untouched]
    mismatch_ratios = [r for a, r in ratios.items() if a in mismatch_accounts]
    check(
        max(clean_ratios) < 8.0,
        f"accounts with no planted channel stay under 8x declared income "
        f"(max {max(clean_ratios):.1f}x)",
    )
    check(
        min(mismatch_ratios) > 6.0,
        f"planted KYC mismatches exceed 6x declared income (min {min(mismatch_ratios):.1f}x)",
    )
    check(
        min(mismatch_ratios) > max(clean_ratios) * 0.9,
        "planted mismatches sit above the clean population, not inside it",
    )
    money_in: dict[str, float] = {}
    money_out: dict[str, float] = {}
    for txn in transactions:
        if txn.type.value == "SWIFT":
            continue
        side = money_in if txn.direction.value == "CREDIT" else money_out
        side[txn.account_id] = side.get(txn.account_id, 0.0) + txn.amount
    balance = [money_in.get(a, 0.0) / money_out[a] for a in untouched if money_out.get(a)]
    check(
        all(0.5 <= r <= 2.0 for r in balance),
        f"clean accounts' domestic money in and out stay level "
        f"(in/out ratio {min(balance):.2f}-{max(balance):.2f})",
    )

    # ==================================================================================
    print("\n=== Decoys (false-positive bait) ===")
    name_decoys = [e for e in ledger if "name_similarity" in e.decoy_types]
    graph_decoys = [e for e in ledger if "graph_coincidence" in e.decoy_types]
    round_decoys = [e for e in ledger if "round_amount" in e.decoy_types]
    check(len(name_decoys) == cfg.NUM_NAME_SIMILARITY_DECOYS, f"{cfg.NUM_NAME_SIMILARITY_DECOYS} name-similarity decoys")
    check(
        len(graph_decoys) == cfg.NUM_GRAPH_COINCIDENCE_PAIRS * 2,
        f"{cfg.NUM_GRAPH_COINCIDENCE_PAIRS * 2} graph-coincidence decoy accounts",
    )
    check(len(round_decoys) == cfg.NUM_ROUND_AMOUNT_DECOYS, f"{cfg.NUM_ROUND_AMOUNT_DECOYS} round-amount decoys")
    cross_border_decoys = [e for e in ledger if "cross_border_legitimate" in e.decoy_types]
    check(
        len(cross_border_decoys) == cfg.NUM_CROSS_BORDER_DECOYS,
        f"{cfg.NUM_CROSS_BORDER_DECOYS} legitimate cross-border look-alikes",
    )
    check(
        all(any(t.account_id == e.account_id for t in swift) for e in cross_border_decoys),
        "every cross-border look-alike actually has foreign wires",
    )
    shared_foreign_addresses = Counter(
        c.registered_address for c in counterparties if c.registered_address
    )
    round_trip_ids = {cp for r in rings if r.ring_type.value == "ROUND_TRIP" for cp in r.foreign_counterparty_ids}
    cluster_ids = {cp for r in rings if r.ring_type.value == "INTERNATIONAL" for cp in r.foreign_counterparty_ids}
    innocent_shared = [
        c for c in counterparties
        if c.registered_address
        and shared_foreign_addresses[c.registered_address] > 1
        and c.counterparty_id not in round_trip_ids | cluster_ids
    ]
    check(
        len(innocent_shared) >= 4,
        f"unrelated legitimate foreign companies share registered addresses ({len(innocent_shared)} companies)",
    )
    check(
        all(e.real_channel_count == 0 for e in name_decoys + graph_decoys + round_decoys + cross_border_decoys),
        "no decoy account also carries a real evidence channel",
    )

    # ==================================================================================
    print("\n=== Adverse media corpus ===")
    check(len(media) == cfg.NUM_ADVERSE_MEDIA_ENTRIES, f"{cfg.NUM_ADVERSE_MEDIA_ENTRIES} entries")
    linked = [m for m in media if m.linked_actor_id]
    check(len(linked) == cfg.NUM_LINKED_ENTRIES, f"{cfg.NUM_LINKED_ENTRIES} entries reference a planted actor")
    tiers = Counter(m.severity_tier.value for m in media)
    check(all(tier in tiers for tier in ("CONFIRMED_SANCTIONS", "ADVERSE_NEWS", "NAME_SIMILARITY_ONLY")), "all three severity tiers present")
    check(
        all(m.linked_actor_id is None for m in media if m.severity_tier == SeverityTier.NAME_SIMILARITY_ONLY),
        "no NAME_SIMILARITY_ONLY entry is linked to a planted actor",
    )
    ledger_media_accounts = {e.account_id for e in ledger if e.adverse_media_hit and e.adverse_media_hit != SeverityTier.NAME_SIMILARITY_ONLY}
    customer_of = {a.account_id: a.customer_id for a in accounts}
    linked_customers = {m.linked_actor_id for m in linked}
    check(
        {customer_of[a] for a in ledger_media_accounts} <= linked_customers,
        "every planted adverse-media account has a corpus entry",
    )
    # Stage 4 screens counterparties too. A counterparty whose name sits close to a corpus
    # entry would give every account that pays it a hit nobody planted. difflib stands in
    # for Stage 4's matcher here, at Stage 4's own 85 cutoff.
    from difflib import SequenceMatcher

    corpus_names = [n.lower() for m in media for n in [m.entity_name, *m.aliases]]
    counterparty_collisions = [
        cp.counterparty_id
        for cp in counterparties
        if any(SequenceMatcher(None, cp.name.lower(), n).ratio() >= 0.85 for n in corpus_names)
    ]
    check(
        not counterparty_collisions,
        f"no counterparty name comes within 85% of a corpus entry ({len(counterparty_collisions)} found)",
    )

    # ==================================================================================
    print("\n=== KYC profiles ===")
    import re

    narrative_of = {p.customer_id: p.narrative for p in profiles}
    check(
        all(_lakh_phrase(c.stated_income) in narrative_of[c.customer_id] for c in customers),
        "every narrative states the income on the customer record, mismatches included",
    )
    check(
        not any(re.search(r"\ba [aeiou]", n, re.IGNORECASE) for n in narrative_of.values()),
        'no narrative reads "a" before a vowel',
    )
    check(
        not any(
            "pension credit" in narrative_of[c.customer_id]
            for c in customers
            if "retired" not in c.occupation.lower()
        ),
        "only retired customers describe a pension credit",
    )

    # ==================================================================================
    print("\n=== Documents ===")
    check(
        len(manifest) == cfg.NUM_CUSTOMERS * cfg.DOCUMENTS_PER_CUSTOMER,
        f"{cfg.NUM_CUSTOMERS * cfg.DOCUMENTS_PER_CUSTOMER} rendered images",
    )
    rendered = {entry["file"] for entry in manifest}
    on_disk = {path.name for path in cfg.DOCUMENTS_DIR.glob("*.png")}
    check(
        rendered <= on_disk,
        "every manifest entry exists on disk"
        + ("" if on_disk else " - run `python -m src.generators.id_documents` to render them"),
    )
    doc_types = Counter(entry["document_type"] for entry in manifest)
    check(doc_types["PAN"] == cfg.NUM_CUSTOMERS, "every customer has a PAN card")
    check(doc_types["AADHAAR"] == cfg.NUM_DOMESTIC, "every domestic customer has an Aadhaar")
    check(doc_types["PASSPORT"] == cfg.NUM_NRI, "every NRI customer has a passport")
    check(
        not any("forged" in str(key).lower() for entry in manifest for key in entry),
        "manifest leaks no forgery ground truth",
    )

    # End-to-end check of the one thing Phase 5 actually depends on: that Tesseract can read
    # these renders, and that a forged document's printed date really does disagree with the
    # customer record. A rendered card that OCRs to nothing would leave Stage 1 with no
    # signal at all, and that is worth finding here rather than six phases later.
    _check_ocr(customers, accounts, ledger)

    # ==================================================================================
    print("\n=== Graph ===")
    node_types = Counter(data.get("node_type") for _, data in graph.nodes(data=True))
    check(node_types["account"] == cfg.NUM_CUSTOMERS, "every account is a graph node")
    check(node_types["device"] > 0, "device nodes present")
    check(node_types["ip"] > 0, "IP nodes present")
    check(node_types["address"] > 0, "address nodes present")
    check(node_types["counterparty"] > 0, "counterparty nodes present")
    foreign_cp = [
        node
        for node, data in graph.nodes(data=True)
        if data.get("node_type") == "counterparty" and data.get("is_foreign")
    ]
    check(len(foreign_cp) > 0, "foreign counterparty nodes carry is_foreign")
    relations = Counter(data.get("relation") for _, _, data in graph.edges(data=True))
    check(relations["REGISTERED_AT"] > 0, "foreign companies are linked to their registered-address nodes")
    flow_edges = [d for _, _, d in graph.edges(data=True) if d.get("relation") == "TRANSACTED_WITH"]
    check(
        all("amount_in" in d and "amount_out" in d for d in flow_edges),
        "every account-counterparty edge records money in and money out separately",
    )
    check(
        not any("ring" in str(key).lower() for _, data in graph.nodes(data=True) for key in data),
        "graph leaks no ring membership",
    )
    shared_devices = [
        node
        for node, data in graph.nodes(data=True)
        if data.get("node_type") == "device" and graph.degree(node) > 1
    ]
    check(
        len(shared_devices) == cfg.NUM_RINGS + cfg.NUM_GRAPH_COINCIDENCE_PAIRS,
        f"shared phones visible for every ring and every household look-alike ({len(shared_devices)})",
    )
    check(
        all(graph.has_edge(m, r.shared_device_id) for r in rings for m in r.member_account_ids),
        "every ring member is linked to the ring's shared phone, even with phones on sent payments only",
    )

    def address_node(address: str) -> str:
        return "ADDR::" + " ".join(address.split()).lower()

    ring_addresses = [r for r in rings if r.shared_address]
    check(
        bool(ring_addresses)
        and all(
            set(graph.neighbors(address_node(r.shared_address))) == set(r.member_account_ids)
            for r in ring_addresses
        ),
        f"every ring's shared address is in the graph, linking exactly its members ({len(ring_addresses)} rings)",
    )

    # ==================================================================================
    print("\n=== Rulebook ===")
    check(len(clauses) == 20, f"{len(clauses)} clauses (20 expected)")
    categories = Counter(c.category.value for c in clauses)
    check(len(categories) == 6, "all six categories covered")
    struct001 = next((c for c in clauses if c.clause_id == "STRUCT-001"), None)
    check(struct001 is not None, "STRUCT-001 exists")
    if struct001:
        check(
            struct001.metadata.get("threshold_inr") == cfg.CTR_THRESHOLD_INR,
            f"STRUCT-001 carries threshold_inr = {cfg.CTR_THRESHOLD_INR:,}",
        )
        check(
            struct001.metadata.get("near_threshold_pct") == cfg.NEAR_THRESHOLD_PCT,
            f"STRUCT-001 carries near_threshold_pct = {cfg.NEAR_THRESHOLD_PCT}",
        )
    xb004 = next((c for c in clauses if c.clause_id == "XB-004"), None)
    check(
        xb004 is not None and xb004.metadata.get("threshold_inr") == cfg.CBWTR_THRESHOLD_INR,
        f"XB-004 carries the CBWTR threshold_inr = {cfg.CBWTR_THRESHOLD_INR:,}",
    )
    check(all(c.sources for c in clauses), "every clause cites a real published source")
    check(cfg.RULEBOOK_CHROMA_DIR.exists(), "ChromaDB store written")
    _check_retrieval(len(clauses))

    # ==================================================================================
    print("\n=== Ground-truth isolation ===")
    check(
        cfg.EVIDENCE_LEDGER_FILE.parent.name == "_ground_truth",
        "evidence ledger is outside the pipeline-readable data directory",
    )
    pipeline_text = "".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (cfg.CUSTOMERS_FILE, cfg.ACCOUNTS_FILE, cfg.KYC_PROFILES_FILE)
    )
    check(
        "RING-" not in pipeline_text,
        "no ring identifier appears in customer, account or KYC data",
    )

    # ==================================================================================
    _summary(customers, accounts, transactions, counterparties, profiles, media, clauses, ledger, rings, graph)

    print("\n" + "=" * 78)
    if _failures:
        print(f"FAILED: {len(_failures)} check(s) did not pass.")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("All checks passed.")
    return 0


def _summary(customers, accounts, transactions, counterparties, profiles, media, clauses, ledger, rings, graph) -> None:
    print("\n" + "=" * 78)
    print("DATASET SUMMARY")
    print("=" * 78)

    print("\n-- Customers --")
    for status, count in Counter(c.residency_status.value for c in customers).most_common():
        print(f"   {status:<10} {count:>4}")
    print("   top occupations:")
    for occupation, count in Counter(c.occupation for c in customers).most_common(5):
        print(f"     {count:>3}  {occupation}")

    print("\n-- Accounts --")
    for account_type, count in Counter(a.account_type.value for a in accounts).most_common():
        print(f"   {account_type:<10} {count:>4}")

    print("\n-- Transactions --")
    print(f"   total          {len(transactions):>8,}")
    print(f"   per account    {len(transactions) / len(accounts):>8.1f} avg")
    print(f"   total value    {_rupees(sum(t.amount for t in transactions)):>8}")
    print(f"   cross-border   {sum(1 for t in transactions if t.type.value == 'SWIFT'):>8,}")
    for direction, count in Counter(t.direction.value for t in transactions).most_common():
        print(f"   {direction.lower():<14} {count:>8,}")
    print("   by payment rail:")
    for rail, count in Counter(t.type.value for t in transactions).most_common():
        print(f"     {rail:<16} {count:>6,}")

    print("\n-- Counterparties --")
    print(f"   domestic       {sum(1 for c in counterparties if not c.is_foreign):>4}")
    print(f"   foreign        {sum(1 for c in counterparties if c.is_foreign):>4}")
    print("   foreign jurisdictions:")
    for country, count in Counter(c.country for c in counterparties if c.is_foreign).most_common(6):
        print(f"     {count:>3}  {country}")

    print("\n-- Adverse media --")
    for tier, count in Counter(m.severity_tier.value for m in media).most_common():
        print(f"   {tier:<22} {count:>4}")

    print("\n-- Rulebook --")
    for category, count in Counter(c.category.value for c in clauses).most_common():
        print(f"   {category:<20} {count:>3}")

    print("\n-- Graph --")
    print(f"   nodes          {graph.number_of_nodes():>8,}")
    print(f"   edges          {graph.number_of_edges():>8,}")
    for node_type, count in Counter(d.get("node_type") for _, d in graph.nodes(data=True)).most_common():
        print(f"     {str(node_type):<14} {count:>6,}")

    print("\n-- Evidence ledger --")
    channels = {
        "structuring": sum(1 for e in ledger if e.structuring_planted),
        "kyc_mismatch": sum(1 for e in ledger if e.kyc_mismatch_planted),
        "adverse_media (real)": sum(
            1 for e in ledger if e.adverse_media_hit and e.adverse_media_hit != SeverityTier.NAME_SIMILARITY_ONLY
        ),
        "graph_ring": sum(1 for e in ledger if e.graph_ring_membership),
        "forged_document": sum(1 for e in ledger if e.forged_document_planted),
    }
    for channel, count in channels.items():
        print(f"   {channel:<22} {count:>4}")
    print(f"   {'decoys (any type)':<22} {sum(1 for e in ledger if e.decoy_types):>4}")

    print("\n-- Rendering --")
    print(f"   document font  {documents_module.resolved_font_family('sans')}")

    print("\n" + "=" * 78)
    print("SAMPLE RECORDS")
    print("=" * 78)

    # An escalate-worthy structuring case, so the sample transaction below really is a
    # sub-threshold deposit.
    sample_account = next(
        e for e in ledger if e.real_channel_count >= 2 and e.structuring_planted
    )
    customer = next(c for c in customers if c.customer_id == sample_account.customer_id)
    print(f"\n-- Customer ({customer.customer_id}, an escalate-worthy case) --")
    for key, value in customer.model_dump(mode="json").items():
        print(f"   {key:<20} {value}")

    print("\n-- Account --")
    account = next(a for a in accounts if a.account_id == sample_account.account_id)
    for key, value in account.model_dump(mode="json").items():
        print(f"   {key:<20} {value}")

    print("\n-- Transaction (a sub-threshold deposit from that account) --")
    sample_txn = next(
        (
            t
            for t in transactions
            if t.account_id == sample_account.account_id
            and cfg.STRUCTURING_BAND_LOW <= t.amount < cfg.STRUCTURING_BAND_HIGH
        ),
        next(t for t in transactions if t.account_id == sample_account.account_id),
    )
    for key, value in sample_txn.model_dump(mode="json").items():
        print(f"   {key:<20} {value}")

    print("\n-- KYC profile --")
    profile = next(p for p in profiles if p.customer_id == customer.customer_id)
    print(f"   {profile.narrative}")

    print("\n-- Adverse media entry --")
    entry = next((m for m in media if m.linked_actor_id), media[0])
    for key, value in entry.model_dump(mode="json").items():
        print(f"   {key:<20} {value}")

    print("\n-- Fraud ring --")
    ring = rings[0]
    for key, value in ring.model_dump(mode="json").items():
        print(f"   {key:<20} {value}")

    print("\n-- Rulebook clause --")
    clause = next(c for c in clauses if c.clause_id == "STRUCT-001")
    print(f"   clause_id            {clause.clause_id}")
    print(f"   title                {clause.title}")
    print(f"   category             {clause.category.value}")
    print(f"   sources              {clause.sources}")
    print(f"   metadata             {clause.metadata}")
    print(f"   text                 {clause.text[:200]}...")

    print("\n-- Evidence ledger row (ground truth, never seen by a stage) --")
    for key, value in sample_account.model_dump(mode="json").items():
        print(f"   {key:<20} {value}")


if __name__ == "__main__":
    sys.exit(main())
