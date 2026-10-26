"""Check the generated dataset, then describe it.

    python -m src.generators.verify

Asserts the data matches its design (370 customers, 13 rings, 160 list entries, 47 accounts
with 2+ channels, 49 ESCALATE labels), then prints a readable summary with one sample record
per generator.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from difflib import SequenceMatcher

import networkx as nx

from src.generators import config as cfg
from src.generators import id_documents as documents_module
from src.schemas import (
    Account,
    AdverseMediaEntry,
    Counterparty,
    Customer,
    Disposition,
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


#: One probe per category, worded like Stage 6's queries. Each lists every clause that fits.
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
    (
        "payments from many unrelated people forwarded to one account within hours of arriving",
        frozenset({"MULE-002", "MULE-001"}),
    ),
    (
        "large NEFT and RTGS transfers moved in and out with no business reason",
        frozenset({"STR-002", "STR-001"}),
    ),
)


def _check_retrieval(clause_count: int) -> None:
    """Probe the ChromaDB index: the right clause must come back near the top."""
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

    # A right clause in the set and a top hit in the right category (not top-1 exact).
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
    # Reported, not asserted: shows whether Stage 6's 0.75 cutoff still fits.
    print(
        f"  [note] top-hit cosine distances "
        f"{min(top_distances):.3f}-{max(top_distances):.3f} "
        f"(Stage 6's 0.75 cutoff would drop "
        f"{sum(1 for d in top_distances if d > 0.75)}/{len(top_distances)} top hits)"
    )


def _check_ocr(customers, accounts, ledger) -> None:
    """OCR a sample of clean and forged cards. Skipped with a note if Tesseract is missing."""
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
    check(
        sum(1 for r in rings if r.ring_type.value == "MULE") == cfg.NUM_MULE_RINGS
        and all(r.collector_counterparty_id for r in rings if r.ring_type.value == "MULE"),
        f"{cfg.NUM_MULE_RINGS} mule networks, each with a collector",
    )
    ring_members = [m for r in rings for m in r.member_account_ids]
    check(len(ring_members) == len(set(ring_members)), "no account belongs to two rings")
    check(all(r.shared_device_id for r in rings), "every ring shares a device identifier")

    # ==================================================================================
    print("\n=== Transactions ===")
    check(len(transactions) > 0, "transactions generated")
    check(
        len({t.transaction_id for t in transactions}) == len(transactions),
        "every transaction id is unique",
    )
    check(
        all(
            re.fullmatch(r"TXN-\d{8}-\d{6}-\d{4}(-\d+)?", t.transaction_id)
            and t.transaction_id[4:19] == t.timestamp.strftime("%Y%m%d-%H%M%S")
            and t.transaction_id[20:24] == t.account_id.split("-")[1]
            for t in transactions
        ),
        "transaction ids encode their own posting time and account (TXN-YYYYMMDD-HHMMSS-NNNN)",
    )
    by_time = sorted(transactions, key=lambda t: (t.timestamp, t.account_id))
    check(
        [t.transaction_id for t in by_time] == sorted(t.transaction_id for t in transactions),
        "transaction ids sort bank-wide in time order",
    )
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
        if cfg.STRUCTURING_BAND_LOW <= t.amount <= cfg.STRUCTURING_BAND_HIGH
    ]
    check(len(band) > 0, "sub-threshold structuring band is populated")
    check(
        all(t.amount < cfg.CTR_THRESHOLD_INR for t in band),
        "no planted structuring transaction reaches the CTR threshold",
    )
    check(all(t.direction.value == "CREDIT" for t in band), "structuring deposits are all credits")
    check(
        all(t.type.value == "CASH_DEPOSIT" for t in band),
        "structuring deposits are all cash (structuring is a cash-only signal)",
    )

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
    print("\n=== Money-mule pass-through ===")
    _check_mules(ledger, rings, transactions, counterparties)

    # ==================================================================================
    print("\n=== Evidence ledger (ground truth) ===")
    check(len(ledger) == cfg.NUM_CUSTOMERS, "one ledger row per account")
    multi = [e for e in ledger if e.real_channel_count >= 2]
    single = [e for e in ledger if e.real_channel_count == 1]
    clean = [e for e in ledger if e.real_channel_count == 0]
    check(len(multi) == cfg.NUM_MULTI_CHANNEL, f"{cfg.NUM_MULTI_CHANNEL} accounts with two or more channels")
    check(len(single) == cfg.NUM_SINGLE_CHANNEL, f"{cfg.NUM_SINGLE_CHANNEL} accounts with exactly one channel")
    check(
        len(clean) == cfg.NUM_CUSTOMERS - cfg.NUM_MULTI_CHANNEL - cfg.NUM_SINGLE_CHANNEL,
        f"{cfg.NUM_CUSTOMERS - cfg.NUM_MULTI_CHANNEL - cfg.NUM_SINGLE_CHANNEL} accounts with no real channel",
    )
    # Labels: channels decide, except that a sanctions hit escalates on its own.
    sanctions_only = [
        e for e in single if e.adverse_media_hit == SeverityTier.CONFIRMED_SANCTIONS
    ]
    labels = Counter(e.expected_disposition for e in ledger)
    check(
        labels[Disposition.SUSPICIOUS_ESCALATE] == len(multi) + len(sanctions_only)
        and labels[Disposition.MONITOR] == len(single) - len(sanctions_only)
        and labels[Disposition.CLEAR] == len(clean),
        f"labels: {labels[Disposition.SUSPICIOUS_ESCALATE]} escalate-worthy "
        f"({len(sanctions_only)} on a confirmed sanctions hit alone), "
        f"{labels[Disposition.MONITOR]} monitor-worthy, {labels[Disposition.CLEAR]} clean",
    )
    check(
        all(
            e.expected_disposition == Disposition.SUSPICIOUS_ESCALATE
            for e in ledger
            if e.adverse_media_hit == SeverityTier.CONFIRMED_SANCTIONS
        ),
        "every confirmed-sanctions account is escalate-worthy",
    )
    check(
        sum(1 for e in ledger if e.graph_ring_membership) == cfg.NUM_RINGS * cfg.RING_SIZE,
        f"{cfg.NUM_RINGS * cfg.RING_SIZE} accounts carry ring membership",
    )

    # Every ring must contain at least one member with no adverse-media footprint at all.
    # Those are the accounts only Stage 5 can reach.
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
    # And no unplanted account in the band (it would be an unlabelled positive).
    check(
        accounts_with_band <= planted_structuring,
        "no unplanted account has transactions in the structuring band",
    )

    # Clean turnover tracks declared income, so a KYC mismatch stands out.
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
    # Only accounts with no planted channel are held to the clean bound.
    untouched = {e.account_id for e in ledger if e.real_channel_count == 0}
    clean_ratios = [r for a, r in ratios.items() if a in untouched]
    mismatch_ratios = [r for a, r in ratios.items() if a in mismatch_accounts]
    pass_through_ratios = [r for a, r in ratios.items() if a in {e.account_id for e in ledger if e.pass_through_planted}]
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
    # Mules stay below KYC-mismatch volume, so no KYC channel is planted by accident.
    check(
        not pass_through_ratios or max(pass_through_ratios) < min(mismatch_ratios),
        f"pass-through accounts stay below planted KYC-mismatch turnover "
        f"(max {max(pass_through_ratios or [0]):.1f}x vs {min(mismatch_ratios):.1f}x)",
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
    merchant_decoys = [e for e in ledger if "merchant_lookalike" in e.decoy_types]
    life_decoys = [e for e in ledger if "life_event" in e.decoy_types]
    check(
        len(merchant_decoys) == cfg.NUM_MERCHANT_LOOKALIKE_DECOYS,
        f"{cfg.NUM_MERCHANT_LOOKALIKE_DECOYS} personal-UPI merchant look-alikes",
    )
    check(len(life_decoys) == cfg.NUM_LIFE_EVENT_DECOYS, f"{cfg.NUM_LIFE_EVENT_DECOYS} life-event look-alikes")
    check(
        all(
            e.real_channel_count == 0
            for e in name_decoys + graph_decoys + round_decoys + cross_border_decoys + merchant_decoys + life_decoys
        ),
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
    # No counterparty name near a list entry (difflib at 85; cheap bounds first for speed).
    corpus_names = [n.lower() for m in media for n in [m.entity_name, *m.aliases]]

    def near_corpus(name: str) -> bool:
        for other in corpus_names:
            matcher = SequenceMatcher(None, name.lower(), other)
            if matcher.real_quick_ratio() >= 0.85 and matcher.quick_ratio() >= 0.85 and matcher.ratio() >= 0.85:
                return True
        return False

    counterparty_collisions = [cp.counterparty_id for cp in counterparties if near_corpus(cp.name)]
    check(
        not counterparty_collisions,
        f"no counterparty name comes within 85% of a corpus entry ({len(counterparty_collisions)} found)",
    )
    planted_actors = {m.linked_actor_id for m in media if m.linked_actor_id}
    later_block_collisions = [
        c.customer_id
        for c in customers
        if cfg.block_of(c.customer_id) not in cfg.CORPUS_SCREENED_BLOCKS
        and c.customer_id not in planted_actors
        and near_corpus(c.name)
    ]
    check(
        not later_block_collisions,
        f"no later-block customer name comes within 85% of a corpus entry ({len(later_block_collisions)} found)",
    )

    # ==================================================================================
    print("\n=== KYC profiles ===")

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

    # Tesseract reads the renders, and a forged card's DOB disagrees with the record.
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
    check(len(clauses) == 24, f"{len(clauses)} clauses (24 expected)")
    categories = Counter(c.category.value for c in clauses)
    check(len(categories) == 8, "all eight categories covered")
    mule002 = next((c for c in clauses if c.clause_id == "MULE-002"), None)
    check(
        mule002 is not None and mule002.metadata.get("txn_floor_inr") == cfg.MULE_TXN_FLOOR_INR,
        f"MULE-002 carries the suspected-mule transaction floor txn_floor_inr = {cfg.MULE_TXN_FLOOR_INR:,}",
    )
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


def _check_mules(ledger, rings, transactions, counterparties) -> None:
    """Mules and their look-alikes differ by where money goes and how long it stays."""
    by_account: dict[str, list] = defaultdict(list)
    for txn in transactions:
        by_account[txn.account_id].append(txn)

    from src.generators.rings import MULE_NETWORK_COLLECTOR_IDS, SOLO_MULE_COLLECTOR_IDS

    mule_rings = [r for r in rings if r.ring_type.value == "MULE"]
    network_collectors = {r.collector_counterparty_id for r in mule_rings}
    all_collectors = set(MULE_NETWORK_COLLECTOR_IDS) | set(SOLO_MULE_COLLECTOR_IDS)
    check(
        len({c.counterparty_id for c in counterparties}) == len(counterparties),
        "every counterparty id is unique",
    )
    # No planted tell: no :00-second postings, no telling collector ids.
    zero_seconds = sum(1 for t in transactions if t.timestamp.second == 0) / len(transactions)
    check(zero_seconds < 0.05, f"postings are not marked by round seconds ({zero_seconds:.1%} on :00)")
    check(
        not any(re.search(r"COL|MULE|VICTIM", c.counterparty_id) for c in counterparties),
        "no counterparty id names its role",
    )
    senders_to: dict[str, set[str]] = defaultdict(set)
    for txn in transactions:
        if txn.counterparty_id in all_collectors and txn.direction.value == "DEBIT":
            senders_to[txn.counterparty_id].add(txn.account_id)

    pass_through = [e for e in ledger if e.pass_through_planted]
    lookalikes = [e for e in ledger if "merchant_lookalike" in e.decoy_types or "life_event" in e.decoy_types]
    check(len(pass_through) == 12, f"12 pass-through accounts ({len(pass_through)})")

    check(
        all(senders_to[r.collector_counterparty_id] >= set(r.member_account_ids) for r in mule_rings),
        "every network mule sweeps into its network's collector",
    )
    ring_of = {m: r.ring_id for r in rings for m in r.member_account_ids}
    complicit = [e.account_id for e in pass_through if e.account_id not in ring_of and any(
        t.counterparty_id in network_collectors for t in by_account[e.account_id])]
    solo = [e.account_id for e in pass_through if e.account_id not in ring_of and e.account_id not in complicit]
    check(
        len(complicit) == 2 and all(len(senders_to[c]) == cfg.RING_SIZE + 1 for c in network_collectors),
        "each network collector also receives from one complicit merchant with no phone in common",
    )
    solo_collectors = all_collectors - network_collectors
    check(
        len(solo) == 2 and all(len(senders_to[c]) == 1 for c in solo_collectors),
        "each solo mule's collector is used by that mule alone - no graph link to lean on",
    )
    check(
        not any(senders_to[c] & {e.account_id for e in lookalikes} for c in all_collectors),
        "no look-alike ever pays a collector",
    )

    def hold_hours(account_id: str, is_onward) -> list[float]:
        """Hours from each payment by a stranger (a victim, a walk-in customer - the account's
        own CPM- payers) until the next onward payment."""
        rows = sorted(by_account[account_id], key=lambda t: t.timestamp)
        onward = [t.timestamp for t in rows if t.direction.value == "DEBIT" and is_onward(t)]
        waits = []
        for t in rows:
            if t.direction.value == "CREDIT" and t.counterparty_id.startswith("CPM-"):
                later = [o for o in onward if o >= t.timestamp]
                if later:
                    waits.append((later[0] - t.timestamp).total_seconds() / 3600)
        return waits

    mule_holds = [
        statistics.median(hold_hours(a, lambda t: t.counterparty_id in all_collectors or t.type.value == "ATM"))
        for a in solo + [m for r in mule_rings for m in r.member_account_ids]
    ]
    merchant_holds = [
        statistics.median(hold_hours(e.account_id, lambda t: t.type.value in ("NEFT", "RTGS")))
        for e in ledger
        if "merchant_lookalike" in e.decoy_types
    ]
    check(
        max(mule_holds) < 4 and min(merchant_holds) > 24,
        f"mules hold money for hours (median at most {max(mule_holds):.1f}h), look-alike merchants "
        f"for days (median at least {min(merchant_holds):.0f}h)",
    )

    last_day = max(t.timestamp for t in transactions).date()
    layer_start = last_day - timedelta(days=cfg.COMPLICIT_LAYER_DAYS)
    check(
        all(
            t.timestamp.date() >= layer_start
            for a in complicit
            for t in by_account[a]
            if t.counterparty_id in network_collectors
        ),
        f"complicit merchants' forwarding sits only in the last {cfg.COMPLICIT_LAYER_DAYS} days",
    )

    def repeat_share(account_id: str) -> float:
        payers = Counter(
            t.counterparty_id for t in by_account[account_id]
            if t.direction.value == "CREDIT" and t.type.value in ("UPI", "IMPS")
        )
        total = sum(payers.values())
        return sum(n for n in payers.values() if n > 1) / total if total else 0.0

    check(
        all(repeat_share(a) == 0 for a in solo + [m for r in mule_rings for m in r.member_account_ids])
        and all(repeat_share(e.account_id) < 0.5 for e in ledger if "merchant_lookalike" in e.decoy_types),
        "mule victims each pay once, and look-alike merchants' payers are mostly one-time too",
    )


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
        "pass_through": sum(1 for e in ledger if e.pass_through_planted),
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
        e
        for e in ledger
        if e.expected_disposition == Disposition.SUSPICIOUS_ESCALATE and e.structuring_planted
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
            and cfg.STRUCTURING_BAND_LOW <= t.amount <= cfg.STRUCTURING_BAND_HIGH
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
