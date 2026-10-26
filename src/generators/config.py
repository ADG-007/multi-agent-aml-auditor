"""Every number the data generators use. verify.py checks the data against them."""

from __future__ import annotations

from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------

#: Root seed. Each random stream = this + a key (streams.py). The data is committed, not rebuilt.
SEED = 42

#: The pinned "today" every date works back from (not the clock, so rebuilds match).
REFERENCE_DATE = date(2026, 9, 22)

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
RULEBOOK_DIR = DATA_DIR / "rulebook"
RULEBOOK_MARKDOWN_DIR = RULEBOOK_DIR / "clauses"
RULEBOOK_CHROMA_DIR = RULEBOOK_DIR / "chroma"

#: The answer key. No pipeline stage reads it; only src/evaluation does.
GROUND_TRUTH_DIR = DATA_DIR / "_ground_truth"

CUSTOMERS_FILE = DATA_DIR / "customers.json"
ACCOUNTS_FILE = DATA_DIR / "accounts.json"
TRANSACTIONS_FILE = DATA_DIR / "transactions.json"
COUNTERPARTIES_FILE = DATA_DIR / "counterparties.json"
KYC_PROFILES_FILE = DATA_DIR / "kyc_profiles.json"
ADVERSE_MEDIA_FILE = DATA_DIR / "adverse_media.json"
GRAPH_FILE = DATA_DIR / "graph.graphml"
DOCUMENT_MANIFEST_FILE = DOCUMENTS_DIR / "manifest.json"
RULEBOOK_JSON_FILE = RULEBOOK_DIR / "clauses.json"
EVIDENCE_LEDGER_FILE = GROUND_TRUTH_DIR / "evidence_ledger.json"
FRAUD_RINGS_FILE = GROUND_TRUTH_DIR / "fraud_rings.json"

# --------------------------------------------------------------------------------------
# Population
# --------------------------------------------------------------------------------------

#: Base population size.
NUM_BASE_CUSTOMERS = 300

#: 75% domestic / 25% NRI across the base 300.
NUM_BASE_DOMESTIC = 225
NUM_NRI = 75

#: The cross-border block, added on top of the base 300 (all domestic residents). Each cohort
#: is either a planted typology or its legitimate look-alike:
#:   round_trip            money sent abroad that comes back from a linked foreign entity
#:   unexplained_inward    foreign money arriving that the customer's profile cannot explain
#:   family_inward         relatives abroad sending household money home (legitimate)
#:   education_outward     parents paying a child's university fees abroad (legitimate)
#:   lrs_outward           gifts and investments abroad within the LRS (legitimate)
CROSS_BORDER_COHORTS: tuple[tuple[str, int], ...] = (
    ("round_trip", 12),
    ("unexplained_inward", 8),
    ("family_inward", 10),
    ("education_outward", 8),
    ("lrs_outward", 6),
)
NUM_CROSS_BORDER_CUSTOMERS = sum(count for _, count in CROSS_BORDER_COHORTS)

#: The mule block, added after the cross-border block. Mule typologies and look-alikes:
#:   network_mule        one of four accounts run from a handler's shared phone, forwarding
#:                       victims' money to the network's collector within minutes to hours
#:   solo_mule           the same pass-through on its own phone and its own collector
#:   complicit_merchant  a genuine busy merchant with a thin layer of victims' money in its
#:                       last 30 days, forwarded to a network's collector
#:   merchant_lookalike  a fuel outlet or roadside electronics shop taking customer payments
#:                       on a personal UPI ID (legitimate)
#:   life_event          a salaried customer with one sudden, legitimate spike (legitimate)
MULE_BLOCK_COHORTS: tuple[tuple[str, int], ...] = (
    ("network_mule", 8),
    ("solo_mule", 2),
    ("complicit_merchant", 2),
    ("merchant_lookalike", 8),
    ("life_event", 6),
)
NUM_MULE_BLOCK_CUSTOMERS = sum(count for _, count in MULE_BLOCK_COHORTS)

#: Blocks in customer-id order. Only appended, each with its own streams, so earlier data stays.
POPULATION_BLOCKS: tuple[tuple[str, int], ...] = (
    ("base", NUM_BASE_CUSTOMERS),
    ("cross_border", NUM_CROSS_BORDER_CUSTOMERS),
    ("mule", NUM_MULE_BLOCK_CUSTOMERS),
)

#: Blocks the blacklist was screened against. Later blocks clear their own names instead.
CORPUS_SCREENED_BLOCKS: tuple[str, ...] = ("base", "cross_border")

NUM_CUSTOMERS = sum(count for _, count in POPULATION_BLOCKS)
NUM_DOMESTIC = NUM_CUSTOMERS - NUM_NRI


def block_of(entity_id: str) -> str:
    """Which population block a CUST-/ACC- id belongs to."""
    index = int(entity_id.rsplit("-", 1)[1])
    start = 0
    for name, count in POPULATION_BLOCKS:
        if index <= start + count:
            return name
        start += count
    raise ValueError(f"{entity_id} is outside every population block")

#: Declared annual income spread, INR.
INCOME_MIN = 300_000
INCOME_MAX = 5_000_000

# --------------------------------------------------------------------------------------
# Money (INR)
# --------------------------------------------------------------------------------------

#: Cash Transaction Report threshold. PMLA (Maintenance of Records) Rules 2005, Rule 3 -
#: cash transactions above ten lakh rupees are reportable to FIU-IND.
CTR_THRESHOLD_INR = 1_000_000

#: Band floor as a share of the threshold. Written into STRUCT-001, which the stages read.
NEAR_THRESHOLD_PCT = 0.9

#: Planted structuring deposits fall in [LOW, HIGH], both inclusive.
STRUCTURING_BAND_LOW = int(CTR_THRESHOLD_INR * NEAR_THRESHOLD_PCT)
STRUCTURING_BAND_HIGH = CTR_THRESHOLD_INR

#: CBWTR threshold (PMLA Rules 2005, Rule 3): each cross-border wire above it is reported.
CBWTR_THRESHOLD_INR = 500_000

#: Values a benign customer plausibly transacts at, and the basis of the round-amount decoy.
ROUND_AMOUNTS = (1_000, 5_000, 10_000, 25_000, 50_000, 100_000)

# --------------------------------------------------------------------------------------
# Transaction history
# --------------------------------------------------------------------------------------

HISTORY_DAYS = 90
TXNS_PER_ACCOUNT_MIN = 30
TXNS_PER_ACCOUNT_MAX = 90

#: Structuring window (Rule 3 aggregates connected cash within a month).
STRUCTURING_WINDOW_DAYS = 30
STRUCTURING_TXN_MIN = 4
STRUCTURING_TXN_MAX = 8

# --------------------------------------------------------------------------------------
# Fraud rings
# --------------------------------------------------------------------------------------

NUM_LOCAL_RINGS = 6
NUM_INTERNATIONAL_RINGS = 2
NUM_ROUND_TRIP_RINGS = 3
#: Mule networks: the mule block's network_mule cohort, four accounts each.
NUM_MULE_RINGS = 2
NUM_RINGS = NUM_LOCAL_RINGS + NUM_INTERNATIONAL_RINGS + NUM_ROUND_TRIP_RINGS + NUM_MULE_RINGS
RING_SIZE = 4

#: Per ring: members with a second channel (ESCALATE). The rest carry the ring alone (MONITOR).
RING_CORE_MEMBERS = 3

# --------------------------------------------------------------------------------------
# Planting density (drives the answer-key labels)
# --------------------------------------------------------------------------------------

#: These count planted channels, not labels (labels: EvidenceLedgerEntry.expected_disposition).

#: Two or more real evidence channels (~13% of accounts). 30 from the base population, plus
#: the 9 round-trip core members (ring + KYC mismatch), plus the 8 network mules (ring +
#: pass-through).
NUM_MULTI_CHANNEL = 30 + NUM_ROUND_TRIP_RINGS * 3 + NUM_MULE_RINGS * 4

#: Exactly one real channel (~20%). 60 from the base population, plus the 3 round-trip solo
#: members (ring only), the unexplained-inward cohort (KYC mismatch only), and the 2 solo
#: mules and 2 complicit merchants (pass-through only).
NUM_SINGLE_CHANNEL = 60 + NUM_ROUND_TRIP_RINGS + 8 + 2 + 2

# --------------------------------------------------------------------------------------
# Deliberate false-positive decoys
# --------------------------------------------------------------------------------------

#: Clean accounts whose name collides with a NAME_SIMILARITY_ONLY corpus entry.
NUM_NAME_SIMILARITY_DECOYS = 15

#: Clean accounts sharing a device or address for a legitimate reason (household, shop).
NUM_GRAPH_COINCIDENCE_PAIRS = 6

#: Clean accounts that transact heavily at round amounts without any sub-threshold pattern.
NUM_ROUND_AMOUNT_DECOYS = 15

#: Legitimate cross-border customers: family_inward + education_outward + lrs_outward.
NUM_CROSS_BORDER_DECOYS = 24

#: Legitimate pass-through look-alikes from the mule block.
NUM_MERCHANT_LOOKALIKE_DECOYS = 8
NUM_LIFE_EVENT_DECOYS = 6

# --------------------------------------------------------------------------------------
# Money mules
# --------------------------------------------------------------------------------------

#: Minutes a mule holds victims' money before sweeping it on.
MULE_SWEEP_DELAY_MINUTES = (5, 180)

#: The share of what arrives that a mule forwards; the rest is the mule's commission.
MULE_FORWARD_SHARE = (0.88, 0.97)

#: Burst size as a share of declared income. Modest, so no KYC mismatch is planted by accident.
MULE_INFLOW_SHARE_OF_INCOME = (0.3, 0.6)

#: The complicit merchant's illicit layer, as a share of its genuine 90-day takings, and
#: the recent window it sits in.
COMPLICIT_LAYER_SHARE = (0.05, 0.10)
COMPLICIT_LAYER_DAYS = 30

#: RBI's draft SOP on suspected money-mule accounts defines a suspected mule transaction as
#: one of Rs 1,000 or more flagged by transaction monitoring. Carried in MULE-002's metadata.
MULE_TXN_FLOOR_INR = 1_000

# --------------------------------------------------------------------------------------
# Adverse media corpus
# --------------------------------------------------------------------------------------

NUM_ADVERSE_MEDIA_ENTRIES = 160
NUM_INDIAN_ENTITY_ENTRIES = 120
NUM_FOREIGN_ENTITY_ENTRIES = 40

#: 20% of the corpus references a planted actor.
NUM_LINKED_ENTRIES = 32

# --------------------------------------------------------------------------------------
# Counterparties
# --------------------------------------------------------------------------------------

NUM_DOMESTIC_COUNTERPARTIES = 180

#: The shared foreign cluster the two international rings funnel into.
FOREIGN_RING_CLUSTER_SIZE = 5

#: Countries NRI customers realistically bank from, weighted toward the actual size of the
#: Indian diaspora in each.
NRI_COUNTRIES = (
    ("United Arab Emirates", 0.26),
    ("United States", 0.20),
    ("Saudi Arabia", 0.14),
    ("United Kingdom", 0.11),
    ("Singapore", 0.09),
    ("Canada", 0.09),
    ("Australia", 0.06),
    ("Qatar", 0.05),
)

# --------------------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------------------

#: Two rendered documents per customer: domestic gets Aadhaar + PAN, NRI gets Passport + PAN.
DOCUMENTS_PER_CUSTOMER = 2
