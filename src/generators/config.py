"""Every tunable quantity the synthetic data layer uses.

One module so a number like the CTR threshold or the escalate-worthy account count has
exactly one definition. phase_1_execution.md fixes these values; verify.py asserts the
generated data actually matches them, which is what catches a generator bug immediately
rather than in Phase 11 when the evaluation numbers look strange.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------

#: Applied to random, Faker and numpy at generator entry. The generated dataset is
#: committed to the repo rather than regenerated per run, so that a library version change
#: in someone else's environment cannot silently shift the ground-truth labels.
SEED = 42

#: The "today" every generator works backwards from.
#:
#: Pinned rather than read from the clock. Invariant 7 requires the dataset to be
#: regenerable from a fixed seed, and a clock-anchored run satisfies that only until
#: midnight: every date of birth, transaction timestamp, document expiry and listing date
#: would shift by a day, so a regenerated dataset would no longer match the committed one
#: and a diff could not distinguish a real generator change from the calendar moving.
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

#: Deliberately separated from the rest of data/. Nothing under this path may be read by a
#: pipeline stage at inference time - it exists only for Phase 11's label derivation.
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

#: 300 rather than the 150 in MASTER_PLAN.md. At 150, misjudging two of roughly fifteen
#: escalate-worthy cases swings Phase 11's reported recall by more than 13 points; 300
#: halves that swing to about 6.7 points, which is the best stability-per-generation-cost
#: point before diminishing returns.
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

NUM_CUSTOMERS = NUM_BASE_CUSTOMERS + NUM_CROSS_BORDER_CUSTOMERS
NUM_DOMESTIC = NUM_BASE_DOMESTIC + NUM_CROSS_BORDER_CUSTOMERS

#: Declared annual income spread, INR.
INCOME_MIN = 300_000
INCOME_MAX = 5_000_000

# --------------------------------------------------------------------------------------
# Money (INR)
# --------------------------------------------------------------------------------------

#: Cash Transaction Report threshold. PMLA (Maintenance of Records) Rules 2005, Rule 3 -
#: cash transactions above ten lakh rupees are reportable to FIU-IND.
CTR_THRESHOLD_INR = 1_000_000

#: Lower edge of the structuring band, as a fraction of the threshold. Externalised into
#: the STRUCT-001 clause metadata so Stage 0's routing check and Stage 2's formula read the
#: same two numbers from one source.
NEAR_THRESHOLD_PCT = 0.9

#: Deposits deliberately parked just under the reporting ceiling.
STRUCTURING_BAND_LOW = int(CTR_THRESHOLD_INR * NEAR_THRESHOLD_PCT)
STRUCTURING_BAND_HIGH = CTR_THRESHOLD_INR

#: Cross-Border Wire Transfer Report threshold. PMLA (Maintenance of Records) Rules 2005,
#: Rule 3 - every cross-border wire above five lakh rupees, in either direction, is reported
#: to FIU-IND monthly. Judged per transfer, not aggregated.
CBWTR_THRESHOLD_INR = 500_000

#: Values a benign customer plausibly transacts at, and the basis of the round-amount decoy.
ROUND_AMOUNTS = (1_000, 5_000, 10_000, 25_000, 50_000, 100_000)

# --------------------------------------------------------------------------------------
# Transaction history
# --------------------------------------------------------------------------------------

HISTORY_DAYS = 90
TXNS_PER_ACCOUNT_MIN = 30
TXNS_PER_ACCOUNT_MAX = 90

#: The structuring window inside the 90-day history. Rule 3 treats a series of integrally
#: connected cash transactions within a month as one reportable transaction, so a month is
#: the window a real analyst would aggregate over.
STRUCTURING_WINDOW_DAYS = 30
STRUCTURING_TXN_MIN = 4
STRUCTURING_TXN_MAX = 8

# --------------------------------------------------------------------------------------
# Fraud rings
# --------------------------------------------------------------------------------------

NUM_LOCAL_RINGS = 6
NUM_INTERNATIONAL_RINGS = 2
NUM_ROUND_TRIP_RINGS = 3
NUM_RINGS = NUM_LOCAL_RINGS + NUM_INTERNATIONAL_RINGS + NUM_ROUND_TRIP_RINGS
RING_SIZE = 4

#: Per ring: this many members sit in the escalate-worthy group with a second planted
#: channel; the remaining member carries ring membership alone and is monitor-worthy. That
#: single-channel member is the case Graph corroboration has to catch on its own.
RING_CORE_MEMBERS = 3

# --------------------------------------------------------------------------------------
# Planting density (drives Phase 11's ground-truth labels)
# --------------------------------------------------------------------------------------

#: Two or more real evidence channels -> SUSPICIOUS-ESCALATE-worthy (~10% of accounts).
#: 30 from the base population, plus the 9 round-trip core members (ring + KYC mismatch).
NUM_ESCALATE_WORTHY = 30 + NUM_ROUND_TRIP_RINGS * 3

#: Exactly one real channel -> MONITOR-worthy (~20%). 60 from the base population, plus the
#: 3 round-trip solo members (ring only) and the unexplained-inward cohort (KYC mismatch only).
NUM_MONITOR_WORTHY = 60 + NUM_ROUND_TRIP_RINGS + 8

# --------------------------------------------------------------------------------------
# Deliberate false-positive decoys (all three types, per Phase 11)
# --------------------------------------------------------------------------------------

#: Clean accounts whose name collides with a NAME_SIMILARITY_ONLY corpus entry.
NUM_NAME_SIMILARITY_DECOYS = 15

#: Clean accounts sharing a device or address for a legitimate reason (household, shop).
NUM_GRAPH_COINCIDENCE_PAIRS = 6

#: Clean accounts that transact heavily at round amounts without any sub-threshold pattern.
NUM_ROUND_AMOUNT_DECOYS = 15

#: Legitimate cross-border customers: family_inward + education_outward + lrs_outward.
NUM_CROSS_BORDER_DECOYS = 24

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
