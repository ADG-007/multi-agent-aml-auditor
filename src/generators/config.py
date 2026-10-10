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

#: Every random stream in the generators is derived from this plus a key naming what the
#: stream is for (streams.py). The generated dataset is committed to the repo rather than
#: regenerated per run, so that a library version change in someone else's environment
#: cannot silently shift the ground-truth labels.
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

#: The mule block (Amendment 2), added on top of the cross-border block (all domestic
#: residents). Planted money-mule typologies and their legitimate look-alikes:
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

#: The population is built in blocks, in customer-id order. A block is only ever appended,
#: never inserted, and each draws from its own random streams (streams.py), so adding a block
#: at any phase leaves every earlier block's data unchanged.
POPULATION_BLOCKS: tuple[tuple[str, int], ...] = (
    ("base", NUM_BASE_CUSTOMERS),
    ("cross_border", NUM_CROSS_BORDER_CUSTOMERS),
    ("mule", NUM_MULE_BLOCK_CUSTOMERS),
)

#: The blocks the adverse-media corpus was screened against when it was built. The corpus is
#: fixed against these; every later block keeps its own names clear of the corpus instead
#: (adverse_media.clear_later_block_names), so adding a block never changes the corpus.
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

#: Lower edge of the structuring band, as a fraction of the threshold. Externalised into
#: the STRUCT-001 clause metadata so Stage 0's routing check and Stage 2's formula read the
#: same two numbers from one source.
NEAR_THRESHOLD_PCT = 0.9

#: Deposits deliberately parked just under the reporting ceiling. The band is
#: [LOW, HIGH], both edges inclusive: the law reports cash of more than the threshold, so
#: an amount exactly at it is not reported and sits at the top of the band.
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
#: Mule networks: the mule block's network_mule cohort, four accounts each.
NUM_MULE_RINGS = 2
NUM_RINGS = NUM_LOCAL_RINGS + NUM_INTERNATIONAL_RINGS + NUM_ROUND_TRIP_RINGS + NUM_MULE_RINGS
RING_SIZE = 4

#: Per ring: this many members sit in the escalate-worthy group with a second planted
#: channel; the remaining member carries ring membership alone and is monitor-worthy. That
#: single-channel member is the case Graph corroboration has to catch on its own.
RING_CORE_MEMBERS = 3

# --------------------------------------------------------------------------------------
# Planting density (drives Phase 11's ground-truth labels)
# --------------------------------------------------------------------------------------

#: These count planted channels, not labels. The label rule lives in one place,
#: EvidenceLedgerEntry.expected_disposition: it adds one exception (Amendment 3, A4), a
#: confirmed sanctions hit escalating on its own, so a few single-channel accounts are
#: escalate-worthy.

#: Two or more real evidence channels (~13% of accounts). 30 from the base population, plus
#: the 9 round-trip core members (ring + KYC mismatch), plus the 8 network mules (ring +
#: pass-through).
NUM_MULTI_CHANNEL = 30 + NUM_ROUND_TRIP_RINGS * 3 + NUM_MULE_RINGS * 4

#: Exactly one real channel (~20%). 60 from the base population, plus the 3 round-trip solo
#: members (ring only), the unexplained-inward cohort (KYC mismatch only), and the 2 solo
#: mules and 2 complicit merchants (pass-through only).
NUM_SINGLE_CHANNEL = 60 + NUM_ROUND_TRIP_RINGS + 8 + 2 + 2

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

#: Legitimate pass-through look-alikes from the mule block.
NUM_MERCHANT_LOOKALIKE_DECOYS = 8
NUM_LIFE_EVENT_DECOYS = 6

# --------------------------------------------------------------------------------------
# Money mules (Amendment 2)
# --------------------------------------------------------------------------------------

#: How long a mule holds victims' money before sweeping it on, in minutes. Mule accounts
#: are rented precisely so money does not sit where it can be frozen: published red flags
#: describe funds leaving within hours of arriving.
MULE_SWEEP_DELAY_MINUTES = (5, 180)

#: The share of what arrives that a mule forwards; the rest is the mule's commission.
MULE_FORWARD_SHARE = (0.88, 0.97)

#: What a mule moves over its burst, as a share of the holder's declared annual income.
#: Kept modest on purpose: the giveaway is the shape of the money, not its size, so the
#: account's turnover stays inside what the profile could plausibly explain and the KYC
#: channel is not silently planted as well.
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
