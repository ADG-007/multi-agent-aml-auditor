"""Make a pack of new accounts. Never touches data/.

Two ways in:
- make_pack(kind, count, seed): a ready pattern, every detail drawn from the seed.
- make_custom_pack(specs, title): one AccountSpec per account (or ring), any detail set by
  hand; a detail left empty is drawn from the seed.
Each account's expected verdict is written before any run. The same inputs always make the
same pack.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from faker import Faker

from src.bank import BankData
from src.generators.id_documents import apply_tampering, render_aadhaar, render_pan
from src.generators.identifiers import aadhaar_number, device_id, ip_address, pan_number
from src.generators.names import indian_name
from src.matching import NameIndex
from src.newdata.pack import DATA_DIR, PACKS_DIR, Pack, read, write
from src.schemas import (
    Account,
    AccountType,
    Counterparty,
    Customer,
    Gender,
    IdDocumentType,
    KYCProfile,
    ResidencyStatus,
    SeverityTier,
    Transaction,
)
from src.schemas import TransactionDirection as D
from src.schemas import TransactionType as T
from src.stages.stage0_triage import config as triage_config


LAKH = 100_000
VERDICTS = ("CLEAR", "MONITOR", "SUSPICIOUS-ESCALATE")


@dataclass(frozen=True)
class Knob:
    """One detail a person can set for a pattern: allowed `low`..`high`, drawn from `draw`
    when left empty."""

    key: str
    label: str
    low: float
    high: float
    draw: tuple[float, float]
    step: float = 1
    help: str = ""


@dataclass(frozen=True)
class Kind:
    kind: str  # clean / trap / bad, as in the 50's answer key
    label: str
    expected: str
    what: str
    occupations: tuple[str, ...]
    income_lakh: tuple[int, int]  # random range, ₹ lakh a year
    profile: str  # KYC text; {name} {age} {occupation} {income} {monthly} {sale} are filled in
    monthly: float | None = None  # expected monthly activity; None = income / 12
    knobs: tuple[Knob, ...] = ()


_INTRO = (
    "Profile updated at the customer's most recent KYC refresh. {name}, aged {age}, works as "
    "{occupation} and declares an annual income of {income}. Customer resides at the address "
    "on record in India. "
)
_CASH = (
    Knob("cash_deposits", "Cash deposits", 1, 8, (4, 6), help="inside one 30-day window"),
    Knob(
        "cash_lakh",
        "Each deposit (₹ lakh)",
        5.0,
        9.99,
        (9.05, 9.95),
        0.05,
        "the report line is ₹10 lakh; Stage 2 counts ₹9–10 lakh",
    ),
)
_MULE = (
    Knob("bursts", "Bursts of incoming money", 1, 6, (3, 3)),
    Knob("credits_per_burst", "Credits per burst", 2, 8, (3, 5), help="from strangers"),
)

KINDS: dict[str, Kind] = {
    "clean": Kind(
        "clean",
        "Ordinary salaried customer",
        "CLEAR",
        "salary in, everyday spending out; the profile fits",
        ("Software engineer", "School teacher", "Accountant", "Nurse"),
        (5, 25),
        _INTRO + "The savings account is used for salary credits and household spending. "
        "Expected activity is in the range of {monthly} per month. Source of funds is stated as "
        "salary income. Mostly UPI and card payments. No adverse information was noted at the "
        "time of verification.",
    ),
    "trap": Kind(
        "trap",
        "Large property sale, declared",
        "CLEAR",
        "one large credit from a flat sale; looks odd, but the profile declares it",
        ("Retired bank officer",),
        (6, 12),
        _INTRO + "The savings account is used for pension credits and household spending. "
        "Expected activity is in the range of {monthly} per month. Source of funds is stated as "
        "pension and the sale of property. The customer declared the sale of an inherited flat "
        "for about {sale} during this period; the sale proceeds will be credited to this "
        "account by RTGS.",
        knobs=(Knob("sale_lakh", "Sale price (₹ lakh)", 10, 200, (40, 79), 5),),
    ),
    "structurer": Kind(
        "bad",
        "Cash under ₹10 lakh, declared cash business",
        "MONITOR",
        "repeated cash deposits just under the ₹10 lakh report line; the profile declares a "
        "cash business",
        ("Wholesale grain trader",),
        (30, 50),
        _INTRO + "The savings account is used for the customer's grain trading business. "
        "Expected activity is in the range of {monthly} per month. Source of funds is stated as "
        "business income. The business takes most payments in cash from farmers and deposits "
        "the takings at the branch.",
        monthly=20 * LAKH,
        knobs=_CASH,
    ),
    "structurer_kyc": Kind(
        "bad",
        "Cash under ₹10 lakh, profile says no cash",
        "SUSPICIOUS-ESCALATE",
        "the same cash deposits, but a salaried profile that expects no cash at all",
        ("Office clerk",),
        (4, 9),
        _INTRO + "The savings account is used only for salary credits and household spending. "
        "Expected activity is in the range of {monthly} per month. Source of funds is stated as "
        "salary income. There is no business use and no cash deposits are expected.",
        knobs=_CASH,
    ),
    "kyc_lie": Kind(
        "bad",
        "Business-size money on a salary profile",
        "MONITOR",
        "large transfers in and straight back out, through a salary-only profile",
        ("Junior lab assistant",),
        (3, 6),
        _INTRO + "The savings account is used only for salary credits and household spending. "
        "Expected activity is in the range of {monthly} per month. Source of funds is stated as "
        "salary income. The customer has a single income source and no business activity.",
        knobs=(
            Knob("transfers", "Transfers in (each sent back out)", 2, 20, (8, 13)),
            Knob("transfer_lakh", "Each transfer (₹ lakh)", 1.0, 10.0, (2.5, 7.0), 0.5),
        ),
    ),
    "mule": Kind(
        "bad",
        "Money mule, working alone",
        "MONITOR",
        "bursts of small credits from strangers, swept on to one collector within hours",
        ("College student",),
        (3, 5),
        _INTRO + "The savings account is used for personal expenses and a monthly allowance. "
        "Expected activity is in the range of {monthly} per month. Source of funds is stated as "
        "family support. There is no business activity on the account.",
        monthly=20_000,
        knobs=_MULE,
    ),
    "mule_ring": Kind(
        "bad",
        "Money mule ring, one shared phone",
        "SUSPICIOUS-ESCALATE",
        "several mule accounts that sweep to the same collector from the same phone",
        ("College student",),
        (3, 5),
        _INTRO + "The savings account is used for personal expenses and a monthly allowance. "
        "Expected activity is in the range of {monthly} per month. Source of funds is stated as "
        "family support. There is no business activity on the account.",
        monthly=20_000,
        knobs=_MULE,
    ),
    "sanctions": Kind(
        "bad",
        "Person on the sanctions list",
        "SUSPICIOUS-ESCALATE",
        "the account holder has the name and birth date of a sanctioned person",
        ("Import-export consultant",),
        (15, 40),
        _INTRO + "The savings account is used for consulting fees and household spending. "
        "Expected activity is in the range of {monthly} per month. Source of funds is stated as "
        "consulting income.",
    ),
    "fake_id": Kind(
        "bad",
        "Forged ID card",
        "MONITOR",
        "ordinary money, but the Aadhaar card image is tampered. It pauses for the admin. The "
        "ID check never scores, so it ends CLEAR (a known miss)",
        ("Software engineer", "School teacher", "Accountant", "Nurse"),
        (5, 25),
        _INTRO + "The savings account is used for salary credits and household spending. "
        "Expected activity is in the range of {monthly} per month. Source of funds is stated as "
        "salary income. Mostly UPI and card payments.",
    ),
}

#: One of each kind (two ordinary customers) and a 3-account ring: 12 accounts.
MIX: tuple[str, ...] = (
    "clean",
    "clean",
    "trap",
    "structurer",
    "structurer_kyc",
    "kyc_lie",
    "mule",
    "mule_ring",
    "sanctions",
    "fake_id",
)
MIX_RING_SIZE = 3
MIX_SIZE = len(MIX) - 1 + MIX_RING_SIZE
FIRST_NUMBER = 9001


@dataclass
class AccountSpec:
    """One account, or one ring for mule_ring. Any field left None is drawn from the seed."""

    pattern: str
    count: int = 1  # ring members for mule_ring (at least 2)
    label: str | None = None  # shown in tables; default the pattern's label
    name: str | None = None
    occupation: str | None = None
    income: int | None = None  # ₹ a year
    profile: str | None = None  # KYC text, placeholders allowed
    expected: str | None = None  # the person's own answer; default the pattern's
    settings: dict[str, float] = field(default_factory=dict)  # Knob.key -> value
    listed_name: str | None = None  # sanctions: which list entry

    def __post_init__(self) -> None:
        if self.pattern not in KINDS:
            raise ValueError(f"unknown pattern {self.pattern!r}: pick one of {[*KINDS]}")
        if self.expected is not None and self.expected not in VERDICTS:
            raise ValueError(f"expected must be one of {VERDICTS}, got {self.expected!r}")


def mix_title(seed: int) -> str:
    return f"One of each pattern ({MIX_SIZE} accounts), variation {seed}"


def make_pack(
    kind: str,
    count: int = 1,
    seed: int = 7,
    name: str | None = None,
    root: Path = PACKS_DIR,
    data_dir: Path = DATA_DIR,
    bank: BankData | None = None,
) -> Pack:
    """`kind` is a KINDS key or "mix". `count` is the ring size for mule_ring (at least 2)."""
    if kind == "mix":
        specs = [AccountSpec(k, MIX_RING_SIZE if k == "mule_ring" else 1) for k in MIX]
        title = mix_title(seed)
    elif kind in KINDS:
        if kind == "mule_ring":
            specs = [AccountSpec(kind, max(count, 2))]
        else:
            specs = [AccountSpec(kind) for _ in range(max(count, 1))]
        n = sum(s.count for s in specs)
        title = f"{KINDS[kind].label} ({n} account{'s' * (n > 1)}), variation {seed}"
    else:
        raise ValueError(f"unknown kind {kind!r}: pick one of {[*KINDS, 'mix']}")
    path = root / (name or f"{kind}-seed{seed}")
    return _build(specs, seed, path, title, "sample", data_dir, bank)


def make_custom_pack(
    specs: list[AccountSpec],
    title: str,
    seed: int = 1,
    root: Path = PACKS_DIR,
    data_dir: Path = DATA_DIR,
    bank: BankData | None = None,
) -> Pack:
    """Accounts built to the person's own settings, in a folder named after the title."""
    if not specs:
        raise ValueError("add at least one account first")
    return _build(specs, seed, free_folder(title, root), title, "custom", data_dir, bank)


def free_folder(title: str, root: Path = PACKS_DIR) -> Path:
    """A folder name made from the title, with -2, -3 ... when it is taken."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:48] or "batch"
    path, n = root / slug, 2
    while path.exists():
        path, n = root / f"{slug}-{n}", n + 1
    return path


def listed_people(bank: BankData) -> list[str]:
    """Names on the confirmed-sanctions list, for the sanctions pattern."""
    return sorted(
        {
            e.entity_name
            for e in bank.adverse_media
            if e.severity_tier == SeverityTier.CONFIRMED_SANCTIONS
        }
    )


def _build(specs, seed, path, title, how, data_dir, bank) -> Pack:
    if path.exists():
        raise FileExistsError(f"{path} already exists: pick another name")
    bank = bank or BankData.load(data_dir)
    numbers = iter(_free_numbers(sum(_size(s) for s in specs), bank))
    b = _Builder(seed, bank)
    for spec in specs:
        b.make(spec, [next(numbers) for _ in range(_size(spec))])
    b.save(path, data_dir, title, how)
    return Pack(path)


def _size(spec: AccountSpec) -> int:
    return max(spec.count, 2) if spec.pattern == "mule_ring" else max(spec.count, 1)


def _free_numbers(count: int, bank: BankData) -> list[int]:
    """Account numbers the bank does not use. Packs are never merged with each other, so
    other packs do not count: the same inputs always give the same numbers."""
    taken = {int(a.rsplit("-", 1)[1]) for a in bank.account_ids()}
    out, n = [], FIRST_NUMBER
    while len(out) < count:
        if n not in taken:
            out.append(n)
        n += 1
    return out


def _lakh(x: float) -> str:
    return f"Rs {x / LAKH:.2f} lakh"


class _Keep(dict):
    """format_map helper: an unknown {placeholder} stays as written."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class _Builder:
    """Collects one pack's rows."""

    def __init__(self, seed: int, bank: BankData) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.faker = Faker("en_IN")
        self.faker.seed_instance(seed)
        self.names = NameIndex(bank.adverse_media)
        self.listed = [
            e for e in bank.adverse_media if e.severity_tier == SeverityTier.CONFIRMED_SANCTIONS
        ]
        # Inside the bank's 90-day frame, and never later than its review date.
        self.end = bank.review_date() - timedelta(hours=2)
        self.accounts: list[Account] = []
        self.customers: list[Customer] = []
        self.transactions: list[Transaction] = []
        self.counterparties: dict[str, Counterparty] = {}
        self.profiles: list[KYCProfile] = []
        self.answers: list[dict] = []
        self.forged: set[str] = set()

    # -- settings ------------------------------------------------------------------------

    def setting(self, spec: AccountSpec, key: str) -> float:
        """The person's value (kept inside the allowed range), else a draw from `draw`."""
        knob = next(k for k in KINDS[spec.pattern].knobs if k.key == key)
        if spec.settings.get(key) is not None:
            return min(max(float(spec.settings[key]), knob.low), knob.high)
        low, high = knob.draw
        if knob.step >= 1:
            return float(self.rng.randint(int(low), int(high)))
        return self.rng.uniform(low, high)

    # -- people and parties --------------------------------------------------------------

    def clear_name(self, gender: str | None = None) -> str:
        """A new name that Stage 0 would not match to the list by chance."""
        while True:
            name = indian_name(self.faker, gender)
            if self.names.best_match(name, cutoff=triage_config.NAME_MATCH_CUTOFF) is None:
                return name

    def own_name(self, spec: AccountSpec) -> str | None:
        """The person's typed name, for the first account only: ring members get their own."""
        if spec.name and spec.name not in {c.name for c in self.customers}:
            return spec.name
        return None

    def party(self, name: str | None = None, prefix: str = "CPX") -> Counterparty:
        cp = Counterparty(
            counterparty_id=f"{prefix}-{self.seed}-{len(self.counterparties) + 1:04d}",
            name=name or self.clear_name(),
            country="India",
            is_foreign=False,
            registered_address=None,
        )
        self.counterparties[cp.counterparty_id] = cp
        return cp

    def open(self, n: int, spec: AccountSpec, name=None, dob=None, **fill) -> tuple:
        kind = KINDS[spec.pattern]
        gender = self.rng.choice(["MALE", "FEMALE"])
        low, high = kind.income_lakh
        income = spec.income or self.rng.randrange(low, high + 1) * LAKH
        c = Customer(
            customer_id=f"CUST-{n}",
            name=name or self.own_name(spec) or self.clear_name(gender),
            dob=dob or self.faker.date_of_birth(minimum_age=22, maximum_age=60),
            gender=Gender(gender),
            occupation=spec.occupation or self.rng.choice(kind.occupations),
            stated_income=income,
            address=self.faker.address().replace("\n", ", "),
            residency_status=ResidencyStatus.DOMESTIC,
            country="India",
            id_document_type=IdDocumentType.AADHAAR,
            id_number=aadhaar_number(self.rng),
            id_expiration_date=None,
        )
        self.customers.append(c)
        year, month = 2019 + self.rng.randrange(6), self.rng.randrange(1, 13)
        self.accounts.append(
            Account(
                account_id=f"ACC-{n}",
                customer_id=c.customer_id,
                opened_date=date(year, month, self.rng.randrange(1, 28)),
                account_type=AccountType.SAVINGS,
            )
        )
        text = (spec.profile or kind.profile).format_map(
            _Keep(
                name=c.name,
                age=(self.end.date() - c.dob).days // 365,
                occupation=c.occupation.lower(),
                income=_lakh(c.stated_income),
                monthly=_lakh(kind.monthly or c.stated_income / 12),
                **fill,
            )
        )
        self.profiles.append(KYCProfile(customer_id=c.customer_id, narrative=text.strip()))
        self.answers.append(
            {
                "account_id": f"ACC-{n}",
                "kind": kind.kind,
                "group": spec.label or kind.label,
                "expected": spec.expected or kind.expected,
                "pattern": spec.pattern,
                "what": kind.what,
            }
        )
        return f"ACC-{n}", c, device_id(self.rng), ip_address(self.rng)

    # -- money ---------------------------------------------------------------------------

    def txn(self, a: str, amount, direction, kind, when, cp, device=None, ip=None) -> None:
        self.transactions.append(
            Transaction(
                transaction_id=f"TXN-{when:%Y%m%d-%H%M%S}-{a[4:]}-{len(self.transactions):04d}",
                account_id=a,
                amount=round(amount, 2),
                direction=direction,
                type=kind,
                timestamp=when,
                device_id=device,
                ip_address=ip,
                counterparty_id=cp.counterparty_id,
                counterparty_name=cp.name,
                purpose_code=None,
            )
        )

    def day(self, days_back: int, hour: int | None = None) -> datetime:
        h = hour if hour is not None else self.rng.randrange(8, 22)
        when = self.end - timedelta(days=days_back)
        return when.replace(hour=h, minute=self.rng.randrange(60), second=self.rng.randrange(60))

    def everyday(self, a: str, c: Customer, dev: str, ip: str, spend_share: float = 0.6) -> None:
        """Salary in each month; everyday UPI and card spending and one ATM visit out."""
        employer = self.party(self.faker.company(), "CPE")
        for back in (85, 55, 25):
            self.txn(a, c.stated_income / 12, D.CREDIT, T.NEFT, self.day(back, 10), employer)
        shops = [self.party(self.faker.company(), "CPM") for _ in range(6)]
        budget = c.stated_income / 4 * spend_share
        n = self.rng.randrange(18, 32)
        for _ in range(n):
            amount = min(budget / n * self.rng.uniform(0.5, 1.5), 40_000)
            kind = self.rng.choice([T.UPI, T.UPI, T.POS])
            when = self.day(self.rng.randrange(1, 88))
            self.txn(a, amount, D.DEBIT, kind, when, self.rng.choice(shops), dev, ip)
        atm = self.party("ATM Cash Withdrawal", "CPA")
        amount = self.rng.randrange(2, 10) * 1000
        self.txn(a, amount, D.DEBIT, T.ATM, self.day(self.rng.randrange(1, 88)), atm, dev, ip)

    def cash(self, a: str, spec: AccountSpec) -> None:
        """Cash deposits just under ₹10 lakh inside one 30-day window."""
        count = int(self.setting(spec, "cash_deposits"))
        branch = self.party("Cash deposit at branch", "CPC")
        first = self.rng.randrange(30, 60)
        for back in sorted(self.rng.sample(range(first - 27, first), count)):
            size = self.setting(spec, "cash_lakh")  # a new draw per deposit unless set
            self.txn(a, size * LAKH, D.CREDIT, T.CASH_DEPOSIT, self.day(back, 11), branch)

    # -- patterns ------------------------------------------------------------------------

    def make(self, spec: AccountSpec, numbers: list[int]) -> None:
        getattr(self, f"_{spec.pattern}")(spec, numbers)

    def _clean(self, spec, numbers) -> None:
        for n in numbers:
            a, c, dev, ip = self.open(n, spec)
            self.everyday(a, c, dev, ip)

    def _fake_id(self, spec, numbers) -> None:
        self._clean(spec, numbers)
        self.forged |= {f"ACC-{n}" for n in numbers}

    def _trap(self, spec, numbers) -> None:
        for n in numbers:
            price = round(self.setting(spec, "sale_lakh")) * LAKH
            a, c, dev, ip = self.open(n, spec, sale=_lakh(price))
            self.everyday(a, c, dev, ip)
            self.txn(a, price, D.CREDIT, T.RTGS, self.day(40, 12), self.party(None, "CPB"))

    def _structurer(self, spec, numbers) -> None:
        for n in numbers:
            a, c, dev, ip = self.open(n, spec)
            self.everyday(a, c, dev, ip)
            self.cash(a, spec)

    _structurer_kyc = _structurer

    def _kyc_lie(self, spec, numbers) -> None:
        for n in numbers:
            a, c, dev, ip = self.open(n, spec)
            self.everyday(a, c, dev, ip)
            firms = [self.party(self.faker.company(), "CPF") for _ in range(4)]
            for _ in range(int(self.setting(spec, "transfers"))):
                back = self.rng.randrange(3, 85)
                amount = self.setting(spec, "transfer_lakh") * LAKH
                self.txn(a, amount, D.CREDIT, T.RTGS, self.day(back, 11), self.rng.choice(firms))
                out = amount * self.rng.uniform(0.9, 0.98)
                when = self.day(max(back - 1, 1), 15)
                self.txn(a, out, D.DEBIT, T.RTGS, when, self.rng.choice(firms), dev, ip)

    def _mules(self, spec, numbers, shared: bool) -> None:
        collector = self.party(None, "CPK")
        ring_dev, ring_ip = device_id(self.rng), ip_address(self.rng)
        for n in numbers:
            a, c, dev, ip = self.open(n, spec)
            self.everyday(a, c, dev, ip, spend_share=0.3)
            mine = collector if shared else self.party(None, "CPK")
            sweep_dev, sweep_ip = (ring_dev, ring_ip) if shared else (dev, ip)
            bursts = int(self.setting(spec, "bursts"))
            for back in sorted(self.rng.sample(range(10, 80), bursts), reverse=True):
                start, total = self.day(back, 9), 0.0
                for k in range(int(self.setting(spec, "credits_per_burst"))):
                    amount = self.rng.uniform(8_000, 45_000)
                    total += amount
                    when = start + timedelta(minutes=40 * k + self.rng.randrange(30))
                    kind = self.rng.choice([T.UPI, T.IMPS])
                    self.txn(a, amount, D.CREDIT, kind, when, self.party(None, "CPV"))
                sweep = total * self.rng.uniform(0.85, 0.95)
                when = start + timedelta(hours=self.rng.randrange(5, 12))
                self.txn(a, sweep, D.DEBIT, T.IMPS, when, mine, sweep_dev, sweep_ip)

    def _mule(self, spec, numbers) -> None:
        for n in numbers:
            self._mules(spec, [n], shared=False)

    def _mule_ring(self, spec, numbers) -> None:
        self._mules(spec, numbers, shared=True)

    def _sanctions(self, spec, numbers) -> None:
        for n in numbers:
            chosen = [e for e in self.listed if e.entity_name == spec.listed_name]
            e = chosen[0] if chosen else self.rng.choice(self.listed)
            a, c, dev, ip = self.open(n, spec, name=e.entity_name, dob=self.dob_for(e))
            self.everyday(a, c, dev, ip)

    def dob_for(self, entry) -> date | None:
        """The listed date of birth; a year-only entry gets a day in that year."""
        listed = entry.date_of_birth
        if not listed:
            return None
        if len(listed) == 4:
            return date(int(listed), self.rng.randrange(1, 13), self.rng.randrange(1, 28))
        return date.fromisoformat(listed)

    # -- save ----------------------------------------------------------------------------

    def save(self, path: Path, data_dir: Path, title: str, how: str) -> None:
        for name, rows in (
            ("accounts", self.accounts),
            ("customers", self.customers),
            ("transactions", self.transactions),
            ("counterparties", list(self.counterparties.values())),
            ("kyc_profiles", self.profiles),
        ):
            dumped = [r.model_dump(mode="json", exclude={"is_cross_border"}) for r in rows]
            write(path / f"{name}.json", dumped)
        write(
            path / "_ground_truth" / "answers.json",
            {
                "title": title,
                "how": how,
                "seed": self.seed,
                "made_at": datetime.now().isoformat(timespec="seconds"),
                "show_cases": {},
                "accounts": self.answers,
            },
        )
        self.documents(path / "documents", data_dir)

    def documents(self, folder: Path, data_dir: Path) -> None:
        """Aadhaar and PAN images for the new customers. The manifest also lists the bank's
        own images by absolute path, so Stage 1 finds both in this one folder."""
        folder.mkdir(parents=True, exist_ok=True)
        committed = data_dir / "documents"
        manifest = [
            {**m, "file": str((committed / m["file"]).resolve())}
            for m in read(committed / "manifest.json")
        ]
        account_of = {a.customer_id: a.account_id for a in self.accounts}
        for c in self.customers:
            rng = random.Random(f"{self.seed}-{c.customer_id}")
            aadhaar = render_aadhaar(c, rng)
            pan = render_pan(c, pan_number(rng, c.name.split()[-1]), rng)
            if account_of[c.customer_id] in self.forged:
                aadhaar, _ = apply_tampering(aadhaar, c, "AADHAAR", rng)
            for doc_type, image in (("AADHAAR", aadhaar), ("PAN", pan)):
                file = f"{c.customer_id}_{doc_type}.png"
                image.save(folder / file, optimize=True)
                manifest.append(
                    {
                        "customer_id": c.customer_id,
                        "account_id": account_of[c.customer_id],
                        "document_type": doc_type,
                        "file": file,
                    }
                )
        write(folder / "manifest.json", manifest)
