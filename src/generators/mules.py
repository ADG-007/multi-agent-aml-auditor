"""The mule block's transaction flows (Amendment 2).

Called from transactions.py for every account in the mule block. Five cohorts, two of them
planted typologies, one a planted typology hidden inside a legitimate one, and two legitimate
look-alikes. They are built to be hard to tell apart on volume alone - what separates them
is the shape of the money:

  network_mule / solo_mule
      A rented account: quiet for weeks, then a burst of payments from strangers - scam
      victims, each paying once - swept on to one collector account within minutes to hours,
      minus the mule's commission. Published red flags describe exactly this: accounts opened
      in the last year or long dormant, inward bursts from unrelated UPI IDs, then an outward
      sweep. A network's four accounts are run from the handler's phone (rings.py).

  merchant_lookalike
      A fuel outlet or a roadside mobile/electronics shop taking customer payments on the
      owner's personal UPI ID. Just as many strangers pay once, and amounts range from a
      hundred rupees to tens of thousands - but the money builds up for days and then goes, in
      a lump, to the same one or two suppliers every time.

  complicit_merchant
      A genuine merchant, generated exactly like a look-alike, with a thin layer of victims'
      money in its last 30 days only, each payment forwarded within hours to a mule network's
      collector. Small against the real trade, so whole-account averages miss it; a
      recent-versus-own-baseline comparison and the shared collector are what catch it.

  life_event
      An ordinary salaried customer with one sudden, legitimate spike: relatives chipping in
      to pay off a moneylender, relatives covering a hospital bill, or a share of a land sale
      spread to new payees. For a few days it looks like pass-through; it happens once, the
      senders share the customer's surname, and the recipients are a lender, a hospital, a
      car dealer - not a collector.

Counterparties created here are the account's own, drawn from the account's own stream, so
the block adds counterparties without touching the base list. Their ids are opaque
(CPM-3F9A1C22): numbering them after the account that introduced them, or calling a collector
"COL-01", would tell a pipeline stage what it is looking at.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import datetime, time, timedelta

from src.generators import config as cfg
from src.generators.names import indian_name
from src.generators.rings import MULE_NETWORK_COLLECTOR_IDS, SOLO_MULE_COLLECTOR_IDS
from src.generators.streams import Streams, opaque_id
from src.generators.transactions import (
    CREDIT,
    DEBIT,
    ordinary_rows,
)
from src.schemas import Counterparty, Customer, TransactionType

UPI = TransactionType.UPI
IMPS = TransactionType.IMPS
NEFT = TransactionType.NEFT
RTGS = TransactionType.RTGS
ATM = TransactionType.ATM

#: The first moment of the 90-day window, and the last day a planted flow may start on - the
#: sweep that follows has to land inside the window too.
WINDOW_START = datetime.combine(cfg.REFERENCE_DATE - timedelta(days=cfg.HISTORY_DAYS), time())
LAST_FLOW_DAY = cfg.HISTORY_DAYS - 2

#: Supplier names by trade. Invented, in the style of real Indian distributors.
SUPPLIER_SUFFIXES = {
    "fuel": ("Petroleum Distributors", "Fuel Depot", "Lubricants & Fuels"),
    "electronics": ("Mobile Distributors", "Electronics Wholesale", "Telecom Accessories"),
}

HOSPITAL_NAMES = (
    "Sanjeevani Multispeciality Hospital",
    "Lifeline Hospital & Research Centre",
    "City Care Multispeciality Hospital",
    "Sri Venkateswara Hospital",
    "Apex Heart & Multispeciality Hospital",
    "Shree Krishna Hospital & Research Centre",
)

#: The average counter payment each trade's ticket mix works out to, used to size how many
#: customers a day it takes to reach the account's turnover.
MEAN_TICKET = {"fuel": 1_850.0, "electronics": 1_950.0}

#: A merchant's annual turnover through this account (money in plus money out), as a multiple
#: of declared income. The personal UPI ID carries only part of the takings - cards, cash and
#: the business's own merchant QR carry the rest - so the account moves a few times what the
#: owner earns rather than the full turnover of the business.
MERCHANT_TURNOVER_MULTIPLE = (2.0, 4.0)

#: Price points a mobile-accessories counter actually charges.
ACCESSORY_PRICES = (99, 149, 199, 249, 299, 349, 399, 499, 599, 699, 799, 999, 1199, 1499)
TWO_WHEELER_FILLS = (100, 150, 200, 250, 300, 400, 500)


RowFn = Callable[..., dict]


def _at(day: int, hour: int, minute: int) -> datetime:
    return WINDOW_START + timedelta(days=day, hours=hour, minutes=minute)


def _rupees(amount: float) -> float:
    return float(round(amount, 2))


def collector_counterparties(streams: Streams) -> list[Counterparty]:
    """The collectors every mule sweeps into.

    Individuals at other banks - usually themselves the next layer of mules - so from this
    bank they are ordinary-looking domestic payees. Nothing about the counterparty record
    marks them out; only the money flowing to them does.
    """
    parties = []
    for counterparty_id in MULE_NETWORK_COLLECTOR_IDS + SOLO_MULE_COLLECTOR_IDS:
        faker = streams.faker("en_IN", "counterparty", counterparty_id)
        parties.append(
            Counterparty(
                counterparty_id=counterparty_id,
                name=indian_name(faker),
                country="India",
                is_foreign=False,
            )
        )
    return parties


class _Book:
    """One mule-block account's own counterparties and the rows built so far."""

    def __init__(self, streams: Streams, rng: random.Random, account_id: str, row: RowFn) -> None:
        self.streams = streams
        self.rng = rng
        self.account_id = account_id
        self.row = row
        self.counterparties: list[Counterparty] = []
        self.planted: list[dict] = []

    def party(self, name: str | None = None, surname: str | None = None, business: str | None = None) -> Counterparty:
        """A new counterparty of this account's own, named from its own stream."""
        counterparty_id = opaque_id("CPM", "counterparty", self.account_id, len(self.counterparties) + 1)
        faker = self.streams.faker("en_IN", "counterparty", counterparty_id)
        if name is None:
            if business:
                name = f"{faker.last_name()} {business}"
            else:
                name = indian_name(faker)
                if surname:
                    name = f"{name.split()[0]} {surname}"
        party = Counterparty(counterparty_id=counterparty_id, name=name, country="India", is_foreign=False)
        self.counterparties.append(party)
        return party

    def add(self, amount: float, rail: TransactionType, when: datetime, party: Counterparty, direction) -> None:
        # Flows are timed to the minute; the seconds are drawn so a planted posting does not
        # stand out from ordinary ones by always landing on :00.
        when = when.replace(second=self.rng.randint(0, 59))
        self.planted.append(self.row(_rupees(amount), rail, when, party, direction))


# --------------------------------------------------------------------------------------
# Mules
# --------------------------------------------------------------------------------------


def _victim_amount(rng: random.Random) -> float:
    """What one scam victim pays: typically a few thousand to a lakh, often a round figure
    the fraudster named."""
    amount = min(max(rng.lognormvariate(10.0, 0.7), 2_000.0), 99_000.0)
    if rng.random() < 0.5:
        amount = round(amount / 500) * 500
    return float(amount)


def _victim_credit(book: _Book, amount: float, when: datetime) -> None:
    rail = UPI if amount <= 100_000 and book.rng.random() < 0.7 else IMPS
    book.add(amount, rail, when, book.party(), CREDIT)


def _sweep(book: _Book, total: float, when: datetime, collector: Counterparty, cash_out_cp) -> None:
    """Send what arrived on to the collector, or occasionally draw it out as cash."""
    rng = book.rng
    if rng.random() < 0.15:
        # The cash-out leg: withdrawn at ATMs in chunks under the per-withdrawal limit.
        remaining = total
        while remaining > 500:
            chunk = min(remaining, float(rng.choice((10_000, 15_000, 20_000))))
            book.add(chunk, ATM, when, cash_out_cp, DEBIT)
            remaining -= chunk
            when += timedelta(minutes=rng.randint(2, 6))
        return
    rail = UPI if total <= 100_000 and rng.random() < 0.4 else IMPS
    book.add(total, rail, when, collector, DEBIT)


def mule_flow(book: _Book, customer: Customer, collector: Counterparty, burst_start: int, cash_out_cp) -> None:
    """Victims' payments arriving in small batches, each batch swept on within hours."""
    rng = book.rng
    target = customer.stated_income * rng.uniform(*cfg.MULE_INFLOW_SHARE_OF_INCOME)
    received = 0.0
    day = burst_start
    while received < target and day <= LAST_FLOW_DAY:
        when = _at(day, rng.randint(8, 21), rng.randint(0, 59))
        batch = 0.0
        for _ in range(rng.choice((1, 1, 2, 2, 3))):
            amount = _victim_amount(rng)
            _victim_credit(book, amount, when)
            batch += amount
            when += timedelta(minutes=rng.randint(3, 90))
            if received + batch >= target:
                break
        received += batch
        sweep_at = when + timedelta(minutes=rng.randint(*cfg.MULE_SWEEP_DELAY_MINUTES))
        _sweep(book, batch * rng.uniform(*cfg.MULE_FORWARD_SHARE), sweep_at, collector, cash_out_cp)
        day += rng.choice((0, 1, 1, 2))


# --------------------------------------------------------------------------------------
# Merchants
# --------------------------------------------------------------------------------------


def _trade(customer: Customer) -> str:
    return "fuel" if "fuel" in customer.occupation.lower() else "electronics"


def _ticket(rng: random.Random, trade: str) -> float:
    """One customer's payment at the counter."""
    roll = rng.random()
    if trade == "fuel":
        if roll < 0.55:
            return float(rng.choice(TWO_WHEELER_FILLS))
        if roll < 0.90:
            amount = rng.uniform(800, 3_500)
            # A "fill it up" is paid to the paisa off the meter; most people ask for a round sum.
            return float(round(amount, -2)) if rng.random() < 0.7 else _rupees(amount)
        return float(round(rng.uniform(4_000, 15_000), -2))  # a diesel fill for a goods vehicle
    if roll < 0.75:
        return float(rng.choice(ACCESSORY_PRICES))
    if roll < 0.95:
        return float(round(rng.uniform(1_500, 6_000), -1))  # a repair, a speaker, a power bank
    return float(rng.choice((7_999, 9_499, 11_999, 13_999, 16_499, 19_999, 24_999, 34_999)))


def merchant_flow(book: _Book, customer: Customer, cash_cps: list[Counterparty]) -> float:
    """Counter takings on a personal UPI ID, paid on to regular suppliers every few days.

    Returns the total taken at the counter, which sizes a complicit merchant's layer.
    """
    rng = book.rng
    trade = _trade(customer)
    suppliers = [
        book.party(business=rng.choice(SUPPLIER_SUFFIXES[trade]))
        for _ in range(rng.randint(1, 2) if trade == "fuel" else rng.randint(2, 3))
    ]
    regulars = [book.party() for _ in range(rng.randint(15, 30))]
    closed_weekday = None if trade == "fuel" else rng.randint(0, 6)
    open_hours = (6, 22) if trade == "fuel" else (10, 21)
    # Customers a day, sized so the takings match the turnover the owner's income supports.
    takings_target = (
        customer.stated_income * rng.uniform(*MERCHANT_TURNOVER_MULTIPLE) / 2 * (cfg.HISTORY_DAYS / 365)
    )
    open_days = cfg.HISTORY_DAYS - 1 if closed_weekday is None else (cfg.HISTORY_DAYS - 1) * 6 / 7
    per_day = takings_target / MEAN_TICKET[trade] / open_days

    takings = 0.0
    unpaid = 0.0
    next_settlement = rng.randint(3, 7)
    for day in range(cfg.HISTORY_DAYS - 1):
        on = (WINDOW_START + timedelta(days=day)).weekday() != closed_weekday
        if on:
            customers_today = int(per_day) + (1 if rng.random() < per_day - int(per_day) else 0)
            customers_today = max(0, customers_today + rng.choice((-1, 0, 0, 1)))
            for _ in range(customers_today):
                amount = _ticket(rng, trade)
                # Most payers are passing trade seen once; a few are regulars.
                payer = rng.choice(regulars) if rng.random() < 0.2 else book.party()
                book.add(amount, UPI, _at(day, rng.randint(*open_hours), rng.randint(0, 59)), payer, CREDIT)
                takings += amount
                unpaid += amount
        if day == next_settlement and unpaid > 0:
            # Stock is paid for in a lump - the money sits for days first, unlike a mule's.
            payment = unpaid * rng.uniform(0.80, 0.95)
            rail = RTGS if payment >= 200_000 else NEFT
            book.add(payment, rail, _at(day, rng.randint(10, 17), rng.randint(0, 59)), rng.choice(suppliers), DEBIT)
            unpaid -= payment
            next_settlement = day + rng.randint(3, 7)
        if on and rng.random() < 0.2:
            # Cash drawn for wages and the day's small expenses.
            book.add(float(rng.choice((2_000, 3_000, 5_000, 10_000))), ATM,
                     _at(day, rng.randint(8, 20), rng.randint(0, 59)), rng.choice(cash_cps), DEBIT)
    return takings


def complicit_layer(book: _Book, takings: float, collector: Counterparty) -> None:
    """Victims' money slipped into the merchant's last 30 days, each payment forwarded alone.

    Timed inside opening hours so it does not stand out by hour of day; what gives it away is
    that these payers are strangers paying scam-sized sums, and that each one's money leaves
    within hours for an account the merchant has never paid before - one a mule network uses.
    """
    rng = book.rng
    target = takings * rng.uniform(*cfg.COMPLICIT_LAYER_SHARE)
    first_day = cfg.HISTORY_DAYS - cfg.COMPLICIT_LAYER_DAYS
    layered = 0.0
    while layered < target:
        amount = min(max(rng.lognormvariate(10.1, 0.6), 5_000.0), 95_000.0)
        amount = float(round(amount / 500) * 500)
        when = _at(rng.randint(first_day, LAST_FLOW_DAY), rng.randint(10, 20), rng.randint(0, 59))
        _victim_credit(book, amount, when)
        forward_at = when + timedelta(minutes=rng.randint(10, cfg.MULE_SWEEP_DELAY_MINUTES[1]))
        book.add(amount * rng.uniform(*cfg.MULE_FORWARD_SHARE), IMPS, forward_at, collector, DEBIT)
        layered += amount


# --------------------------------------------------------------------------------------
# Life events
# --------------------------------------------------------------------------------------


def _split(rng: random.Random, total: float, parts: int) -> list[float]:
    weights = [rng.uniform(0.6, 1.4) for _ in range(parts)]
    scale = total / sum(weights)
    return [float(round(w * scale, -2)) for w in weights]


def _transfer_rail(amount: float, rng: random.Random) -> TransactionType:
    if amount >= 200_000:
        return RTGS if rng.random() < 0.5 else NEFT
    if amount <= 100_000 and rng.random() < 0.4:
        return UPI
    return IMPS if rng.random() < 0.6 else NEFT


def life_event_flow(book: _Book, customer: Customer, kind: str) -> None:
    """One legitimate spike, sized modestly against the customer's income."""
    rng = book.rng
    surname = customer.name.split()[-1]
    total = customer.stated_income * rng.uniform(0.15, 0.35)
    day = rng.randint(20, cfg.HISTORY_DAYS - 12)

    if kind in ("moneylender_payoff", "medical_emergency"):
        # Family chips in over a day or two.
        when = _at(day, rng.randint(8, 12), rng.randint(0, 59))
        for amount in _split(rng, total, rng.randint(3, 5) if kind == "moneylender_payoff" else rng.randint(2, 5)):
            book.add(amount, _transfer_rail(amount, rng), when, book.party(surname=surname), CREDIT)
            when += timedelta(hours=rng.randint(1, 20))

        if kind == "moneylender_payoff":
            lender = book.party(business="Finance") if rng.random() < 0.5 else book.party()
            owed = total * rng.uniform(0.97, 1.0)
            pay_at = when + timedelta(days=rng.randint(0, 2), hours=rng.randint(1, 5))
            for amount in _split(rng, owed, rng.randint(1, 2)):
                book.add(amount, _transfer_rail(amount, rng), pay_at, lender, DEBIT)
                pay_at += timedelta(minutes=rng.randint(5, 60))
        else:
            hospital = book.party(name=rng.choice(HOSPITAL_NAMES))
            bill = total * rng.uniform(0.95, 1.10)  # the family's own savings top it up
            deposit = float(round(bill * rng.uniform(0.3, 0.5), -2))
            first = _at(day, rng.randint(13, 20), rng.randint(0, 59))
            book.add(deposit, _transfer_rail(deposit, rng), first, hospital, DEBIT)
            settle = bill - deposit
            book.add(settle, _transfer_rail(settle, rng),
                     first + timedelta(days=rng.randint(2, 5), hours=rng.randint(0, 3)), hospital, DEBIT)
        return

    # windfall: a share of a family land sale, then spread to people and places never paid before.
    windfall = total
    book.add(windfall, RTGS if windfall >= 200_000 else NEFT, _at(day, rng.randint(11, 15), rng.randint(0, 59)),
             book.party(), CREDIT)
    payees = [
        book.party(surname=surname),
        book.party(surname=surname),
        book.party(business="Motors"),
        book.party(business="Home Finance Ltd"),
    ]
    rng.shuffle(payees)
    spent = windfall * rng.uniform(0.70, 0.95)
    for payee, amount in zip(payees[: rng.randint(3, 4)], _split(rng, spent, 4)):
        book.add(amount, _transfer_rail(amount, rng),
                 _at(day + rng.randint(1, 7), rng.randint(9, 20), rng.randint(0, 59)), payee, DEBIT)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def account_activity(
    streams: Streams,
    rng: random.Random,
    account_id: str,
    customer: Customer,
    cohort: str,
    plan,
    collectors: dict[str, Counterparty],
    domestic_cps: list[Counterparty],
    row: RowFn,
    burst_start: int,
) -> tuple[list[dict], list[dict], list[Counterparty]]:
    """Build one mule-block account's history.

    Returns ordinary rows (directions still to be assigned by the caller, as for every other
    account), planted rows, and the counterparties this account introduced.
    """
    book = _Book(streams, rng, account_id, row)
    income = customer.stated_income

    # Everyone also has a little ordinary life: groceries, bills, a transfer to family.
    # Mules are quiet (a rented account is not anyone's main account); merchants' personal
    # spending is small beside their takings; salaried customers look like anyone else.
    if cohort in ("network_mule", "solo_mule"):
        count, multiple = rng.randint(12, 24), rng.uniform(0.3, 0.8)
    elif cohort in ("merchant_lookalike", "complicit_merchant"):
        count, multiple = rng.randint(8, 15), rng.uniform(0.2, 0.5)
    else:
        count, multiple = rng.randint(cfg.TXNS_PER_ACCOUNT_MIN, cfg.TXNS_PER_ACCOUNT_MAX), rng.uniform(0.6, 1.5)
    mean_amount = max(600.0, income * multiple * (cfg.HISTORY_DAYS / 365) / count)
    ordinary = ordinary_rows(rng, row, count, mean_amount, domestic_cps)

    if cohort in ("network_mule", "solo_mule"):
        mule_flow(book, customer, collectors[plan.mule_collector[account_id]], burst_start, rng.choice(domestic_cps))
    elif cohort in ("merchant_lookalike", "complicit_merchant"):
        takings = merchant_flow(book, customer, domestic_cps)
        if cohort == "complicit_merchant":
            complicit_layer(book, takings, collectors[plan.mule_collector[account_id]])
    elif cohort == "life_event":
        life_event_flow(book, customer, plan.life_events[account_id])

    return ordinary, book.planted, book.counterparties


def burst_start_day(streams: Streams, account_id: str, ring_id: str | None) -> int:
    """When a mule's burst begins.

    A handler activates a network's accounts together, so members start within days of one
    another; a solo mule starts on its own schedule. Either way the burst sits in the back
    half of the window, leaving weeks of quiet baseline before it.
    """
    if ring_id:
        start = streams.rng("mule_burst", ring_id).randint(45, 60)
        return start + streams.rng("mule_burst", account_id).randint(0, 4)
    return streams.rng("mule_burst", account_id).randint(40, 62)

