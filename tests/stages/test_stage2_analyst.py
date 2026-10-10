"""Stage 2's entry point: schema-valid output, max of the sub-scores, short trail, determinism."""

from datetime import datetime, timedelta

import pytest

from src.bank import BankData
from src.rulebook import RulebookLookup
from src.schemas import Stage2TransactionPatternOutput, TransactionDirection, TransactionType
from src.stages.stage0_triage import TriageContext
from src.stages.stage2_transaction_pattern import config, pass_through, run, structuring
from tests.factories import bank, txn


AS_OF = datetime(2026, 9, 21, 23, 0)
DEBIT = TransactionDirection.DEBIT


@pytest.fixture(scope="module")
def rulebook() -> RulebookLookup:
    return RulebookLookup.load()


@pytest.fixture(scope="module")
def real_ctx(rulebook) -> TriageContext:
    return TriageContext.build(BankData.load(), rulebook)


def _ctx(rows, rulebook):
    return TriageContext.build(bank(transactions=rows), rulebook)


def _structurer_and_mule():
    """Five in-band cash deposits (k = 5) and four fast-swept strangers (n = 4)."""
    rows = [
        txn(amount=950_000, type=TransactionType.CASH_DEPOSIT, timestamp=f"2026-08-0{d}T10:00:00")
        for d in range(1, 6)
    ]
    start = datetime(2026, 9, 1, 10)
    for i in range(4):
        when = start + timedelta(days=i)
        rows += [
            txn(amount=20_000, timestamp=when.isoformat(), counterparty_id=f"CP-V{i}"),
            txn(
                amount=19_000,
                direction=DEBIT,
                timestamp=(when + timedelta(hours=1)).isoformat(),
                counterparty_id="CP-COLLECTOR",
            ),
        ]
    return rows


def test_output_validates_and_confidence_is_the_max(rulebook):
    rows = _structurer_and_mule()
    out = run("ACC-T001", _ctx(rows, rulebook), AS_OF)
    Stage2TransactionPatternOutput.model_validate(out.model_dump())

    s = structuring.score(rows, AS_OF, rulebook).score
    p = pass_through.score(rows, AS_OF, rulebook).score
    assert s == pytest.approx(0.65) and p == pytest.approx(0.8)
    assert out.confidence == round(max(s, p), 4)
    assert len(out.structuring_indicators) == 1
    assert out.structuring_indicators[0] == (
        "5 cash deposits of ₹9,00,000–₹10,00,000 between 1 Aug and 5 Aug 2026, "
        "each under the CTR (STRUCT-001)"
    )
    assert out.pass_through_indicators and all(
        "MULE-002" in line for line in out.pass_through_indicators
    )


def test_trail_is_short_and_fixed(rulebook):
    out = run("ACC-T001", _ctx(_structurer_and_mule(), rulebook), AS_OF)
    assert len(out.reasoning_trail) <= config.MAX_TRAIL_STEPS
    assert [s.step for s in out.reasoning_trail] == [
        "Structuring",
        "Pass-through",
        "Combined confidence",
        "Velocity",
        "Round-amount share",
        "Calendar-month cash totals",
        "In-band non-cash payments",
        "Stranger share",
    ]
    text = " ".join(f"{s.finding} {s.impact}" for s in out.reasoning_trail)
    assert "TXN-" not in text and "TXN-" not in " ".join(out.pass_through_indicators)


def test_no_activity_in_the_frame_gives_zero(rulebook):
    old = txn(timestamp="2026-05-01T10:00:00", amount=950_000, type=TransactionType.CASH_DEPOSIT)
    out = run("ACC-T001", _ctx([old], rulebook), AS_OF)
    assert out.confidence == 0.0
    assert out.structuring_indicators == [] and out.pass_through_indicators == []
    assert out.velocity_summary.startswith("No transactions in the 90 days")


def test_a_bank_without_any_transactions_gives_zero(rulebook):
    ctx = _ctx([], rulebook)
    assert ctx.review_date is None
    out = run("ACC-T001", ctx, ctx.review_date)
    assert out.confidence == 0.0 and out.velocity_summary == "No transactions in the 90 days."


def test_rows_after_as_of_are_not_seen(rulebook):
    rows = _structurer_and_mule()
    out = run("ACC-T001", _ctx(rows, rulebook), datetime(2026, 8, 3, 12))
    assert out.confidence == pytest.approx(0.40)  # only the deposits of 1-3 Aug are in the frame


def test_a_live_transaction_counts_whether_or_not_it_is_posted(rulebook):
    rows = _structurer_and_mule()[:2]  # two in-band deposits posted: k = 2
    live = txn(
        amount=950_000,
        type=TransactionType.CASH_DEPOSIT,
        timestamp="2026-08-03T10:00:00",
        transaction_id="T-LIVE",
    )
    ctx = _ctx(rows, rulebook)
    unposted = run("ACC-T001", ctx, live.timestamp, live)
    assert unposted.confidence == pytest.approx(0.40)  # k = 3 with the live deposit

    ctx.bank.add_transaction(live)
    posted = run("ACC-T001", ctx, live.timestamp, live)
    assert posted.confidence == pytest.approx(0.40)  # counted once, not twice


def test_velocity_summary_compares_busiest_30_days_with_the_average(rulebook):
    out = run("ACC-T001", _ctx(_structurer_and_mule(), rulebook), AS_OF)
    # 13 rows in the frame, 4.3 per 30 days on average; the busiest 30 days (from 1 Sep)
    # hold the 8 rows of the sweep burst.
    assert out.velocity_summary == (
        "Busiest 30 days from 1 Sep 2026: 8 transactions against an average of 4.3 per 30 "
        "days over the 90-day frame (ratio 1.85)."
    )


def test_deterministic_on_the_real_bank(real_ctx):
    for account_id in ("ACC-0001", "ACC-0347", real_ctx.bank.account_ids()[-1]):
        first = run(account_id, real_ctx, real_ctx.review_date)
        second = run(account_id, real_ctx, real_ctx.review_date)
        assert first == second
        assert 0.0 <= first.confidence <= 1.0
