"""Stage 0's monthly-cash rule (Amendment 4, S12): review and live modes, routes and wording."""

from datetime import datetime, timedelta

import pytest

from src.orchestration.run_state import RunState
from src.rulebook import RulebookLookup
from src.schemas import (
    StageName,
    TransactionDirection,
    TransactionType,
    TriageEvent,
    TriageEventKind,
    TriageRule,
)
from src.stages.stage0_triage import TriageContext, rules, triage
from tests.factories import bank, txn


DEPOSIT = TransactionType.CASH_DEPOSIT


@pytest.fixture(scope="module")
def rulebook():
    return RulebookLookup.load()


def _deposits(total, start, n=10):
    first = datetime.fromisoformat(start)
    return [
        txn(
            amount=total / n,
            type=DEPOSIT,
            direction=TransactionDirection.CREDIT,
            timestamp=f"{first + timedelta(days=i):%Y-%m-%dT11:00:00}",
        )
        for i in range(n)
    ]


def test_review_fires_once_per_band_month_on_the_deposit_that_reached_the_floor(rulebook):
    rows = _deposits(950_000, "2026-07-01") + _deposits(940_000, "2026-08-03")
    hits = rules.monthly_cash(rows, rulebook)
    assert [h.fired.rule for h in hits] == [TriageRule.MONTHLY_CASH] * 2
    # 95,000 a deposit: the 10th deposit takes July to ₹9,50,000.
    assert hits[0].fired.transaction_id == rows[9].transaction_id
    assert hits[0].fired.clause_id == "STRUCT-002"
    assert "Jul 2026" in hits[0].reason and "₹9,50,000" in hits[0].reason


def test_review_fires_on_a_window_across_a_month_end(rulebook):
    rows = _deposits(700_000, "2026-07-20", n=7) + _deposits(700_000, "2026-08-03", n=7)
    (hit,) = rules.monthly_cash(rows, rulebook)
    assert "no calendar month above" in hit.reason


def test_review_is_silent_below_the_band_and_above_the_threshold(rulebook):
    assert rules.monthly_cash(_deposits(800_000, "2026-07-01"), rulebook) == []
    assert rules.monthly_cash(_deposits(1_200_000, "2026-07-01"), rulebook) == []


def test_live_fires_once_the_month_to_date_total_is_in_the_band(rulebook):
    rows = _deposits(950_000, "2026-07-01")
    assert rules.monthly_cash(rows[:9], rulebook, within=rows[8]) == []
    (hit,) = rules.monthly_cash(rows, rulebook, within=rows[9])
    assert hit.fired.transaction_id == rows[9].transaction_id
    assert "to date" in hit.reason


def test_live_ignores_non_cash(rulebook):
    rows = _deposits(950_000, "2026-07-01")
    upi = txn(amount=5_000, timestamp="2026-07-12T11:00:00")
    assert rules.monthly_cash(rows + [upi], rulebook, within=upi) == []


def test_routes_like_the_cash_threshold_rule(rulebook):
    rows = _deposits(950_000, "2026-07-01")
    out = triage(
        TriageEvent(kind=TriageEventKind.PERIODIC_REVIEW, account_id="ACC-T001"),
        TriageContext.build(bank(transactions=rows), rulebook),
        RunState(),
    )
    assert TriageRule.MONTHLY_CASH in {f.rule for f in out.fired_rules}
    assert {StageName.TRANSACTION_PATTERN, StageName.ADVERSE_MEDIA, StageName.GRAPH} <= set(
        out.routed_stages
    )
    assert (
        rules.RULE_ROUTES[TriageRule.MONTHLY_CASH] == rules.RULE_ROUTES[TriageRule.CASH_THRESHOLD]
    )
