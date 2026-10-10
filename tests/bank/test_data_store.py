"""The bank data store: real counts, time order, live additions and the answer-key guard."""

from pathlib import Path

import pytest

from src.bank import BankData
from tests.factories import bank, txn


@pytest.fixture(scope="module")
def real_bank() -> BankData:
    return BankData.load("data")


def test_loads_the_committed_dataset(real_bank):
    assert len(real_bank.account_ids()) == 370
    assert sum(len(real_bank.transactions(a)) for a in real_bank.account_ids()) == 23_450
    assert len(real_bank.adverse_media) == 160


def test_transactions_are_time_ordered(real_bank):
    for account_id in real_bank.account_ids():
        rows = real_bank.transactions(account_id)
        assert [t.timestamp for t in rows] == sorted(t.timestamp for t in rows)


def test_ground_truth_links_are_blanked(real_bank):
    assert all(entry.linked_actor_id is None for entry in real_bank.adverse_media)


def test_refuses_to_read_ground_truth():
    with pytest.raises(PermissionError):
        BankData.load(Path("data") / "_ground_truth")


def test_lookups_by_account(real_bank):
    account_id = real_bank.account_ids()[0]
    customer = real_bank.customer_for(account_id)
    assert customer.customer_id == real_bank.account(account_id).customer_id
    assert real_bank.kyc_profile_for(account_id).customer_id == customer.customer_id
    with pytest.raises(KeyError):
        real_bank.account("ACC-NOPE")


def test_add_transaction_keeps_time_order_and_rejects_duplicates():
    b = bank(transactions=[txn(timestamp="2026-08-01T10:00:00", transaction_id="T-A")])
    b.add_transaction(txn(timestamp="2026-07-01T10:00:00", transaction_id="T-B"))
    assert [t.transaction_id for t in b.transactions("ACC-T001")] == ["T-B", "T-A"]

    with pytest.raises(ValueError):
        b.add_transaction(txn(transaction_id="T-A"))
    with pytest.raises(KeyError):
        b.add_transaction(txn(account_id="ACC-UNKNOWN"))
