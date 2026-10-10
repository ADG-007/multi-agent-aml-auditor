"""Small builders for hand-made test records.

Tests use these instead of the generated dataset wherever a rule or boundary has to be pinned
exactly; the real data is only used where the point is "this works on the actual bank".
"""

from __future__ import annotations

from datetime import date, datetime
from itertools import count

from src.bank.data_store import BankData
from src.schemas import (
    Account,
    AccountType,
    AdverseMediaEntry,
    Counterparty,
    Customer,
    Gender,
    IdDocumentType,
    KYCProfile,
    ResidencyStatus,
    SeverityTier,
    Transaction,
    TransactionDirection,
    TransactionType,
)


_txn_ids = count(1)


def txn(
    account_id: str = "ACC-T001",
    amount: float = 1_000.0,
    direction: TransactionDirection = TransactionDirection.CREDIT,
    type: TransactionType = TransactionType.UPI,
    timestamp: str = "2026-08-01T10:00:00",
    counterparty_id: str = "CP-0001",
    counterparty_name: str = "Ordinary Payer",
    device_id: str | None = None,
    ip_address: str | None = None,
    transaction_id: str | None = None,
) -> Transaction:
    return Transaction(
        transaction_id=transaction_id or f"TXN-TEST-{next(_txn_ids):05d}",
        account_id=account_id,
        amount=amount,
        direction=direction,
        type=type,
        timestamp=datetime.fromisoformat(timestamp),
        device_id=device_id,
        ip_address=ip_address,
        counterparty_id=counterparty_id,
        counterparty_name=counterparty_name,
    )


def customer(
    customer_id: str = "CUST-T001",
    name: str = "Asha Verma",
    address: str = "12 Lake Road, Pune 411001",
    dob: str = "1985-04-12",
) -> Customer:
    return Customer(
        customer_id=customer_id,
        name=name,
        dob=date.fromisoformat(dob),
        gender=Gender.FEMALE,
        occupation="school teacher",
        stated_income=600_000,
        address=address,
        residency_status=ResidencyStatus.DOMESTIC,
        country="India",
        id_document_type=IdDocumentType.AADHAAR,
        id_number="1234 5678 9012",
    )


def account(account_id: str = "ACC-T001", customer_id: str = "CUST-T001") -> Account:
    return Account(
        account_id=account_id,
        customer_id=customer_id,
        opened_date=date(2020, 1, 1),
        account_type=AccountType.SAVINGS,
    )


def media_entry(
    entry_id: str = "AM-T001",
    entity_name: str = "Vikram Sethi",
    tier: SeverityTier = SeverityTier.CONFIRMED_SANCTIONS,
    aliases: list[str] | None = None,
    date_of_birth: str | None = None,
) -> AdverseMediaEntry:
    return AdverseMediaEntry(
        entry_id=entry_id,
        entity_name=entity_name,
        severity_tier=tier,
        description="test entry",
        aliases=aliases or [],
        date_of_birth=date_of_birth,
    )


def bank(
    transactions: list[Transaction] | None = None,
    customers: list[Customer] | None = None,
    accounts: list[Account] | None = None,
    media: list[AdverseMediaEntry] | None = None,
    counterparties: list[Counterparty] | None = None,
    kyc_profiles: list[KYCProfile] | None = None,
) -> BankData:
    """A tiny in-memory bank. Defaults to one account (ACC-T001) with one customer."""
    return BankData(
        accounts=accounts if accounts is not None else [account()],
        customers=customers if customers is not None else [customer()],
        transactions=transactions or [],
        counterparties=counterparties or [],
        kyc_profiles=kyc_profiles or [],
        adverse_media=media or [],
    )
