"""The simulated bank, as the pipeline may see it. Every stage reads data through here.

It never reads `data/_ground_truth/` and blanks the blacklist's answer-key link on load.
Only `src/evaluation/` reads the answer key.
"""

from __future__ import annotations

import bisect
import json
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from src.schemas import (
    Account,
    AdverseMediaEntry,
    Counterparty,
    Customer,
    KYCProfile,
    Transaction,
)


GROUND_TRUTH_DIR_NAME = "_ground_truth"


class BankData:
    """In-memory bank. Each account's transactions stay in time order (ties by id)."""

    def __init__(
        self,
        accounts: Iterable[Account],
        customers: Iterable[Customer],
        transactions: Iterable[Transaction],
        counterparties: Iterable[Counterparty],
        kyc_profiles: Iterable[KYCProfile],
        adverse_media: Iterable[AdverseMediaEntry],
    ) -> None:
        self._accounts = {a.account_id: a for a in accounts}
        self._customers = {c.customer_id: c for c in customers}
        self._counterparties = {c.counterparty_id: c for c in counterparties}
        self._kyc_by_customer = {k.customer_id: k for k in kyc_profiles}
        self._adverse_media = tuple(
            entry.model_copy(update={"linked_actor_id": None}) for entry in adverse_media
        )

        grouped: dict[str, list[Transaction]] = defaultdict(list)
        for t in transactions:
            grouped[t.account_id].append(t)
        self._transactions = {
            account_id: sorted(rows, key=_order_key) for account_id, rows in grouped.items()
        }
        self._transaction_ids = {
            t.transaction_id for rows in self._transactions.values() for t in rows
        }

    @classmethod
    def load(cls, data_dir: str | Path = "data") -> BankData:
        """Load the committed dataset from `data_dir`."""
        root = Path(data_dir)
        return cls(
            accounts=(Account(**r) for r in _read(root / "accounts.json")),
            customers=(Customer(**r) for r in _read(root / "customers.json")),
            transactions=(Transaction(**r) for r in _read(root / "transactions.json")),
            counterparties=(Counterparty(**r) for r in _read(root / "counterparties.json")),
            kyc_profiles=(KYCProfile(**r) for r in _read(root / "kyc_profiles.json")),
            adverse_media=(AdverseMediaEntry(**r) for r in _read(root / "adverse_media.json")),
        )

    # ----------------------------------------------------------------------------------
    # Lookups
    # ----------------------------------------------------------------------------------

    def account_ids(self) -> list[str]:
        return sorted(self._accounts)

    def account(self, account_id: str) -> Account:
        try:
            return self._accounts[account_id]
        except KeyError:
            raise KeyError(f"unknown account {account_id!r}") from None

    def customer_for(self, account_id: str) -> Customer:
        return self._customers[self.account(account_id).customer_id]

    def transactions(self, account_id: str) -> tuple[Transaction, ...]:
        """The account's transactions, oldest first."""
        self.account(account_id)
        return tuple(self._transactions.get(account_id, ()))

    def review_date(self) -> datetime | None:
        """The latest transaction time across the bank. Periodic reviews use it as "now"."""
        latest = [rows[-1].timestamp for rows in self._transactions.values() if rows]
        return max(latest, default=None)

    def kyc_profile_for(self, account_id: str) -> KYCProfile | None:
        return self._kyc_by_customer.get(self.account(account_id).customer_id)

    def counterparty(self, counterparty_id: str) -> Counterparty | None:
        return self._counterparties.get(counterparty_id)

    @property
    def adverse_media(self) -> tuple[AdverseMediaEntry, ...]:
        """The screening corpus, with its ground-truth links removed."""
        return self._adverse_media

    # ----------------------------------------------------------------------------------
    # Live additions
    # ----------------------------------------------------------------------------------

    def add_transaction(self, transaction: Transaction) -> None:
        """Post a new transaction to an existing account, keeping time order."""
        self.account(transaction.account_id)
        if transaction.transaction_id in self._transaction_ids:
            raise ValueError(f"duplicate transaction id {transaction.transaction_id!r}")
        rows = self._transactions.setdefault(transaction.account_id, [])
        bisect.insort(rows, transaction, key=_order_key)
        self._transaction_ids.add(transaction.transaction_id)


def _order_key(t: Transaction) -> tuple:
    return (t.timestamp, t.transaction_id)


def _read(path: Path) -> list[dict]:
    if GROUND_TRUTH_DIR_NAME in path.resolve().parts:
        raise PermissionError(
            f"{path} is ground truth. Pipeline code never reads it; only src/evaluation/ does."
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)
