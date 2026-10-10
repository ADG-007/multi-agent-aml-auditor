"""Read-only access to the simulated bank's data, as the pipeline is allowed to see it."""

from src.bank.data_store import BankData


__all__ = ["BankData"]
