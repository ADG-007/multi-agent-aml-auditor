"""The answer-key label rule on EvidenceLedgerEntry (Amendment 3, A4)."""

import pytest

from src.schemas import Disposition, EvidenceLedgerEntry, SeverityTier


def entry(**planted) -> EvidenceLedgerEntry:
    return EvidenceLedgerEntry(account_id="ACC-T001", customer_id="CUST-T001", **planted)


@pytest.mark.parametrize(
    "planted, label",
    [
        ({}, Disposition.CLEAR),
        ({"structuring_planted": True}, Disposition.MONITOR),
        (
            {"structuring_planted": True, "kyc_mismatch_planted": True},
            Disposition.SUSPICIOUS_ESCALATE,
        ),
        ({"adverse_media_hit": SeverityTier.ADVERSE_NEWS}, Disposition.MONITOR),
        # A confirmed sanctions hit escalates on its own: the veto must escalate it.
        ({"adverse_media_hit": SeverityTier.CONFIRMED_SANCTIONS}, Disposition.SUSPICIOUS_ESCALATE),
        # The decoy tier is not a real channel.
        ({"adverse_media_hit": SeverityTier.NAME_SIMILARITY_ONLY}, Disposition.CLEAR),
    ],
)
def test_expected_disposition(planted, label):
    assert entry(**planted).expected_disposition == label
