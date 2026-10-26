"""Stage 4: tier rules, the date-of-birth check, counterparties and the highest-wins rule."""

from src.rulebook import RulebookLookup
from src.schemas import SeverityTier, TransactionDirection
from src.stages import stage4_adverse_media as s4
from src.stages.stage0_triage import TriageContext
from tests.factories import bank, customer, media_entry, txn


T = SeverityTier


def _run(media, transactions=None, name="Vikram Sethi", dob="1980-05-01"):
    b = bank(customers=[customer(name=name, dob=dob)], media=media, transactions=transactions)
    ctx = TriageContext.build(b, RulebookLookup.load())
    return s4.run("ACC-T001", ctx, ctx.review_date or None)


def test_sanctions_with_agreeing_dob_vetoes():
    out = _run([media_entry(date_of_birth="1980-05-01")])
    assert out.confidence == 0.95 and out.severity_tier == T.CONFIRMED_SANCTIONS


def test_sanctions_without_dob_still_vetoes():
    assert _run([media_entry()]).confidence == 0.90


def test_sanctions_with_other_dob_is_a_likely_name_clash():
    out = _run([media_entry(date_of_birth="1990-01-01")])
    assert out.confidence == 0.50
    assert "likely name clash" in out.matched_entries[0]


def test_year_only_dob_compares_the_year():
    assert _run([media_entry(date_of_birth="1980")]).confidence == 0.95
    assert _run([media_entry(date_of_birth="1981")]).confidence == 0.50


def test_adverse_news_is_capped_at_070():
    assert _run([media_entry(tier=T.ADVERSE_NEWS, date_of_birth="1980-05-01")]).confidence == 0.70


def test_lookalike_counts_only_at_92():
    # "Vikram Sethy" scores 91.7: a candidate at 85, but under the look-alike bar of 92.
    near = _run([media_entry(entity_name="Vikram Sethy", tier=T.NAME_SIMILARITY_ONLY)])
    assert near.confidence == 0.0 and near.matched_entries == []
    assert any("no hit" in step.finding for step in near.reasoning_trail)
    exact = _run([media_entry(tier=T.NAME_SIMILARITY_ONLY)])
    assert exact.confidence == 0.15
    far = _run([media_entry(entity_name="Vikrim Sathe", tier=T.NAME_SIMILARITY_ONLY)])
    assert far.confidence == 0.0


def test_a_sanction_never_hides_behind_a_lookalike():
    media = [
        media_entry(entry_id="AM-T1", tier=T.NAME_SIMILARITY_ONLY),
        media_entry(entry_id="AM-T2", entity_name="Vikram Sethii", date_of_birth="1980-05-01"),
    ]
    out = _run(media)
    assert out.confidence == 0.95 and len(out.matched_entries) == 2


def test_a_sanctioned_counterparty_vetoes():
    paid = txn(counterparty_name="Rahul Batra", direction=TransactionDirection.DEBIT)
    out = _run([media_entry(entity_name="Rahul Batra")], transactions=[paid], name="Asha Verma")
    assert out.confidence == 0.90
    assert "counterparty" in out.matched_entries[0]


def test_news_about_a_counterparty_weighs_040():
    paid = txn(counterparty_name="Rahul Batra")
    media = [media_entry(entity_name="Rahul Batra", tier=T.ADVERSE_NEWS)]
    assert _run(media, transactions=[paid], name="Asha Verma").confidence == 0.40


def test_no_match_is_zero_with_a_trail():
    out = _run([media_entry(entity_name="Somebody Else")])
    assert out.confidence == 0.0 and out.severity_tier is None and out.reasoning_trail
