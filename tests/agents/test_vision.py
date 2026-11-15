"""The Vision agent: two-read ID check, claim extraction guards, Python KYC rules."""

import json

import pytest

from src.agents.vision import claims as cf
from src.agents.vision import config, rules
from src.agents.vision.agent import AGREES, FLAGGED, ONE_READ, UNCONFIRMED, VisionAgent
from src.agents.vision.kyc_facts import Activity
from src.llm import LLMMode
from src.rulebook import RulebookLookup
from src.schemas import KYCProfile, TransactionDirection, TransactionType
from src.stages.stage0_triage import TriageContext
from tests import factories as f


PROFILE = (
    "Asha Verma works as a school teacher with a declared annual income of Rs 6.00 lakh. "
    "Customer stated that no cash deposits would be made into the account at any point. "
    "Expected activity is in the range of Rs 75,000 per month across roughly 20 transactions."
)
SHOP = (
    "Runs a shop. The shop's cash takings are deposited at the branch on most working days, "
    "around Rs 9.08 lakh a month, and paid on to suppliers."
)
GOOD_CARD = "Name\nAsha Verma\nDate of Birth\n12/04/1985\n1234 5678 9012\n"
FORGED_CARD = "Name\nAsha Verma\nDate of Birth\n12/04/1981\n1234 5678 9012\n"


def answer(**quotes) -> str:
    """A claim read as the model returns it: every quote field, empty unless given."""
    return json.dumps({name: quotes.get(name, "") for name in cf.ClaimsForm.model_fields})


NO_CASH = {"no_cash": "no cash deposits would be made into the account"}
ACTIVITY = {"expected_activity": "Expected activity is in the range of Rs 75,000 per month"}


def activity(**kwargs) -> Activity:
    base = dict(
        income=600_000,
        money_in_month=40_000,
        money_out_month=35_000,
        cash_deposits=0,
        cash_deposit_total=0.0,
        busiest_cash_month=0.0,
        cross_border_wires=0,
        payers=15,
        transactions=60,
    )
    return Activity(**(base | kwargs))


def claims(**values) -> cf.Claims:
    c = cf.Claims()
    for name, (value, quote) in values.items():
        c.values[name] = value
        c.quotes[name] = quote
    return c


# --------------------------------------------------------------------------------------
# Claims: guard, two reads, amounts
# --------------------------------------------------------------------------------------


def test_amounts_are_parsed_by_python():
    assert cf.parse_inr("around Rs 9.08 lakh a month") == [908_000]
    assert cf.parse_inr("Rs 89,583 per month and ₹1.2 crore") == [89_583, 12_000_000]
    assert cf.parse_inr("around 55 a month") == []


def test_claim_kept_only_when_both_reads_agree_and_the_quote_is_real():
    one = cf.ClaimsForm.model_validate_json(answer(**NO_CASH, **ACTIVITY))
    two = cf.ClaimsAnswers.model_validate_json(answer(**NO_CASH, no_foreign="no overseas dealings"))
    got = cf.reconcile(one, two, PROFILE)
    assert got.values["cash"] == "NO_CASH"
    assert got.unsure == ["expected_activity"]  # one read only
    assert got.values["expected_activity"] == cf.NOT_STATED
    assert got.values["foreign"] == cf.NOT_STATED
    assert got.dropped == ["read 2 NO_FOREIGN (quote not in the profile)"]


def test_a_denial_in_the_wrong_field_is_dropped():
    wrong = cf.ClaimsForm.model_validate_json(answer(cash_takings=NO_CASH["no_cash"]))
    values, dropped = cf.guarded(wrong, PROFILE)
    assert values["cash"] == (cf.NOT_STATED, "")
    assert dropped == ["CASH_DECLARED (the quoted words say the opposite)"]
    both = cf.ClaimsForm.model_validate_json(
        answer(**NO_CASH, cash_takings="Expected activity is in the range of Rs 75,000")
    )
    assert cf.guarded(both, PROFILE)[0]["cash"] == (cf.NOT_STATED, "")


def test_backup_read_settles_unsure_fields():
    one = cf.ClaimsForm.model_validate_json(answer(**ACTIVITY))
    two = cf.ClaimsAnswers.model_validate_json(answer())
    got = cf.reconcile(one, two, PROFILE)
    third = cf.ClaimsForm.model_validate_json(answer(**ACTIVITY))
    assert cf.resolve(got, third, PROFILE) == ["expected_activity"]
    assert got.unsure == [] and got.amount("expected_activity") == 75_000


def test_no_reads_means_nothing_is_stated():
    got = cf.reconcile(None, None, PROFILE)
    assert got.stated() == {} and got.unsure == []


# --------------------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------------------


def test_volume_levels_at_1_5x_and_3x():
    assert rules.judge(claims(), activity(money_in_month=74_000), False).level == "NONE"
    moderate = rules.judge(claims(), activity(money_in_month=75_000), False)
    assert (moderate.level, moderate.score) == ("MODERATE", 0.0)
    high = rules.judge(claims(), activity(money_in_month=150_000), False)
    assert (high.level, high.score) == ("HIGH", config.LEVEL_SCORES["HIGH"])


def test_declared_turnover_is_the_base_when_larger_than_income():
    stated = claims(expected_activity=("STATED", "activity of Rs 2.00 lakh per month"))
    assert rules.judge(stated, activity(money_in_month=250_000), False).level == "NONE"
    assert rules.judge(claims(), activity(money_in_month=250_000), False).level == "HIGH"


def test_cash_already_scored_is_not_counted_twice():
    structurer = activity(money_in_month=700_000, cash_deposits=30, cash_deposit_total=1_950_000)
    assert rules.judge(claims(), structurer, cash_scored=True).level == "NONE"
    assert rules.judge(claims(), structurer, cash_scored=False).level == "HIGH"


def test_no_cash_against_cash_deposits_is_high_and_two_highs_are_severe():
    lied = claims(cash=("NO_CASH", NO_CASH["no_cash"]))
    some_cash = activity(cash_deposits=4, cash_deposit_total=60_000)
    assert rules.judge(lied, some_cash, False).level == "HIGH"
    both = activity(cash_deposits=4, cash_deposit_total=60_000, money_in_month=200_000)
    assert rules.judge(lied, both, False).level == "SEVERE"
    tiny = activity(cash_deposits=1, cash_deposit_total=3_000)
    assert rules.judge(lied, tiny, False).level == "NONE"


def test_declared_cash_takings_explain_cash_within_the_margin():
    shop = claims(cash=("CASH_DECLARED", "cash takings around Rs 9.08 lakh a month"))
    near = rules.judge(shop, activity(busiest_cash_month=964_000), True)
    assert near.cash_explained and "explained" in near.cash_note
    far = rules.judge(shop, activity(busiest_cash_month=2_000_000), True)
    assert not far.cash_explained
    no_amount = claims(cash=("CASH_DECLARED", "cash takings are deposited"))
    assert not rules.judge(no_amount, activity(busiest_cash_month=900_000), True).cash_explained
    assert not rules.judge(claims(), activity(busiest_cash_month=900_000), True).cash_explained


def test_two_moderate_concerns_make_high():
    said = claims(
        foreign=("NO_FOREIGN", "does not deal with overseas counterparties"),
        sources=("SINGLE_SOURCE", "one income source"),
    )
    one = rules.judge(said, activity(cross_border_wires=3), False)
    assert [c.key for c in one.concerns] == ["foreign"] and one.level == "MODERATE"
    two = rules.judge(said, activity(cross_border_wires=3, payers=80), False)
    assert two.level == "HIGH"


def test_one_off_event_cancels_a_covered_excess():
    event = claims(one_off_event=("EVENT_DECLARED", "sale of a flat for Rs 30 lakh"))
    covered = rules.judge(event, activity(money_in_month=600_000), False)
    assert covered.level == "NONE" and covered.cancelled
    too_big = claims(one_off_event=("EVENT_DECLARED", "a gift of Rs 50,000"))
    assert rules.judge(too_big, activity(money_in_month=600_000), False).level == "HIGH"


# --------------------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------------------


@pytest.fixture
def ctx():
    rows = [
        f.txn(amount=20_000, type=TransactionType.CASH_DEPOSIT, timestamp="2026-08-10T10:00:00"),
        f.txn(amount=20_000, type=TransactionType.CASH_DEPOSIT, timestamp="2026-08-20T10:00:00"),
        f.txn(amount=30_000, timestamp="2026-09-01T10:00:00"),
        f.txn(amount=5_000, direction=TransactionDirection.DEBIT, timestamp="2026-09-02T10:00:00"),
    ]
    profile = KYCProfile(customer_id="CUST-T001", narrative=PROFILE)
    bank = f.bank(transactions=rows, kyc_profiles=[profile])
    return TriageContext.build(bank, RulebookLookup.load())


@pytest.fixture
def documents(tmp_path):
    folder = tmp_path / "documents"
    folder.mkdir()
    manifest = [{"customer_id": "CUST-T001", "document_type": "AADHAAR", "file": "card.png"}]
    (folder / config.MANIFEST_NAME).write_text(json.dumps(manifest))
    (folder / "card.png").write_bytes(b"not a real image")
    return folder


def run(ctx, documents, client, card=GOOD_CARD):
    return VisionAgent(client).run(
        "ACC-T001", ctx, ctx.review_date, documents_dir=documents, reader=lambda path: card
    )


def test_clean_card_needs_no_image_read_and_kyc_lie_is_high(tmp_path, ctx, documents):
    client = f.fake_client(tmp_path, answer(**NO_CASH), answer(**NO_CASH))
    result = run(ctx, documents, client)
    assert result.vision.forgery_confidence == 0.0
    assert all(r.result == AGREES for r in result.id_fields)
    assert len(client.fake.chats()) == 2  # two claim reads, no image read
    assert result.kyc.confidence == config.LEVEL_SCORES["HIGH"]
    assert "cash deposits" in result.kyc.profile_behavior_mismatches[0]
    assert result.kyc.reasoning_trail[-1].step == "Level to score"
    assert not result.cash_explained


def test_field_is_flagged_only_when_both_reads_disagree(tmp_path, ctx, documents):
    image = '{"name": "Asha Verma", "date_of_birth": "12/04/1981", "id_number": "1234 5678 9012"}'
    client = f.fake_client(tmp_path, image, answer(), answer())
    result = run(ctx, documents, client, card=FORGED_CARD)
    assert [r.result for r in result.id_fields] == [AGREES, FLAGGED, AGREES]
    assert result.vision.forgery_confidence == config.BOTH_READS_DISAGREE
    assert client.fake.chats()[0]["messages"][-1]["images"]
    assert "Asha" not in client.fake.prompts()[0]  # the model never sees the record


def test_image_read_that_agrees_clears_a_tesseract_misread(tmp_path, ctx, documents):
    image = '{"name": "Asha Verma", "date_of_birth": "12/04/1985", "id_number": "1234 5678 9012"}'
    result = run(ctx, documents, f.fake_client(tmp_path, image, answer(), answer()), FORGED_CARD)
    assert result.id_fields[1].result == UNCONFIRMED
    assert result.vision.forgery_confidence == config.UNCONFIRMED


def test_models_off_one_read_never_pauses(tmp_path, ctx, documents):
    result = run(ctx, documents, f.fake_client(tmp_path, mode=LLMMode.OFF), FORGED_CARD)
    assert result.id_fields[1].result == ONE_READ
    assert result.vision.forgery_confidence == config.ONE_READ_DISAGREES
    assert result.kyc.confidence == 0.0 and not result.run.queued


def test_reads_that_disagree_are_queued_and_the_backup_settles_them(tmp_path, ctx, documents):
    client = f.fake_client(tmp_path, answer(**NO_CASH), answer(), answer(**NO_CASH))
    agent = VisionAgent(client)
    result = agent.run(
        "ACC-T001", ctx, ctx.review_date, documents_dir=documents, reader=lambda p: GOOD_CARD
    )
    assert result.kyc.confidence == 0.0
    assert [j.kind for j in result.run.queued] == ["KYC claims"]
    assert agent.recheck(result, ctx, ctx.review_date) is True
    assert result.kyc.confidence == config.LEVEL_SCORES["HIGH"]
    assert client.fake.chats()[-1]["model"] == client.model_for(agent.spec.role.BACKUP)


def test_shop_profile_explains_its_cash(tmp_path, documents):
    rows = [
        f.txn(amount=45_000, type=TransactionType.CASH_DEPOSIT, timestamp=f"2026-08-{d:02d}T10:00")
        for d in range(1, 21)
    ]
    bank = f.bank(
        transactions=rows, kyc_profiles=[KYCProfile(customer_id="CUST-T001", narrative=SHOP)]
    )
    ctx = TriageContext.build(bank, RulebookLookup.load())
    said = (
        "cash takings are deposited at the branch on most working days, around Rs 9.08 lakh a month"
    )
    client = f.fake_client(tmp_path, answer(cash_takings=said), answer(cash_takings=said))
    result = run(ctx, documents, client)
    assert result.cash_explained
    assert any(s.step == "Declared cash" for s in result.kyc.reasoning_trail)
