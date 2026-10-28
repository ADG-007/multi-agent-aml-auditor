"""Generated packs: repeatable, inside the bank's frame, never colliding, custom-built, runnable."""

import json
from datetime import timedelta

import pytest

from src.bank import BankData
from src.llm.cache import LLMMode
from src.matching import NameIndex
from src.newdata import (
    KINDS,
    MIX_SIZE,
    AccountSpec,
    list_packs,
    listed_people,
    make_custom_pack,
    make_pack,
    run_pack,
)
from src.rulebook import RulebookLookup
from src.schemas import Disposition, TransactionType
from src.stages.stage0_triage import config as triage_config


@pytest.fixture(scope="module")
def bank():
    return BankData.load()


def _rows(pack, name):
    return json.loads((pack.path / f"{name}.json").read_text(encoding="utf-8"))


def test_the_same_seed_makes_the_same_pack(tmp_path, bank):
    a = make_pack("mix", seed=3, root=tmp_path / "a", bank=bank)
    b = make_pack("mix", seed=3, root=tmp_path / "b", bank=bank)
    assert _rows(a, "transactions") == _rows(b, "transactions")
    assert _rows(a, "kyc_profiles") == _rows(b, "kyc_profiles")
    assert a.answers["accounts"] == b.answers["accounts"]
    assert a.title == "One of each pattern (12 accounts), variation 3"


def test_mix_plants_every_pattern_inside_the_frame(tmp_path, bank):
    pack = make_pack("mix", seed=5, root=tmp_path, bank=bank)
    answers = pack.answers["accounts"]
    assert len(answers) == MIX_SIZE
    assert {r["pattern"] for r in answers} == set(KINDS)
    assert not set(pack.account_ids) & set(bank.account_ids())

    merged, end = pack.bank(), bank.review_date()
    assert merged.review_date() == end  # new rows never move the bank's "now"
    for account_id in pack.account_ids:
        for t in merged.transactions(account_id):
            assert end - timedelta(days=90) <= t.timestamp <= end


def test_only_the_sanctions_holder_matches_the_list(tmp_path, bank):
    pack = make_pack("mix", seed=9, root=tmp_path, bank=bank)
    names = NameIndex(bank.adverse_media)
    pattern = {r["account_id"]: r["pattern"] for r in pack.answers["accounts"]}
    for c in _rows(pack, "customers"):
        hit = names.best_match(c["name"], cutoff=triage_config.NAME_MATCH_CUTOFF)
        assert (hit is not None) == (
            pattern[c["customer_id"].replace("CUST", "ACC")] == "sanctions"
        )


def test_numbers_do_not_depend_on_other_packs(tmp_path, bank):
    first = make_pack("clean", count=2, seed=1, root=tmp_path, bank=bank)
    again = make_pack("clean", count=2, seed=1, name="again", root=tmp_path, bank=bank)
    assert first.account_ids == again.account_ids  # same inputs, same accounts
    assert not set(first.account_ids) & set(bank.account_ids())
    assert {p.name for p in list_packs(tmp_path)} == {first.name, "again"}
    with pytest.raises(FileExistsError):
        make_pack("clean", seed=1, name=first.name, root=tmp_path, bank=bank)


def test_a_custom_account_follows_its_settings(tmp_path, bank):
    spec = AccountSpec(
        "structurer_kyc",
        label="My cash test",
        name="Meera Pillai",
        income=6 * 100_000,
        profile="{name} is a {occupation} earning {income}. No cash at all.",
        expected="MONITOR",
        settings={"cash_deposits": 7, "cash_lakh": 9.5},
    )
    pack = make_custom_pack([spec], "My cash test", root=tmp_path, bank=bank)
    (answer,) = pack.answers["accounts"]
    assert answer["group"] == "My cash test" and answer["expected"] == "MONITOR"
    assert pack.title == "My cash test" and pack.name == "my-cash-test"

    cash = [
        t for t in _rows(pack, "transactions") if t["type"] == TransactionType.CASH_DEPOSIT.value
    ]
    assert len(cash) == 7 and {t["amount"] for t in cash} == {950_000.0}
    (profile,) = _rows(pack, "kyc_profiles")
    assert (
        profile["narrative"]
        == "Meera Pillai is a office clerk earning Rs 6.00 lakh. No cash at all."
    )

    again = make_custom_pack([spec], "My cash test", root=tmp_path, bank=bank)
    assert again.name == "my-cash-test-2"


def test_a_custom_sanctions_account_uses_the_chosen_person(tmp_path, bank):
    person = listed_people(bank)[0]
    pack = make_custom_pack(
        [AccountSpec("sanctions", listed_name=person)], "Listed", root=tmp_path, bank=bank
    )
    assert _rows(pack, "customers")[0]["name"] == person


def test_a_typed_name_goes_to_the_first_ring_member_only(tmp_path, bank):
    spec = AccountSpec("mule_ring", count=3, name="Ravi Kumar")
    pack = make_custom_pack([spec], "Ring", root=tmp_path, bank=bank)
    names = [c["name"] for c in _rows(pack, "customers")]
    assert names[0] == "Ravi Kumar" and len(set(names)) == 3


def test_a_bad_expected_verdict_is_refused():
    with pytest.raises(ValueError):
        AccountSpec("clean", expected="MAYBE")


def test_a_sanctions_pack_escalates_with_models_off(tmp_path, bank):
    pack = make_pack("sanctions", seed=4, root=tmp_path, bank=bank)
    _, result = run_pack(pack, RulebookLookup.load(), kyc=LLMMode.OFF)
    report = result.latest[pack.account_ids[0]]
    assert report.disposition == Disposition.SUSPICIOUS_ESCALATE
    assert report.judge_trail and (pack.path / "outputs" / "run.jsonl").exists()
