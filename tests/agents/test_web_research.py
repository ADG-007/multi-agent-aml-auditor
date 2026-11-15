"""The Web Research agent: Python screening scores and the lower-only same-person call."""

import json

from src.agents.web_research import config
from src.agents.web_research.agent import WebResearchAgent, dob_agrees
from src.llm import LLMMode, Role
from src.rulebook import RulebookLookup
from src.schemas import SeverityTier, TransactionDirection
from src.stages.stage0_triage import TriageContext
from tests import factories as f


def ctx_with(media, rows=None):
    bank = f.bank(media=media, transactions=rows or [f.txn(timestamp="2026-08-01T10:00:00")])
    return TriageContext.build(bank, RulebookLookup.load())


def say(verdict, fact_ids=("F4",), more=()) -> str:
    return json.dumps(
        {"verdict": verdict, "fact_ids": list(fact_ids), "reason": "because", "more": list(more)}
    )


def run(ctx, client, **kwargs):
    return WebResearchAgent(client).run("ACC-T001", ctx, ctx.review_date, **kwargs)


def test_no_hit_means_no_model_call(tmp_path):
    client = f.fake_client(tmp_path)
    result = run(ctx_with([f.media_entry(entity_name="Someone Else")]), client)
    assert result.output.confidence == 0.0 and result.output.severity_tier is None
    assert client.fake.requests == []


def test_sanctions_hit_with_matching_dob_cannot_be_dismissed(tmp_path):
    ctx = ctx_with([f.media_entry(entity_name="Asha Verma", date_of_birth="1985-04-12")])
    result = run(ctx, f.fake_client(tmp_path, say("DIFFERENT")))
    assert result.output.confidence == config.HOLDER_SANCTIONS_DOB_AGREES
    assert "cannot be dismissed" in result.output.reasoning_trail[2].finding


def test_different_lowers_a_sanctions_hit_whose_dob_disagrees(tmp_path):
    entry = f.media_entry(entity_name="Asha Verma", date_of_birth="1960-01-01")
    result = run(ctx_with([entry]), f.fake_client(tmp_path, say("DIFFERENT")))
    assert result.output.confidence == config.DISMISSED_SCORE
    assert "Lowered from 0.5" in result.output.reasoning_trail[2].finding


def test_a_news_hit_is_never_lowered(tmp_path):
    entry = f.media_entry(
        entity_name="Asha Verma", tier=SeverityTier.ADVERSE_NEWS, date_of_birth="1960-01-01"
    )
    result = run(ctx_with([entry]), f.fake_client(tmp_path, say("DIFFERENT")))
    assert result.output.confidence == config.HOLDER_NEWS
    assert "recorded for the analyst" in result.output.reasoning_trail[2].finding


def test_same_never_raises_and_unknown_fact_ids_are_ignored(tmp_path):
    entry = f.media_entry(entity_name="Asha Verma", date_of_birth="1960-01-01")
    same = run(ctx_with([entry]), f.fake_client(tmp_path, say("SAME")))
    assert same.output.confidence == config.HOLDER_SANCTIONS_DOB_DISAGREES
    made_up = run(ctx_with([entry]), f.fake_client(tmp_path / "b", say("DIFFERENT", ["F99"])))
    assert made_up.output.confidence == config.HOLDER_SANCTIONS_DOB_DISAGREES
    assert "No real fact id" in made_up.output.reasoning_trail[2].finding


def test_unsure_keeps_the_score_and_queues_the_backup_when_it_could_matter(tmp_path):
    entry = f.media_entry(entity_name="Asha Verma")  # sanctions, no date of birth
    result = run(ctx_with([entry]), f.fake_client(tmp_path, say("UNSURE")))
    assert result.output.confidence == config.HOLDER_SANCTIONS_NO_DOB
    assert [j.kind for j in result.run.queued] == ["Same-person call"]
    client = f.fake_client(tmp_path, say("UNSURE"))
    again = run(ctx_with([entry]), client, role=Role.BACKUP)
    assert not again.run.queued
    assert client.fake.chats()[0]["model"] == client.model_for(Role.BACKUP)
    news = f.media_entry(entity_name="Asha Verma", tier=SeverityTier.ADVERSE_NEWS)
    quiet = run(ctx_with([news]), f.fake_client(tmp_path / "n", say("UNSURE")))
    assert not quiet.run.queued  # nothing the backup says could change a news score


def test_models_off_python_score_stands(tmp_path):
    entry = f.media_entry(entity_name="Asha Verma", tier=SeverityTier.ADVERSE_NEWS)
    result = run(ctx_with([entry]), f.fake_client(tmp_path, mode=LLMMode.OFF))
    assert result.output.confidence == config.HOLDER_NEWS and not result.run.queued


def test_alias_lookup_is_the_only_optional_tool(tmp_path):
    entry = f.media_entry(
        entity_name="A. Verma Asha", tier=SeverityTier.ADVERSE_NEWS, aliases=["Asha Verma"]
    )
    client = f.fake_client(
        tmp_path, say("UNSURE", more=["alias_lookup", "web_search"]), say("SAME")
    )
    result = run(ctx_with([entry]), client)
    assert result.run.tools_run[-1] == "alias_lookup"
    assert "Aliases of AM-T001" in client.fake.prompts()[1]
    assert "web_search" in next(s.finding for s in result.run.trail if s.step == "Optional tools")


def test_lookalike_entry_below_the_cutoff_is_no_hit(tmp_path):
    entry = f.media_entry(entity_name="Asha Varma Rao", tier=SeverityTier.NAME_SIMILARITY_ONLY)
    client = f.fake_client(tmp_path)
    result = run(ctx_with([entry]), client)
    assert result.output.confidence == 0.0 and client.fake.requests == []


def test_sanctioned_counterparty_scores_without_a_model_call(tmp_path):
    rows = [
        f.txn(
            direction=TransactionDirection.DEBIT,
            counterparty_name="Vikram Sethi",
            timestamp="2026-08-01T10:00:00",
        )
    ]
    client = f.fake_client(tmp_path)
    result = run(ctx_with([f.media_entry()], rows), client)
    assert result.output.confidence == config.COUNTERPARTY_SANCTIONS
    assert client.fake.requests == []


def test_dob_compare():
    from datetime import date

    assert dob_agrees("1985-04-12", date(1985, 4, 12)) is True
    assert dob_agrees("1985", date(1985, 4, 12)) is True
    assert dob_agrees("1990-01-01", date(1985, 4, 12)) is False
    assert dob_agrees(None, date(1985, 4, 12)) is None
