"""The agent base: facts, tools, guards, the model call and its fallbacks."""

import pytest
from pydantic import BaseModel

from src.agents.base import Agent, AgentSpec, FactSheet, known_ids, quote_in
from src.llm import LLMMode, Role
from tests.factories import fake_client


class Pick(BaseModel):
    label: str
    fact_ids: list[str]


class Toy(Agent):
    spec = AgentSpec(
        name="toy",
        role=Role.GRAPH,
        prompt_version="toy-v1",
        required_tools=("count",),
        optional_tools=("extra", "more", "third"),
    )


def test_facts_are_numbered_and_rendered():
    sheet = FactSheet()
    sheet.add("cash", "3 cash deposits", value=3)
    sheet.add("income", "money in 2.0x income", value=2.0)
    assert sheet.render() == "F1: 3 cash deposits\nF2: money in 2.0x income"
    assert sheet.get(" f2 ").value == 2.0 and sheet.get("F9") is None
    assert [f.fact_id for f in sheet.topic("cash")] == ["F1"]


def test_guard_drops_unknown_fact_ids():
    sheet = FactSheet()
    sheet.add("a", "one")
    sheet.add("a", "two")
    assert known_ids(["f2", "F7", "F2", "F1", "made up"], sheet) == (
        ["F2", "F1"],
        ["F7", "made up"],
    )


def test_quote_must_be_in_the_source_word_for_word():
    profile = "Customer stated that no cash deposits  would be made.\nSalary income only."
    assert quote_in("no cash deposits would be made", profile)
    assert quote_in('"Salary income only."', profile)
    assert not quote_in("no business use", profile)
    assert not quote_in("no", profile)  # too short to prove anything
    assert not quote_in(None, profile)


def test_model_answer_lands_in_the_run_and_trail(tmp_path):
    agent = Toy(fake_client(tmp_path, '{"label": "X", "fact_ids": ["F1"]}'))
    run = agent.start("ACC-1")
    answer = agent.ask(run, "prompt", Pick, "Label")
    assert answer.label == "X"
    assert run.model_calls == 1 and not run.model_failed
    assert "new answer" in run.trail[-1].finding


def test_models_off_gives_none_without_a_failure(tmp_path):
    agent = Toy(fake_client(tmp_path, mode=LLMMode.OFF))
    run = agent.start("ACC-1")
    assert agent.ask(run, "prompt", Pick, "Label") is None
    assert not run.model_failed and not run.queued


def test_schema_failure_is_queued_for_the_backup(tmp_path):
    agent = Toy(fake_client(tmp_path, "junk", "junk"))
    run = agent.start("ACC-1")
    assert agent.ask(run, "prompt", Pick, "Label") is None
    assert run.model_failed
    assert [(j.agent, j.account_id, j.kind) for j in run.queued] == [("toy", "ACC-1", "Label")]


def test_cache_miss_is_not_queued(tmp_path):
    agent = Toy(fake_client(tmp_path, mode=LLMMode.CACHED))
    run = agent.start("ACC-1")
    assert agent.ask(run, "prompt", Pick, "Label") is None
    assert run.model_failed and not run.queued


def test_optional_tools_known_names_only_and_at_most_two(tmp_path):
    agent = Toy(fake_client(tmp_path))
    run = agent.start("ACC-1")
    ran = []
    tools = {name: (lambda n=name: ran.append(n)) for name in ("extra", "more", "third")}
    chosen = agent.optional(run, ["hack", "more", "extra", "third", "more"], tools)
    assert chosen == ran == ["more", "extra"]
    assert "hack" in run.trail[-1].finding and "third" in run.trail[-1].finding


def test_required_tools_are_tracked(tmp_path):
    agent = Toy(fake_client(tmp_path))
    run = agent.start("ACC-1")
    assert agent.missing_required(run) == ["count"]
    agent.tool(run, "count")
    assert agent.missing_required(run) == []
    with pytest.raises(ValueError):
        agent.tool(run, "not_a_tool")
