"""The live engine: events in order, the pause, the queue and its limits."""

import json
import threading

import pytest

from src.agents.vision import ocr
from src.api.live import LiveJobs, NotAllowed, QueueFull, node_weights, run_case
from src.llm import LLMClient, LLMMode
from src.orchestration.pipeline import NODES, Pipeline
from src.orchestration.steps import Agents
from tests.api.conftest import VISITOR
from tests.factories import fake_client


FORGED_CARD = "Name\nAsha Verma\nDate of Birth\n12/04/1981\n1234 5678 9012\n"
FORGED_READ = '{"name": "Asha Verma", "date_of_birth": "12/04/1981", "id_number": "1234 5678 9012"}'


def _collect(jobs: LiveJobs, job_id: str) -> list[dict]:
    return [e for e in jobs.events(job_id, keep_alive=0.05) if e is not None]


def _types(events):
    return [e["type"] for e in events]


def test_replay_job_emits_every_node_in_order_then_done(ctx):
    jobs = LiveJobs(ctx)
    job, ahead = jobs.submit("ACC-T001", "replay")
    events = _collect(jobs, job.id)
    assert ahead == 0 and [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert _types(events)[:2] == ["queued", "started"] and events[-1]["type"] == "done"
    done = [e for e in events if e["type"] == "node_done"]
    assert [e["node"] for e in done] == list(NODES)
    assert all(e["seconds"] >= 0 and e["source"] == "python" for e in done)
    notes = {e["node"]: e["note"] for e in done}
    assert notes["admin_gate"] == notes["backup"] == "skipped"
    assert notes["judge"].split()[0] == events[-1]["case"]["disposition"]
    assert events[-1]["report"]["target_account"] == "ACC-T001"
    assert job.status == "done"


def test_events_can_be_read_again_from_a_sequence_number(ctx):
    jobs = LiveJobs(ctx)
    job, _ = jobs.submit("ACC-T001", "replay")
    everything = _collect(jobs, job.id)
    rest = [e for e in jobs.events(job.id, after=5) if e is not None]
    assert rest == everything[5:]


@pytest.fixture
def pausing(ctx, data_dir, tmp_path, monkeypatch):
    """Jobs over a bank whose one ID card both reads flag."""
    monkeypatch.setattr(ocr, "tesseract", lambda path: FORGED_CARD)

    def make(timeout=5.0):
        return LiveJobs(
            ctx,
            client_factory=lambda mode: fake_client(tmp_path, FORGED_READ),
            agents_factory=lambda c: Agents.build(c, documents_dir=data_dir / "documents"),
            pause_timeout=timeout,
        )

    return make


def _wait_for(job, status, seconds=10):
    with job.changed:
        assert job.changed.wait_for(lambda: job.status == status, seconds), job.status


def test_paused_case_waits_then_resumes_on_the_decision(pausing):
    jobs = pausing()
    job, _ = jobs.submit("ACC-T001", "live")
    _wait_for(job, "paused")
    paused = job.events[-1]
    assert paused["type"] == "paused" and paused["evidence"]["field_discrepancies"]
    jobs.decide(job.id, "block")
    events = _collect(jobs, job.id)
    kinds = _types(events)
    assert kinds.index("paused") < kinds.index("resumed") < kinds.index("done")
    assert events[-1]["case"]["disposition"] == "BLOCKED_BY_ADMIN_REVIEW"
    gate = next(e for e in events if e["type"] == "node_done" and e["node"] == "admin_gate")
    assert gate["note"] == "block"
    with pytest.raises(NotAllowed):
        jobs.decide(job.id, "approve")


def test_undecided_pause_is_abandoned_and_frees_the_queue(pausing):
    jobs = pausing(timeout=0.2)
    first, _ = jobs.submit("ACC-T001", "live")
    second, ahead = jobs.submit("ACC-T002", "replay")
    assert ahead == 1
    assert _collect(jobs, first.id)[-1] == {
        "seq": len(first.events),
        "type": "error",
        "message": "abandoned",
    }
    assert first.status == "abandoned"
    assert _collect(jobs, second.id)[-1]["type"] == "done"


def test_queue_holds_one_running_and_two_waiting(pausing):
    jobs = pausing()
    running, _ = jobs.submit("ACC-T001", "live")
    _wait_for(running, "paused")
    waiting = [jobs.submit("ACC-T002", "replay") for _ in range(2)]
    assert [ahead for _, ahead in waiting] == [1, 2]
    with pytest.raises(QueueFull):
        jobs.submit("ACC-T002", "replay")
    jobs.cancel(waiting[1][0].id)
    assert waiting[1][0].status == "cancelled"
    jobs.cancel(running.id)
    assert _collect(jobs, running.id)[-1]["message"] == "cancelled"
    assert _collect(jobs, waiting[0][0].id)[-1]["type"] == "done"


def test_recorded_seconds_come_from_the_saved_answers(ctx, tmp_path, data_dir, monkeypatch):
    monkeypatch.setattr(ocr, "tesseract", lambda path: FORGED_CARD)
    first = fake_client(tmp_path, FORGED_READ)
    agents = Agents.build(first, documents_dir=data_dir / "documents")
    events: list[dict] = []
    run_case(
        Pipeline(ctx, agents), first, "ACC-T001", lambda **e: events.append(e), lambda p: "approve"
    )
    live = next(e for e in events if e.get("node") == "vision")
    assert live["source"] == "live" and live["calls"] == 1

    again = LLMClient(LLMMode.CACHED, cache_root=first.cache.root, audit_path=None)
    agents = Agents.build(again, documents_dir=data_dir / "documents")
    replay: list[dict] = []
    run_case(
        Pipeline(ctx, agents), again, "ACC-T001", lambda **e: replay.append(e), lambda p: "approve"
    )
    recorded = next(e for e in replay if e.get("node") == "vision")
    assert recorded["source"] == "recorded"
    assert recorded["seconds"] == round(again.calls[0]["seconds"], 2)
    assert replay[-1]["case"] == events[-1]["case"]


def test_stream_route_sends_events_and_ends(make_client):
    c = make_client()
    started = c.post("/api/live", json={"account_id": "ACC-T001", "mode": "replay"})
    assert started.status_code == 200 and started.json()["position"] == 0
    text = c.get(f"/api/live/{started.json()['job_id']}/events").text
    events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: {"')]
    assert events[0]["type"] == "queued" and events[-1]["type"] == "done"
    assert "event: end" in text
    assert c.get("/api/live/nope/events").status_code == 404
    assert c.post("/api/live", json={"account_id": "ACC-X", "mode": "replay"}).status_code == 404


def test_replay_is_open_to_a_visitor_who_may_decide_only_a_real_pause(make_client):
    c = make_client(client=VISITOR)
    started = c.post("/api/live", json={"account_id": "ACC-T001", "mode": "replay"})
    assert started.status_code == 200
    job = started.json()["job_id"]
    assert c.post(f"/api/live/{job}/decision", json={"decision": "block"}).status_code == 409


def test_full_queue_is_a_429(make_client, pausing):
    jobs = pausing()
    c = make_client(jobs=jobs)
    first = c.post("/api/live", json={"account_id": "ACC-T001", "mode": "replay"}).json()
    _wait_for(jobs.get(first["job_id"]), "paused")
    for _ in range(2):
        assert (
            c.post("/api/live", json={"account_id": "ACC-T002", "mode": "replay"}).status_code
            == 200
        )
    full = c.post("/api/live", json={"account_id": "ACC-T002", "mode": "replay"})
    assert full.status_code == 429 and "3 ahead" in full.json()["detail"]
    assert (
        c.post(f"/api/live/{first['job_id']}/decision", json={"decision": "approve"}).status_code
        == 200
    )
    assert threading.active_count() >= 1


def test_node_weights_use_measured_seconds_per_call():
    summary = {"model_calls": {"rag (gemma)": {"seconds_per_call": 28.01}}}
    weights = node_weights(summary)
    assert weights["rag"] == 28.01 and weights["judge"] == 0.3 and set(weights) == set(NODES)
    assert set(node_weights({}).values()) == {0.3}
