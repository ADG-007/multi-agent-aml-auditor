"""The site's read routes, the presenter gate, decisions and sign-off."""

import json

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.evidence import DocumentIndex, build_evidence
from src.api.live import Job, LiveJobs
from src.api.store import RunStore, UnknownRun
from tests.api.conftest import RUN, VISITOR


def test_runs_and_summary_without_a_scoreboard(make_client):
    c = make_client()
    runs = c.get("/api/runs").json()
    assert [r["id"] for r in runs] == [RUN] and runs[0]["has_scoreboard"] is False
    body = c.get(f"/api/runs/{RUN}/summary").json()
    assert body["scoreboard"] is None and body["summary"]["accounts"] == 2


def test_scoreboard_is_served_when_the_file_exists(make_client, config):
    board = {"set": "tuning", "mode": "off", "metrics": {"accounts": 2}}
    (config.outputs / "run_a" / "live" / "scoreboard.json").write_text(json.dumps(board))
    c = make_client()
    assert c.get(f"/api/runs/{RUN}/summary").json()["scoreboard"] == board
    assert c.get("/api/runs").json()[0]["set"] == "tuning"


def test_cases_filter_search_sort_and_page(make_client):
    c = make_client()
    everything = c.get(f"/api/runs/{RUN}/cases").json()
    assert everything["total"] == 2 and sum(everything["counts"].values()) == 2
    assert c.get(f"/api/runs/{RUN}/cases?q=vikram").json()["rows"][0]["account"] == "ACC-T002"
    assert c.get(f"/api/runs/{RUN}/cases?q=acc-t001").json()["total"] == 1
    assert c.get(f"/api/runs/{RUN}/cases?verdict=BLOCKED_BY_ADMIN_REVIEW").json()["total"] == 0
    page = c.get(f"/api/runs/{RUN}/cases?page=2&page_size=1&sort=account&order=desc").json()
    assert [r["account"] for r in page["rows"]] == ["ACC-T001"] and page["total"] == 2
    assert c.get(f"/api/runs/{RUN}/cases?id_pause=true").json()["total"] == 0


def test_case_page_has_report_documents_and_draft_status(make_client):
    body = make_client().get(f"/api/runs/{RUN}/cases/ACC-T001").json()
    assert body["report"]["target_account"] == "ACC-T001"
    assert body["signed"] is False and body["admin_decision"] is None
    assert body["documents"] == [{"type": "AADHAAR", "file": "card.png"}]
    found = body["report"]["rulebook_findings"] or {}
    assert set(body["clauses"]) == set(found.get("retrieved_clause_ids", []))


def test_unknown_run_account_and_api_route_are_404(make_client):
    c = make_client()
    assert c.get("/api/runs/nope/summary").status_code == 404
    assert c.get("/api/runs/..~..~secret/cases").status_code == 404
    assert c.get(f"/api/runs/{RUN}/cases/ACC-X").status_code == 404
    assert c.get("/api/accounts/ACC-X/evidence").status_code == 404
    assert c.get("/api/nothing/here").status_code == 404


def test_run_id_cannot_leave_the_outputs_folder(config, ctx):
    store = RunStore(config.outputs, ctx.bank)
    with pytest.raises(UnknownRun):
        store.folder("..~..~secret")


def test_document_route_serves_listed_files_only(make_client):
    c = make_client()
    assert c.get("/api/documents/card.png").content == b"image"
    assert c.get("/api/documents/manifest.json").status_code == 404
    assert c.get("/api/documents/..%2F..%2Fcustomers.json").status_code == 404


def test_presenter_and_models_show_in_capabilities(make_client):
    here = make_client().get("/api/capabilities").json()
    assert here["presenter"] and here["live_available"] and here["reason"] is None
    assert set(here["node_weights"]) == set(here["nodes"])
    away = make_client(client=VISITOR).get("/api/capabilities").json()
    assert not away["presenter"] and not away["live_available"] and away["replay_available"]
    off = make_client(online=False).get("/api/capabilities").json()
    assert off["presenter"] and not off["live_available"] and "offline" in off["reason"]


def test_decisions_and_signoff_persist(make_client, config):
    c = make_client()
    assert c.put(f"/api/runs/{RUN}/admin/ACC-T001", json={"decision": "maybe"}).status_code == 422
    assert c.put(f"/api/runs/{RUN}/admin/ACC-T001", json={"decision": "block"}).status_code == 200
    saved = json.loads((config.outputs / "run_a" / "live" / "admin_decisions.json").read_text())
    assert saved == {"ACC-T001": "block"}
    assert c.post(f"/api/runs/{RUN}/cases/ACC-T001/signoff").status_code == 200
    page = c.get(f"/api/runs/{RUN}/cases/ACC-T001").json()
    assert page["signed"] is True and page["admin_decision"] == "block"
    assert c.get(f"/api/runs/{RUN}/cases?q=T001").json()["rows"][0]["signed"] is True


def test_writes_are_refused_for_a_visitor(make_client):
    c = make_client(client=VISITOR)
    assert c.put(f"/api/runs/{RUN}/admin/ACC-T001", json={"decision": "block"}).status_code == 403
    assert c.post(f"/api/runs/{RUN}/cases/ACC-T001/signoff").status_code == 403
    assert c.post("/api/live", json={"account_id": "ACC-T001", "mode": "live"}).status_code == 403


def test_live_is_refused_when_the_models_are_offline(make_client):
    c = make_client(online=False)
    assert c.post("/api/live", json={"account_id": "ACC-T001", "mode": "live"}).status_code == 403


def test_health_is_ready_and_a_missing_build_says_so(make_client):
    c = make_client()
    assert c.get("/api/health").json()["ready"] is True
    assert c.get("/cases/ACC-T001").status_code == 503


def test_built_site_is_served_on_any_page_path(config, ctx, tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text('<script src="./assets/app.js"></script>')
    (dist / "assets" / "app.js").write_text("1")
    config.static_dir = dist
    c = TestClient(create_app(config, ctx=ctx, probe=lambda: True))
    assert '"/assets/app.js"' in c.get("/cases/ACC-T001").text
    assert c.get("/assets/app.js").text == "1"
    assert c.get("/api/missing").status_code == 404


def test_evidence_months_cap_counterparties_and_graph(ctx, data_dir):
    docs = DocumentIndex(data_dir)
    ev = build_evidence(ctx.bank, ctx.rulebook, docs, "ACC-T002", {"ACC-T001": "MONITOR"})
    august = next(m for m in ev["months"] if m["month"] == "2026-08")
    assert august["cash_in"] == 40_000 and august["credits"] == 40_900
    cap = ctx.rulebook.monthly_structuring()
    assert ev["cap"] == {"threshold_inr": cap.threshold_inr, "band_floor_inr": cap.band_floor_inr}
    totals = [c["total"] for c in ev["counterparties"]]
    assert totals == sorted(totals, reverse=True) and len(totals) <= 8
    nodes = {n["id"]: n for n in ev["graph"]["nodes"]}
    assert nodes["ACC-T002"]["is_target"] and nodes["ACC-T001"]["verdict"] == "MONITOR"
    assert nodes["device:DEV-RING"]["kind"] == "device"
    edge = {"source": "ACC-T001", "target": "device:DEV-RING", "relation": "device"}
    assert edge in ev["graph"]["edges"]
    assert ev["documents"] == []


def test_document_index_refuses_unlisted_names(data_dir):
    with pytest.raises(KeyError):
        DocumentIndex(data_dir).path("../../customers.json")


def test_only_the_presenter_can_cancel_a_live_model_run(make_client, ctx):
    jobs = LiveJobs(ctx)
    jobs._jobs["job-1"] = Job("job-1", "ACC-T001", "live", None)
    assert make_client(client=VISITOR, jobs=jobs).delete("/api/live/job-1").status_code == 403
    assert jobs.get("job-1").cancel is False
    assert make_client(jobs=jobs).delete("/api/live/job-1").status_code == 200
    assert jobs.get("job-1").cancel is True
