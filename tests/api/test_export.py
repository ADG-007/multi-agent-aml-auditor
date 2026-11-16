"""The replay build: script-wrapped data, recorded timelines, no answer key."""

import json

import pytest

from src.agents.vision import ocr
from src.api.app import Site
from src.api.export import export_site, record_timelines
from src.llm import LLMClient, LLMMode
from src.orchestration.steps import Agents
from tests.api.conftest import RUN
from tests.api.test_live import FORGED_CARD, FORGED_READ
from tests.factories import fake_client


@pytest.fixture
def dist(tmp_path):
    folder = tmp_path / "dist"
    (folder / "assets").mkdir(parents=True)
    (folder / "index.html").write_text(
        '<body>\n    <script defer src="./assets/app.js"></script>\n</body>'
    )
    (folder / "assets" / "app.js").write_text("1")
    return folder


def _load(path) -> dict:
    """What a data script puts in window.__AML__."""
    text = path.read_text(encoding="utf-8").strip().rstrip(";")
    assert text.startswith("window.__AML__")
    if text.startswith("window.__AML__="):
        return json.loads(text.removeprefix("window.__AML__="))
    key, _, body = text.removeprefix("window.__AML__[").partition("]=")
    return {json.loads(key): json.loads(body)}


def _export(config, ctx, dist, out):
    site = Site.build(config, ctx)
    client = LLMClient(LLMMode.OFF, audit_path=None)
    return export_site(site, [RUN], out, dist, client=client)


def test_export_writes_every_page_as_a_script(config, ctx, dist, tmp_path):
    out = tmp_path / "replay"
    report = _export(config, ctx, dist, out)
    assert report.accounts == 2 and report.timelines == 2
    assert report.files == len([p for p in out.rglob("*") if p.is_file()])
    boot = _load(out / "data" / "boot.js")
    assert boot["index"][0]["id"] == RUN
    assert boot["capabilities"]["static"] and not boot["capabilities"]["live_available"]
    for account in ("ACC-T001", "ACC-T002"):
        page = _load(out / "data" / "case" / "run_a" / "live" / f"{account}.js")
        assert page[f"case:{RUN}:{account}"]["report"]["target_account"] == account
        assert (out / "data" / "evidence" / "run_a" / "live" / f"{account}.js").exists()
    rows = _load(out / "data" / "cases" / "run_a" / "live.js")[f"cases:{RUN}"]
    assert [r["account"] for r in rows] == ["ACC-T001", "ACC-T002"]
    html = (out / "index.html").read_text()
    assert html.index("data/boot.js") < html.index("assets/app.js")
    assert [p.name for p in (out / "docs").iterdir()] == ["card.png"]


def test_timeline_ends_with_the_verdict_the_run_saved(config, ctx, dist, tmp_path):
    out = tmp_path / "replay"
    _export(config, ctx, dist, out)
    site = Site.build(config, ctx)
    for account in ("ACC-T001", "ACC-T002"):
        timeline = _load(out / "data" / "timeline" / "run_a" / "live" / f"{account}.js")
        events = timeline[f"timeline:{RUN}:{account}"]["default"]
        assert events[0]["type"] == "started" and events[-1]["type"] == "done"
        saved = site.store.report(RUN, account)
        assert events[-1]["case"]["disposition"] == saved.disposition.value


def test_export_holds_no_answer_key(config, ctx, dist, tmp_path):
    out = tmp_path / "replay"
    _export(config, ctx, dist, out)
    for path in out.rglob("*"):
        assert "_ground_truth" not in path.as_posix()
        if path.suffix in (".js", ".html"):
            text = path.read_text(encoding="utf-8")
            assert "_ground_truth" not in text and "evidence_ledger" not in text


def test_export_is_the_same_twice(config, ctx, dist, tmp_path):
    def snapshot(out):
        _export(config, ctx, dist, out)
        keep = [p for p in sorted(out.rglob("*.js")) if "timeline" not in p.parts]
        return {p.relative_to(out).as_posix(): p.read_bytes() for p in keep}

    assert snapshot(tmp_path / "a") == snapshot(tmp_path / "b")


def test_missing_build_is_a_clear_error(config, ctx, tmp_path):
    with pytest.raises(FileNotFoundError, match="npm run build"):
        _export(config, ctx, tmp_path / "nothing", tmp_path / "replay")


def test_paused_account_is_recorded_for_both_decisions(
    config, ctx, data_dir, tmp_path, monkeypatch
):
    monkeypatch.setattr(ocr, "tesseract", lambda path: FORGED_CARD)
    site = Site.build(config, ctx)
    client = fake_client(tmp_path, FORGED_READ)
    agents = Agents.build(client, documents_dir=data_dir / "documents")
    folder = config.outputs / "run_a" / "live"
    recorded = record_timelines(site, folder, ["ACC-T001", "ACC-T002"], client, agents)
    assert set(recorded["ACC-T002"]) == {"default"}
    approve, block = recorded["ACC-T001"]["approve"], recorded["ACC-T001"]["block"]
    stop = [e["type"] for e in approve].index("paused")
    assert [e["type"] for e in block][: stop + 1] == [e["type"] for e in approve][: stop + 1]
    assert block[-1]["case"]["disposition"] == "BLOCKED_BY_ADMIN_REVIEW"
    assert approve[-1]["case"]["disposition"] != "BLOCKED_BY_ADMIN_REVIEW"
