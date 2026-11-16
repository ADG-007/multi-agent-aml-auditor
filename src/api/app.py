"""The demo site's server.

    python -m src.api            # http://127.0.0.1:8000

It serves the built site (web/dist) and a small JSON API over the saved run folders, the
bank's own data and the live graph. The bank loads in a background thread, so the page opens
at once and shows what is loading. Only this laptop (a loopback client) may run the models or
write a decision; anyone else gets replay and read-only pages. Every model is local, on
Ollama; on this 4 GB laptop a new answer takes seconds to tens of seconds.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agents.graph.links import build_graph
from src.api.evidence import DocumentIndex, build_evidence
from src.api.live import LiveJobs, NotAllowed, QueueFull, UnknownJob, node_weights
from src.api.store import RunStore, UnknownAccount, UnknownRun
from src.bank import BankData
from src.llm import config as llm_config
from src.orchestration.pipeline import NODES
from src.rulebook import RulebookLookup
from src.stages.stage0_triage import TriageContext


LOOPBACK = ("127.0.0.1", "::1")
BOOT_STAGES = (
    "Reading accounts and transactions",
    "Loading the rulebook",
    "Building the link graph",
    "Finding saved runs",
)
HARDWARE_NOTE = (
    "Every model runs on this laptop (4 GB GPU); performance is reduced by hardware limits."
)


@dataclass
class ApiConfig:
    outputs: Path = Path("outputs")
    data_dir: Path = Path("data")
    static_dir: Path | None = Path("web/dist")
    lan: bool = False
    ollama_url: str = llm_config.OLLAMA_URL


@dataclass
class Site:
    """Everything the routes read, built once."""

    ctx: TriageContext
    store: RunStore
    documents: DocumentIndex
    jobs: LiveJobs

    @classmethod
    def build(cls, config: ApiConfig, ctx: TriageContext, jobs: LiveJobs | None = None) -> Site:
        audit = config.outputs / "audit" / "llm_calls.jsonl"
        return cls(
            ctx=ctx,
            store=RunStore(config.outputs, ctx.bank),
            documents=DocumentIndex(config.data_dir),
            jobs=jobs or LiveJobs(ctx, audit_path=audit),
        )


@dataclass
class Boot:
    stage: int = 0
    site: Site | None = None
    error: str | None = None
    started: float = field(default_factory=time.monotonic)

    def load(self, config: ApiConfig) -> None:
        try:
            bank = BankData.load(config.data_dir)
            self.stage = 1
            rulebook = RulebookLookup.load(config.data_dir / "rulebook" / "clauses.json")
            self.stage = 2
            ctx = TriageContext.build(bank, rulebook)
            build_graph(bank)
            self.stage = 3
            site = Site.build(config, ctx)
            site.store.folders()
            self.stage = len(BOOT_STAGES)
            self.site = site
        except Exception as exc:  # shown on the loading screen instead of a dead page
            self.error = f"{type(exc).__name__}: {exc}"


class Probe:
    """Is Ollama answering? Asked at most once every few seconds."""

    def __init__(self, url: str, ttl: float = 5.0) -> None:
        self.url, self.ttl = url, ttl
        self._at, self._ok = 0.0, False

    def __call__(self) -> bool:
        if time.monotonic() - self._at > self.ttl:
            try:
                self._ok = httpx.get(self.url + "/api/tags", timeout=1.0).status_code == 200
            except httpx.HTTPError:
                self._ok = False
            self._at = time.monotonic()
        return self._ok


class LiveRequest(BaseModel):
    account_id: str
    mode: str = "replay"
    run: str | None = None


class Decision(BaseModel):
    decision: str


def create_app(
    config: ApiConfig | None = None,
    ctx: TriageContext | None = None,
    jobs: LiveJobs | None = None,
    probe=None,
) -> FastAPI:
    """The app. With `ctx` the site is ready at once (tests); without, it loads in a thread."""
    config = config or ApiConfig()
    probe = probe or Probe(config.ollama_url)
    boot = Boot()
    if ctx is not None:
        boot.site, boot.stage = Site.build(config, ctx, jobs), len(BOOT_STAGES)
    else:
        threading.Thread(target=boot.load, args=(config,), daemon=True).start()
    app = FastAPI(title="AML Auditor v2", docs_url=None, redoc_url=None)
    app.state.boot = boot

    def site() -> Site:
        if boot.site is None:
            raise HTTPException(503, boot.error or "The bank is still loading.")
        return boot.site

    def presenter(request: Request) -> bool:
        return bool(request.client) and request.client.host in LOOPBACK

    def writer(request: Request) -> None:
        if not presenter(request):
            raise HTTPException(403, "Read-only: only the presenter's laptop can change a case.")

    @app.exception_handler(UnknownRun)
    async def _no_run(_, exc):
        return JSONResponse({"detail": f"No saved run {exc.args[0]!r}."}, status_code=404)

    @app.exception_handler(UnknownAccount)
    async def _no_account(_, exc):
        return JSONResponse({"detail": f"No account {exc.args[0]!r} in this run."}, 404)

    @app.exception_handler(UnknownJob)
    async def _no_job(_, exc):
        return JSONResponse({"detail": "No such live run."}, status_code=404)

    # ----------------------------------------------------------------------------------
    # Status
    # ----------------------------------------------------------------------------------

    @app.get("/api/health")
    def health():
        return {
            "ready": boot.site is not None,
            "stage": boot.stage,
            "stages": list(BOOT_STAGES),
            "error": boot.error,
            "seconds": round(time.monotonic() - boot.started, 1),
        }

    @app.get("/api/capabilities")
    def capabilities(request: Request, run: str | None = None):
        s = site()
        is_presenter, online = presenter(request), probe()
        reason = None
        if not is_presenter:
            reason = "Live model runs are only available on the presenter's laptop."
        elif not online:
            reason = "Ollama is not answering, so the local models are offline. Replay still works."
        runs = s.store.folders()
        run_id = run if run in runs else next(iter(runs), None)
        summary = s.store.summary(run_id)["summary"] if run_id else {}
        return {
            "presenter": is_presenter,
            "models_online": online,
            "live_available": is_presenter and online,
            "replay_available": True,
            "static": False,
            "lan": config.lan,
            "reason": reason,
            "nodes": list(NODES),
            "node_weights": node_weights(summary),
            "hardware": HARDWARE_NOTE,
            "queue": s.jobs.busy(),
        }

    # ----------------------------------------------------------------------------------
    # Saved runs
    # ----------------------------------------------------------------------------------

    @app.get("/api/runs")
    def runs():
        return site().store.list_runs()

    @app.get("/api/runs/{run}/summary")
    def summary(run: str):
        return site().store.summary(run)

    @app.get("/api/runs/{run}/cases")
    def cases(
        run: str,
        request: Request,
        label: str | None = None,
        id_pause: bool | None = None,
        q: str | None = None,
        sort: str = "account",
        order: str = "asc",
        page: int = 1,
        page_size: int = 25,
    ):
        verdicts = set(request.query_params.getlist("verdict")) or None
        return site().store.rows(
            run,
            verdicts=verdicts,
            label=label,
            id_pause=id_pause,
            q=q,
            sort=sort,
            order=order,
            page=page,
            page_size=min(max(page_size, 1), 500),
        )

    @app.get("/api/runs/{run}/cases/{account}")
    def case_page(run: str, account: str):
        return build_case_page(site(), run, account)

    @app.get("/api/runs/{run}/cases/{account}/evidence")
    def evidence(run: str, account: str):
        s = site()
        s.store.report(run, account)
        return build_evidence(
            s.ctx.bank, s.ctx.rulebook, s.documents, account, s.store.verdicts(run)
        )

    @app.get("/api/accounts/{account}/evidence")
    def account_evidence(account: str, run: str | None = None):
        s = site()
        try:
            verdicts = s.store.verdicts(run) if run else {}
            return build_evidence(s.ctx.bank, s.ctx.rulebook, s.documents, account, verdicts)
        except KeyError as exc:
            if isinstance(exc, UnknownRun):
                raise
            raise HTTPException(404, f"No account {account!r}.") from None

    @app.get("/api/accounts")
    def accounts():
        bank = site().ctx.bank
        return [
            {"account": a, "customer_name": bank.customer_for(a).name} for a in bank.account_ids()
        ]

    @app.get("/api/documents/{file}")
    def document(file: str):
        try:
            return FileResponse(site().documents.path(file), media_type="image/png")
        except KeyError, FileNotFoundError:
            raise HTTPException(404, "No such document.") from None

    # ----------------------------------------------------------------------------------
    # Admin decisions and sign-off
    # ----------------------------------------------------------------------------------

    @app.get("/api/runs/{run}/admin")
    def admin(run: str):
        s = site()
        return [
            {**item, "documents": s.documents.for_account(item["account_id"])}
            for item in s.store.admin_queue(run)
        ]

    @app.put("/api/runs/{run}/admin/{account}")
    def decide_admin(run: str, account: str, body: Decision, request: Request):
        writer(request)
        try:
            site().store.set_decision(run, account, body.decision)
        except ValueError:
            raise HTTPException(422, "The decision is approve or block.") from None
        return {"account": account, "decision": body.decision}

    @app.post("/api/runs/{run}/cases/{account}/signoff")
    def signoff(run: str, account: str, request: Request):
        writer(request)
        site().store.sign(run, account)
        return {"account": account, "signed": True}

    # ----------------------------------------------------------------------------------
    # Live runs
    # ----------------------------------------------------------------------------------

    @app.post("/api/live")
    def start_live(body: LiveRequest, request: Request):
        s = site()
        if body.mode not in ("replay", "live"):
            raise HTTPException(422, "The mode is replay or live.")
        if body.mode == "live":
            if not presenter(request):
                raise HTTPException(
                    403, "Live model runs are only available on the presenter's laptop."
                )
            if not probe():
                raise HTTPException(
                    403, "Ollama is not answering, so the local models are offline."
                )
        folder = s.store.folder(body.run) if body.run else None
        try:
            job, ahead = s.jobs.submit(body.account_id, body.mode, folder)
        except QueueFull as full:
            raise HTTPException(
                429, f"Another case is running ({full.args[0]} ahead). Try again in a moment."
            ) from None
        except KeyError:
            raise HTTPException(404, f"No account {body.account_id!r}.") from None
        return {"job_id": job.id, "position": ahead}

    @app.get("/api/live/{job}/events")
    def live_events(job: str, request: Request, after: int = 0):
        jobs_ = site().jobs
        jobs_.get(job)
        last = request.headers.get("last-event-id")
        start = int(last) if last and last.isdigit() else after

        def stream():
            for event in jobs_.events(job, start):
                if event is None:
                    yield ": keep-alive\n\n"
                else:
                    yield f"id: {event['seq']}\ndata: {json.dumps(event)}\n\n"
            yield "event: end\ndata: {}\n\n"

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)

    @app.post("/api/live/{job}/decision")
    def live_decision(job: str, body: Decision, request: Request):
        writer(request)
        try:
            site().jobs.decide(job, body.decision)
        except ValueError:
            raise HTTPException(422, "The decision is approve or block.") from None
        except NotAllowed as exc:
            raise HTTPException(409, str(exc)) from None
        return {"job_id": job, "decision": body.decision}

    @app.delete("/api/live/{job}")
    def live_cancel(job: str):
        site().jobs.cancel(job)
        return {"job_id": job, "cancelled": True}

    @app.get("/api/{rest:path}")
    def api_miss(rest: str):
        raise HTTPException(404, "No such API route.")

    # ----------------------------------------------------------------------------------
    # The built site
    # ----------------------------------------------------------------------------------

    static = config.static_dir
    if static is not None and (static / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

        @app.get("/{rest:path}", response_class=HTMLResponse)
        def page(rest: str):
            html = (static / "index.html").read_text(encoding="utf-8")
            return html.replace('"./assets/', '"/assets/')
    else:

        @app.get("/{rest:path}", response_class=HTMLResponse)
        def no_site(rest: str):
            return HTMLResponse(
                "<h1>The site is not built yet</h1><p>Run <code>npm install</code> and "
                "<code>npm run build</code> in the <code>web</code> folder, then reload.</p>",
                status_code=503,
            )

    return app


def build_case_page(site: Site, run: str, account: str) -> dict:
    """One saved case with what the page needs beside it: clause texts, documents, status."""
    report = site.store.report(run, account)
    clause_ids = report.rulebook_findings.retrieved_clause_ids if report.rulebook_findings else []
    clauses = {}
    for clause_id in clause_ids:
        try:
            clause = site.ctx.rulebook.clause(clause_id)
        except KeyError:
            continue
        clauses[clause_id] = {"title": clause.title, "text": clause.text}
    queue = {item["account_id"]: item for item in site.store.admin_queue(run)}
    return {
        "report": report.model_dump(mode="json"),
        "customer_name": site.ctx.bank.customer_for(account).name,
        "admin_decision": site.store.decisions(run).get(account),
        "admin_item": queue.get(account),
        "signed": report.case_id in site.store.signed(run),
        "documents": site.documents.for_account(account),
        "clauses": clauses,
    }
