"""Write a replay build of the site: a folder that opens in a browser with no server.

    python -m src.api.export                                   # the newest graded run
    python -m src.api.export --run outputs/demo/tuning --out dist/replay

The folder holds the built site (web/dist), one data file per page, the ID images of the
exported accounts, and a recorded timeline of every account's case (both admin branches when
the ID check pauses it). Timelines are recorded with saved answers only: no model is called.
Data files are scripts that fill window.__AML__, because a browser refuses to fetch JSON from
a folder (file://). Open index.html, or copy the folder to a USB stick or any static host.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.api.app import HARDWARE_NOTE, ApiConfig, Site, build_case_page, judge_rules
from src.api.evidence import build_evidence
from src.api.live import node_weights, run_case
from src.bank import BankData
from src.llm import LLMClient, LLMMode
from src.orchestration import batch
from src.orchestration.pipeline import NODES, Pipeline
from src.orchestration.run_state import RunState
from src.orchestration.steps import Agents
from src.rulebook import RulebookLookup
from src.stages.stage0_triage import TriageContext


BOOT_TAG = '<script src="./data/boot.js"></script>'
APP_TAG = '<script defer src="./assets/app.js"></script>'


@dataclass
class ExportReport:
    files: int
    bytes: int
    accounts: int
    timelines: int


def record_timelines(
    site: Site, run_folder: Path, account_ids: list[str], client: LLMClient, agents: Agents
) -> dict[str, dict[str, list[dict]]]:
    """Account -> {"default": events}, or {"approve": events, "block": events} if it pauses."""
    cases = batch.load_cases(run_folder)
    out: dict[str, dict[str, list[dict]]] = {}

    def record(account_id: str, branch: str) -> tuple[list[dict], bool]:
        events: list[dict] = []
        state = RunState.from_cases(cases, site.ctx.bank, skip=account_id)

        def emit(**event) -> None:
            events.append({"seq": len(events) + 1, **event})

        run_case(
            Pipeline(site.ctx, agents, state=state), client, account_id, emit, lambda p: branch
        )
        return events, any(e["type"] == "paused" for e in events)

    for account_id in account_ids:
        events, paused = record(account_id, "approve")
        if paused:
            out[account_id] = {"approve": events, "block": record(account_id, "block")[0]}
        else:
            out[account_id] = {"default": events}
    return out


def export_site(
    site: Site,
    run_ids: list[str],
    out: Path,
    dist: Path,
    *,
    client: LLMClient | None = None,
    agents_factory: Callable[[LLMClient], Agents] = Agents.build,
    progress: Callable[[str], None] = lambda line: None,
) -> ExportReport:
    if not (dist / "index.html").exists():
        raise FileNotFoundError(
            f"{dist} has no built site. Run `npm install` and `npm run build` in web/ first."
        )
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(dist, out)
    html = (out / "index.html").read_text(encoding="utf-8")
    (out / "index.html").write_text(
        html.replace(APP_TAG, f"{BOOT_TAG}\n    {APP_TAG}"), encoding="utf-8"
    )
    client = client or LLMClient(LLMMode.CACHED, audit_path=None)
    agents = agents_factory(client)
    store, data = site.store, out / "data"
    recorded, documents, accounts, timelines = 0, set(), 0, 0

    def write(key: str, value) -> None:
        path = data / (key.replace(":", "/").replace("~", "/") + ".js")
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        path.write_text(f"window.__AML__[{json.dumps(key)}]={body};\n", encoding="utf-8")

    runs = [r for r in store.list_runs() if r["id"] in run_ids]
    for run in runs:
        run_id, folder = run["id"], store.folder(run["id"])
        recorded = max(recorded, run["modified"])
        rows = store.all_rows(run_id)
        verdicts = store.verdicts(run_id)
        write(f"summary:{run_id}", store.summary(run_id))
        write(f"cases:{run_id}", rows)
        queue = store.admin_queue(run_id)
        for item in queue:
            item["documents"] = site.documents.for_account(item["account_id"])
        write(f"admin:{run_id}", queue)
        ids = [r["account"] for r in rows]
        progress(f"{run_id}: {len(ids)} accounts; recording their timelines from saved answers")
        recordings = record_timelines(site, folder, ids, client, agents)
        for n, account_id in enumerate(ids, 1):
            write(f"case:{run_id}:{account_id}", build_case_page(site, run_id, account_id))
            bank, rulebook = site.ctx.bank, site.ctx.rulebook
            evidence = build_evidence(bank, rulebook, site.documents, account_id, verdicts)
            write(f"evidence:{run_id}:{account_id}", evidence)
            write(f"timeline:{run_id}:{account_id}", recordings[account_id])
            documents.update(d["file"] for d in evidence["documents"])
            if n % 25 == 0:
                progress(f"{run_id}: {n} of {len(ids)} accounts written")
        accounts += len(ids)
        timelines += sum(len(branches) for branches in recordings.values())

    first = store.summary(runs[0]["id"])["summary"] if runs else {}
    boot = {
        "index": runs,
        "capabilities": {
            "presenter": False,
            "models_online": False,
            "live_available": False,
            "replay_available": True,
            "static": True,
            "reason": "Read-only replay.",
            "nodes": list(NODES),
            "node_weights": node_weights(first),
            "hardware": HARDWARE_NOTE,
            "rules": judge_rules(),
            "recorded": time.strftime("%d %b %Y", time.localtime(recorded)) if recorded else "",
        },
    }
    data.mkdir(parents=True, exist_ok=True)
    (data / "boot.js").write_text(
        "window.__AML__=" + json.dumps(boot, separators=(",", ":")) + ";\n", encoding="utf-8"
    )
    (out / "docs").mkdir(exist_ok=True)
    for file in sorted(documents):
        shutil.copyfile(site.documents.path(file), out / "docs" / file)
    files = [p for p in out.rglob("*") if p.is_file()]
    return ExportReport(len(files), sum(p.stat().st_size for p in files), accounts, timelines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a replay build of the demo site.")
    parser.add_argument("--run", type=Path, action="append", help="a saved run folder")
    parser.add_argument("--out", type=Path, default=Path("dist/replay"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--dist", type=Path, default=Path("web/dist"))
    args = parser.parse_args(argv)
    config = ApiConfig(outputs=args.outputs)
    ctx = TriageContext.build(BankData.load(config.data_dir), RulebookLookup.load())
    site = Site.build(config, ctx)
    folders = site.store.folders()
    if args.run:
        wanted = {p.resolve() for p in args.run}
        run_ids = [run_id for run_id, folder in folders.items() if folder.resolve() in wanted]
    else:
        run_ids = [r["id"] for r in site.store.list_runs() if r["has_scoreboard"]][:1]
    if not run_ids:
        print("No saved run to export. Give one with --run outputs/<folder>.")
        return 2
    report = export_site(site, run_ids, args.out, args.dist, progress=print)
    print(
        f"Written to {args.out}: {report.files} files, {report.bytes / 1e6:.1f} MB, "
        f"{report.accounts} accounts, {report.timelines} timelines. Open {args.out / 'index.html'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
