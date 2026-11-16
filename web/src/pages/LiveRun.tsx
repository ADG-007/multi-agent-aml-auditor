import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useApp, useAsync } from "../app";
import { NODE_INFO, SPEED_LABEL, VERDICT } from "../lib/copy";
import { clock, nextNode, paceMs, progress, roughly, type Speed } from "../lib/progress";
import type { LiveHandle } from "../lib/source";
import type { PipelineEvent } from "../lib/types";
import { Crumbs, Explain, Icon, IdCards, Modal, Section, Term, VerdictBadge, fmt2 } from "../ui";
import { ReportBody, Why } from "./CaseFile";

type Status = "idle" | "starting" | "queued" | "running" | "paused" | "done" | "error";

/** Feeds events to the screen: at once in a live run, paced by recorded seconds in a replay. */
function useRunner() {
  const [shown, setShown] = useState<PipelineEvent[]>([]);
  const [queued, setQueued] = useState<PipelineEvent[]>([]);
  const [paced, setPaced] = useState<{ node: string; since: number; ms: number; seconds: number } | null>(null);
  const [lastAt, setLastAt] = useState(0);
  const queue = useRef<PipelineEvent[]>([]);
  const timer = useRef<number>();
  const pace = useRef<{ replay: boolean; speed: Speed }>({ replay: true, speed: "fast" });

  const pump = useCallback(() => {
    if (timer.current !== undefined) return;
    while (queue.current.length) {
      const event = queue.current[0];
      const ms = event.type === "node_done" && pace.current.replay && event.note !== "skipped" ? paceMs(event.seconds ?? 0, pace.current.speed) : 0;
      const reveal = () => {
        queue.current.shift();
        setQueued([...queue.current]);
        setShown((s) => [...s, event]);
        setLastAt(performance.now());
        setPaced(null);
      };
      if (ms > 0) {
        setPaced({ node: event.node!, since: performance.now(), ms, seconds: event.seconds ?? 0 });
        timer.current = window.setTimeout(() => {
          timer.current = undefined;
          reveal();
          pump();
        }, ms);
        return;
      }
      reveal();
    }
  }, []);

  const push = useCallback(
    (event: PipelineEvent) => {
      queue.current.push(event);
      setQueued([...queue.current]);
      pump();
    },
    [pump],
  );
  const reset = useCallback((replay: boolean, speed: Speed) => {
    clearTimeout(timer.current);
    timer.current = undefined;
    queue.current = [];
    pace.current = { replay, speed };
    setShown([]);
    setQueued([]);
    setPaced(null);
    setLastAt(performance.now());
  }, []);
  useEffect(() => () => clearTimeout(timer.current), []);
  return { shown, queued, paced, lastAt, push, reset };
}

export default function LiveRun() {
  const params = useParams();
  const go = useNavigate();
  const { source, run, caps } = useApp();
  const rows = useAsync(() => (run ? source.cases(run, { page_size: 500 }) : Promise.resolve(null)), [run]);
  const [account, setAccount] = useState(params.account ?? "");
  const [mode, setMode] = useState<"replay" | "live">("replay");
  const [speed, setSpeed] = useState<Speed>("fast");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string>();
  const [startedAt, setStartedAt] = useState(0);
  const [endedAt, setEndedAt] = useState(0);
  const [now, setNow] = useState(0);
  const [deciding, setDeciding] = useState(false);
  const handle = useRef<LiveHandle>();
  const runner = useRunner();
  const { shown, queued, paced } = runner;

  useEffect(() => {
    if (params.account) setAccount(params.account);
  }, [params.account]);
  useEffect(() => {
    if (!account && rows.data?.rows.length) setAccount((rows.data.rows.find((r) => r.verdict !== "CLEAR") ?? rows.data.rows[0]).account);
  }, [rows.data, account]);
  useEffect(() => () => handle.current?.cancel(), []);

  // What the screen shows follows the events revealed so far.
  const last = shown[shown.length - 1];
  useEffect(() => {
    if (!last) return;
    if (last.type === "queued" && (last.position ?? 0) > 0) setStatus("queued");
    if (last.type === "started" || last.type === "resumed" || last.type === "node_done") setStatus("running");
    if (last.type === "paused") setStatus("paused");
    if (last.type === "done") (setStatus("done"), setEndedAt(performance.now()));
    if (last.type === "error") (setStatus("error"), setError(last.message === "abandoned" ? "Nobody decided the ID pause in time, so the case was given up." : `The run stopped: ${last.message}`));
  }, [last]);

  const active = status === "starting" || status === "queued" || status === "running";
  useEffect(() => {
    if (!active) return;
    const t = window.setInterval(() => setNow(performance.now()), 100);
    return () => clearInterval(t);
  }, [active]);

  const start = async () => {
    handle.current?.cancel();
    runner.reset(mode === "replay", speed);
    setError(undefined);
    setStatus("starting");
    setStartedAt(performance.now());
    setNow(performance.now());
    setDeciding(false);
    if (params.account !== account) go(`/live/${account}`, { replace: true });
    try {
      handle.current = await source.startLive({ account, mode, run }, runner.push, (problem) => problem && (setStatus("error"), setError(problem)));
    } catch (e) {
      setStatus("error");
      setError((e as Error).message);
    }
  };
  const decide = async (choice: "approve" | "block") => {
    setDeciding(true);
    try {
      await handle.current?.decide(choice);
    } catch (e) {
      setError((e as Error).message);
      setDeciding(false);
    }
  };
  const cancel = () => {
    handle.current?.cancel();
    setStatus("idle");
    runner.reset(mode === "replay", speed);
  };

  const replay = mode === "replay";
  const liveNode = !replay && status === "running" ? nextNode(caps.nodes, shown) : null;
  const running = paced?.node ?? liveNode;
  // In a replay the bar follows screen time, so every figure is paced the way the steps are.
  const onScreen = (seconds: number) => (replay ? paceMs(seconds, speed) / 1000 : seconds);
  const scaled = (events: PipelineEvent[]) => (replay ? events.map((e) => (e.type === "node_done" ? { ...e, seconds: onScreen(e.seconds ?? 0) } : e)) : events);
  const weights = Object.fromEntries(caps.nodes.map((n) => [n, onScreen(caps.node_weights[n] ?? 0.3)]));
  const runningSeconds = paced ? (now - paced.since) / 1000 : liveNode ? (now - runner.lastAt) / 1000 : 0;
  const p = progress(caps.nodes, weights, scaled(shown), scaled(queued), running, Math.max(runningSeconds, 0));
  const modelSeconds = shown.reduce((sum, e) => sum + (e.type === "node_done" ? (e.seconds ?? 0) : 0), 0);
  const pct = status === "idle" ? 0 : Math.round(p.fraction * 100);
  const elapsed = ((status === "done" || status === "error" ? endedAt || now : now) - startedAt) / 1000;
  const done = new Map(shown.filter((e) => e.type === "node_done").map((e) => [e.node!, e]));
  const pausedEvent = status === "paused" ? last : undefined;
  const final = shown.find((e) => e.type === "done");
  const typical = caps.nodes.reduce((s, n) => s + (caps.node_weights[n] ?? 0), 0);
  const doing = running ? NODE_INFO[running]?.doing : undefined;
  const line = doing ? doing[Math.floor((now - (paced?.since ?? runner.lastAt)) / 1700) % doing.length] : undefined;
  const idDocs = useAsync(() => (pausedEvent ? source.evidence(run, account).then((e) => e.documents) : Promise.resolve([])), [!!pausedEvent, account]);
  const known = rows.data?.rows.some((r) => r.account === account);

  return (
    <>
      <Crumbs items={[["Overview", "/"], ["Live run"]]} />
      <header className="page-head">
        <h1>Watch a case run</h1>
        <p className="lead">Pick an account and press Run. The pipeline goes step by step, and the bar shows how far it is.</p>
      </header>
      <Explain>
        <b>Replay</b> uses the answers the models gave the first time, so it is quick and always the same. <b>Live</b> asks the models on this laptop again, which takes{" "}
        {roughly(typical)} for a flagged case. Either way the Python steps run for real.
      </Explain>

      <div className="card runform">
        <label className="field">
          <span>Account</span>
          <input className="input mono" list="accounts" value={account} onChange={(e) => setAccount(e.target.value.trim().toUpperCase())} disabled={active} placeholder="ACC-…" />
          <datalist id="accounts">
            {rows.data?.rows.map((r) => (
              <option key={r.account} value={r.account}>
                {r.customer_name} · {VERDICT[r.verdict].short}
              </option>
            ))}
          </datalist>
        </label>
        <fieldset className="field">
          <legend>Mode</legend>
          <div className="seg">
            <button className={replay ? "on" : ""} onClick={() => setMode("replay")} disabled={active} aria-pressed={replay}>
              <Term k="replay">Replay</Term> saved answers
            </button>
            <button className={!replay ? "on" : ""} onClick={() => setMode("live")} disabled={active || !caps.live_available} aria-pressed={!replay} title={caps.reason ?? ""}>
              Live on local models
            </button>
          </div>
        </fieldset>
        {replay && (
          <fieldset className="field">
            <legend>Replay speed</legend>
            <div className="seg">
              {(Object.keys(SPEED_LABEL) as Speed[]).map((s) => (
                <button key={s} className={speed === s ? "on" : ""} onClick={() => setSpeed(s)} disabled={active} aria-pressed={speed === s}>
                  {SPEED_LABEL[s]}
                </button>
              ))}
            </div>
          </fieldset>
        )}
        <div className="runform-go">
          {active || status === "paused" ? (
            <button className="btn btn-lg" onClick={cancel}>
              Stop
            </button>
          ) : (
            <button className="btn btn-primary btn-lg" onClick={start} disabled={!account}>
              <Icon name="play" size={18} /> {status === "idle" ? "Run case" : "Run again"}
            </button>
          )}
        </div>
        {!caps.live_available && !caps.static && <p className="muted small runform-note">Live is off: {caps.reason}</p>}
        {replay && rows.data && account && !known && (
          <p className="muted small runform-note">This account is not in the saved run, so a replay may have no saved answers for it. Agents without an answer show as failed.</p>
        )}
      </div>

      {status !== "idle" && (
        <div className={`card runpanel status-${status}`}>
          <div className="runhead">
            <div>
              <div className="runhead-title" aria-live="polite">
                {status === "starting" && "Starting…"}
                {status === "queued" && `Waiting in line: ${last?.position} case(s) ahead`}
                {status === "running" && (running ? `${NODE_INFO[running]?.label ?? running}: ${line ?? "working"}…` : "Working…")}
                {status === "paused" && "Paused: waiting for the admin's decision"}
                {status === "done" && (final?.case ? "Done" : "Done: nothing new on this account, so no case was opened")}
                {status === "error" && "Stopped"}
              </div>
              <div className="muted small">
                {replay ? `Replaying recorded timings (${SPEED_LABEL[speed].toLowerCase()})` : "Live: measured seconds on this laptop"}
                {p.overdue && status === "running" && " · this step is taking longer than usual"}
              </div>
            </div>
            <div className="runhead-nums">
              <div>
                <span className="mono big">{pct}%</span>
                <small>{status === "done" ? "complete" : "estimate"}</small>
              </div>
              <div>
                <span className="mono big">{clock(elapsed)}</span>
                <small>elapsed</small>
              </div>
              <div>
                <span className="mono big">{status === "done" ? clock(modelSeconds) : status === "running" ? roughly(p.leftSeconds).replace("about ", "~") : "–"}</span>
                <small>{status === "done" ? (replay ? "model time, first run" : "pipeline time") : "left"}</small>
              </div>
            </div>
          </div>
          <div className="bar bar-lg" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct} aria-label="Pipeline progress">
            <div className={`bar-fill ${active ? "striped" : ""} ${status === "paused" ? "held" : ""}`} style={{ width: `${Math.max(pct, 2)}%` }} />
          </div>

          <ol className="steps">
            {caps.nodes.map((n) => {
              const info = NODE_INFO[n];
              const e = done.get(n);
              const isRunning = running === n && status === "running";
              const isPaused = status === "paused" && n === "admin_gate";
              const skipped = e?.note === "skipped";
              const state = isPaused ? "paused" : e ? (skipped ? "skipped" : "done") : isRunning ? "running" : "waiting";
              return (
                <li key={n} className={`step step-${state}`}>
                  <span className="step-dot">
                    {state === "done" && <Icon name="check" size={14} />}
                    {state === "running" && <span className="spinner" />}
                    {state === "paused" && <Icon name="pause" size={14} />}
                  </span>
                  <div className="step-main">
                    <div className="step-title">
                      <b>{info?.label ?? n}</b>
                      <span className={`flow-kind kind-${info?.kind}`}>{info?.kind === "python" ? "Python" : "local model"}</span>
                      {n === "judge" && e?.note && <VerdictBadge verdict={e.note.split(" ")[0]} size="sm" />}
                    </div>
                    <div className="step-text">
                      {state === "waiting" && info?.short}
                      {state === "running" && (line ?? info?.short)}
                      {state === "paused" && "The ID card was flagged by both reads. A human decides."}
                      {state === "skipped" && (n === "admin_gate" ? "Not needed: the ID card was not flagged." : "Not needed: no answer was unsure.")}
                      {state === "done" && (e?.detail ?? (n === "admin_gate" ? `Admin decision: ${e?.note}` : info?.short))}
                    </div>
                  </div>
                  <span className="step-time mono">
                    {e && !skipped && (e.seconds! < 0.1 ? "<0.1 s" : `${e.seconds! < 10 ? e.seconds!.toFixed(1) : Math.round(e.seconds!)} s`)}
                    {e && !skipped && e.source === "recorded" && <small>recorded</small>}
                    {isRunning && <small>typically {roughly(caps.node_weights[n] ?? 0.3).replace("about ", "")}</small>}
                  </span>
                </li>
              );
            })}
          </ol>
          {status === "error" && (
            <div className="notice notice-warn" role="alert">
              <Icon name="alert" /> {error}
              <button className="btn btn-sm" onClick={start}>
                Try again
              </button>
            </div>
          )}
        </div>
      )}

      {final?.report && final.case && (
        <Section title="The verdict" kicker="Result">
          <div className={`casehead reveal tone-${VERDICT[final.case.disposition].tone}`}>
            <div>
              <div className="mono muted">{final.case.case_id}</div>
              <h2>{account}</h2>
              <p>{VERDICT[final.case.disposition].plain}</p>
            </div>
            <div className="casehead-verdict">
              <VerdictBadge verdict={final.case.disposition} size="lg" />
              <div className="conf-big">
                <span className="mono">{fmt2(final.case.confidence)}</span>
                <small>confidence</small>
              </div>
              {known && (
                <Link className="btn btn-primary btn-sm" to={`/cases/${account}`}>
                  Open full case file <Icon name="arrow" size={14} />
                </Link>
              )}
            </div>
          </div>
          <Why report={final.report} />
          <div className="card">
            <ReportBody report={final.report} clauses={{}} />
          </div>
        </Section>
      )}

      <Modal open={!!pausedEvent} title="The ID check flagged this card" wide>
        <p>
          Both reads of the ID card disagree with the bank's record, so the pipeline stopped. The <Term k="pause">ID check</Term> never changes the score: it asks a human. If you
          approve, the case continues to{" "}
          {pausedEvent?.evidence && (
            <>
              <VerdictBadge verdict={pausedEvent.evidence.disposition_if_approved} size="sm" /> at {fmt2(pausedEvent.evidence.score_if_approved)}
            </>
          )}
          . If you block, the case ends as BLOCKED.
        </p>
        <IdCards documents={idDocs.data ?? []} discrepancies={pausedEvent?.evidence?.field_discrepancies ?? []} />
        {error && <p className="error-text">{error}</p>}
        <footer className="modal-actions">
          <button className="btn" onClick={cancel}>
            Stop the run
          </button>
          <button className="btn btn-danger" disabled={deciding} onClick={() => decide("block")}>
            <Icon name="lock" size={16} /> Block
          </button>
          <button className="btn btn-primary" disabled={deciding} onClick={() => decide("approve")}>
            <Icon name="check" size={16} /> Approve and continue
          </button>
        </footer>
      </Modal>
    </>
  );
}
