// The frame around every page: loading screen, header, top loading bar, tour ribbon, footer.
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import { useApp } from "./app";
import type { Health, Row } from "./lib/types";
import { Icon, Modal, VerdictBadge } from "./ui";

export function BootScreen({ health, error }: { health: Health | null; error?: string }) {
  const stages = health?.stages ?? [];
  const at = health?.stage ?? 0;
  const pct = stages.length ? Math.round((at / stages.length) * 100) : 6;
  return (
    <div className="boot">
      <div className="boot-card">
        <div className="logo-mark big" />
        <h1>AML Auditor</h1>
        {error ? (
          <div className="state state-error" role="alert">
            <p>{error}</p>
            <button className="btn" onClick={() => location.reload()}>
              Try again
            </button>
          </div>
        ) : (
          <>
            <p className="muted">Opening the bank. This takes a few seconds, once.</p>
            <div className="bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct} aria-label="Loading the bank">
              <div className="bar-fill striped" style={{ width: `${Math.max(pct, 6)}%` }} />
            </div>
            <ul className="boot-steps">
              {stages.map((s, i) => (
                <li key={s} className={i < at ? "done" : i === at ? "now" : ""}>
                  <span className="tick">{i < at ? <Icon name="check" size={14} /> : i === at ? <span className="spinner" /> : null}</span>
                  {s}
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  );
}

const NAV: [string, string][] = [
  ["/", "Overview"],
  ["/how-it-works", "How it works"],
  ["/live", "Live run"],
  ["/cases", "Cases"],
  ["/results", "Results"],
  ["/admin", "Admin"],
];

function StatusPill() {
  const { caps } = useApp();
  const [tone, text] = caps.static
    ? ["replay", "Replay build"]
    : caps.live_available
      ? ["ok", "Models online"]
      : caps.presenter
        ? ["warn", "Models offline, replay only"]
        : ["replay", "Replay only"];
  return (
    <span className={`pill pill-${tone}`} title={caps.reason ?? caps.hardware}>
      <i />
      {text}
    </span>
  );
}

function Palette({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { source, run } = useApp();
  const go = useNavigate();
  const [q, setQ] = useState("");
  const [rows, setRows] = useState<Row[]>([]);
  const [at, setAt] = useState(0);
  useEffect(() => {
    if (open && run) void source.cases(run, { page_size: 500 }).then((p) => setRows(p.rows), () => setRows([]));
    if (open) (setQ(""), setAt(0));
  }, [open, run, source]);
  const hits = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const pages = NAV.filter(([, label]) => label.toLowerCase().includes(needle)).map(([to, label]) => ({ to, label, row: undefined as Row | undefined }));
    const accounts = rows
      .filter((r) => !needle || `${r.account} ${r.customer_name}`.toLowerCase().includes(needle))
      .slice(0, 8)
      .map((r) => ({ to: `/cases/${r.account}`, label: `${r.account} · ${r.customer_name}`, row: r }));
    return needle ? [...accounts, ...pages] : [...pages, ...accounts];
  }, [q, rows]);
  const pick = (to: string) => (onClose(), go(to));
  return (
    <Modal open={open} title="Jump to a page or an account" onClose={onClose}>
      <input
        className="input palette-input"
        placeholder="Type an account id, a customer name or a page…"
        value={q}
        onChange={(e) => (setQ(e.target.value), setAt(0))}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") (e.preventDefault(), setAt((i) => Math.min(i + 1, hits.length - 1)));
          if (e.key === "ArrowUp") (e.preventDefault(), setAt((i) => Math.max(i - 1, 0)));
          if (e.key === "Enter" && hits[at]) pick(hits[at].to);
        }}
        aria-label="Search"
      />
      <ul className="palette-list">
        {hits.map((h, i) => (
          <li key={h.to}>
            <button className={i === at ? "on" : ""} onClick={() => pick(h.to)} onMouseEnter={() => setAt(i)} tabIndex={-1}>
              <span>{h.label}</span>
              {h.row ? <VerdictBadge verdict={h.row.verdict} size="sm" /> : <span className="muted">page</span>}
            </button>
          </li>
        ))}
        {!hits.length && <li className="muted pad">Nothing matches.</li>}
      </ul>
    </Modal>
  );
}

function Header() {
  const { runs, run, setRun, guide, setGuide, theme, setTheme, caps } = useApp();
  const [palette, setPalette] = useState(false);
  const [menu, setMenu] = useState(false);
  const { pathname } = useLocation();
  useEffect(() => {
    setMenu(false);
  }, [pathname]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") (e.preventDefault(), setPalette(true));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const dark = theme === "dark" || (theme === "auto" && matchMedia("(prefers-color-scheme: dark)").matches);
  return (
    <header className="header">
      <div className="header-in">
        <Link to="/" className="logo" aria-label="AML Auditor, overview">
          <span className="logo-mark" />
          <span>
            AML Auditor <small>v2</small>
          </span>
        </Link>
        <button className="icon-btn nav-toggle" onClick={() => setMenu((m) => !m)} aria-label="Menu" aria-expanded={menu}>
          <Icon name={menu ? "x" : "dot"} />
          Menu
        </button>
        <nav className={`nav ${menu ? "open" : ""}`} aria-label="Sections">
          {NAV.map(([to, label]) => (
            <NavLink key={to} to={to} end={to === "/"}>
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="header-tools">
          <button className="search-btn" onClick={() => setPalette(true)} aria-label="Search accounts and pages">
            <Icon name="search" size={16} />
            <span>Search</span>
            <kbd>Ctrl K</kbd>
          </button>
          {runs.length > 1 && (
            <select className="input select-sm" value={run} onChange={(e) => setRun(e.target.value)} aria-label="Saved run">
              {runs.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.label}
                  {r.has_scoreboard ? " (graded)" : ""}
                </option>
              ))}
            </select>
          )}
          <StatusPill />
          <label className="switch" title="Plain-word notes on every page for someone new to the project">
            <input type="checkbox" checked={guide} onChange={(e) => setGuide(e.target.checked)} />
            <i />
            <span>Guide</span>
          </label>
          <button className="icon-btn" onClick={() => setTheme(dark ? "light" : "dark")} aria-label={dark ? "Light theme" : "Dark theme"}>
            <Icon name={dark ? "sun" : "moon"} />
          </button>
        </div>
      </div>
      {caps.static && (
        <div className="banner">
          <Icon name="info" size={16} /> Replay build{caps.recorded ? `, recorded ${caps.recorded}` : ""}. Everything here is a saved run: no server, no model, read-only.
        </div>
      )}
      <Palette open={palette} onClose={() => setPalette(false)} />
    </header>
  );
}

export interface TourStep {
  title: string;
  text: string;
  to: string;
}

export function useTourSteps(): TourStep[] {
  const { board } = useApp();
  const first = board.data?.scoreboard?.show_cases[0]?.account;
  return [
    { title: "See the idea", text: "One picture of the five agents and the Python judge.", to: "/how-it-works" },
    { title: "Watch one case run", text: "Press Run and follow the progress bar to the verdict.", to: first ? `/live/${first}` : "/live" },
    { title: "Read the case file", text: "The verdict, why, the evidence and the report.", to: first ? `/cases/${first}` : "/cases" },
    { title: "Check the scoreboard", text: "How often the system was right, and where it was not.", to: "/results" },
  ];
}

function TourRibbon() {
  const { tour, setTour } = useApp();
  const steps = useTourSteps();
  const go = useNavigate();
  if (tour === null) return null;
  const step = steps[Math.min(tour, steps.length - 1)];
  const move = (to: number) => {
    if (to >= steps.length) return setTour(null);
    setTour(to);
    go(steps[to].to);
  };
  return (
    <div className="tour" role="region" aria-label="Guided tour">
      <div className="tour-dots" aria-hidden="true">
        {steps.map((s, i) => (
          <i key={s.title} className={i < tour ? "done" : i === tour ? "now" : ""} />
        ))}
      </div>
      <div className="tour-text">
        <b>
          Tour {tour + 1} of {steps.length}: {step.title}
        </b>
        <span>{step.text}</span>
      </div>
      <div className="tour-actions">
        {tour > 0 && (
          <button className="btn btn-ghost" onClick={() => move(tour - 1)}>
            Back
          </button>
        )}
        <button className="btn btn-primary" onClick={() => move(tour + 1)}>
          {tour + 1 < steps.length ? "Next" : "Finish"} <Icon name="arrow" size={16} />
        </button>
        <button className="icon-btn" onClick={() => setTour(null)} aria-label="End the tour">
          <Icon name="x" />
        </button>
      </div>
    </div>
  );
}

function TopBar() {
  const { busy } = useApp();
  const [show, setShow] = useState(false);
  const timer = useRef<number>();
  // Only show the bar when a load takes longer than a blink; hide it a moment after it ends.
  useEffect(() => {
    clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setShow(busy), busy ? 120 : 250);
    return () => clearTimeout(timer.current);
  }, [busy]);
  return <div className={`topbar ${show ? "on" : ""}`} aria-hidden="true" />;
}

export function Shell({ children }: { children: ReactNode }) {
  const { caps, run, tour } = useApp();
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return (
    <div className={`shell ${tour !== null ? "with-tour" : ""}`}>
      <a className="skip" href="#main">
        Skip to the page
      </a>
      <TopBar />
      <Header />
      <main id="main" className="main">
        {children}
      </main>
      <footer className="footer">
        <span>
          <Icon name="cpu" size={15} /> Local only. {caps.hardware}
        </span>
        <span className="mono">{run ? `run ${run.replace(/~/g, " / ")}` : "no saved run"}</span>
      </footer>
      <TourRibbon />
    </div>
  );
}
