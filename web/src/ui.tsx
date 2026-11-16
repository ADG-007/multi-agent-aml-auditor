// Shared pieces: icons, verdict badge, score bar, stat tile, glossary term, states, modal.
import { useEffect, useRef, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { useApp } from "./app";
import { GLOSSARY, VERDICT } from "./lib/copy";
import type { Disposition, Step } from "./lib/types";

const PATHS: Record<string, string> = {
  check: "M5 12.5l4.5 4.5L19 7.5",
  x: "M6 6l12 12M18 6L6 18",
  alert: "M12 8v5m0 3.5v.5M10.3 3.9L2.6 17.5A2 2 0 004.3 20.5h15.4a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z",
  eye: "M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12zm10 3a3 3 0 100-6 3 3 0 000 6z",
  play: "M7 4.5v15l12-7.5z",
  arrow: "M5 12h14m-6-6l6 6-6 6",
  back: "M19 12H5m6 6l-6-6 6-6",
  search: "M11 18a7 7 0 100-14 7 7 0 000 14zm9 3l-4-4",
  sun: "M12 17a5 5 0 100-10 5 5 0 000 10zM12 2v2m0 16v2M4.2 4.2l1.4 1.4m12.8 12.8l1.4 1.4M2 12h2m16 0h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4",
  moon: "M20 14.5A8.5 8.5 0 019.5 4a7 7 0 1010.5 10.5z",
  lock: "M7 11V8a5 5 0 0110 0v3m-11 0h12a1 1 0 011 1v8a1 1 0 01-1 1H6a1 1 0 01-1-1v-8a1 1 0 011-1z",
  cpu: "M9 3v3m6-3v3M9 18v3m6-3v3M3 9h3m-3 6h3m12-6h3m-3 6h3M7 6h10a1 1 0 011 1v10a1 1 0 01-1 1H7a1 1 0 01-1-1V7a1 1 0 011-1zm3 4h4v4h-4z",
  code: "M8 7l-5 5 5 5m8-10l5 5-5 5M14 4l-4 16",
  clock: "M12 21a9 9 0 100-18 9 9 0 000 18zm0-14v5l3 2",
  book: "M4 5a2 2 0 012-2h13v16H6a2 2 0 00-2 2V5zm0 14a2 2 0 012-2h13",
  user: "M12 12a4 4 0 100-8 4 4 0 000 8zm-8 9a8 8 0 0116 0",
  info: "M12 21a9 9 0 100-18 9 9 0 000 18zm0-10v6m0-9v.5",
  pause: "M8 5v14M16 5v14",
  flag: "M5 21V4m0 1h11l-2 4 2 4H5",
  dot: "M12 13a1 1 0 100-2 1 1 0 000 2z",
};

export function Icon({ name, size = 18 }: { name: string; size?: number }) {
  return (
    <svg className="icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={PATHS[name] ?? PATHS.dot} />
    </svg>
  );
}

const VERDICT_ICON: Record<string, string> = { clear: "check", monitor: "eye", escalate: "alert", blocked: "lock" };

export function VerdictBadge({ verdict, size = "md" }: { verdict: Disposition | string; size?: "sm" | "md" | "lg" }) {
  const meta = VERDICT[verdict as Disposition] ?? { short: verdict, tone: "blocked", plain: "" };
  return (
    <span className={`badge badge-${meta.tone} badge-${size}`} title={meta.plain}>
      <Icon name={VERDICT_ICON[meta.tone]} size={size === "lg" ? 20 : 14} />
      {meta.short}
    </span>
  );
}

export function fmt2(value: number): string {
  return value.toFixed(2);
}

export function inr(value: number): string {
  return "₹" + Math.round(value).toLocaleString("en-IN");
}

export function inrShort(value: number): string {
  if (value >= 1e7) return `₹${(value / 1e7).toFixed(1)} cr`;
  if (value >= 1e5) return `₹${(value / 1e5).toFixed(1)} lakh`;
  if (value >= 1e3) return `₹${Math.round(value / 1e3)}k`;
  return `₹${Math.round(value)}`;
}

/** A 0 to 1 bar. Marks are drawn at the given values (the judge's thresholds). */
export function ScoreBar({ value, marks = [], label, tone }: { value: number; marks?: number[]; label?: string; tone?: string }) {
  const v = Math.min(Math.max(value, 0), 1);
  return (
    <div className="scorebar" role="meter" aria-valuemin={0} aria-valuemax={1} aria-valuenow={v} aria-label={label ?? "score"}>
      <div className={`scorebar-fill ${tone ? `tone-${tone}` : ""}`} style={{ width: `${v * 100}%` }} />
      {marks.map((m) => (
        <i key={m} className="scorebar-mark" style={{ left: `${m * 100}%` }} title={`threshold ${m}`} />
      ))}
    </div>
  );
}

export function toneFor(value: number, rules: { clear_below: number; escalate_from: number }): string {
  return value >= rules.escalate_from ? "escalate" : value >= rules.clear_below ? "monitor" : "clear";
}

export function Stat({ label, value, sub, tone }: { label: ReactNode; value: ReactNode; sub?: ReactNode; tone?: string }) {
  return (
    <div className={`stat ${tone ? `stat-${tone}` : ""}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

/** A glossary word: dotted underline, plain-word meaning on hover or focus. */
export function Term({ k, children }: { k: string; children?: ReactNode }) {
  const entry = GLOSSARY[k];
  if (!entry) return <>{children}</>;
  return (
    <span className="term" tabIndex={0}>
      {children ?? entry.term}
      <span className="term-pop" role="tooltip">
        <b>{entry.term}</b>
        {entry.plain}
      </span>
    </span>
  );
}

/** A "what am I looking at" note. Shown only while the beginner guide is on. */
export function Explain({ title = "What am I looking at?", children }: { title?: string; children: ReactNode }) {
  const { guide } = useApp();
  if (!guide) return null;
  return (
    <aside className="explain">
      <Icon name="info" size={18} />
      <div>
        <b>{title}</b> {children}
      </div>
    </aside>
  );
}

export function Skeleton({ lines = 3, height }: { lines?: number; height?: number }) {
  if (height) return <div className="skeleton" style={{ height }} aria-hidden="true" />;
  return (
    <div className="skeleton-lines" aria-busy="true" aria-label="Loading">
      {Array.from({ length: lines }, (_, i) => (
        <div key={i} className="skeleton" style={{ width: `${92 - ((i * 17) % 40)}%` }} />
      ))}
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: Error; onRetry?: () => void }) {
  return (
    <div className="state state-error" role="alert">
      <Icon name="alert" size={28} />
      <h3>That did not load</h3>
      <p>{error.message}</p>
      {onRetry && (
        <button className="btn" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, hint, action }: { title: string; hint?: ReactNode; action?: ReactNode }) {
  return (
    <div className="state">
      <Icon name="search" size={28} />
      <h3>{title}</h3>
      {hint && <p>{hint}</p>}
      {action}
    </div>
  );
}

export function Section({ id, title, kicker, children, aside }: { id?: string; title: ReactNode; kicker?: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section id={id} className="section">
      <header className="section-head">
        <div>
          {kicker && <div className="kicker">{kicker}</div>}
          <h2>{title}</h2>
        </div>
        {aside}
      </header>
      {children}
    </section>
  );
}

export function Trail({ steps }: { steps: Step[] }) {
  if (!steps.length) return <p className="muted">No steps recorded.</p>;
  return (
    <ol className="trail">
      {steps.map((s, i) => (
        <li key={i}>
          <div className="trail-step">{s.step}</div>
          <div className="trail-finding">{s.finding}</div>
          <div className="trail-impact">{s.impact}</div>
        </li>
      ))}
    </ol>
  );
}

export function Modal({ open, title, onClose, children, wide }: { open: boolean; title: string; onClose?: () => void; children: ReactNode; wide?: boolean }) {
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const before = document.activeElement as HTMLElement | null;
    box.current?.querySelector<HTMLElement>("button, [href], input, select")?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && onClose) onClose();
      if (e.key !== "Tab" || !box.current) return;
      const items = box.current.querySelectorAll<HTMLElement>("button:not([disabled]), [href], input, select");
      if (!items.length) return;
      const [first, last] = [items[0], items[items.length - 1]];
      if (e.shiftKey && document.activeElement === first) (e.preventDefault(), last.focus());
      else if (!e.shiftKey && document.activeElement === last) (e.preventDefault(), first.focus());
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      before?.focus();
    };
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && onClose?.()}>
      <div className={`modal ${wide ? "modal-wide" : ""}`} role="dialog" aria-modal="true" aria-label={title} ref={box}>
        <header>
          <h2>{title}</h2>
          {onClose && (
            <button className="icon-btn" onClick={onClose} aria-label="Close">
              <Icon name="x" />
            </button>
          )}
        </header>
        {children}
      </div>
    </div>
  );
}

export function IdCards({ documents, discrepancies }: { documents: { type: string; file: string }[]; discrepancies: string[] }) {
  const { source } = useApp();
  if (!documents.length) return <p className="muted">No ID image on file.</p>;
  return (
    <div className="idcards">
      <div className="idcards-row">
        {documents.map((d) => (
          <figure key={d.file} className={discrepancies.some((x) => x.startsWith(d.type)) ? "flagged" : ""}>
            <img src={source.documentUrl(d.file)} alt={`${d.type} card on file`} loading="lazy" />
            <figcaption>
              {d.type}
              {discrepancies.some((x) => x.startsWith(d.type)) && <span className="chip chip-escalate">disagrees with the record</span>}
            </figcaption>
          </figure>
        ))}
      </div>
      {discrepancies.length > 0 && (
        <ul className="callouts">
          {discrepancies.map((line) => (
            <li key={line}>
              <Icon name="flag" size={16} />
              {line}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function Crumbs({ items }: { items: [string, string?][] }) {
  return (
    <nav className="crumbs" aria-label="Breadcrumb">
      {items.map(([label, to], i) => (
        <span key={label}>
          {i > 0 && <span className="crumbs-sep">/</span>}
          {to ? <Link to={to}>{label}</Link> : <span aria-current="page">{label}</span>}
        </span>
      ))}
    </nav>
  );
}
