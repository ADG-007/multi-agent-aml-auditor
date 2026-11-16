// Drawings: monthly cash against the cap, the link graph, the confusion grid.
import { useNavigate } from "react-router-dom";
import { VERDICT } from "./lib/copy";
import type { Disposition, Evidence, Month } from "./lib/types";
import { inr, inrShort } from "./ui";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const monthName = (m: string) => `${MONTHS[Number(m.slice(5, 7)) - 1]} ${m.slice(2, 4)}`;

export function CashChart({ months, cap }: { months: Month[]; cap: Evidence["cap"] }) {
  if (!months.length) return <p className="muted">No transactions in the review window.</p>;
  const top = Math.max(...months.map((m) => m.cash_in));
  const showCap = top >= cap.band_floor_inr * 0.4;
  const max = showCap ? Math.max(top, cap.threshold_inr) * 1.12 : Math.max(top, 1) * 1.3;
  const [w, h, left, bottom, topPad] = [720, 280, 64, 34, 14];
  const plotH = h - bottom - topPad;
  const y = (v: number) => topPad + plotH * (1 - v / max);
  const slot = (w - left - 12) / months.length;
  const bar = Math.min(slot * 0.56, 72);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => t * max);
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Cash deposited each month, against the monthly cash cap">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={left} x2={w - 8} y1={y(t)} y2={y(t)} className="grid" />
            <text x={left - 8} y={y(t) + 4} textAnchor="end" className="axis">
              {inrShort(t)}
            </text>
          </g>
        ))}
        {showCap && (
          <>
            <rect x={left} width={w - left - 8} y={y(cap.threshold_inr)} height={y(cap.band_floor_inr) - y(cap.threshold_inr)} className="band" />
            <line x1={left} x2={w - 8} y1={y(cap.threshold_inr)} y2={y(cap.threshold_inr)} className="capline" />
            <text x={w - 12} y={y(cap.threshold_inr) - 6} textAnchor="end" className="caplabel">
              report above {inrShort(cap.threshold_inr)} a month
            </text>
          </>
        )}
        {months.map((m, i) => {
          const x = left + slot * i + (slot - bar) / 2;
          const inBand = m.cash_in >= cap.band_floor_inr && m.cash_in <= cap.threshold_inr;
          return (
            <g key={m.month}>
              <rect x={x} y={y(m.cash_in)} width={bar} height={Math.max(y(0) - y(m.cash_in), 1)} rx="5" className={inBand ? "bar bar-hot" : "bar"}>
                <title>{`${monthName(m.month)}: ${inr(m.cash_in)} cash in, ${m.count} transactions`}</title>
              </rect>
              {m.cash_in > 0 && (
                <text x={x + bar / 2} y={y(m.cash_in) - 6} textAnchor="middle" className="barlabel">
                  {inrShort(m.cash_in)}
                </text>
              )}
              <text x={x + bar / 2} y={h - 12} textAnchor="middle" className="axis">
                {monthName(m.month)}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption>
        <span className="key key-bar" /> cash deposited in the month
        {showCap ? (
          <>
            <span className="key key-hot" /> month inside the band just under the cap
            <span className="key key-band" /> the band ({inrShort(cap.band_floor_inr)} to {inrShort(cap.threshold_inr)})
          </>
        ) : (
          <span className="muted"> · the monthly cap ({inrShort(cap.threshold_inr)}) is far above this account's cash, so it is off the chart</span>
        )}
      </figcaption>
    </figure>
  );
}

const KIND_LETTER: Record<string, string> = { device: "D", ip: "IP", address: "A" };
const KIND_NAME: Record<string, string> = { device: "shared device", ip: "shared IP address", address: "shared home address" };

export function LinkGraph({ graph }: { graph: Evidence["graph"] }) {
  const go = useNavigate();
  const target = graph.nodes.find((n) => n.is_target);
  const shared = graph.nodes.filter((n) => n.kind !== "account");
  const others = graph.nodes.filter((n) => n.kind === "account" && !n.is_target);
  if (!target || !others.length) {
    return <p className="muted">This account shares no device, IP address or home address with any other account.</p>;
  }
  const [w, h, cx, cy] = [720, 420, 360, 210];
  const pos = new Map<string, [number, number]>([[target.id, [cx, cy]]]);
  shared.forEach((n, i) => {
    const a = (i / shared.length) * Math.PI * 2 - Math.PI / 2;
    pos.set(n.id, [cx + Math.cos(a) * 105, cy + Math.sin(a) * 82]);
  });
  // Each account sits outside the shared values it touches.
  const angle = (id: string) => {
    const touched = graph.edges.filter((e) => e.source === id).map((e) => pos.get(e.target)!);
    const [x, y] = touched.reduce(([sx, sy], [px, py]) => [sx + px - cx, sy + py - cy], [0, 0]);
    return Math.atan2(y, x);
  };
  [...others]
    .sort((a, b) => angle(a.id) - angle(b.id))
    .forEach((n, i, all) => {
      const a = (i / all.length) * Math.PI * 2 + angle(all[0].id);
      pos.set(n.id, [cx + Math.cos(a) * 290, cy + Math.sin(a) * 172]);
    });
  const tone = (v: Disposition | null) => (v ? VERDICT[v].tone : "unknown");
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`Accounts that share a device, IP address or home address with ${target.id}`}>
        {graph.edges.map((e, i) => {
          const [a, b] = [pos.get(e.source), pos.get(e.target)];
          return a && b ? <line key={i} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} className={`edge edge-${e.relation}`} /> : null;
        })}
        {shared.map((n) => {
          const [x, y] = pos.get(n.id)!;
          return (
            <g key={n.id} className="gshared">
              <title>{`${KIND_NAME[n.kind]}: ${n.label}`}</title>
              <rect x={x - 15} y={y - 12} width="30" height="24" rx="7" />
              <text x={x} y={y + 4} textAnchor="middle">
                {KIND_LETTER[n.kind]}
              </text>
            </g>
          );
        })}
        {[target, ...others].map((n) => {
          const [x, y] = pos.get(n.id)!;
          return (
            <g
              key={n.id}
              className={`gacct tone-${tone(n.verdict)} ${n.is_target ? "gtarget" : ""}`}
              tabIndex={n.is_target ? -1 : 0}
              role={n.is_target ? undefined : "link"}
              aria-label={`${n.id}, ${n.label}, ${n.verdict ? VERDICT[n.verdict].short : "not in this run"}`}
              onClick={() => !n.is_target && go(`/cases/${n.id}`)}
              onKeyDown={(e) => e.key === "Enter" && !n.is_target && go(`/cases/${n.id}`)}
            >
              <title>{`${n.id} · ${n.label} · ${n.verdict ? VERDICT[n.verdict].short : "not in this run"}`}</title>
              <circle cx={x} cy={y} r={n.is_target ? 26 : 17} />
              <text x={x} y={y + (n.is_target ? 44 : 33)} textAnchor="middle">
                {n.id}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption>
        <span className="key key-target" /> this account
        <span className="key key-escalate" /> escalated
        <span className="key key-monitor" /> monitored
        <span className="key key-clear" /> clear
        <span className="key key-shared" /> D device · IP address · A home address
        {graph.linked_total > others.length && <span className="muted"> · showing {others.length} of {graph.linked_total} linked accounts</span>}
      </figcaption>
    </figure>
  );
}

export function ConfusionGrid({ confusion }: { confusion: Record<string, Record<string, number>> }) {
  const labels = (["CLEAR", "MONITOR", "SUSPICIOUS-ESCALATE", "BLOCKED_BY_ADMIN_REVIEW"] as Disposition[]).filter(
    (l) => confusion[l] || Object.values(confusion).some((row) => row[l]),
  );
  const max = Math.max(1, ...Object.values(confusion).flatMap((row) => Object.values(row)));
  return (
    <div className="confusion" role="table" aria-label="Answer key against the result">
      <div className="confusion-corner">
        answer key ↓ · result →
      </div>
      {labels.map((l) => (
        <div key={l} className="confusion-head" role="columnheader">
          {VERDICT[l].short}
        </div>
      ))}
      {labels.map((key) => (
        <div key={key} className="confusion-row" role="row">
          <div className="confusion-head" role="rowheader">
            {VERDICT[key].short}
          </div>
          {labels.map((got) => {
            const n = confusion[key]?.[got] ?? 0;
            const right = key === got;
            return (
              <div
                key={got}
                role="cell"
                className={`confusion-cell ${right ? "right" : n ? "wrong" : "zero"}`}
                style={{ ["--heat" as string]: n ? 0.18 + 0.82 * (n / max) : 0 }}
                aria-label={`Key ${VERDICT[key].short}, got ${VERDICT[got].short}: ${n}`}
              >
                {n}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}
