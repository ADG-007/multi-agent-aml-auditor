import { useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useApp, useAsync } from "../app";
import { INPUT_NAMES, VERDICT } from "../lib/copy";
import type { CaseQuery, Disposition } from "../lib/types";
import { Crumbs, EmptyState, ErrorState, Explain, Icon, ScoreBar, Skeleton, Term, VerdictBadge, fmt2, toneFor } from "../ui";

const PAGE = 25;
const VERDICTS: Disposition[] = ["CLEAR", "MONITOR", "SUSPICIOUS-ESCALATE", "BLOCKED_BY_ADMIN_REVIEW"];
const COLUMNS: [string, string][] = [
  ["account", "Account"],
  ["customer_name", "Customer"],
  ["verdict", "Verdict"],
  ["confidence", "Confidence"],
];

export default function Cases() {
  const { source, run, caps } = useApp();
  const [params, setParams] = useSearchParams();
  const go = useNavigate();
  const [typed, setTyped] = useState(params.get("q") ?? "");

  const query: CaseQuery = {
    q: params.get("q") ?? undefined,
    verdicts: params.getAll("verdict"),
    label: params.get("label") ?? undefined,
    id_pause: params.get("id_pause") === "true" ? true : undefined,
    sort: params.get("sort") ?? "account",
    order: (params.get("order") as "asc" | "desc") ?? "asc",
    page: Number(params.get("page") ?? 1),
    page_size: PAGE,
  };
  const change = (edit: (p: URLSearchParams) => void, keepPage = false) => {
    const next = new URLSearchParams(params);
    edit(next);
    if (!keepPage) next.delete("page");
    setParams(next, { replace: true });
  };

  // Search waits for a pause in typing, so each key press is not a request.
  useEffect(() => {
    const t = setTimeout(() => {
      if (typed !== (params.get("q") ?? "")) change((p) => (typed ? p.set("q", typed) : p.delete("q")));
    }, 250);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [typed]);

  const page = useAsync(() => source.cases(run, query), [run, params.toString()]);
  const data = page.data;
  const filtered = !!(query.q || query.verdicts?.length || query.label || query.id_pause);
  const from = data ? (data.page - 1) * PAGE + 1 : 0;
  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE)) : 1;
  const toggleVerdict = (v: string) =>
    change((p) => {
      const has = p.getAll("verdict");
      p.delete("verdict");
      (has.includes(v) ? has.filter((x) => x !== v) : [...has, v]).forEach((x) => p.append("verdict", x));
    });
  const sortBy = (key: string) =>
    change((p) => {
      p.set("order", query.sort === key && query.order === "asc" ? "desc" : "asc");
      p.set("sort", key);
    });

  return (
    <>
      <Crumbs items={[["Overview", "/"], ["Cases"]]} />
      <header className="page-head">
        <h1>Cases</h1>
        <p className="lead">Every account in this run, with its verdict. Click a row to open the case file.</p>
      </header>
      <Explain>
        One row is one bank account. The <b>verdict</b> is the system's recommendation. The four small bars are the four checks the judge reads, each from 0 to 1. Try the{" "}
        <b>ESCALATE</b> chip to see only the serious cases.
      </Explain>

      <div className="toolbar">
        <label className="search">
          <Icon name="search" size={16} />
          <input value={typed} onChange={(e) => setTyped(e.target.value)} placeholder="Search account id or customer name" aria-label="Search cases" />
        </label>
        <div className="chips" role="group" aria-label="Verdict">
          {VERDICTS.filter((v) => data?.counts[v]).map((v) => (
            <button key={v} className={`chip chip-btn chip-${VERDICT[v].tone} ${query.verdicts?.includes(v) ? "on" : ""}`} aria-pressed={query.verdicts?.includes(v)} onClick={() => toggleVerdict(v)}>
              {VERDICT[v].short} <b>{data?.counts[v]}</b>
            </button>
          ))}
        </div>
        <select className="input select-sm" value={query.label ?? ""} onChange={(e) => change((p) => (e.target.value ? p.set("label", e.target.value) : p.delete("label")))} aria-label="Pattern label">
          <option value="">Any pattern label</option>
          {data?.labels.map((l) => (
            <option key={l}>{l}</option>
          ))}
        </select>
        <label className="check">
          <input type="checkbox" checked={!!query.id_pause} onChange={(e) => change((p) => (e.target.checked ? p.set("id_pause", "true") : p.delete("id_pause")))} />
          <Term k="pause">ID check paused</Term>
        </label>
      </div>

      {page.error ? (
        <ErrorState error={page.error} onRetry={page.reload} />
      ) : !data ? (
        <div className="card">
          <Skeleton lines={10} />
        </div>
      ) : !data.total ? (
        <EmptyState
          title="No case matches these filters"
          hint="Try fewer filters or a shorter search."
          action={
            <button className="btn" onClick={() => (setTyped(""), setParams({}, { replace: true }))}>
              Clear filters
            </button>
          }
        />
      ) : (
        <div className={`card table-card ${page.loading ? "is-loading" : ""}`}>
          <table className="table cases">
            <thead>
              <tr>
                {COLUMNS.map(([key, label]) => (
                  <th key={key} aria-sort={query.sort === key ? (query.order === "asc" ? "ascending" : "descending") : "none"}>
                    <button onClick={() => sortBy(key)}>
                      {label} <span className="sort">{query.sort === key ? (query.order === "asc" ? "↑" : "↓") : ""}</span>
                    </button>
                  </th>
                ))}
                <th>The four checks</th>
                <th>Pattern label</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((r) => (
                <tr key={r.account} onClick={() => go(`/cases/${r.account}`)} className="rowlink">
                  <td className="mono">
                    <Link to={`/cases/${r.account}`} onClick={(e) => e.stopPropagation()}>
                      {r.account}
                    </Link>
                    {r.id_pause && (
                      <span className="chip chip-blocked" title="The ID check paused this case for an admin">
                        <Icon name="pause" size={12} /> ID
                      </span>
                    )}
                  </td>
                  <td>
                    {r.customer_name}
                    <div className="muted small">{r.occupation}</div>
                  </td>
                  <td>
                    <VerdictBadge verdict={r.verdict} size="sm" />
                  </td>
                  <td className="conf">
                    <span className="mono">{fmt2(r.confidence)}</span>
                    <ScoreBar value={r.confidence} tone={toneFor(r.confidence, caps.rules)} label="confidence" />
                  </td>
                  <td>
                    <div className="mini">
                      {Object.keys(INPUT_NAMES).map((k) => (
                        <span key={k} title={`${INPUT_NAMES[k]}: ${k in r.inputs ? fmt2(r.inputs[k]) : "did not run"}`}>
                          <ScoreBar value={r.inputs[k] ?? 0} tone={toneFor(r.inputs[k] ?? 0, caps.rules)} label={INPUT_NAMES[k]} />
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="small">{r.label ? r.label.replace(/_/g, " ").toLowerCase() : <span className="muted">none</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <footer className="pager">
            <span>
              {from} to {from + data.rows.length - 1} of {data.total}
              {filtered && " (filtered)"}
            </span>
            <div>
              <button className="btn btn-sm" disabled={data.page <= 1} onClick={() => change((p) => p.set("page", String(data.page - 1)), true)}>
                Previous
              </button>
              <span className="muted">
                page {data.page} of {pages}
              </span>
              <button className="btn btn-sm" disabled={data.page >= pages} onClick={() => change((p) => p.set("page", String(data.page + 1)), true)}>
                Next
              </button>
            </div>
          </footer>
        </div>
      )}
    </>
  );
}
