import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useApp, useAsync } from "../app";
import { CashChart, LinkGraph } from "../charts";
import { INPUT_NAMES, NODE_INFO, VERDICT } from "../lib/copy";
import type { CasePage, Evidence, Report } from "../lib/types";
import { Crumbs, EmptyState, ErrorState, Explain, Icon, IdCards, ScoreBar, Section, Skeleton, Term, Trail, VerdictBadge, fmt2, inr, toneFor } from "../ui";

const AGENTS: [string, string][] = [
  ["vision", "vision"],
  ["web_research", "web_research"],
  ["graph", "graph_label"],
  ["rag", "rag"],
  ["auditor", "auditor"],
];
const NAV: [string, string][] = [
  ["verdict", "Verdict"],
  ["why", "Why"],
  ["evidence", "Evidence"],
  ["agents", "Agents"],
  ["report", "Report"],
  ["signoff", "Sign-off"],
];

/** Report text with [CLAUSE-001: title] references drawn as chips that show the clause. */
function WithClauses({ text, clauses }: { text: string; clauses: CasePage["clauses"] }) {
  const parts = text.split(/(\[[A-Z]+-\d+[^\]]*\])/g);
  return (
    <>
      {parts.map((part, i) => {
        const id = /^\[([A-Z]+-\d+)/.exec(part)?.[1];
        return id ? <ClauseChip key={i} id={id} clause={clauses[id]} /> : <span key={i}>{part}</span>;
      })}
    </>
  );
}

function ClauseChip({ id, clause }: { id: string; clause?: { title: string; text: string } }) {
  return (
    <span className="term clause" tabIndex={0}>
      {id}
      {clause && (
        <span className="term-pop" role="tooltip">
          <b>{clause.title}</b>
          {clause.text}
        </span>
      )}
    </span>
  );
}

export function ReportBody({ report, clauses }: { report: Report; clauses: CasePage["clauses"] }) {
  return (
    <div className="prose">
      <h4>Summary</h4>
      <p>
        <WithClauses text={report.executive_summary} clauses={clauses} />
      </p>
      <h4>Narrative</h4>
      <p>
        <WithClauses text={report.investigation_narrative} clauses={clauses} />
      </p>
      <h4>Conclusion</h4>
      <p>
        <WithClauses text={report.compliance_conclusion} clauses={clauses} />
      </p>
      {report.str_narrative && (
        <>
          <h4>
            <Term k="str">STR draft</Term>
          </h4>
          <p className="str">
            <WithClauses text={report.str_narrative} clauses={clauses} />
          </p>
        </>
      )}
    </div>
  );
}

export function Why({ report }: { report: Report }) {
  const { caps } = useApp();
  const r = caps.rules;
  return (
    <div className="grid2">
      <div className="card">
        <h3>The four checks the judge read</h3>
        {Object.keys(INPUT_NAMES).map((k) => {
          const ran = k in report.judge_inputs;
          const v = report.judge_inputs[k] ?? 0;
          return (
            <div className="input-row" key={k}>
              <div className="input-name">
                {INPUT_NAMES[k]}
                <span className="mono">{ran ? fmt2(v) : "did not run"}</span>
              </div>
              <ScoreBar value={v} marks={[r.corroboration_min, r.escalate_from, r.solo_min]} tone={toneFor(v, r)} label={INPUT_NAMES[k]} />
            </div>
          );
        })}
        <p className="muted small">
          Marks at {r.corroboration_min} (counts as a signal), {r.escalate_from} (can escalate) and {r.solo_min} (strong on its own).
        </p>
      </div>
      <div className="card">
        <h3>The judge's steps (Python, no model)</h3>
        <Trail steps={report.judge_trail} />
      </div>
    </div>
  );
}

function EvidenceTabs({ ev, page }: { ev: Evidence; page: CasePage }) {
  const [tab, setTab] = useState("id");
  const tabs: [string, string][] = [
    ["id", "ID document"],
    ["cash", "Cash by month"],
    ["links", `Links (${ev.graph.linked_total})`],
    ["payees", "Counterparties"],
    ["kyc", "KYC profile"],
  ];
  const vision = page.report.vision_findings;
  return (
    <div className="card">
      <div className="tabs" role="tablist">
        {tabs.map(([id, label]) => (
          <button key={id} role="tab" aria-selected={tab === id} className={tab === id ? "on" : ""} onClick={() => setTab(id)}>
            {label}
          </button>
        ))}
      </div>
      <div role="tabpanel" className="tabpanel">
        {tab === "id" && (
          <>
            <IdCards documents={ev.documents} discrepancies={vision?.field_discrepancies ?? []} />
            {vision && (
              <p className="muted small">
                ID check {fmt2(vision.forgery_confidence)}. A field is flagged only when both reads (text reader and image model) disagree with the bank's record.
                {!vision.field_discrepancies.length && " Every field agrees with the record."}
              </p>
            )}
          </>
        )}
        {tab === "cash" && (
          <>
            <CashChart months={ev.months} cap={ev.cap} />
            <Explain>
              A bank must report cash above the <Term k="cap">monthly cap</Term>. Honest customers land anywhere. Someone <Term k="structuring">structuring</Term> lands just under the line, month
              after month. Highlighted bars are inside that band.
            </Explain>
          </>
        )}
        {tab === "links" && (
          <>
            <LinkGraph graph={ev.graph} />
            <Explain>
              Two accounts are linked when they used the same phone, the same IP address or the same home address. A link only counts against an account when the other account was
              flagged on its own evidence. Click an account to open its case.
            </Explain>
          </>
        )}
        {tab === "payees" && (
          <table className="table">
            <thead>
              <tr>
                <th>Counterparty</th>
                <th>Country</th>
                <th className="num">Transactions</th>
                <th className="num">Total</th>
              </tr>
            </thead>
            <tbody>
              {ev.counterparties.map((c) => (
                <tr key={c.id}>
                  <td>
                    {c.name} <span className="muted mono small">{c.id}</span>
                  </td>
                  <td>
                    {c.country}
                    {c.is_foreign && <span className="chip chip-monitor">foreign</span>}
                  </td>
                  <td className="num mono">{c.count}</td>
                  <td className="num mono">{inr(c.total)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {tab === "kyc" && (
          <>
            <dl className="facts">
              <dt>Occupation</dt>
              <dd>{ev.customer.occupation}</dd>
              <dt>Declared income</dt>
              <dd>{inr(ev.customer.stated_income)} a year</dd>
              <dt>Residency</dt>
              <dd>
                {ev.customer.residency.toLowerCase()}, {ev.customer.country}
              </dd>
              <dt>Account</dt>
              <dd>
                {ev.customer.account_type.toLowerCase()}, opened {ev.customer.opened}, {ev.transactions} transactions in the window
              </dd>
            </dl>
            <blockquote>{ev.kyc_narrative || "No KYC narrative on file."}</blockquote>
          </>
        )}
      </div>
    </div>
  );
}

export default function CaseFile() {
  const { account = "" } = useParams();
  const { source, run, caps } = useApp();
  const page = useAsync(() => source.casePage(run, account), [run, account]);
  const ev = useAsync(() => source.evidence(run, account), [run, account]);
  const [signing, setSigning] = useState<string>();

  if (page.error) {
    return (
      <>
        <Crumbs items={[["Overview", "/"], ["Cases", "/cases"], [account]]} />
        <EmptyState
          title={`No case for ${account} in this run`}
          hint={page.error.message}
          action={
            <Link className="btn" to="/cases">
              Back to the cases
            </Link>
          }
        />
      </>
    );
  }
  if (!page.data) {
    return (
      <>
        <Crumbs items={[["Overview", "/"], ["Cases", "/cases"], [account]]} />
        <Skeleton height={120} />
        <div className="card">
          <Skeleton lines={8} />
        </div>
      </>
    );
  }
  const { report, clauses } = page.data;
  const signed = page.data.signed || !report.is_draft;
  const flags = [
    report.single_signal_capped && "One signal only: held at MONITOR by the second-signal gate.",
    report.evidence_incomplete && "Evidence incomplete: an agent failed, so the case is held at MONITOR or higher.",
    report.policy_gap_identified && "Policy gap: real evidence, but no rulebook clause was strong enough to cite.",
  ].filter(Boolean) as string[];
  const sign = async () => {
    setSigning("busy");
    try {
      await source.signOff(run, account);
      page.reload();
      setSigning(undefined);
    } catch (e) {
      setSigning((e as Error).message);
    }
  };

  return (
    <>
      <Crumbs items={[["Overview", "/"], ["Cases", "/cases"], [account]]} />
      <header className={`casehead tone-${VERDICT[report.disposition].tone}`}>
        <div>
          <div className="mono muted">{report.case_id}</div>
          <h1>
            {account} <span className="casehead-name">{page.data.customer_name}</span>
          </h1>
          <div className="casehead-meta">
            <span className={`chip ${signed ? "chip-clear" : ""}`}>{signed ? "SIGNED OFF" : "DRAFT"}</span>
            {report.graph_label && <span className="chip">{report.graph_label.replace(/_/g, " ").toLowerCase()}</span>}
            {page.data.admin_item && <span className="chip chip-blocked">ID check paused this case</span>}
          </div>
        </div>
        <div className="casehead-verdict">
          <VerdictBadge verdict={report.disposition} size="lg" />
          <div className="conf-big">
            <span className="mono">{fmt2(report.confidence)}</span>
            <small>
              <Term k="confidence">confidence</Term>
            </small>
          </div>
          <Link className="btn btn-sm" to={`/live/${account}`}>
            <Icon name="play" size={14} /> Watch it run
          </Link>
        </div>
      </header>
      {report.forgery_notice && (
        <div className="notice notice-warn">
          <Icon name="alert" /> {report.forgery_notice}
        </div>
      )}
      {flags.map((f) => (
        <div className="notice" key={f}>
          <Icon name="info" /> {f}
        </div>
      ))}

      <div className="casebody">
        <nav className="mininav" aria-label="Case sections">
          {NAV.map(([id, label]) => (
            <a key={id} href={`#${id}`} onClick={(e) => (e.preventDefault(), document.getElementById(id)?.scrollIntoView({ behavior: "smooth" }))}>
              {label}
            </a>
          ))}
        </nav>
        <div className="casesections">
          <Section id="verdict" title="Verdict" kicker="1">
            <div className="card verdict-card">
              <p className="verdict-plain">{VERDICT[report.disposition].plain}</p>
              <p>
                <WithClauses text={report.executive_summary} clauses={clauses} />
              </p>
              {report.triage_findings && report.triage_findings.trigger_reasons.length > 0 && (
                <>
                  <h4>Why this account was reviewed</h4>
                  <ul className="bullets">
                    {report.triage_findings.trigger_reasons.map((t) => (
                      <li key={t}>{t}</li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          </Section>

          <Section id="why" title="Why this verdict" kicker="2">
            <Explain>
              The left card shows how strongly each check fired. The right card is the judge's working, line by line. It is plain Python: you can follow every step yourself.
            </Explain>
            <Why report={report} />
          </Section>

          <Section id="evidence" title="Evidence" kicker="3">
            {ev.error ? <ErrorState error={ev.error} onRetry={ev.reload} /> : ev.data ? <EvidenceTabs ev={ev.data} page={page.data} /> : <Skeleton height={320} />}
          </Section>

          <Section id="agents" title="What each agent found" kicker="4">
            <Explain>Every agent leaves a trail: what it looked at, what it saw, and why that matters. Nothing is a black box.</Explain>
            <div className="agents">
              {AGENTS.map(([key, node], i) => {
                const steps = report.agent_trails[key] ?? [];
                const info = NODE_INFO[node];
                return (
                  <details className="card agent" key={key} open={i === 0}>
                    <summary>
                      <span>
                        <b>{info.label}</b>
                        <span className="muted small"> {info.who}</span>
                      </span>
                      <span className="chip">{steps.length} steps</span>
                    </summary>
                    <Trail steps={steps} />
                  </details>
                );
              })}
            </div>
          </Section>

          <Section id="report" title="The report" kicker="5">
            <div className="card">
              <ReportBody report={report} clauses={clauses} />
              {Object.keys(clauses).length > 0 && (
                <p className="clauses">
                  Rulebook clauses cited:{" "}
                  {Object.keys(clauses).map((id) => (
                    <ClauseChip key={id} id={id} clause={clauses[id]} />
                  ))}
                </p>
              )}
              <details className="outline">
                <summary>The outline the text was written from</summary>
                <p className="muted small">Python builds this outline. A check confirms that every number and every clause in the text appears here.</p>
                <ul>
                  {report.report_outline.map((line, i) => (
                    <li key={i} className={line.startsWith("-") ? "" : "head"}>
                      {line.replace(/^- /, "")}
                    </li>
                  ))}
                </ul>
              </details>
            </div>
          </Section>

          <Section id="signoff" title="Sign-off" kicker="6">
            <div className="card signoff">
              <p>The system recommends. A human analyst decides. A report stays a draft until it is signed off.</p>
              {page.data.admin_item && (
                <p>
                  Admin decision on the ID pause: <b>{page.data.admin_decision ?? `${page.data.admin_item.batch_decision ?? "none"} (the batch default)`}</b>.{" "}
                  <Link to="/admin">Open the admin queue</Link>
                </p>
              )}
              {signed ? (
                <span className="chip chip-clear">
                  <Icon name="check" size={14} /> Signed off
                </span>
              ) : (
                <button className="btn btn-primary" disabled={!caps.presenter || caps.static || signing === "busy"} onClick={sign}>
                  Sign off as analyst
                </button>
              )}
              {!signed && (caps.static || !caps.presenter) && <p className="muted small">{caps.static ? "Read-only replay." : "Only the presenter's laptop can sign off."}</p>}
              {signing && signing !== "busy" && <p className="error-text">{signing}</p>}
            </div>
          </Section>
        </div>
      </div>
    </>
  );
}
