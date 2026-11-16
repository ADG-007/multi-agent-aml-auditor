import { useState } from "react";
import { Link } from "react-router-dom";
import { useApp, useAsync } from "../app";
import { Crumbs, EmptyState, ErrorState, Explain, Icon, IdCards, Skeleton, Term, VerdictBadge, fmt2 } from "../ui";

export default function Admin() {
  const { source, run, caps } = useApp();
  const queue = useAsync(() => source.admin(run), [run]);
  const [problem, setProblem] = useState<string>();
  const locked = caps.static ? "Read-only replay." : !caps.presenter ? "Only the presenter's laptop can decide." : null;
  const decide = async (account: string, choice: "approve" | "block") => {
    try {
      await source.setDecision(run, account, choice);
      setProblem(undefined);
      queue.reload();
    } catch (e) {
      setProblem((e as Error).message);
    }
  };
  return (
    <>
      <Crumbs items={[["Overview", "/"], ["Admin queue"]]} />
      <header className="page-head">
        <h1>Admin queue</h1>
        <p className="lead">Cases the ID check paused. A human looks at the card and approves or blocks.</p>
      </header>
      <Explain>
        When both reads of an ID card disagree with the bank's record, the system does not guess. It stops and asks. This <Term k="pause">ID pause</Term> never changes the score. A
        decision made here is saved beside the run; the saved report is not rewritten (run the case live to see a block take effect).
      </Explain>
      {problem && <p className="error-text">{problem}</p>}
      {queue.error ? (
        <ErrorState error={queue.error} onRetry={queue.reload} />
      ) : !queue.data ? (
        <Skeleton height={260} />
      ) : !queue.data.length ? (
        <EmptyState title="No case is waiting for the admin in this run" />
      ) : (
        <div className="adminlist">
          {queue.data.map((item) => (
            <article className="card admincard" key={item.account_id}>
              <header>
                <div>
                  <Link className="mono" to={`/cases/${item.account_id}`}>
                    {item.account_id}
                  </Link>{" "}
                  <b>{item.customer_name}</b>
                  <div className="muted small">
                    ID check {fmt2(item.forgery_confidence)} · verdict if approved <VerdictBadge verdict={item.disposition_if_approved} size="sm" />
                  </div>
                </div>
                <span className={`chip ${item.decision === "block" ? "chip-escalate" : item.decision === "approve" ? "chip-clear" : ""}`}>
                  {item.decision ? `decided: ${item.decision}` : `waiting (batch default: ${item.batch_decision ?? "none"})`}
                </span>
              </header>
              <IdCards documents={item.documents} discrepancies={item.field_discrepancies} />
              <footer className="modal-actions">
                {locked && <span className="muted small">{locked}</span>}
                <button className="btn btn-danger" disabled={!!locked || item.decision === "block"} onClick={() => decide(item.account_id, "block")}>
                  <Icon name="lock" size={16} /> Block
                </button>
                <button className="btn btn-primary" disabled={!!locked || item.decision === "approve"} onClick={() => decide(item.account_id, "approve")}>
                  <Icon name="check" size={16} /> Approve
                </button>
              </footer>
            </article>
          ))}
        </div>
      )}
    </>
  );
}
