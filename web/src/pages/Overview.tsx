import { Link, useNavigate } from "react-router-dom";
import { useApp } from "../app";
import { NODE_INFO, VERDICT } from "../lib/copy";
import { useTourSteps } from "../shell";
import { EmptyState, ErrorState, Explain, Icon, Section, Skeleton, Stat, Term, VerdictBadge, fmt2 } from "../ui";

export function Flow({ nodes, link = true }: { nodes: string[]; link?: boolean }) {
  return (
    <ol className="flow">
      {nodes.map((n, i) => {
        const info = NODE_INFO[n];
        const body = (
          <>
            <span className="flow-n">{i + 1}</span>
            <span className="flow-name">{info?.label ?? n}</span>
            <span className={`flow-kind kind-${info?.kind}`}>
              <Icon name={info?.kind === "python" ? "code" : "cpu"} size={13} />
              {info?.kind === "python" ? "Python" : "local model"}
            </span>
          </>
        );
        return (
          <li key={n} className={`flow-item kind-${info?.kind}`}>
            {link ? <Link to={`/how-it-works#${n}`}>{body}</Link> : <div>{body}</div>}
          </li>
        );
      })}
    </ol>
  );
}

const TRICKS: [string, string, string, string][] = [
  ["structuring", "Split the cash", "Deposit cash in pieces that stay just under the amount the bank must report.", "flag"],
  ["ring", "Hide in a group", "Run many accounts from the same phone, IP address or home, and pass money between them.", "user"],
  ["kyc", "Lie on the form", "Say you are a teacher with a small salary, then move a trader's worth of money.", "book"],
];

export default function Overview() {
  const { board, caps, setTour, run } = useApp();
  const steps = useTourSteps();
  const go = useNavigate();
  const sb = board.data?.scoreboard;
  const m = sb?.metrics;
  const startTour = () => (setTour(0), go(steps[0].to));
  return (
    <>
      <section className="hero">
        <div className="hero-text">
          <div className="kicker">Anti-money-laundering review, on one laptop</div>
          <h1>
            Five local agents read every account. <em>Python makes the call.</em>
          </h1>
          <p className="lead">
            AML Auditor reviews a bank account the way an analyst would: it reads the ID card, the customer's profile, the transactions, the sanctions list and the rulebook, then
            writes a case file. Small local models do the reading. Fixed Python rules do the scoring, so the same evidence always gives the same verdict.
          </p>
          <div className="hero-actions">
            <button className="btn btn-primary btn-lg" onClick={startTour}>
              <Icon name="play" size={18} /> Start the 4-step tour
            </button>
            <Link className="btn btn-lg" to="/live">
              Watch a case run
            </Link>
          </div>
          <p className="hero-note">
            <Icon name="lock" size={14} /> Nothing leaves this machine. {caps.hardware}
          </p>
        </div>
        <div className="hero-card" aria-hidden="true">
          <div className="hero-card-top">
            <span className="mono">case file</span>
            <span className="badge badge-escalate badge-sm">
              <Icon name="alert" size={14} /> ESCALATE
            </span>
          </div>
          {["Transaction pattern", "KYC mismatch", "Sanctions and news", "Links"].map((label, i) => (
            <div className="hero-row" key={label}>
              <span>{label}</span>
              <div className="scorebar">
                <div className={`scorebar-fill tone-${i < 2 ? "escalate" : "clear"} grow`} style={{ width: i < 2 ? "85%" : "3%", animationDelay: `${i * 0.25}s` }} />
              </div>
            </div>
          ))}
          <div className="hero-card-foot">
            <Icon name="code" size={14} /> scored by Python · explained by the agents
          </div>
          <small>illustration</small>
        </div>
      </section>

      <Section kicker="The problem, in plain words" title="Three ways dirty money hides in a bank">
        <div className="grid3">
          {TRICKS.map(([k, title, text, icon]) => (
            <article className="card trick" key={k}>
              <span className="trick-icon">
                <Icon name={icon} size={22} />
              </span>
              <h3>{title}</h3>
              <p>{text}</p>
              <p className="muted">
                The word for it: <Term k={k} />
              </p>
            </article>
          ))}
        </div>
        <Explain title="New to this?">
          A bank must spot these patterns and file a report (an <Term k="str" />). A human analyst needs hours per account. This project shows how small AI models, kept on a short leash
          by plain Python rules, can prepare the case file for the analyst. Words with a dotted underline explain themselves when you point at them.
        </Explain>
      </Section>

      <Section
        kicker="The proof"
        title={sb ? `Results on ${sb.accounts} test accounts` : "Results"}
        aside={
          <Link className="btn btn-ghost" to="/results">
            Full results <Icon name="arrow" size={16} />
          </Link>
        }
      >
        {board.loading && !board.data ? (
          <div className="grid4">
            {[0, 1, 2, 3].map((i) => (
              <Skeleton key={i} height={128} />
            ))}
          </div>
        ) : board.error ? (
          <ErrorState error={board.error} onRetry={board.reload} />
        ) : m ? (
          <>
            <div className="grid4">
              <Stat label="Exact verdict" value={<>{m.exact_key}<small> / {m.accounts}</small></>} sub="accounts given exactly the answer key's label" />
              <Stat tone="clear" label="Bad accounts caught" value={<>{m.caught}<small> / {m.bad}</small></>} sub="planted launderers that were flagged" />
              <Stat label={<Term k="falsealarm">False alarms</Term>} value={<>{m.false_alarms}<small> / {m.clean}</small></>} sub="clean accounts flagged by mistake" />
              <Stat label="Forged IDs paused" value={<>{m.fake_ids_paused}<small> / {m.fake_ids}</small></>} sub="fake ID cards stopped for a human" />
            </div>
            <p className="muted small">
              Measured on the <Term k="tuning">tuning set</Term> of run <span className="mono">{run.replace(/~/g, " / ")}</span>. The data is synthetic, so the{" "}
              <Term k="key">answer key</Term> is known.
            </p>
          </>
        ) : (
          <EmptyState title="No scoreboard for this run" hint="This run was not graded. Pick a graded run in the header, or run the evaluation script." />
        )}
      </Section>

      <Section kicker="The pipeline" title="How one case flows">
        <Flow nodes={caps.nodes} />
        <Explain>
          Each box is one step. <b>Python</b> steps follow fixed rules. <b>Local model</b> steps use a small AI model on this laptop to read or write text. Click a box to see what it
          reads and what it is never allowed to do.
        </Explain>
      </Section>

      {sb && sb.show_cases.length > 0 && (
        <Section kicker="Try it" title="Show cases">
          <div className="cases-grid">
            {sb.show_cases.map((c) => (
              <article className={`card showcase tone-${VERDICT[c.got].tone}`} key={c.account}>
                <div className="showcase-top">
                  <span className="mono">{c.account}</span>
                  <VerdictBadge verdict={c.got} size="sm" />
                </div>
                <h3>{c.title}</h3>
                <p className="muted">
                  {VERDICT[c.got].plain} Confidence {fmt2(c.confidence)}.
                </p>
                <div className="showcase-actions">
                  <Link className="btn btn-sm" to={`/cases/${c.account}`}>
                    Open case
                  </Link>
                  <Link className="btn btn-sm btn-ghost" to={`/live/${c.account}`}>
                    <Icon name="play" size={14} /> Watch it run
                  </Link>
                </div>
              </article>
            ))}
          </div>
        </Section>
      )}
    </>
  );
}
