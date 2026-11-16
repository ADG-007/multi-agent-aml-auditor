import { Link } from "react-router-dom";
import { useApp } from "../app";
import { ConfusionGrid } from "../charts";
import { VERDICT } from "../lib/copy";
import type { Disposition } from "../lib/types";
import { Crumbs, EmptyState, ErrorState, Explain, Section, Skeleton, Stat, Term, VerdictBadge, fmt2 } from "../ui";

const pct = (a: number, b: number) => (b ? `${((a / b) * 100).toFixed(1)}%` : "n/a");

function Frac({ a, b }: { a: number; b: number }) {
  return (
    <>
      {a}
      <small> / {b}</small>
    </>
  );
}

export default function Results() {
  const { board, run, caps } = useApp();
  if (board.error) return <ErrorState error={board.error} onRetry={board.reload} />;
  if (!board.data) return <Skeleton height={400} />;
  const { summary, scoreboard: sb } = board.data;
  const m = sb?.metrics;
  const calls = Object.entries(summary.model_calls ?? {});
  const styles = m ? Object.keys(m.structurers).sort() : [];
  return (
    <>
      <Crumbs items={[["Overview", "/"], ["Results"]]} />
      <header className="page-head">
        <h1>Results</h1>
        <p className="lead">
          How the run <span className="mono">{run.replace(/~/g, " / ")}</span> did{sb ? `, graded on the ${sb.set.replace("_", "-")} set of ${sb.accounts} accounts` : ""}.
        </p>
      </header>

      {!sb || !m ? (
        <EmptyState title="No scoreboard for this run" hint="This run was saved without a grade. The timing below is still real. Pick a graded run in the header to see the scores." />
      ) : (
        <>
          <Explain>
            The data is synthetic: launderers were planted on purpose, so the right answer for every account is known (the <Term k="key">answer key</Term>). These numbers compare
            the system's verdicts with that key. Nothing on this page is typed in: it all comes from the run's files.
          </Explain>
          <div className="grid4">
            <Stat label="Exact verdict (answer key)" value={<Frac a={m.exact_key} b={m.accounts} />} sub={pct(m.exact_key, m.accounts)} />
            <Stat label={<Term k="design">Exact verdict (design label)</Term>} value={<Frac a={m.exact_design} b={m.accounts} />} sub={pct(m.exact_design, m.accounts)} />
            <Stat tone="clear" label="Catch rate" value={<Frac a={m.caught} b={m.bad} />} sub={`${pct(m.caught, m.bad)} of bad accounts flagged`} />
            <Stat label={<Term k="falsealarm">False-alarm rate</Term>} value={<Frac a={m.false_alarms} b={m.clean} />} sub={`${pct(m.false_alarms, m.clean)} of clean accounts flagged`} />
            <Stat label="KYC mismatches found" value={<Frac a={m.kyc_found} b={m.kyc_planted} />} sub={`${m.kyc_false_high} honest profile(s) scored high by mistake`} />
            <Stat label="Cash-business traps flagged" value={<Frac a={m.cash_business_alarms} b={m.cash_business} />} sub="honest cash shops that look like structuring (reported apart)" />
            <Stat label="Forged IDs paused" value={<Frac a={m.fake_ids_paused} b={m.fake_ids} />} sub={`${m.false_pauses} genuine card(s) paused by mistake`} />
            <Stat label="Sent to the bigger model" value={`${((summary.backup_share ?? 0) * 100).toFixed(1)}%`} sub="share of accounts the 9B re-checked" />
          </div>

          <div className="grid2">
            <Section title="Answer key against result" kicker="Confusion grid">
              <div className="card">
                <ConfusionGrid confusion={sb.confusion} />
                <p className="muted small">Read a row: of the accounts whose right answer was X, how many got each verdict. The diagonal is correct.</p>
              </div>
            </Section>
            <Section title="Structurers caught, by style" kicker="Recall">
              <div className="card">
                {styles.map((s) => {
                  const [got, all] = [m.structurers_caught[s] ?? 0, m.structurers[s]];
                  return (
                    <div className="input-row" key={s}>
                      <div className="input-name">
                        {s}
                        <span className="mono">
                          {got} / {all}
                        </span>
                      </div>
                      <div className="scorebar">
                        <div className="scorebar-fill tone-clear" style={{ width: `${(got / all) * 100}%` }} />
                      </div>
                    </div>
                  );
                })}
                <p className="muted small">
                  Four ways of <Term k="structuring">structuring</Term> under the <Term k="cap">monthly cap</Term>, from obvious (naive) to disguised (mixed).
                </p>
              </div>
            </Section>
          </div>

          <Section title={`Misses (${sb.misses.length})`} kicker="Where the result differs from the answer key">
            {sb.misses.length ? (
              <div className="card table-card">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Account</th>
                      <th>Answer key</th>
                      <th>Design label</th>
                      <th>Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sb.misses.map((x) => (
                      <tr key={x.account}>
                        <td className="mono">
                          <Link to={`/cases/${x.account}`}>{x.account}</Link>
                        </td>
                        <td>
                          <VerdictBadge verdict={x.key} size="sm" />
                        </td>
                        <td>
                          <VerdictBadge verdict={x.design} size="sm" />
                        </td>
                        <td>
                          <VerdictBadge verdict={x.got} size="sm" />
                          {x.design === x.got && <span className="muted small"> matches the design label</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="muted">No miss in this run.</p>
            )}
          </Section>

          {sb.show_cases.length > 0 && (
            <Section title="Show cases" kicker="Five accounts worth opening">
              <div className="card table-card">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Account</th>
                      <th>What it is</th>
                      <th>Answer key</th>
                      <th>Result</th>
                      <th className="num">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sb.show_cases.map((c) => (
                      <tr key={c.account}>
                        <td className="mono">
                          <Link to={`/cases/${c.account}`}>{c.account}</Link>
                        </td>
                        <td>{c.title}</td>
                        <td>{VERDICT[c.key_label as Disposition]?.short ?? c.key_label}</td>
                        <td>
                          <VerdictBadge verdict={c.got} size="sm" />
                        </td>
                        <td className="num mono">{fmt2(c.confidence)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          )}
        </>
      )}

      <Section title="Speed" kicker="Seconds per model call">
        {calls.length ? (
          <div className="card table-card">
            <table className="table">
              <thead>
                <tr>
                  <th>Agent (model)</th>
                  <th className="num">Calls</th>
                  <th className="num">From saved answers</th>
                  <th className="num">Failed</th>
                  <th className="num">Seconds per call</th>
                </tr>
              </thead>
              <tbody>
                {calls.map(([name, c]) => (
                  <tr key={name}>
                    <td className="mono">{name}</td>
                    <td className="num mono">{c.calls}</td>
                    <td className="num mono">{c.from_cache}</td>
                    <td className="num mono">{c.failed}</td>
                    <td className="num mono">{c.seconds_per_call ?? "n/a"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted">This run recorded no model call.</p>
        )}
      </Section>

      <Section title="Limits and gaps" kicker="Read this before trusting the numbers">
        <ul className="card bullets limits">
          {(summary.warnings ?? []).map((w) => (
            <li key={w}>{w}</li>
          ))}
          <li>Seconds are as measured when each answer was first made on this laptop; a replay returns the same answers without the wait.</li>
          <li>{caps.hardware}</li>
          {sb?.set === "tuning" && (
            <li>
              This is the <Term k="tuning">tuning set</Term>: thresholds and prompts were set on these accounts, so these scores are the best case. The held-out accounts are the fair
              test.
            </li>
          )}
          <li>The data is synthetic. Real bank data is messier, and no result here is a claim about real customers.</li>
        </ul>
      </Section>
    </>
  );
}
