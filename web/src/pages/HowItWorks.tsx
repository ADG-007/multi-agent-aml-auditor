import { useEffect } from "react";
import { useLocation } from "react-router-dom";
import { useApp } from "../app";
import { GLOSSARY, INPUT_NAMES, NODE_INFO } from "../lib/copy";
import { Crumbs, Explain, Icon, Section } from "../ui";
import { Flow } from "./Overview";

export default function HowItWorks() {
  const { caps } = useApp();
  const { hash } = useLocation();
  useEffect(() => {
    if (!hash) return;
    const el = document.getElementById(hash.slice(1));
    el?.scrollIntoView({ block: "center" });
    el?.classList.add("flash");
    const t = setTimeout(() => el?.classList.remove("flash"), 1600);
    return () => clearTimeout(t);
  }, [hash]);
  const r = caps.rules;
  return (
    <>
      <Crumbs items={[["Overview", "/"], ["How it works"]]} />
      <header className="page-head">
        <h1>How it works</h1>
        <p className="lead">Models read. Python scores. A human signs. Every case goes through the same ten steps, in this order.</p>
      </header>

      <div className="legend">
        <span className="flow-kind kind-python">
          <Icon name="code" size={14} /> Python: fixed rules, same answer every time
        </span>
        <span className="flow-kind kind-model">
          <Icon name="cpu" size={14} /> Local model: reads or writes text, never scores
        </span>
      </div>
      <Flow nodes={caps.nodes} />

      <Section kicker="Step by step" title="What each step reads, and what it can never do">
        <div className="nodes">
          {caps.nodes.map((n, i) => {
            const info = NODE_INFO[n];
            if (!info) return null;
            return (
              <article className={`card node kind-${info.kind}`} id={n} key={n}>
                <div className="node-head">
                  <span className="flow-n">{i + 1}</span>
                  <div>
                    <h3>{info.label}</h3>
                    <span className={`flow-kind kind-${info.kind}`}>
                      <Icon name={info.kind === "python" ? "code" : "cpu"} size={13} /> {info.who}
                    </span>
                  </div>
                </div>
                <p>{info.what}</p>
                <dl>
                  <dt>Reads</dt>
                  <dd>{info.reads}</dd>
                  <dt>Never</dt>
                  <dd className="never">
                    <Icon name="lock" size={14} /> {info.never}
                  </dd>
                </dl>
              </article>
            );
          })}
        </div>
      </Section>

      <Section kicker="The judge" title="How Python turns findings into a verdict">
        <Explain>
          Four checks each give a number from 0 (nothing found) to 1 (certain). The judge combines them with the fixed rules below. These numbers come straight from the code that
          scores.
        </Explain>
        <div className="grid2">
          <div className="card">
            <h3>The four inputs and their weights</h3>
            <table className="table">
              <tbody>
                {Object.entries(r.weights)
                  .filter(([k]) => INPUT_NAMES[k])
                  .map(([k, w]) => (
                    <tr key={k}>
                      <td>{INPUT_NAMES[k]}</td>
                      <td className="num mono">{w}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
            <p className="muted small">The average is taken over the checks that ran and found something.</p>
          </div>
          <div className="card">
            <h3>The rules, in order</h3>
            <ol className="rules">
              <li>
                <b>Veto.</b> Sanctions or links at {r.veto} or more escalate on their own.
              </li>
              <li>
                <b>Thresholds.</b> Below {r.clear_below} is CLEAR. From {r.escalate_from} it can ESCALATE. In between is MONITOR.
              </li>
              <li>
                <b>Second signal.</b> To escalate, two checks must be at {r.corroboration_min} or more. One signal alone is held at MONITOR.
              </li>
              <li>
                <b>ID check.</b> At {r.id_pause} or more the case pauses for an admin. It never changes the score.
              </li>
              <li>
                <b>Failures.</b> If an agent fails, the case is held at MONITOR or higher. It never crashes the run.
              </li>
            </ol>
          </div>
        </div>
      </Section>

      <Section kicker="Glossary" title="The words, in plain words">
        <dl className="glossary">
          {Object.entries(GLOSSARY).map(([k, g]) => (
            <div key={k}>
              <dt>{g.term}</dt>
              <dd>{g.plain}</dd>
            </div>
          ))}
        </dl>
      </Section>
    </>
  );
}
