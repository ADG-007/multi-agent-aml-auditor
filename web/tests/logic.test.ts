import { describe, expect, it } from "vitest";
import { clock, nextNode, paceMs, progress } from "../src/lib/progress";
import { StaticSource, caseQueryString, keyPath, pageRows } from "../src/lib/source";
import type { PipelineEvent, Row } from "../src/lib/types";

const NODES = ["triage", "vision", "judge", "admin_gate", "auditor"];
const WEIGHTS = { triage: 1, vision: 4, judge: 1, admin_gate: 1, auditor: 13 };
const done = (node: string, seconds: number, note?: string): PipelineEvent => ({ seq: 0, type: "node_done", node, seconds, note });
const started: PipelineEvent = { seq: 0, type: "started" };

describe("progress", () => {
  it("is 0 before anything ran and 1 after done", () => {
    expect(progress(NODES, WEIGHTS, [], [], null, 0).fraction).toBe(0);
    expect(progress(NODES, WEIGHTS, [done("triage", 1), { seq: 0, type: "done" }], [], null, 0).fraction).toBe(1);
  });
  it("counts finished nodes against the estimate of the rest", () => {
    const p = progress(NODES, WEIGHTS, [done("triage", 1), done("vision", 4)], [], null, 0);
    expect(p.fraction).toBeCloseTo(5 / 20);
    expect(p.leftSeconds).toBe(15);
  });
  it("adds the running node's time but never its whole estimate", () => {
    const half = progress(NODES, WEIGHTS, [done("triage", 1)], [], "vision", 2);
    expect(half.fraction).toBeCloseTo(3 / 20);
    const late = progress(NODES, WEIGHTS, [done("triage", 1)], [], "vision", 60);
    expect(late.overdue).toBe(true);
    expect(late.fraction).toBeCloseTo((1 + 4 * 0.95) / 20);
  });
  it("uses the recorded seconds of answers already received (replay)", () => {
    const p = progress(NODES, WEIGHTS, [done("triage", 1)], [done("auditor", 33), done("admin_gate", 0, "skipped")], null, 0);
    expect(p.leftSeconds).toBeCloseTo(4 + 1 + 0 + 33);
  });
  it("stays under 100 percent until the done event", () => {
    const all = NODES.map((n) => done(n, 1));
    expect(progress(NODES, WEIGHTS, all, [], null, 0).fraction).toBe(0.99);
  });
});

describe("pacing", () => {
  it("replays a 20 s node in 2 s at 10x, with a floor and a ceiling", () => {
    expect(paceMs(20, "fast")).toBe(2000);
    expect(paceMs(0.1, "fast")).toBe(350);
    expect(paceMs(300, "fast")).toBe(4000);
    expect(paceMs(20, "real")).toBe(20000);
    expect(paceMs(20, "instant")).toBe(0);
  });
  it("knows which node a live run is on", () => {
    expect(nextNode(NODES, [])).toBeNull();
    expect(nextNode(NODES, [started])).toBe("triage");
    expect(nextNode(NODES, [started, done("triage", 1)])).toBe("vision");
    expect(nextNode(NODES, [started, done("judge", 1), { seq: 0, type: "paused" }])).toBeNull();
    expect(nextNode(NODES, [started, done("judge", 1), { seq: 0, type: "paused" }, { seq: 0, type: "resumed" }])).toBe("admin_gate");
  });
  it("formats the clock", () => {
    expect(clock(75.4)).toBe("01:15");
  });
});

const row = (account: string, verdict: Row["verdict"], extra: Partial<Row> = {}): Row => ({
  account,
  case_id: `CASE-${account}-1`,
  customer_name: "Asha Verma",
  occupation: "",
  verdict,
  confidence: 0,
  label: "",
  id_pause: false,
  inputs: {},
  clauses: [],
  signed: false,
  admin_decision: null,
  ...extra,
});
const ROWS = [
  row("ACC-1", "CLEAR"),
  row("ACC-2", "MONITOR", { customer_name: "Vikram Sethi", confidence: 0.5, id_pause: true }),
  row("ACC-3", "SUSPICIOUS-ESCALATE", { confidence: 0.9, label: "RING" }),
];

describe("rows in a replay build", () => {
  it("filters like the server", () => {
    expect(pageRows(ROWS, { q: "vikram" }).rows.map((r) => r.account)).toEqual(["ACC-2"]);
    expect(pageRows(ROWS, { verdicts: ["CLEAR", "MONITOR"] }).total).toBe(2);
    expect(pageRows(ROWS, { id_pause: true }).total).toBe(1);
    expect(pageRows(ROWS, { label: "RING" }).rows[0].account).toBe("ACC-3");
    expect(pageRows(ROWS, {}).counts).toEqual({ CLEAR: 1, MONITOR: 1, "SUSPICIOUS-ESCALATE": 1 });
  });
  it("sorts and pages", () => {
    expect(pageRows(ROWS, { sort: "confidence", order: "desc" }).rows[0].account).toBe("ACC-3");
    expect(pageRows(ROWS, { sort: "verdict", order: "desc" }).rows[0].account).toBe("ACC-3");
    const second = pageRows(ROWS, { page: 2, page_size: 2 });
    expect(second.rows.map((r) => r.account)).toEqual(["ACC-3"]);
    expect(second.total).toBe(3);
  });
  it("builds the server query string", () => {
    expect(caseQueryString({ verdicts: ["CLEAR", "MONITOR"], q: "a b", page: 2 })).toBe("verdict=CLEAR&verdict=MONITOR&q=a+b&page=2");
  });
});

describe("static source", () => {
  it("maps keys to script files", () => {
    expect(keyPath("case:demo~tuning:ACC-0044")).toBe("data/case/demo/tuning/ACC-0044.js");
  });
  it("loads a data script once and reads what it stored", async () => {
    const store: Record<string, unknown> = {};
    const loaded: string[] = [];
    const source = new StaticSource(store, async (path) => {
      loaded.push(path);
      store["cases:r"] = ROWS;
    });
    const [a, b] = await Promise.all([source.cases("r", {}), source.cases("r", { q: "vikram" })]);
    expect(a.total).toBe(3);
    expect(b.total).toBe(1);
    expect(loaded).toEqual(["data/cases/r.js"]);
  });
  it("replays a recording up to the pause, then the chosen branch", async () => {
    const head: PipelineEvent[] = [{ seq: 1, type: "started" }, { seq: 2, type: "paused" }];
    const timeline = {
      approve: [...head, { seq: 3, type: "done", message: "approved" } as PipelineEvent],
      block: [...head, { seq: 3, type: "done", message: "blocked" } as PipelineEvent],
    };
    const source = new StaticSource({ "timeline:r:ACC-1": timeline }, async () => {});
    const seen: PipelineEvent[] = [];
    let closed = 0;
    const handle = await source.startLive({ account: "ACC-1", mode: "replay", run: "r" }, (e) => seen.push(e), () => closed++);
    await new Promise((r) => setTimeout(r, 5));
    expect(seen.map((e) => e.type)).toEqual(["started", "paused"]);
    expect(closed).toBe(0);
    await handle.decide("block");
    await new Promise((r) => setTimeout(r, 5));
    expect(seen[2].message).toBe("blocked");
    expect(closed).toBe(1);
  });
  it("refuses writes", async () => {
    const source = new StaticSource({}, async () => {});
    await expect(source.signOff()).rejects.toThrow("Read-only replay.");
  });
});
