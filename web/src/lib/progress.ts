// The live run's progress bar and replay pacing. Pure functions, so they can be tested.
import type { PipelineEvent } from "./types";

export type Speed = "real" | "fast" | "instant";

/** How long a replayed node stays "running" on screen, in milliseconds. */
export function paceMs(seconds: number, speed: Speed): number {
  if (speed === "instant") return 0;
  if (speed === "real") return Math.max(seconds, 0.25) * 1000;
  return Math.min(Math.max(seconds / 10, 0.35), 4) * 1000;
}

export interface Progress {
  fraction: number; // 0 to 1
  running: string | null;
  doneSeconds: number; // pipeline seconds finished so far
  leftSeconds: number; // estimate of the pipeline seconds still to come
  overdue: boolean; // the running node has taken longer than its estimate
}

/**
 * Finished nodes count their own seconds. The running node counts its elapsed time, capped
 * just under its estimate. Nodes still to come count their estimate: the recorded seconds
 * when the answer is already known (replay), else the typical seconds from a saved run.
 */
export function progress(
  nodes: string[],
  weights: Record<string, number>,
  shown: PipelineEvent[],
  queued: PipelineEvent[],
  running: string | null,
  runningSeconds: number,
): Progress {
  if (shown.some((e) => e.type === "done")) {
    return { fraction: 1, running: null, doneSeconds: total(shown), leftSeconds: 0, overdue: false };
  }
  const finished = new Map<string, number>();
  for (const e of shown) if (e.type === "node_done" && e.node) finished.set(e.node, cost(e));
  const known = new Map<string, number>();
  for (const e of queued) if (e.type === "node_done" && e.node) known.set(e.node, cost(e));
  const estimate = (node: string) => known.get(node) ?? weights[node] ?? 0.3;

  let done = 0;
  let left = 0;
  let overdue = false;
  for (const node of nodes) {
    if (finished.has(node)) {
      done += finished.get(node)!;
    } else if (node === running) {
      const est = estimate(node);
      overdue = runningSeconds > est;
      const counted = Math.min(runningSeconds, est * 0.95);
      done += counted;
      left += est - counted;
    } else {
      left += estimate(node);
    }
  }
  const whole = done + left;
  const fraction = whole > 0 ? Math.min(done / whole, 0.99) : 0;
  return { fraction, running, doneSeconds: done, leftSeconds: left, overdue };
}

function cost(e: PipelineEvent): number {
  return e.note === "skipped" ? 0 : Math.max(e.seconds ?? 0, 0.05);
}

function total(events: PipelineEvent[]): number {
  return events.reduce((sum, e) => sum + (e.type === "node_done" ? (e.seconds ?? 0) : 0), 0);
}

/** In a live run the node after the last finished one is the one running now. */
export function nextNode(nodes: string[], shown: PipelineEvent[]): string | null {
  const last = [...shown].reverse().find((e) => e.type === "node_done");
  const state = [...shown].reverse().find((e) => e.type !== "node_done")?.type;
  if (!shown.some((e) => e.type === "started")) return null;
  if (shown.some((e) => e.type === "done" || e.type === "error")) return null;
  if (state === "paused" && shown[shown.length - 1].type === "paused") return null;
  if (!last) return nodes[0] ?? null;
  const at = nodes.indexOf(last.node ?? "");
  return nodes[at + 1] ?? null;
}

export function clock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

export function roughly(seconds: number): string {
  if (seconds < 1) return "under a second";
  if (seconds < 90) return `about ${Math.round(seconds)} s`;
  return `about ${Math.round(seconds / 60)} min`;
}
