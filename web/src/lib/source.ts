// Where the site's data comes from: the server (HttpSource) or the files of a replay
// export (StaticSource). Every page reads through this one interface.
import type {
  AdminItem,
  Capabilities,
  CasePage,
  CaseQuery,
  Evidence,
  Health,
  PipelineEvent,
  Row,
  RowPage,
  RunInfo,
  RunSummary,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export interface LiveHandle {
  decide(choice: "approve" | "block"): Promise<void>;
  cancel(): void;
}

export interface DataSource {
  mode: "server" | "static";
  health(): Promise<Health>;
  capabilities(run?: string): Promise<Capabilities>;
  runs(): Promise<RunInfo[]>;
  summary(run: string): Promise<RunSummary>;
  cases(run: string, query: CaseQuery): Promise<RowPage>;
  casePage(run: string, account: string): Promise<CasePage>;
  evidence(run: string, account: string): Promise<Evidence>;
  admin(run: string): Promise<AdminItem[]>;
  setDecision(run: string, account: string, decision: "approve" | "block"): Promise<void>;
  signOff(run: string, account: string): Promise<void>;
  documentUrl(file: string): string;
  startLive(
    req: { account: string; mode: "replay" | "live"; run: string },
    onEvent: (event: PipelineEvent) => void,
    onClose: (error?: string) => void,
  ): Promise<LiveHandle>;
}

// ---------------------------------------------------------------------------------------
// Rows: the same filter, sort and paging the server does (used by the static export)
// ---------------------------------------------------------------------------------------

const VERDICT_ORDER: Record<string, number> = {
  CLEAR: 0,
  MONITOR: 1,
  "SUSPICIOUS-ESCALATE": 2,
  BLOCKED_BY_ADMIN_REVIEW: 3,
};

export function pageRows(all: Row[], query: CaseQuery): RowPage {
  const labels = [...new Set(all.map((r) => r.label).filter(Boolean))].sort();
  const counts: Record<string, number> = {};
  for (const r of all) counts[r.verdict] = (counts[r.verdict] ?? 0) + 1;
  const needle = (query.q ?? "").trim().toLowerCase();
  let rows = all.filter(
    (r) =>
      (!query.verdicts?.length || query.verdicts.includes(r.verdict)) &&
      (!query.label || r.label === query.label) &&
      (query.id_pause === undefined || r.id_pause === query.id_pause) &&
      (!needle || `${r.account} ${r.customer_name}`.toLowerCase().includes(needle)),
  );
  const key = (query.sort ?? "account") as keyof Row;
  const value = (r: Row): string | number =>
    key === "verdict" ? (VERDICT_ORDER[r.verdict] ?? 9) : (r[key] as string | number);
  rows = [...rows].sort((a, b) => {
    const [x, y] = [value(a), value(b)];
    const first = x < y ? -1 : x > y ? 1 : a.account < b.account ? -1 : 1;
    return query.order === "desc" ? -first : first;
  });
  const page = Math.max(query.page ?? 1, 1);
  const size = query.page_size ?? 25;
  return { rows: rows.slice((page - 1) * size, page * size), total: rows.length, page, page_size: size, labels, counts };
}

export function caseQueryString(query: CaseQuery): string {
  const p = new URLSearchParams();
  for (const v of query.verdicts ?? []) p.append("verdict", v);
  if (query.label) p.set("label", query.label);
  if (query.id_pause !== undefined) p.set("id_pause", String(query.id_pause));
  if (query.q) p.set("q", query.q);
  if (query.sort) p.set("sort", query.sort);
  if (query.order) p.set("order", query.order);
  if (query.page) p.set("page", String(query.page));
  if (query.page_size) p.set("page_size", String(query.page_size));
  return p.toString();
}

// ---------------------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------------------

export class HttpSource implements DataSource {
  mode = "server" as const;
  constructor(private notify: (delta: number) => void = () => {}) {}

  private async call<T>(path: string, init?: RequestInit): Promise<T> {
    this.notify(1);
    try {
      const response = await fetch(path, init);
      if (!response.ok) {
        let detail = response.statusText;
        try {
          detail = (await response.json()).detail ?? detail;
        } catch {
          /* not JSON */
        }
        throw new ApiError(response.status, String(detail));
      }
      return (await response.json()) as T;
    } catch (error) {
      if (error instanceof ApiError) throw error;
      throw new ApiError(0, "The server is not answering. Is `python -m src.api` still running?");
    } finally {
      this.notify(-1);
    }
  }

  private send<T>(method: string, path: string, body?: unknown): Promise<T> {
    return this.call<T>(path, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  }

  health = () => fetch("/api/health").then((r) => r.json() as Promise<Health>);
  capabilities = (run?: string) =>
    this.call<Capabilities>(`/api/capabilities${run ? `?run=${encodeURIComponent(run)}` : ""}`);
  runs = () => this.call<RunInfo[]>("/api/runs");
  summary = (run: string) => this.call<RunSummary>(`/api/runs/${run}/summary`);
  cases = (run: string, query: CaseQuery) =>
    this.call<RowPage>(`/api/runs/${run}/cases?${caseQueryString(query)}`);
  casePage = (run: string, account: string) => this.call<CasePage>(`/api/runs/${run}/cases/${account}`);
  evidence = (run: string, account: string) =>
    this.call<Evidence>(`/api/accounts/${account}/evidence?run=${encodeURIComponent(run)}`);
  admin = (run: string) => this.call<AdminItem[]>(`/api/runs/${run}/admin`);
  setDecision = async (run: string, account: string, decision: "approve" | "block") => {
    await this.send("PUT", `/api/runs/${run}/admin/${account}`, { decision });
  };
  signOff = async (run: string, account: string) => {
    await this.send("POST", `/api/runs/${run}/cases/${account}/signoff`);
  };
  documentUrl = (file: string) => `/api/documents/${encodeURIComponent(file)}`;

  async startLive(
    req: { account: string; mode: "replay" | "live"; run: string },
    onEvent: (event: PipelineEvent) => void,
    onClose: (error?: string) => void,
  ): Promise<LiveHandle> {
    const { job_id } = await this.send<{ job_id: string }>("POST", "/api/live", {
      account_id: req.account,
      mode: req.mode,
      run: req.run || null,
    });
    const stream = new EventSource(`/api/live/${job_id}/events`);
    let failures = 0;
    stream.onmessage = (message) => {
      failures = 0;
      onEvent(JSON.parse(message.data) as PipelineEvent);
    };
    stream.addEventListener("end", () => {
      stream.close();
      onClose();
    });
    stream.onerror = () => {
      // EventSource reconnects by itself and resends Last-Event-ID; give up after a few tries.
      if (++failures > 5) {
        stream.close();
        onClose("The connection to the server was lost.");
      }
    };
    return {
      decide: async (choice) => {
        await this.send("POST", `/api/live/${job_id}/decision`, { decision: choice });
      },
      cancel: () => {
        stream.close();
        void fetch(`/api/live/${job_id}`, { method: "DELETE" });
      },
    };
  }
}

// ---------------------------------------------------------------------------------------
// Replay export: data files are scripts that fill window.__AML__ (fetch is blocked on file://)
// ---------------------------------------------------------------------------------------

type Store = Record<string, unknown>;
type Timeline = Record<string, PipelineEvent[]>;

export function keyPath(key: string): string {
  return `data/${key.replace(/[:~]/g, "/")}.js`;
}

export class StaticSource implements DataSource {
  mode = "static" as const;
  private loading = new Map<string, Promise<unknown>>();
  constructor(
    private store: Store,
    private loadScript: (path: string) => Promise<void>,
    private notify: (delta: number) => void = () => {},
  ) {}

  private get<T>(key: string): Promise<T> {
    if (key in this.store) return Promise.resolve(this.store[key] as T);
    let pending = this.loading.get(key);
    if (!pending) {
      this.notify(1);
      pending = this.loadScript(keyPath(key))
        .then(() => {
          if (!(key in this.store)) throw new ApiError(404, "This replay build does not hold that page.");
          return this.store[key];
        })
        .catch((error) => {
          this.loading.delete(key);
          throw error instanceof ApiError ? error : new ApiError(404, "This replay build does not hold that page.");
        })
        .finally(() => this.notify(-1));
      this.loading.set(key, pending);
    }
    return pending as Promise<T>;
  }

  health = async (): Promise<Health> => ({ ready: true, stage: 0, stages: [], error: null });
  capabilities = () => this.get<Capabilities>("capabilities");
  runs = () => this.get<RunInfo[]>("index");
  summary = (run: string) => this.get<RunSummary>(`summary:${run}`);
  cases = async (run: string, query: CaseQuery) => pageRows(await this.get<Row[]>(`cases:${run}`), query);
  casePage = (run: string, account: string) => this.get<CasePage>(`case:${run}:${account}`);
  evidence = (run: string, account: string) => this.get<Evidence>(`evidence:${run}:${account}`);
  admin = (run: string) => this.get<AdminItem[]>(`admin:${run}`);
  setDecision = async () => {
    throw new ApiError(403, "Read-only replay.");
  };
  signOff = async () => {
    throw new ApiError(403, "Read-only replay.");
  };
  documentUrl = (file: string) => `docs/${file}`;

  async startLive(
    req: { account: string; mode: "replay" | "live"; run: string },
    onEvent: (event: PipelineEvent) => void,
    onClose: (error?: string) => void,
  ): Promise<LiveHandle> {
    let timeline: Timeline;
    try {
      timeline = await this.get<Timeline>(`timeline:${req.run}:${req.account}`);
    } catch {
      throw new ApiError(404, "This replay build has no recording for that account. Pick a flagged account or a show case.");
    }
    const first = timeline.default ?? timeline.approve;
    const stop = first.findIndex((e) => e.type === "paused");
    const head = stop < 0 ? first : first.slice(0, stop + 1);
    let cancelled = false;
    const emit = (events: PipelineEvent[], close: boolean) =>
      setTimeout(() => {
        if (cancelled) return;
        events.forEach(onEvent);
        if (close) onClose();
      }, 0);
    emit(head, stop < 0);
    return {
      decide: async (choice) => {
        emit((timeline[choice] ?? first).slice(stop + 1), true);
      },
      cancel: () => {
        cancelled = true;
      },
    };
  }
}
