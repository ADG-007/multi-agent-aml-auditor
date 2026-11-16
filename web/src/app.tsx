// App-wide state: the data source, the chosen run, the beginner guide switch, the theme.
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { HttpSource, StaticSource, type DataSource } from "./lib/source";
import type { Capabilities, Health, RunInfo, RunSummary } from "./lib/types";

declare global {
  interface Window {
    __AML__?: Record<string, unknown>;
  }
}

export const IS_STATIC = typeof window !== "undefined" && !!window.__AML__;

function useRemember<T extends string>(key: string, fallback: T): [T, (v: T) => void] {
  const read = (): T => {
    try {
      return (localStorage.getItem(key) as T) ?? fallback;
    } catch {
      return fallback;
    }
  };
  const [value, set] = useState<T>(read);
  const save = (v: T) => {
    set(v);
    try {
      localStorage.setItem(key, v);
    } catch {
      /* private window */
    }
  };
  return [value, save];
}

export interface Async<T> {
  data?: T;
  error?: Error;
  loading: boolean;
  reload: () => void;
}

export function useAsync<T>(load: () => Promise<T>, deps: unknown[]): Async<T> {
  const [state, setState] = useState<{ data?: T; error?: Error; loading: boolean }>({ loading: true });
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let live = true;
    setState((s) => ({ data: s.data, loading: true }));
    load().then(
      (data) => live && setState({ data, loading: false }),
      (error) => live && setState({ error, loading: false }),
    );
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  return { ...state, reload: useCallback(() => setTick((t) => t + 1), []) };
}

interface AppState {
  source: DataSource;
  caps: Capabilities;
  runs: RunInfo[];
  run: string;
  setRun: (id: string) => void;
  board: Async<RunSummary>;
  guide: boolean;
  setGuide: (on: boolean) => void;
  theme: string;
  setTheme: (t: string) => void;
  tour: number | null;
  setTour: (step: number | null) => void;
  busy: boolean;
}

const Ctx = createContext<AppState | null>(null);
export const useApp = () => useContext(Ctx)!;

function loadScript(path: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const tag = document.createElement("script");
    tag.src = path;
    tag.onload = () => resolve();
    tag.onerror = () => reject(new Error(path));
    document.head.appendChild(tag);
  });
}

export function AppProvider({ children, boot }: { children: ReactNode; boot: (h: Health | null, error?: string) => ReactNode }) {
  const [pending, setPending] = useState(0);
  const source = useMemo<DataSource>(() => {
    const notify = (d: number) => setPending((n) => Math.max(0, n + d));
    return IS_STATIC ? new StaticSource(window.__AML__!, loadScript, notify) : new HttpSource(notify);
  }, []);

  const [health, setHealth] = useState<Health | null>(null);
  const [fatal, setFatal] = useState<string>();
  const [base, setBase] = useState<{ caps: Capabilities; runs: RunInfo[] }>();
  const [runChoice, setRunChoice] = useRemember<string>("aml.run", "");
  const [guideChoice, setGuideChoice] = useRemember<"on" | "off">("aml.guide", "on");
  const [theme, setTheme] = useRemember<string>("aml.theme", "auto");
  const [tourChoice, setTourChoice] = useRemember<string>("aml.tour", "");
  const tries = useRef(0);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  // Wait for the server to finish loading the bank, showing which stage it is at.
  useEffect(() => {
    let live = true;
    let timer: number;
    const poll = async () => {
      try {
        const h = await source.health();
        if (!live) return;
        tries.current = 0;
        setHealth(h);
        if (h.error) return setFatal(h.error);
        if (h.ready) {
          const [caps, runs] = await Promise.all([source.capabilities(), source.runs()]);
          if (live) setBase({ caps, runs });
          return;
        }
      } catch {
        if (++tries.current > 20) return setFatal("The server is not answering. Start it with: python -m src.api");
      }
      timer = window.setTimeout(poll, 350);
    };
    void poll();
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [source]);

  const run = base ? (base.runs.find((r) => r.id === runChoice)?.id ?? base.runs[0]?.id ?? "") : "";
  const board = useAsync<RunSummary>(
    () => (run ? source.summary(run) : Promise.resolve({ run: "", summary: {}, scoreboard: null })),
    [run, !!base],
  );

  if (!base) return <>{boot(health, fatal)}</>;
  const value: AppState = {
    source,
    caps: base.caps,
    runs: base.runs,
    run,
    setRun: setRunChoice,
    board,
    guide: guideChoice === "on",
    setGuide: (on) => setGuideChoice(on ? "on" : "off"),
    theme,
    setTheme,
    tour: tourChoice === "" ? null : Number(tourChoice),
    setTour: (step) => setTourChoice(step === null ? "" : String(step)),
    busy: pending > 0,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
