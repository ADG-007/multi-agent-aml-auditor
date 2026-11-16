import "@fontsource-variable/inter";
import "@fontsource-variable/fraunces";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/600.css";
import "./styles.css";

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, HashRouter, Link, Route, Routes } from "react-router-dom";
import { AppProvider, IS_STATIC } from "./app";
import Admin from "./pages/Admin";
import CaseFile from "./pages/CaseFile";
import Cases from "./pages/Cases";
import HowItWorks from "./pages/HowItWorks";
import LiveRun from "./pages/LiveRun";
import Overview from "./pages/Overview";
import Results from "./pages/Results";
import { BootScreen, Shell } from "./shell";
import { EmptyState } from "./ui";

// A replay export is opened from a folder, where only hash URLs survive a reload.
const Router = IS_STATIC ? HashRouter : BrowserRouter;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Router>
      <AppProvider boot={(health, error) => <BootScreen health={health} error={error} />}>
        <Shell>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/how-it-works" element={<HowItWorks />} />
            <Route path="/live" element={<LiveRun />} />
            <Route path="/live/:account" element={<LiveRun />} />
            <Route path="/cases" element={<Cases />} />
            <Route path="/cases/:account" element={<CaseFile />} />
            <Route path="/results" element={<Results />} />
            <Route path="/admin" element={<Admin />} />
            <Route
              path="*"
              element={
                <EmptyState
                  title="There is no page here"
                  action={
                    <Link className="btn" to="/">
                      Go to the overview
                    </Link>
                  }
                />
              }
            />
          </Routes>
        </Shell>
      </AppProvider>
    </Router>
  </StrictMode>,
);
