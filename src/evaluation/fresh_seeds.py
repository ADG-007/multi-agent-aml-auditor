"""Fresh-seed run (Amendment 4, V7): does the generator hold up on seeds it was not tuned on?

    python -m src.evaluation.fresh_seeds            # seeds 43, 44, 45
    python -m src.evaluation.fresh_seeds 46 47      # any others

For each seed, builds the dataset (with its rulebook index) into a temporary folder (never
into data/), then runs, in
that folder: `src.generators.verify`, `src.evaluation.stage2_subset` (picks that build's
subset) and `src.evaluation.stage2_separation`. Prints one line per seed and step. The result
is reported, not judged: a failure on a fresh seed is a finding about the generator, recorded
in the amendment's report. Exits non-zero if any step failed, so the finding is not missed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from src.generators import config as cfg
from src.generators.rebuild_check import build


DEFAULT_SEEDS = (43, 44, 45)
STEPS = (
    ("verify", "src.generators.verify"),
    ("stage2_subset", "src.evaluation.stage2_subset"),
    ("stage2_separation", "src.evaluation.stage2_separation"),
)


def run_step(module: str, root: Path) -> tuple[bool, str]:
    """Run one module with `root` as the working folder (its data/ is the fresh build)."""
    env = {
        **os.environ,
        "AML_DATA_DIR": str(root / "data"),
        "PYTHONPATH": str(cfg.PROJECT_ROOT),
        "PYTHONIOENCODING": "utf-8",
    }
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    out = (result.stdout + result.stderr).strip().splitlines()
    return result.returncode == 0, _summary(module, out)


def _summary(module: str, out: list[str]) -> str:
    fails = [
        line.strip()
        for line in out
        if "[FAIL]" in line or "<- FAIL" in line or line.startswith("FAILED")
    ]
    if module.endswith("verify"):
        oks = sum("[ok]" in line for line in out)
        failed = [line for line in fails if line.startswith("[FAIL]")]
        return f"{oks} checks ok, {len(failed)} failed" + (
            ": " + " | ".join(failed[:4]) if failed else ""
        )
    if module.endswith("separation"):
        verdicts = [line.strip() for line in out if line.strip().startswith(("PASS", "FAIL"))]
        return " | ".join(verdicts + fails[:3])
    return out[-1] if out else "(no output)"


def main(seeds: tuple[int, ...] = DEFAULT_SEEDS) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    failed = 0
    with tempfile.TemporaryDirectory(prefix="aml-seeds-") as tmp:
        for seed in seeds:
            root = Path(tmp) / f"seed{seed}"
            print(f"Seed {seed}: building ...", flush=True)
            build(root / "data", seed, embeddings=True)
            for name, module in STEPS:
                ok, summary = run_step(module, root)
                failed += not ok
                print(f"  {'PASS' if ok else 'FAIL'}  {name}: {summary}", flush=True)
    print("\nAll fresh-seed steps passed." if not failed else f"\n{failed} step(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    args = tuple(int(a) for a in sys.argv[1:]) or DEFAULT_SEEDS
    raise SystemExit(main(args))
