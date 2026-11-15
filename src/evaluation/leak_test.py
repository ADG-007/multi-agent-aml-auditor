"""Leak test: no pipeline code reads the answer key (Amendment 4, V5).

    python -m src.evaluation.leak_test

Two checks:

1. **Static.** Every pipeline module (everything under src/ except src/evaluation/ and
   src/generators/) is parsed. None may import src.evaluation or src.generators, and none may
   name an answer-key file or folder in a string. The one allowed name is the bank's own guard
   that refuses to read the folder (src/bank/data_store.py).
2. **Run time.** In a child process with a Python audit hook on every file open, Stage 0 and
   Stage 2 run on a sample of accounts. No opened path may lie under data/_ground_truth/.

Exits non-zero if either check fails.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys

from src.generators import config as cfg


SRC = cfg.PROJECT_ROOT / "src"
NOT_PIPELINE = ("evaluation", "generators")
FORBIDDEN_IMPORTS = ("src.evaluation", "src.generators")
FORBIDDEN_NAMES = (
    "_ground_truth",
    "evidence_ledger.json",
    "fraud_rings.json",
    "stage_subsets.json",
)
#: (file, the string allowed there): the bank's guard against reading the folder.
ALLOWED = {("bank/data_store.py", "_ground_truth")}

#: The child process: an audit hook records every open, then the stages run.
_PROBE = r"""
import sys
opened = []
def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        opened.append(str(args[0]))
sys.addaudithook(hook)
from src.bank import BankData
from src.rulebook import RulebookLookup
from src.orchestration.run_state import RunState
from src.schemas import TriageEvent, TriageEventKind
from src.stages.stage0_triage import TriageContext, triage
from src.stages.stage2_transaction_pattern import run
bank = BankData.load()
ctx = TriageContext.build(bank, RulebookLookup.load())
state = RunState()
for account_id in bank.account_ids()[::int(sys.argv[1])]:
    triage(TriageEvent(kind=TriageEventKind.PERIODIC_REVIEW, account_id=account_id), ctx, state)
    run(account_id, ctx, ctx.review_date)
hits = [p for p in opened if "_ground_truth" in p.replace("\\", "/")]
print(len(opened), len(hits))
for p in hits[:5]:
    print(p)
"""


def static_findings() -> list[str]:
    findings = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel.split("/")[0] in NOT_PIPELINE:
            continue
        findings += scan(rel, path.read_text(encoding="utf-8"))
    return findings


def scan(rel: str, source: str) -> list[str]:
    """Findings in one module's source; `rel` is its path under src/."""
    findings = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            names = []
        for name in names:
            if name.startswith(FORBIDDEN_IMPORTS):
                findings.append(f"src/{rel}:{node.lineno} imports {name}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for word in FORBIDDEN_NAMES:
                if (
                    word in node.value
                    and (rel, word) not in ALLOWED
                    and not _is_docstring(tree, node)
                ):
                    findings.append(f"src/{rel}:{node.lineno} names {word!r}")
    return findings


def _is_docstring(tree: ast.AST, node: ast.Constant) -> bool:
    """Docstrings may describe the rule; only code that could open a path counts."""
    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and body[0].value is node
        ):
            return True
    return False


def runtime_findings(step: int = 5) -> tuple[int, list[str]]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(step)],
        cwd=cfg.PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    lines = result.stdout.strip().splitlines()
    opened, hits = (int(x) for x in lines[0].split())
    return opened, lines[1 : 1 + hits]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    static = static_findings()
    print("Leak test (V5)\n")
    print(
        f"  {'PASS' if not static else 'FAIL'}  static: no pipeline module imports "
        "src.evaluation or src.generators, or names an answer-key file"
    )
    for f in static:
        print(f"        {f}")
    opened, hits = runtime_findings()
    print(
        f"  {'PASS' if not hits else 'FAIL'}  run time: Stage 0 + Stage 2 on every 5th account "
        f"opened {opened} files, {len(hits)} under data/_ground_truth/"
    )
    for h in hits:
        print(f"        {h}")
    if static or hits:
        print("\nFAILED")
        return 1
    print("\nNo leak found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
