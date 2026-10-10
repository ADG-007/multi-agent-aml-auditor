"""Stage 2 separation on its saved test subset - do planted patterns and decoys come apart?

    python -m src.evaluation.stage2_separation

Reads the subset saved by `stage2_subset` (Stage 2's slots plus the shared linked set), runs
Stage 2 on each account at the bank's review date, and prints one row per account: slot,
structuring sub-score, pass-through sub-score, confidence and the band the answer key expects.
Only the subset is run (decisions_and_progress.md Part 11); the full 370-account evaluation
happens once, in Phase 11.

Hard checks (phase_3_execution.md step 8, from Q7), applied to every subset account by its
answer-key role, slotted or shared:

- every structurer and every network or solo mule scores at least 0.40;
- every round-amount, merchant look-alike, life-event and legitimate cross-border decoy scores
  below 0.40.

Reported only: complicit merchants' scores (their path is Graph's collector link, Phase 7), how
many subset accounts reach 0.75, and the shared set's scores. Exits non-zero if a hard check
fails. Pipeline code never imports this module.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from src.bank import BankData
from src.evaluation.stage2_subset import Truth, load_subsets, load_truth
from src.rulebook import RulebookLookup
from src.stages.stage0_triage import TriageContext
from src.stages.stage2_transaction_pattern import pass_through, run, structuring


#: The Q7 line between "Stage 2 sees it" and "Stage 2 clears it".
SEPARATION_LINE = 0.40
#: A clear-cut case reaches this (reported only).
CLEAR_CUT = 0.75

MUST_REACH = frozenset({"structurer", "network_mule", "solo_mule"})
MUST_STAY_BELOW = frozenset(
    {"round_amount", "merchant_lookalike", "life_event", "cross_border_legitimate"}
)


@dataclass(frozen=True)
class Row:
    account_id: str
    slot: str
    role: str
    structuring: float
    pass_through: float
    confidence: float

    @property
    def expected(self) -> str:
        if self.role in MUST_REACH:
            return f">= {SEPARATION_LINE:.2f}"
        if self.role in MUST_STAY_BELOW:
            return f"< {SEPARATION_LINE:.2f}"
        return "reported"

    @property
    def passes(self) -> bool:
        if self.role in MUST_REACH:
            return self.confidence >= SEPARATION_LINE
        if self.role in MUST_STAY_BELOW:
            return self.confidence < SEPARATION_LINE
        return True


@dataclass
class Separation:
    lines: list[str]
    failures: list[str]


def measure(ctx: TriageContext, subsets: dict[str, list], truth: dict[str, Truth]) -> Separation:
    as_of = ctx.review_date
    slots = [(s["account_id"], s["slot"]) for s in subsets["stage2"]]
    slotted = {a for a, _ in slots}
    shared = subsets["shared"]
    slots += [(a, "shared") for a in shared if a not in slotted]

    rows = []
    for account_id, slot in slots:
        history = ctx.bank.transactions(account_id)
        rows.append(
            Row(
                account_id=account_id,
                slot=slot,
                role=truth[account_id].role,
                structuring=structuring.score(history, as_of, ctx.rulebook).score,
                pass_through=pass_through.score(history, as_of, ctx.rulebook).score,
                confidence=run(account_id, ctx, as_of).confidence,
            )
        )

    lines = [
        f"Stage 2 separation - {len(rows)} subset accounts at the review date "
        f"{as_of:%Y-%m-%d %H:%M}",
        "",
        f"  {'Account':<9} {'Slot':<24} {'Role':<24} "
        f"{'Struct':>6} {'Pass':>6} {'Conf':>6}  Expected",
    ]
    for r in rows:
        lines.append(
            f"  {r.account_id:<9} {r.slot:<24} {r.role:<24} {r.structuring:>6.4f} "
            f"{r.pass_through:>6.4f} {r.confidence:>6.4f}  {r.expected}"
            + ("" if r.passes else "  <- FAIL")
        )

    reach = [r for r in rows if r.role in MUST_REACH]
    below = [r for r in rows if r.role in MUST_STAY_BELOW]
    failures = [
        f"{r.account_id} ({r.role}) {r.confidence:.4f}, expected {r.expected}"
        for r in rows
        if not r.passes
    ]
    lines += ["", "  Hard checks (phase_3_execution.md step 8, Q7)"]
    for label, group in (
        (f"structurers and network/solo mules >= {SEPARATION_LINE:.2f}", reach),
        (
            "round-amount, merchant look-alike, life-event and legitimate cross-border decoys "
            f"< {SEPARATION_LINE:.2f}",
            below,
        ),
    ):
        ok = sum(r.passes for r in group)
        verdict = "PASS" if ok == len(group) else "FAIL"
        lines.append(f"  {verdict}  {label} ({ok}/{len(group)})")

    complicit = [r for r in rows if r.role == "complicit_merchant"]
    clear_cut = [r for r in rows if r.confidence >= CLEAR_CUT]
    by_id = {r.account_id: r for r in rows}
    shared_rows = [by_id[a] for a in shared]
    lines += [
        "",
        "  Reported only",
        "  complicit merchants: "
        + (", ".join(f"{r.account_id} {r.confidence:.4f}" for r in complicit) or "none"),
        f"  accounts at >= {CLEAR_CUT:.2f}: {len(clear_cut)} of {len(rows)} "
        f"({', '.join(r.account_id for r in clear_cut) or 'none'})",
        "  shared set: "
        + ", ".join(f"{r.account_id} ({r.role}) {r.confidence:.4f}" for r in shared_rows),
    ]
    return Separation(lines=lines, failures=failures)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    bank = BankData.load()
    rulebook = RulebookLookup.load()
    subsets = load_subsets()
    accounts = {s["account_id"] for s in subsets["stage2"]} | set(subsets["shared"])
    truth = load_truth(bank, rulebook, accounts)
    separation = measure(TriageContext.build(bank, rulebook), subsets, truth)
    print("\n".join(separation.lines))
    if separation.failures:
        print("\nFAILED: " + "; ".join(separation.failures))
        return 1
    print("\nAll hard checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
