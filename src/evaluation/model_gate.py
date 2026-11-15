"""The model gate: does each local model fit this laptop and return valid JSON fast enough?

    python -m src.evaluation.model_gate                 # every candidate that is installed
    python -m src.evaluation.model_gate --only qwen4b   # one candidate

Two task groups, each on fixed inputs from the tuning set:

- extract (the Qwen roles): the KYC claim form on 6 profiles and the image read of 2 ID cards;
- label and write (the Gemma roles, the 9B backup): the pattern label on 4 fact sheets and a
  short report text on 2 outlines.

Logged per candidate: the share of the model on the GPU (from Ollama), the valid-JSON rate,
the median and worst seconds per call, tokens per second, and a small accuracy figure (ID
fields that agree with the record; labels the facts support). Nothing is cached, so every
figure is a fresh measurement. Results go to outputs/model_gate.json.

Limits (plan section 4): a Gemma-role model passes with a median of at most 60 s per call and
a valid-JSON rate of at least 95%.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from src.agents.graph.agent import PROMPT as LABEL_PROMPT
from src.agents.graph.agent import PatternLabel
from src.agents.vision import claims as claim_forms
from src.agents.vision import config as vision_config
from src.agents.vision import ocr
from src.agents.vision.agent import ID_PROMPT, IdCardRead
from src.bank import BankData
from src.evaluation import split
from src.llm import LLMClient, LLMMode, LLMStageFailure, ModelSpec, Role
from src.llm import config as llm_config


OUT_PATH = Path("outputs") / "model_gate.json"
MAX_MEDIAN_SECONDS = 60.0
MIN_JSON_RATE = 0.95

CANDIDATES: dict[str, tuple[str, ModelSpec]] = {
    "qwen4b": ("extract", llm_config.QWEN_4B),
    "qwen4b-ollama-split": (
        "extract",
        ModelSpec("qwen3.5:4b-q4_K_M", think=False, num_ctx=2048),
    ),
    "qwen2b": ("extract", ModelSpec("qwen3.5:2b-q4_K_M", think=False, num_ctx=2048, num_gpu=99)),
    "gemma-B": ("write", llm_config.GEMMA),
    "gemma-B-ollama-split": ("write", ModelSpec("gemma4:e4b-it-qat")),
    "gemma-A": ("write", ModelSpec("gemma3:4b-it-q4_K_M", num_gpu=99)),
    "qwen9b": ("write", llm_config.QWEN_9B),
}

#: Fact sheets with the label the Python facts support.
FACT_SHEETS: tuple[tuple[str, set[str]], ...] = (
    (
        "F1: Single cash deposits in the band: k = 6 in the busiest 30 days (sub-score 0.8).\n"
        "F2: Monthly cash totals in the band: m = 0 (sub-score 0).\n"
        "F3: Pass-through sub-score 0. No fast forwarding.\n"
        "F4: Shares a device, IP or address with 0 account(s) (link score 0).",
        {"NAIVE_CASH_STRUCTURING"},
    ),
    (
        "F1: Single cash deposits in the band: k = 0 in the busiest 30 days (sub-score 0).\n"
        "F2: Monthly cash totals in the band: m = 2 (sub-score 0.6).\n"
        "F3: Pass-through sub-score 0. No fast forwarding.\n"
        "F4: Shares a device, IP or address with 1 account(s); none flagged (link score 0).",
        {"MONTHLY_CAP_STRUCTURING"},
    ),
    (
        "F1: Single cash deposits in the band: k = 0 in the busiest 30 days (sub-score 0).\n"
        "F2: Monthly cash totals in the band: m = 0 (sub-score 0).\n"
        "F3: Pass-through sub-score 0.92. 9 credits from first-time payers sent on within 3 "
        "hours.\n"
        "F4: Shares a device with 3 account(s); flagged on their own evidence: 2 (link score "
        "0.75).",
        {"MIXED_PATTERN", "PASS_THROUGH", "LINKED_TO_FLAGGED_ACCOUNTS"},
    ),
    (
        "F1: Single cash deposits in the band: k = 0 in the busiest 30 days (sub-score 0).\n"
        "F2: Monthly cash totals in the band: m = 0 (sub-score 0).\n"
        "F3: Pass-through sub-score 0. No fast forwarding.\n"
        "F4: Shares a device with 1 account(s); flagged on their own evidence: 1 (link score "
        "0.6).",
        {"LINKED_TO_FLAGGED_ACCOUNTS"},
    ),
)

OUTLINES = (
    "Verdict: MONITOR at 0.60.\n- Two calendar months of cash deposits totalled 9.75 lakh and "
    "9.42 lakh, just under the 10 lakh monthly threshold (STRUCT-002).\n- No link to a flagged "
    "account.",
    "Verdict: SUSPICIOUS-ESCALATE at 0.95.\n- The holder's name and date of birth match a "
    "confirmed sanctions entry (SANC-001).\n- Money in is 0.7x the declared income.",
)


class GateText(BaseModel):
    summary: str = Field(description="At most 3 sentences.")
    conclusion: str = Field(description="One sentence that cites the clause id given.")


@dataclass
class GateRow:
    candidate: str
    model: str
    group: str
    installed: bool
    gpu_share: float | None = None
    calls: int = 0
    valid: int = 0
    median_seconds: float | None = None
    worst_seconds: float | None = None
    tokens_per_second: float | None = None
    accuracy: str = ""
    passed: bool | None = None

    @property
    def json_rate(self) -> float:
        return self.valid / self.calls if self.calls else 0.0


def installed_models() -> set[str]:
    tags = httpx.get(llm_config.OLLAMA_URL + "/api/tags", timeout=30).json()
    return {m["name"] for m in tags.get("models", [])}


def gpu_share(model: str) -> float | None:
    """The share of the loaded model that sits in GPU memory, as Ollama reports it."""
    loaded = httpx.get(llm_config.OLLAMA_URL + "/api/ps", timeout=30).json().get("models", [])
    for m in loaded:
        if m.get("name") == model and m.get("size"):
            return round(m.get("size_vram", 0) / m["size"], 2)
    return None


def run_candidate(name: str, bank: BankData) -> GateRow:
    group, spec = CANDIDATES[name]
    row = GateRow(name, spec.model, group, spec.model in installed_models())
    if not row.installed:
        return row
    role = Role.VISION if group == "extract" else Role.AUDITOR
    with tempfile.TemporaryDirectory() as scratch:
        client = LLMClient(
            LLMMode.LIVE, cache_root=Path(scratch), audit_path=None, models={role: spec}
        )
        seconds: list[float] = []
        good = total = 0

        def call(prompt, schema, images=()):
            row.calls += 1
            started = time.perf_counter()
            try:
                result = client.call(role, prompt, schema, "gate", images=images)
            except LLMStageFailure:
                seconds.append(time.perf_counter() - started)
                return None
            row.valid += result.attempts == 1
            seconds.append(result.seconds)
            return result.value

        if group == "extract":
            accounts = split.tuning_ids()[:6]
            for account_id in accounts:
                profile = bank.kyc_profile_for(account_id).narrative
                call(claim_forms.form_prompt(profile), claim_forms.ClaimsForm)
            for account_id in accounts[:2]:
                customer = bank.customer_for(account_id)
                doc_type, file = ocr.documents_for(
                    vision_config.DOCUMENTS_DIR, customer.customer_id
                )[0]
                read = call(
                    ID_PROMPT.format(doc_type=doc_type),
                    IdCardRead,
                    images=[vision_config.DOCUMENTS_DIR / file],
                )
                if read is not None:
                    fields = {
                        "name": read.name,
                        "dob": read.date_of_birth,
                        "id_number": read.id_number,
                    }
                    checks = ocr.compare(doc_type, fields, customer)
                    total += len(checks)
                    good += sum(c.result == ocr.MATCH for c in checks)
            row.accuracy = f"ID fields read correctly {good}/{total}"
        else:
            for facts, supported in FACT_SHEETS:
                answer = call(LABEL_PROMPT.format(facts=facts), PatternLabel)
                total += 1
                good += answer is not None and answer.label in supported
            for outline in OUTLINES:
                call(
                    "Write a short case note for a bank's lead analyst from this outline. Use "
                    "only what the outline says.\n\n" + outline,
                    GateText,
                )
            row.accuracy = f"labels the facts support {good}/{total}"

        row.gpu_share = gpu_share(spec.model)
        ok = [c for c in client.calls if c["ok"] and c["output_tokens"]]
        if ok:
            row.tokens_per_second = round(
                sum(c["output_tokens"] for c in ok) / sum(c["seconds"] for c in ok), 1
            )
        client.unload()
    # The first call pays the model load; it is reported as the worst call, not the median.
    row.median_seconds = round(statistics.median(seconds), 1)
    row.worst_seconds = round(max(seconds), 1)
    row.passed = row.median_seconds <= MAX_MEDIAN_SECONDS and row.json_rate >= MIN_JSON_RATE
    return row


def render(rows: list[GateRow]) -> str:
    lines = [
        f"{'candidate':<20} {'model':<24} {'GPU':>5} {'JSON':>6} {'median s':>9} "
        f"{'worst s':>8} {'tok/s':>6}  accuracy / result"
    ]
    for r in rows:
        if not r.installed:
            lines.append(f"{r.candidate:<20} {r.model:<24} not installed")
            continue
        gpu = f"{r.gpu_share:.0%}" if r.gpu_share is not None else "?"
        lines.append(
            f"{r.candidate:<20} {r.model:<24} {gpu:>5} {r.valid}/{r.calls:<4} "
            f"{r.median_seconds:>9} {r.worst_seconds:>8} {r.tokens_per_second or 0:>6}  "
            f"{r.accuracy}; {'PASS' if r.passed else 'FAIL'}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", action="append", choices=sorted(CANDIDATES))
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    bank = BankData.load()
    rows = [run_candidate(name, bank) for name in (args.only or CANDIDATES)]
    print(render(rows))
    saved = json.loads(OUT_PATH.read_text(encoding="utf-8")) if OUT_PATH.exists() else {}
    saved.update({r.candidate: asdict(r) | {"json_rate": r.json_rate} for r in rows})
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
