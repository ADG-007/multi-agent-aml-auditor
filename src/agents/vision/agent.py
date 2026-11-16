"""The Vision agent: the ID check and the KYC claims.

    VisionAgent(client).run(account_id, ctx, as_of, transaction) -> VisionResult

ID check (never scored, it only pauses). Tesseract reads every card and Python compares name,
date of birth and id number with the record. A field Tesseract did not confirm is read again
from the image by the model. A field is flagged only when both reads disagree with the
record. The model is never shown the record.

KYC (weight 0.20). Python builds the account's facts. The model copies the profile's claims
in two reads; the guard keeps a claim only when both reads agree and the quote is in the
profile. Python rules set the level and the score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from src.agents.base import Agent, AgentRun, AgentSpec, FactSheet
from src.agents.vision import claims as claim_forms
from src.agents.vision import config, kyc_facts, ocr, rules
from src.agents.vision.claims import Claims
from src.agents.vision.ocr import MATCH, MISMATCH, UNREAD, FieldCheck
from src.llm import Role
from src.schemas import (
    FORGERY_PAUSE_THRESHOLD,
    ReasoningStep,
    Stage1VisionOutput,
    Stage3KYCOutput,
    Transaction,
)
from src.stages.stage0_triage import TriageContext
from src.stages.stage2_transaction_pattern.analyst import sub_scores


FLAGGED, ONE_READ, UNCONFIRMED, AGREES = "FLAGGED", "one read disagrees", "unconfirmed", "agrees"

BACKUP_KIND = "KYC claims"


class IdCardRead(BaseModel):
    """What is printed on one identity card. The model copies; it never compares."""

    name: str | None = Field(description="The holder's name as printed, or null.")
    date_of_birth: str | None = Field(description="As DD/MM/YYYY, or null.")
    id_number: str | None = Field(description="The card's own number as printed, or null.")


ID_PROMPT = """This image is an Indian identity document ({doc_type}). Copy what is printed.
- name: the holder's name (on a passport: the given names, then the surname). Not the \
father's name.
- date_of_birth: the date of birth, written as DD/MM/YYYY.
- id_number: the document's own number (Aadhaar: 12 digits in groups of 4; PAN: 10 \
letters and digits; passport: the passport number).
Use null for anything you cannot read clearly. Do not guess and do not correct anything."""


@dataclass
class FieldResult:
    document: str
    field: str
    expected: str
    tesseract: FieldCheck
    image: FieldCheck | None
    result: str

    def line(self) -> str:
        second = f", image read {self.image.read!r}" if self.image is not None else ""
        return (
            f"{self.document} {self.field}: Tesseract read {self.tesseract.read!r}{second}, "
            f"record {self.expected!r} ({self.result})"
        )


@dataclass
class VisionResult:
    """Both halves of the agent. A half that failed is None, with its error beside it."""

    vision: Stage1VisionOutput | None
    kyc: Stage3KYCOutput | None
    #: The profile's declared cash takings explain the account's monthly cash (plan D14).
    cash_explained: bool
    claims: Claims
    verdict: rules.KycVerdict
    run: AgentRun
    id_fields: list[FieldResult] = field(default_factory=list)
    id_error: str | None = None
    kyc_error: str | None = None


class VisionAgent(Agent):
    spec = AgentSpec(
        name="vision",
        role=Role.VISION,
        prompt_version=config.PROMPT_VERSION,
        required_tools=(
            "tesseract_read",
            "field_compare",
            "image_read",
            "kyc_facts",
            "claim_read_1",
            "claim_read_2",
            "claim_rules",
        ),
        optional_tools=("narrow_profile_question",),
    )

    def run(
        self,
        account_id: str,
        ctx: TriageContext,
        as_of: datetime | None,
        transaction: Transaction | None = None,
        documents_dir: Path | None = None,
        reader=None,
    ) -> VisionResult:
        """`reader` turns an image path into text; tests pass a fake, the pipeline Tesseract."""
        run = self.start(account_id)
        result = VisionResult(None, None, False, Claims(), rules.KycVerdict(), run)
        try:
            result.vision, result.id_fields = self._id_check(
                run, account_id, ctx, documents_dir, reader
            )
        except Exception as exc:  # noqa: BLE001 - a failed ID check must not stop the KYC half
            result.id_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            run.step("ID check", f"Failed: {result.id_error}", "The ID check is FAILED.")
        try:
            result.kyc, result.claims, result.verdict = self._kyc(
                run, account_id, ctx, as_of, transaction
            )
            result.cash_explained = result.verdict.cash_explained
        except Exception as exc:  # noqa: BLE001
            result.kyc_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            run.step("KYC", f"Failed: {result.kyc_error}", "KYC is FAILED.")
        return result

    # ----------------------------------------------------------------------------------
    # ID check
    # ----------------------------------------------------------------------------------

    def _id_check(
        self, run: AgentRun, account_id: str, ctx: TriageContext, documents_dir, reader
    ) -> tuple[Stage1VisionOutput, list[FieldResult]]:
        folder = documents_dir or config.DOCUMENTS_DIR
        read_text = reader or ocr.tesseract
        customer = ctx.bank.customer_for(account_id)
        documents = ocr.documents_for(folder, customer.customer_id)
        if not documents:
            raise FileNotFoundError(f"no ID document on file for {customer.customer_id}")

        extracted: dict[str, str] = {}
        results: list[FieldResult] = []
        trail: list[ReasoningStep] = []
        for doc_type, file in documents:
            fields = ocr.extract(doc_type, read_text(folder / file))
            extracted.update({f"{doc_type}.{k}": v for k, v in fields.items()})
            first = ocr.compare(doc_type, fields, customer)
            second: dict[str, FieldCheck] = {}
            if any(c.result != MATCH for c in first):
                second = self._image_read(run, doc_type, folder / file, customer)
            doc_results = [
                FieldResult(
                    doc_type, c.field, c.expected, c, second.get(c.field), _combine(c, second)
                )
                for c in first
            ]
            results += doc_results
            trail.append(
                ReasoningStep(
                    step=f"ID read {doc_type}",
                    finding="; ".join(
                        f"{r.field} {r.result}" + (f" ({r.line()})" if r.result != AGREES else "")
                        for r in doc_results
                    ),
                    impact="Tesseract first; a field it did not confirm is read again from "
                    "the image. Both reads must disagree with the record to flag a field "
                    "(DOC-001).",
                )
            )
        self.tool(run, "tesseract_read")
        self.tool(run, "field_compare")

        confidence = max((_SCORE[r.result] for r in results), default=0.0)
        flagged = [r for r in results if r.result == FLAGGED]
        trail.append(
            ReasoningStep(
                step="ID check result",
                finding=f"{len(flagged)} field(s) flagged by both reads; "
                f"{sum(1 for r in results if r.result == ONE_READ)} by one read; "
                f"{sum(1 for r in results if r.result == UNCONFIRMED)} unconfirmed.",
                impact=_impact(confidence),
            )
        )
        run.trail += trail
        return (
            Stage1VisionOutput(
                document_type="+".join(doc_type for doc_type, _ in documents),
                extracted_fields=extracted,
                field_discrepancies=[r.line() for r in results if r.result in (FLAGGED, ONE_READ)],
                forgery_confidence=confidence,
                reasoning_trail=trail,
            ),
            results,
        )

    def _image_read(self, run: AgentRun, doc_type: str, path: Path, customer) -> dict:
        """The second read: the model copies the card; Python compares it with the record."""
        answer = self.ask(
            run,
            ID_PROMPT.format(doc_type=doc_type),
            IdCardRead,
            f"Image read {doc_type}",
            images=[path],
            queue_on_failure=False,
        )
        if answer is None:
            return {}
        self.tool(run, "image_read")
        fields = {"name": answer.name, "dob": answer.date_of_birth, "id_number": answer.id_number}
        return {c.field: c for c in ocr.compare(doc_type, fields, customer)}

    # ----------------------------------------------------------------------------------
    # KYC
    # ----------------------------------------------------------------------------------

    def _kyc(
        self,
        run: AgentRun,
        account_id: str,
        ctx: TriageContext,
        as_of: datetime | None,
        transaction: Transaction | None,
    ) -> tuple[Stage3KYCOutput, Claims, rules.KycVerdict]:
        profile_record = ctx.bank.kyc_profile_for(account_id)
        profile = profile_record.narrative if profile_record else ""
        activity = kyc_facts.build(run.facts, account_id, ctx, as_of, transaction)
        self.tool(run, "kyc_facts")
        cash_scored = sub_scores(account_id, ctx, as_of, transaction).cash > 0

        read_1 = read_2 = None
        if profile:
            read_1 = self.ask(
                run, claim_forms.form_prompt(profile), claim_forms.ClaimsForm, "Claim read 1"
            )
            read_2 = self.ask(
                run,
                claim_forms.questions_prompt(profile),
                claim_forms.ClaimsAnswers,
                "Claim read 2",
            )
            self.tool(run, "claim_read_1")
            self.tool(run, "claim_read_2")
        parsed = claim_forms.reconcile(read_1, read_2, profile)
        parsed.unread = bool(profile) and not self.client.off and read_1 is None and read_2 is None
        verdict = rules.judge(parsed, activity, cash_scored)
        if parsed.unsure:
            # The backup model is asked only when the disagreement could change the result.
            results = {
                _result(rules.judge(claim_forms.single(read, profile), activity, cash_scored))
                for read in (read_1, read_2)
            } | {_result(verdict)}
            if len(results) > 1 and not any(j.kind == BACKUP_KIND for j in run.queued):
                run.queue(BACKUP_KIND, "the two reads disagree on: " + ", ".join(parsed.unsure))
                parsed.matters = True
        self.tool(run, "claim_rules")
        self._narrow_check(run, profile, verdict)
        return self._kyc_output(run, parsed, verdict), parsed, verdict

    def _narrow_check(self, run: AgentRun, profile: str, verdict: rules.KycVerdict) -> None:
        """Borderline volume: ask one narrow question before the moderate concern stands."""
        borderline = [c for c in verdict.concerns if c.key == "volume" and c.level == "MODERATE"]
        if not borderline or not profile or self.client.off:
            return

        def ask() -> None:
            answer = self.ask(
                run,
                claim_forms.narrow_prompt(profile),
                claim_forms.ReasonClaim,
                "Narrow profile question",
                queue_on_failure=False,
            )
            if (
                answer is not None
                and answer.value == "REASON_GIVEN"
                and claim_forms.quote_in(answer.quote, profile)
            ):
                rules.drop_concern(
                    verdict, "volume", f"Cancelled: the profile gives a reason ('{answer.quote}')."
                )

        self.optional(run, ["narrow_profile_question"], {"narrow_profile_question": ask})

    def _kyc_output(
        self, run: AgentRun, parsed: Claims, verdict: rules.KycVerdict
    ) -> Stage3KYCOutput:
        facts = run.facts
        stated = "; ".join(
            f"{name}={value} ('{parsed.quotes.get(name, '')}')"
            for name, value in parsed.stated().items()
        )
        trail = [
            ReasoningStep(
                step="Account facts",
                finding=" ".join(
                    f"{f.fact_id}: {f.text}" for t in ("profile", "income") for f in facts.topic(t)
                ),
                impact="Python computed every number (KYC-002, STRUCT-003).",
            ),
            ReasoningStep(
                step="Cash, foreign money, payers",
                finding=" ".join(
                    f"{f.fact_id}: {f.text}"
                    for t in ("cash", "foreign", "payers")
                    for f in facts.topic(t)
                ),
                impact="Compared with what the profile says about them.",
            ),
            ReasoningStep(
                step="Profile claims",
                finding=(stated or "The profile states none of the six claims.")
                + (
                    f" Unsure (reads disagree): {', '.join(parsed.unsure)}."
                    if parsed.unsure
                    else ""
                )
                + (
                    f" Dropped by the guard: {', '.join(parsed.dropped)}." if parsed.dropped else ""
                ),
                impact="A claim counts only when both reads agree and its quote is in the "
                "profile word for word.",
            ),
        ]
        for concern in verdict.concerns:
            trail.append(
                ReasoningStep(
                    step=f"Concern: {concern.key}",
                    finding=concern.text,
                    impact=f"A {concern.level.lower()} concern.",
                )
            )
        for line in verdict.cancelled:
            trail.append(
                ReasoningStep(step="Concern cancelled", finding=line, impact="Not counted.")
            )
        if verdict.cash_note:
            trail.append(
                ReasoningStep(
                    step="Declared cash",
                    finding=verdict.cash_note,
                    impact="Explained cash sets the monthly-cash finding aside (STRUCT-002)."
                    if verdict.cash_explained
                    else "The monthly-cash finding stands.",
                )
            )
        trail.append(
            ReasoningStep(
                step="Level to score",
                finding=f"{len(verdict.concerns)} concern(s): level {verdict.level}.",
                impact=f"Python maps the level to {verdict.score:g}.",
            )
        )
        run.trail += trail
        return Stage3KYCOutput(
            profile_behavior_mismatches=[c.text for c in verdict.concerns],
            confidence=verdict.score,
            reasoning_trail=trail[-config.MAX_TRAIL_STEPS :],
        )

    # ----------------------------------------------------------------------------------
    # Backup model (queue phase)
    # ----------------------------------------------------------------------------------

    def recheck(self, result: VisionResult, ctx: TriageContext, as_of: datetime | None) -> bool:
        """A third claim read by the backup model. True when the KYC score changed."""
        account_id = result.run.account_id
        profile_record = ctx.bank.kyc_profile_for(account_id)
        profile = profile_record.narrative if profile_record else ""
        if not profile:
            return False
        read_3 = self.ask(
            result.run,
            claim_forms.form_prompt(profile),
            claim_forms.ClaimsForm,
            "Claim read by the backup model",
            role=Role.BACKUP,
            queue_on_failure=False,
        )
        if read_3 is None:
            return False
        if result.claims.unsure:
            claim_forms.resolve(result.claims, read_3, profile)
        else:
            fresh, _ = claim_forms.guarded(read_3, profile)
            for name, (value, quote) in fresh.items():
                result.claims.values[name] = value
                if value != claim_forms.NOT_STATED:
                    result.claims.quotes[name] = quote
        before = (result.kyc.confidence, result.cash_explained)
        activity = kyc_facts.build(FactSheet(), account_id, ctx, as_of)
        cash_scored = sub_scores(account_id, ctx, as_of).cash > 0
        verdict = rules.judge(result.claims, activity, cash_scored)
        result.kyc = self._kyc_output(result.run, result.claims, verdict)
        result.verdict = verdict
        result.cash_explained = verdict.cash_explained
        return before != (result.kyc.confidence, result.cash_explained)


def _result(verdict: rules.KycVerdict) -> tuple[str, bool]:
    return verdict.level, verdict.cash_explained


_SCORE = {
    FLAGGED: config.BOTH_READS_DISAGREE,
    ONE_READ: config.ONE_READ_DISAGREES,
    UNCONFIRMED: config.UNCONFIRMED,
    AGREES: 0.0,
}


def _combine(first: FieldCheck, second: dict[str, FieldCheck]) -> str:
    """One field's result from the Tesseract read and, when there is one, the image read."""
    if first.result == MATCH:
        return AGREES
    other = second.get(first.field)
    other_result = other.result if other is not None else UNREAD
    if other_result == MATCH:
        return AGREES if first.result == UNREAD else UNCONFIRMED
    if first.result == MISMATCH and other_result == MISMATCH:
        return FLAGGED
    if MISMATCH in (first.result, other_result):
        return ONE_READ
    return UNCONFIRMED


def _impact(confidence: float) -> str:
    if confidence >= FORGERY_PAUSE_THRESHOLD:
        return (
            "Both reads disagree with the record: the case pauses for the admin. "
            "Never scored (DOC-002)."
        )
    if confidence > 0:
        return "Not confirmed by both reads: noted for the analyst, no pause. Never scored."
    return "Every field agrees with the record. Never scored."
