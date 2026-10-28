"""A pack: a folder of new bank data, read beside data/ and never written into it.

data/packs/<name>/accounts.json ... kyc_profiles.json  same shapes as data/
                 documents/                 ID images + manifest.json
                 _ground_truth/answers.json what was planted (the pipeline never reads it)
                 llm_cache/                 this pack's saved KYC and writer answers
                 outputs/run.jsonl          the last run's reports
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from src.bank import BankData
from src.llm.cache import LLMMode
from src.rulebook import RulebookLookup
from src.schemas import Account, AdverseMediaEntry, Counterparty, Customer, KYCProfile, Transaction
from src.stages.stage0_triage import TriageContext


DATA_DIR = Path("data")
PACKS_DIR = DATA_DIR / "packs"
FILES = ("accounts", "customers", "transactions", "counterparties", "kyc_profiles")


def read(path: Path) -> list | dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


@dataclass(frozen=True)
class Pack:
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    @cached_property
    def answers(self) -> dict:
        """The planted answer key: {"seed", "show_cases", "accounts": [...]}. Read once: it is
        written before any run and never changes."""
        return read(self.path / "_ground_truth" / "answers.json")

    @property
    def title(self) -> str:
        """The readable name shown on the page (the folder name for an older pack)."""
        answers = self.answers
        return answers.get("title") or self.name if isinstance(answers, dict) else self.name

    @property
    def account_ids(self) -> list[str]:
        return [r["account_id"] for r in self.answers["accounts"]]

    @property
    def documents_dir(self) -> Path:
        return self.path / "documents"

    @property
    def cache_root(self) -> Path:
        return self.path / "llm_cache"

    def bank(self, data_dir: Path = DATA_DIR) -> BankData:
        """The committed bank plus this pack, merged in memory."""
        rows = {f: read(data_dir / f"{f}.json") + read(self.path / f"{f}.json") for f in FILES}
        return BankData(
            accounts=(Account(**r) for r in rows["accounts"]),
            customers=(Customer(**r) for r in rows["customers"]),
            transactions=(Transaction(**r) for r in rows["transactions"]),
            counterparties=(Counterparty(**r) for r in rows["counterparties"]),
            kyc_profiles=(KYCProfile(**r) for r in rows["kyc_profiles"]),
            adverse_media=(AdverseMediaEntry(**r) for r in read(data_dir / "adverse_media.json")),
        )

    def context(
        self,
        rulebook: RulebookLookup,
        kyc: LLMMode = LLMMode.LIVE,
        writer: LLMMode = LLMMode.OFF,
        data_dir: Path = DATA_DIR,
    ) -> TriageContext:
        """A run context that reads this pack's ID images and keeps its own saved answers."""
        return TriageContext.build(
            self.bank(data_dir),
            rulebook,
            llm_mode=kyc,
            writer_mode=writer,
            documents_dir=self.documents_dir,
            cache_root=self.cache_root,
        )


def list_packs(root: Path = PACKS_DIR) -> list[Pack]:
    """Every pack with an answer key, newest first."""
    found = [Pack(p.parent.parent) for p in root.glob("*/_ground_truth/answers.json")]
    return sorted(found, key=lambda p: p.path.stat().st_mtime, reverse=True)
