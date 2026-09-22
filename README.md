# Autonomous FinTech Compliance & AML Auditing System

A TRL 3/4 proof-of-concept multi-agent pipeline that takes a flagged transaction, gathers
document, behavioural, network, media and policy evidence across eight stages, and produces a
structured, auditable `InvestigationReport` — alongside a stateful simulated bank environment
that evaluators can inject new data into live.

It is an **assistant to a human lead analyst, not an autonomous decision-maker**, and a
**low-fidelity lab validation, not a production system**. Known limitations are documented
rather than engineered around.

## Documentation

This repository keeps its technical content in one place on purpose — the README is a pointer,
not a second source of truth.

| File | What it holds |
|---|---|
| `MASTER_PLAN.md` | **Authoritative** technical reference: problem statement, architecture, all 12 phases, invariants, tech stack, known limitations. |
| `decisions_and_progress.md` | Why each decision was made, including superseded ones, and phase-by-phase progress. |
| `phase_<N>_execution.md` | The finalized, approved spec implemented in phase N. |

Where `MASTER_PLAN.md` and `decisions_and_progress.md` appear to disagree, `MASTER_PLAN.md` wins.

## Repository layout

```
src/
  schemas/        Pydantic domain models + per-stage I/O models
  generators/     Synthetic data generators (Phase 1) + verify.py
  llm/            Shared LLM abstraction used by the LLM-backed stages
  stages/         The eight pipeline stages
  orchestration/  Pipeline wiring
  evaluation/     Scoring / evaluation harness
  demo/           Demo + live-injection surface
data/             Generated, seed-reproducible dataset (committed)
  _ground_truth/  Evidence ledger — evaluation only, never read by a pipeline stage
```

## Setup

Requires Python and, for OCR in later phases, the Tesseract binary installed at OS level.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Regenerating the dataset

All synthetic data is generated deterministically under a fixed seed:

```bash
python -m src.generators        # runs every generator in dependency order
python -m src.generators.verify # sanity assertions + human-readable summary
```
