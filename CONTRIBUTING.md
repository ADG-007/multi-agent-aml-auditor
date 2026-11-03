# Contributing to AML Auditor

Thanks for taking a look. This is a small research project, so the process is light.

## Before you start

- Open an issue first for anything bigger than a typo or a one-line fix, so we can agree on the approach.
- Use only the synthetic data in this repository. Never add real customer, transaction or ID data.
- Never commit `.env` or any API key.

## Set up

Follow the Quick start in the [README](README.md). You need Python 3.14, Tesseract and Ollama for a full run. The unit tests and the linter need only the Python packages and the generated rulebook index and ID images.

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m src.generators.rulebook
python -m src.generators.id_documents
```

## Before you open a pull request

```powershell
ruff check .
python -m pytest -q
```

Both should pass. If you change a score, a threshold or a stage's behaviour, run `python -m src.demo.cli run50` and say in the pull request how the graded results moved.

## Rules of the codebase

These keep the verdicts explainable, so pull requests that break them will be asked to change.

1. **Scoring is Python.** A language model may pick a category or write text. It never does arithmetic that reaches a score.
2. **The ID check (Stage 1) never changes the score.** It only pauses the case for an admin.
3. **The report writer may cite only clause IDs that the rulebook search returned.** The citation guard enforces this.
4. **Every stage output has a `reasoning_trail`.**
5. **No hardcoding of test accounts.** Do not special-case an account ID to make output look right. If a case comes out wrong, report it.
6. **Failures degrade.** A failed stage becomes `FAILED` and the case floors at Monitor. It does not crash the run.
7. **Thresholds live in config files,** not in the middle of a function.
8. **Confidence is a number from 0 to 1 everywhere.**
9. **Keep it deterministic by default.** Seed anything random.

## Style

- Line length is 100. Ruff settings are in `pyproject.toml`.
- Add or update tests next to the code you change. The `tests/` folders mirror `src/`.
- Keep commits small, with a message that says what changed and why.

## Pull requests

- Branch from `v1` and open the pull request against `v1`.
- Fill in the pull request template.
- Expect review comments. They are about the code, not about you.
