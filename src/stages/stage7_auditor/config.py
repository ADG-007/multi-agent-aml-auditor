"""Stage 7's tunable numbers."""

PROMPT_VERSION: str = "auditor-v1"

#: Free-tier Gemini allows few requests a minute: space calls at least this far apart.
MIN_SECONDS_BETWEEN_CALLS: float = 6.5

#: Each stage's trail is cut to this many steps in the prompt, to keep the context small.
TRAIL_STEPS_PER_STAGE: int = 4
