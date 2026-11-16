"""The Auditor's numbers. It never scores."""

PROMPT_VERSION: str = "auditor-v1"

#: Lines per agent given to the Auditor as facts.
LINES_PER_AGENT: int = 5

#: Outline points kept at most.
MAX_POINTS: int = 8

#: Rewrites asked for when the text breaks the outline (a number or a clause not in it).
MAX_REWRITES: int = 1

#: The agents whose model answer can be checked again by the backup model.
RECHECKS: tuple[str, ...] = ("KYC_CLAIMS", "SAME_PERSON")
