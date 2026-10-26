"""Stage 4's numbers. The 85 cutoff is shared with Stage 0."""

from src.stages.stage0_triage.config import NAME_MATCH_CUTOFF


#: Any tier: a name or alias at or above this is a candidate match.
CANDIDATE_CUTOFF: float = NAME_MATCH_CUTOFF

#: NAME_SIMILARITY_ONLY entries count as a hit only at or above this (stricter on purpose).
LOOKALIKE_CUTOFF: float = 92.0

# Holder matches. The DOB decides between a sanctions veto and a name clash (SANC-002).
HOLDER_SANCTIONS_DOB_AGREES: float = 0.95
HOLDER_SANCTIONS_NO_DOB: float = 0.90
HOLDER_SANCTIONS_DOB_DISAGREES: float = 0.50
#: Adverse news is an allegation: capped below the veto and the solo branch.
HOLDER_NEWS: float = 0.70
HOLDER_LOOKALIKE: float = 0.15

# Counterparty matches. Counterparties carry no date of birth.
COUNTERPARTY_SANCTIONS: float = 0.90
COUNTERPARTY_NEWS: float = 0.40
COUNTERPARTY_LOOKALIKE: float = 0.0

MAX_TRAIL_STEPS: int = 8
