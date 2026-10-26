"""Stage 3's numbers."""

#: Bump on any prompt wording change, so old saved answers stop matching.
PROMPT_VERSION: str = "kyc-v2"

#: The model picks a level, code maps it to a score. LOW and MODERATE carry no signal from
#: this 8B, so only HIGH and SEVERE count.
LEVEL_SCORES: dict[str, float] = {
    "NONE": 0.0,
    "LOW": 0.0,
    "MODERATE": 0.0,
    "HIGH": 0.65,
    "SEVERE": 0.85,
}

#: Money in at or above this many times the declared income lifts the level to at least
#: INCOME_FLOOR_LEVEL. The 8B quotes the ratio but often still says MODERATE.
INCOME_FLOOR_MULTIPLE: float = 3.0
INCOME_FLOOR_LEVEL: str = "HIGH"

#: A level scoring at or above this needs a mismatch citing a real fact id, else it drops.
GUARDED_FROM: float = 0.40
GUARD_FALLBACK_LEVEL: str = "LOW"

MAX_MISMATCHES: int = 4
MAX_MODEL_STEPS: int = 3
