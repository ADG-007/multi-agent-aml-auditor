"""Format-plausible Indian identifier generation.

phase_1_execution.md calibrates this deliberately: reproduce the real field set, length and
visible pattern of each identifier, but not its internal algorithm. So an Aadhaar number
here is twelve digits grouped 4-4-4 and never starting 0 or 1, exactly as a real one looks,
but it does not carry a valid Verhoeff check digit; a PAN matches AAAAA9999A including the
real fourth-character holder-type code, but its tenth character is not a computed check
letter.

The line was drawn there because no evaluator hand-verifies a synthetic Aadhaar's checksum,
while everyone notices an ID number of the wrong shape. Nothing downstream depends on
checksum validity either: Stage 1's forgery detection works from visual tampering and from
cross-checking OCR output against the KYC record, neither of which touches the check digit.
"""

from __future__ import annotations

import random
import string

#: Real PAN fourth-character holder-type codes. P is an individual, C a company, and so on.
#: Every customer here is a natural person, so P is the only correct value.
PAN_HOLDER_TYPE_INDIVIDUAL = "P"


def aadhaar_number(rng: random.Random) -> str:
    """Twelve digits, grouped 4-4-4, first digit 2-9.

    UIDAI never issues a number beginning 0 or 1, which is the one structural rule a reader
    might actually notice. No Verhoeff check digit - see the module docstring.
    """
    first = rng.randint(2, 9)
    rest = "".join(str(rng.randint(0, 9)) for _ in range(11))
    digits = f"{first}{rest}"
    return f"{digits[0:4]} {digits[4:8]} {digits[8:12]}"


def pan_number(rng: random.Random, surname: str) -> str:
    """AAAAA9999A.

    Characters one to three are a sequence code, the fourth is the holder type (P for an
    individual) and the fifth is the first letter of the holder's surname - both real rules
    worth honouring because they make the PAN consistent with the name printed beside it.
    The trailing letter is a plausible stand-in, not a computed check character.
    """
    prefix = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
    surname_initial = (surname[:1] or "X").upper()
    if surname_initial not in string.ascii_uppercase:
        surname_initial = "X"
    digits = "".join(str(rng.randint(0, 9)) for _ in range(4))
    check = rng.choice(string.ascii_uppercase)
    return f"{prefix}{PAN_HOLDER_TYPE_INDIVIDUAL}{surname_initial}{digits}{check}"


def passport_number(rng: random.Random) -> str:
    """One letter followed by seven digits.

    Q, X and Z are not used as the leading letter on Indian passports.
    """
    letter = rng.choice([c for c in string.ascii_uppercase if c not in "QXZ"])
    digits = "".join(str(rng.randint(0, 9)) for _ in range(7))
    return f"{letter}{digits}"


def account_number(rng: random.Random) -> str:
    """Eleven to sixteen digits, the range Indian banks actually issue in."""
    length = rng.choice((11, 12, 13, 14, 15, 16))
    return "".join(str(rng.randint(0, 9)) for _ in range(length))


def device_id(rng: random.Random) -> str:
    """A stable per-device fingerprint of the kind a banking app reports."""
    return "DEV-" + "".join(rng.choice("0123456789ABCDEF") for _ in range(12))


def ip_address(rng: random.Random) -> str:
    """An IPv4 address inside ranges allocated to Indian consumer ISPs.

    Real allocations, so the addresses look like retail broadband and mobile data rather
    than arbitrary integers - the graph treats an IP as a shared identifier, and a plausible
    one makes the shared-IP finding read correctly in a report.
    """
    prefix = rng.choice(("49", "103", "106", "117", "122", "157", "182", "223"))
    return f"{prefix}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
