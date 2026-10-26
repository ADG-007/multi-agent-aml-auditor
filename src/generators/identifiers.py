"""Indian identifiers with the real shape but no real check digit (nothing depends on it)."""

from __future__ import annotations

import random
import string

#: Real PAN fourth-character holder-type codes. P is an individual, C a company, and so on.
#: Every customer here is a natural person, so P is the only correct value.
PAN_HOLDER_TYPE_INDIVIDUAL = "P"


def aadhaar_number(rng: random.Random) -> str:
    """Twelve digits, 4-4-4, first digit 2-9 (as UIDAI issues). No Verhoeff digit."""
    first = rng.randint(2, 9)
    rest = "".join(str(rng.randint(0, 9)) for _ in range(11))
    digits = f"{first}{rest}"
    return f"{digits[0:4]} {digits[4:8]} {digits[8:12]}"


def pan_number(rng: random.Random, surname: str) -> str:
    """AAAAA9999A: 4th letter P (individual), 5th = surname initial. Last letter not computed."""
    prefix = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
    surname_initial = (surname[:1] or "X").upper()
    if surname_initial not in string.ascii_uppercase:
        surname_initial = "X"
    digits = "".join(str(rng.randint(0, 9)) for _ in range(4))
    check = rng.choice(string.ascii_uppercase)
    return f"{prefix}{PAN_HOLDER_TYPE_INDIVIDUAL}{surname_initial}{digits}{check}"


def passport_number(rng: random.Random) -> str:
    """One letter (not Q, X or Z) and seven digits."""
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
    """An IPv4 address in real Indian consumer-ISP ranges."""
    prefix = rng.choice(("49", "103", "106", "117", "122", "157", "182", "223"))
    return f"{prefix}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
