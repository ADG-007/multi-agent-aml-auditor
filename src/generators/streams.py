"""Isolated random streams: seed + a key such as ("transactions", "ACC-0042").

A stream depends only on its key, so adding data never changes existing data.
1. Per-entity draws use a stream keyed by the entity's id.
2. Pool draws use a stream keyed by the pool, and a pool holds only its own block.
Keys hash with SHA-256 (Python's hash() is salted per process).
"""

from __future__ import annotations

import hashlib
import random

from faker import Faker


def opaque_id(prefix: str, *keys: object) -> str:
    """A stable id that reveals nothing (no "COL-01"), derived from `keys` alone."""
    material = "|".join(str(key) for key in keys).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(material).hexdigest()[:8].upper()}"


class Streams:
    """Hands out independent, reproducible random streams for one global seed."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._fakers: dict[str, Faker] = {}

    def seed_for(self, *keys: object) -> int:
        """A 64-bit seed determined only by the global seed and `keys`."""
        material = "|".join([str(self.seed), *(str(key) for key in keys)]).encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")

    def rng(self, *keys: object) -> random.Random:
        """A fresh random stream for `keys`."""
        return random.Random(self.seed_for(*keys))

    def faker(self, locale: str, *keys: object) -> Faker:
        """A shared Faker for `locale`, reseeded for `keys` (building one is slow). Use it
        before asking for the next one."""
        faker = self._fakers.get(locale)
        if faker is None:
            faker = self._fakers[locale] = Faker(locale)
        faker.seed_instance(self.seed_for("faker", locale, *keys))
        return faker
