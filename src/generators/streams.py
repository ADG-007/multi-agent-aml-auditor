"""Isolated random streams - one per account, per planted group, per generator decision.

Phase 1 originally drew every random choice from one shared stream, in order. That made the
dataset reproducible, but fragile: adding a single customer shifted every draw after it, so
every existing account's amounts, counterparties and even which account was the structuring
case changed silently. Phase 1's own close showed the cost - filtering Western first names
out of the name list redrew every account's transaction count.

Here each decision draws from a stream of its own, derived from the global seed plus a
stable key naming what the stream is for: ("transactions", "ACC-0042"), ("ring", "RING-03"),
("planting", "base"). A stream's numbers depend only on the seed and its key, never on what
was generated before it. So a new block of customers appended to the population - at any
phase - leaves every existing account's data exactly as it was.

Two rules keep that guarantee, and every generator follows them:
  1. Anything per-entity (a customer, an account's history, a document) draws from a stream
     keyed by that entity's id.
  2. Anything drawn from a pool (which accounts join a ring, which get planted) draws from a
     stream keyed by the pool's name, and the pool itself only ever contains the population
     block it belongs to. A later block plants its own cases from its own pools.

Keys are hashed with SHA-256 rather than Python's built-in hash(), which is salted per
process and would make every run different.
"""

from __future__ import annotations

import hashlib
import random

from faker import Faker


def opaque_id(prefix: str, *keys: object) -> str:
    """A stable identifier that says nothing about what it identifies.

    Used where a readable id would leak ground truth - a counterparty numbered "COL-01", or
    one numbered after the account that introduced it, would tell a pipeline stage what it
    is looking at without any analysis. Derived from `keys` alone, so it is the same in every
    run and never shifts when something else is added.
    """
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
        """A Faker for `locale`, reseeded for `keys`.

        One Faker per locale is kept and reseeded rather than built per call, because
        constructing a Faker loads every provider and is slow at the thousands of names the
        mule block needs. The instance is shared, so use it before asking for the next one.
        """
        faker = self._fakers.get(locale)
        if faker is None:
            faker = self._fakers[locale] = Faker(locale)
        faker.seed_instance(self.seed_for("faker", locale, *keys))
        return faker
