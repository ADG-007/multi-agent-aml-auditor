"""Indian names: Faker en_IN with its Western first names (Wyatt, Liam ...) filtered out."""

from __future__ import annotations

from faker import Faker
from faker.providers.person.en_IN import Provider as _EnInPersons

#: First names in Faker's en_IN lists that are not Indian given names.
WESTERN_FIRST_NAMES = frozenset(
    {
        "Alexander", "Andrew", "Anthony", "Benjamin", "Caleb", "Charles", "Christopher",
        "Daniel", "David", "Dominic", "Elijah", "Ethan", "Finn", "Frado", "Frederick",
        "Gabriel", "Gavin", "George", "Harrison", "Henry", "Isaac", "Isaiah", "Jack",
        "Jackson", "Jacob", "Jason", "Jeremiah", "Jonathan", "Joshua", "Kai", "Kevin",
        "Laban", "Liam", "Logan", "Luke", "Mason", "Matthew", "Max", "Michael", "Nathan",
        "Nathaniel", "Nicholas", "Noah", "Oliver", "Oscar", "Owen", "Patrick", "Peter",
        "Quincy", "Robert", "Ryan", "Samuel", "Simon", "Theodore", "Thomas", "Timothy",
        "Tristan", "Victor", "Vincent", "William", "Wyatt", "Xavier",
    }
)

#: Indian male first names, for fields that print a name other than the holder's own (the
#: father's name on a PAN card).
INDIAN_MALE_FIRST_NAMES: tuple[str, ...] = tuple(
    sorted(set(_EnInPersons.first_names_male) - WESTERN_FIRST_NAMES)
)


def _is_indian(name: str) -> bool:
    return name.split()[0] not in WESTERN_FIRST_NAMES


def indian_name(faker: Faker, gender: str | None = None) -> str:
    """A full name from an en_IN Faker, redrawn until the first name is Indian.

    `gender` is "MALE", "FEMALE" or None for either. Redrawing keeps the result
    deterministic under a seeded Faker.
    """
    draw = {"MALE": faker.name_male, "FEMALE": faker.name_female}.get(gender or "", faker.name)
    while True:
        name = draw()
        if _is_indian(name):
            return name
