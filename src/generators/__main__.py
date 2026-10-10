"""Run every generator, in dependency order, under one fixed seed.

    python -m src.generators

The order is not arbitrary. Customers come first because every other artefact keys off a
customer or account id. Rings come next because ring membership is one of the five evidence
channels and has to be settled before the planting plan is drawn. The plan itself is then
fixed once, and every generator downstream reads it rather than deciding independently -
which is what keeps a structuring case in the transaction data and a structuring row in the
ledger describing the same account.

The adverse-media corpus is built straight after the transactions, because it is fixed
against the founding population and every later block's customer and counterparty names are
then cleared against it before anything else - KYC narratives, document images, the graph -
uses them.

Every random choice comes from an isolated stream keyed by what it is for (streams.py), so a
population block can be appended at any phase without changing existing data.
`--exclude-block` generates without a block, which is how that guarantee is checked.

Output lands in data/, and the JSON is committed to the repository rather than regenerated
per run. That is deliberate: a library version change in a different environment would
otherwise shift the data underneath the ground-truth labels without anything appearing to
break. The two binary outputs are the exception - the document images and the rulebook's
search index are rebuilt locally from the committed JSON:

    python -m src.generators.id_documents
    python -m src.generators.rulebook
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter

import numpy as np
from faker import Faker

from src.generators import adverse_media as adverse_media_module
from src.generators import config as cfg
from src.generators import customers as customers_module
from src.generators import evidence_ledger as ledger_module
from src.generators import graph as graph_module
from src.generators import id_documents as documents_module
from src.generators import kyc_profiles as kyc_module
from src.generators import rings as rings_module
from src.generators import rulebook as rulebook_module
from src.generators import transactions as transactions_module
from src.generators.streams import Streams


def _write_models(path, models) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([model.model_dump(mode="json") for model in models], indent=2),
        encoding="utf-8",
    )


def _step(index: int, total: int, label: str) -> float:
    print(f"[{index}/{total}] {label} ...", flush=True)
    return time.perf_counter()


def _done(started: float, detail: str) -> None:
    print(f"        {detail}  ({time.perf_counter() - started:.1f}s)", flush=True)


def main(
    seed: int = cfg.SEED,
    skip_embeddings: bool = False,
    exclude_blocks: tuple[str, ...] = (),
) -> None:
    """Generate the full dataset."""
    # Every generator draws from its own Streams; the global seeds are only a guard against
    # a library quietly using the module-level generators.
    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)
    streams = Streams(seed)
    blocks = tuple(name for name, _ in cfg.POPULATION_BLOCKS if name not in exclude_blocks)

    total = 9
    print(f"Generating the AML Auditor synthetic dataset (seed={seed}, blocks={', '.join(blocks)})\n")

    started = _step(1, total, "customers and accounts")
    customers, accounts, cohorts = customers_module.generate(streams, blocks)
    _done(started, f"{len(customers)} customers, {len(accounts)} accounts")

    started = _step(2, total, "fraud rings")
    fraud_rings = rings_module.generate(streams, customers, accounts, cohorts)
    _done(started, f"{len(fraud_rings)} rings covering {sum(len(r.member_account_ids) for r in fraud_rings)} accounts")

    started = _step(3, total, "planting plan (drives every channel below)")
    plan = ledger_module.build_planting_plan(streams, accounts, fraud_rings, cohorts)
    multi = sum(1 for a in accounts if plan.real_channels_for(a.account_id) >= 2)
    single = sum(1 for a in accounts if plan.real_channels_for(a.account_id) == 1)
    _done(started, f"{multi} accounts with two or more channels, {single} with one")

    started = _step(4, total, "transactions and counterparties")
    transactions, counterparties = transactions_module.generate(
        streams, customers, accounts, fraud_rings, plan
    )
    _done(started, f"{len(transactions)} transactions, {len(counterparties)} counterparties")

    started = _step(5, total, "adverse media corpus, and later-block name clearance")
    entries = adverse_media_module.generate(streams, customers, accounts, plan, counterparties)
    _write_models(cfg.ADVERSE_MEDIA_FILE, entries)
    customers, counterparties, transactions, renamed = adverse_media_module.clear_later_block_names(
        streams, entries, customers, counterparties, transactions
    )
    _write_models(cfg.CUSTOMERS_FILE, customers)
    _write_models(cfg.ACCOUNTS_FILE, accounts)
    _write_models(cfg.TRANSACTIONS_FILE, transactions)
    _write_models(cfg.COUNTERPARTIES_FILE, counterparties)
    _done(started, f"{len(entries)} entries; {renamed} later-block names redrawn clear of the corpus")

    started = _step(6, total, "KYC profiles")
    profiles = kyc_module.generate(streams, customers, accounts, transactions, counterparties, plan)
    _write_models(cfg.KYC_PROFILES_FILE, profiles)
    _done(started, f"{len(profiles)} narratives")

    started = _step(7, total, "identity document images")
    manifest = documents_module.generate(customers, accounts, plan.forged_document, seed)
    _done(started, f"{len(manifest)} images rendered")

    started = _step(8, total, "entity graph")
    graph = graph_module.build(customers, accounts, transactions, counterparties, plan, fraud_rings)
    graph_module.write(graph)
    _done(started, f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    started = _step(9, total, "rulebook (markdown, json, embeddings)")
    clauses = rulebook_module.build_clauses()
    rulebook_module.write_markdown(clauses)
    rulebook_module.write_json(clauses)
    if skip_embeddings:
        _done(started, f"{len(clauses)} clauses written; embeddings skipped")
    else:
        count = rulebook_module.embed(clauses)
        _done(started, f"{len(clauses)} clauses written, {count} embedded")

    # Written last so it records the plan exactly as every other generator consumed it.
    ledger = ledger_module.write(accounts, plan, fraud_rings)
    print(f"\nGround truth: {len(ledger)} ledger rows -> {cfg.EVIDENCE_LEDGER_FILE}")
    labels = Counter(e.expected_disposition.value for e in ledger)
    print(
        f"Labels: {labels['SUSPICIOUS-ESCALATE']} escalate-worthy, "
        f"{labels['MONITOR']} monitor-worthy, {labels['CLEAR']} clean"
    )
    print("Done. Run `python -m src.generators.verify` to check the output.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.SEED,
        help="Override the global seed. The committed dataset uses the default.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip the ChromaDB step, which downloads an embedding model on first run.",
    )
    parser.add_argument(
        "--exclude-block",
        action="append",
        default=[],
        choices=[name for name, _ in cfg.POPULATION_BLOCKS if name not in cfg.CORPUS_SCREENED_BLOCKS],
        help=(
            "Generate without a population block. Only for checking that adding a block "
            "leaves existing data unchanged - the committed dataset includes every block."
        ),
    )
    args = parser.parse_args()
    main(seed=args.seed, skip_embeddings=args.skip_embeddings, exclude_blocks=tuple(args.exclude_block))
