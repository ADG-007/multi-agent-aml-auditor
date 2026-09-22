"""Generator 8 of 9 - the policy rulebook, as markdown and as a ChromaDB collection.

Consumed by Stage 6 through retrieval, and by Stages 0 and 2 through a direct metadata
lookup on STRUCT-001.

Everything in the simulated bank is invented, but the rulebook's *content* is not: each
clause paraphrases a real published source - the PMLA Maintenance of Records Rules, the RBI
Master Direction on KYC, UAPA Section 51A, FEMA's Liberalised Remittance Scheme, and the
FATF Recommendations. That single exception to the synthetic-data invariant is deliberate
and recorded in MASTER_PLAN.md: a rulebook of invented rules would make every compliance
citation meaningless, while inventing customers costs nothing.

The clause text is paraphrase rather than transcription - the substance of each obligation
in the register a bank's internal policy manual uses.

Eighteen clauses across six categories, three each. The categories deliberately mirror what
the evidence stages actually detect, so policy_gap_identified stays rare in practice rather
than firing on every case because the rulebook never covered the finding.
"""

from __future__ import annotations

import json

import chromadb

from src.generators import config as cfg
from src.schemas import RulebookClause, RulebookCategory

#: The collection Stage 6 queries.
COLLECTION_NAME = "rulebook"


def build_clauses() -> list[RulebookClause]:
    """The authored clause set.

    STRUCT-001 is the only clause carrying machine-readable metadata, and it carries both
    numbers rather than one: the reporting ceiling and the lower edge of the structuring
    band. Stage 0's routing check and Stage 2's confidence formula both read from here, so a
    policy update injected at demo time moves both together and can never move only one.
    """
    return [
        # ---------------------------------------------------------------- STRUCTURING ---
        RulebookClause(
            clause_id="STRUCT-001",
            title="Cash transaction reporting threshold and sub-threshold structuring",
            category=RulebookCategory.STRUCTURING,
            sources=[
                "PMLA (Maintenance of Records) Rules, 2005 - Rule 3(1)(A)",
                "FATF Recommendation 20 (Reporting of suspicious transactions)",
            ],
            text=(
                "All cash transactions of a value exceeding ten lakh rupees, or their "
                "equivalent in foreign currency, must be reported to the Financial "
                "Intelligence Unit. Deposits deliberately kept below this ceiling in order "
                "to avoid the reporting obligation constitute structuring and are "
                "reportable as suspicious irrespective of the individual amounts involved. "
                "Activity concentrated in the band immediately beneath the threshold is to "
                "be treated as an indicator warranting examination, not as compliant "
                "conduct."
            ),
            metadata={
                "threshold_inr": cfg.CTR_THRESHOLD_INR,
                "near_threshold_pct": cfg.NEAR_THRESHOLD_PCT,
            },
        ),
        RulebookClause(
            clause_id="STRUCT-002",
            title="Series of integrally connected cash transactions",
            category=RulebookCategory.STRUCTURING,
            sources=[
                "PMLA (Maintenance of Records) Rules, 2005 - Rule 3(1)(B)",
                "FIU-IND Cash Transaction Report filing guidance",
            ],
            text=(
                "Where a series of cash transactions is integrally connected to one another "
                "and the aggregate value within a calendar month exceeds ten lakh rupees, "
                "the series is to be treated as a single transaction for reporting "
                "purposes. Connectedness is assessed on the facts, including proximity in "
                "time, common counterparties, common handling and the absence of an "
                "independent commercial rationale for splitting the value."
            ),
        ),
        RulebookClause(
            clause_id="STRUCT-003",
            title="Sudden surge in account turnover",
            category=RulebookCategory.STRUCTURING,
            sources=[
                "RBI Master Direction on KYC - transaction monitoring requirements",
                "FATF Recommendation 20",
            ],
            text=(
                "A marked increase in the volume or value of transactions passing through "
                "an account, when set against that account's own established pattern and "
                "the customer's declared profile, requires examination. A surge that cannot "
                "be explained by a change in the customer's stated circumstances is an "
                "indicator of the account being used as a conduit rather than for the "
                "purpose for which it was opened."
            ),
        ),
        # --------------------------------------------------------------------- KYC/CDD ---
        RulebookClause(
            clause_id="KYC-001",
            title="Customer due diligence and risk categorisation",
            category=RulebookCategory.KYC_CDD,
            sources=[
                "RBI Master Direction on Know Your Customer, 2016 (as amended)",
                "FATF Recommendation 10 (Customer due diligence)",
            ],
            text=(
                "Regulated entities must identify every customer, verify that identity "
                "using independent and reliable documents, understand the purpose and "
                "intended nature of the relationship, and assign a risk categorisation of "
                "low, medium or high. The categorisation must be reviewed when the "
                "customer's circumstances or account behaviour change materially. Due "
                "diligence is a continuing obligation, not a one-time step completed at "
                "account opening."
            ),
        ),
        RulebookClause(
            clause_id="KYC-002",
            title="Inconsistency between declared profile and observed activity",
            category=RulebookCategory.KYC_CDD,
            sources=[
                "RBI Master Direction on KYC - ongoing due diligence",
                "FATF Recommendation 10",
            ],
            text=(
                "Transactions inconsistent with a customer's declared occupation, declared "
                "income or stated source of funds must be examined, and the rationale for "
                "the activity established and recorded. Where the customer cannot "
                "satisfactorily explain activity materially exceeding the declared profile, "
                "or where the stated purpose of the account excludes the activity actually "
                "observed, the relationship must be escalated for suspicious transaction "
                "reporting consideration."
            ),
        ),
        RulebookClause(
            clause_id="KYC-003",
            title="Periodic updation and change in customer profile",
            category=RulebookCategory.KYC_CDD,
            sources=["RBI Master Direction on KYC - periodic updation requirements"],
            text=(
                "Customer records must be updated periodically, at intervals determined by "
                "the assigned risk category, and whenever the entity becomes aware of a "
                "change in the customer's identity, address, occupation or beneficial "
                "ownership. A profile updated to state materially different circumstances "
                "requires the account's existing activity to be re-assessed against the "
                "revised profile rather than the superseded one."
            ),
        ),
        # ------------------------------------------------------------------- SANCTIONS ---
        RulebookClause(
            clause_id="SANC-001",
            title="Designated persons and entities under Section 51A",
            category=RulebookCategory.SANCTIONS,
            sources=[
                "Unlawful Activities (Prevention) Act, 1967 - Section 51A",
                "UN Security Council Consolidated List (1267/1989 Committee)",
                "FATF Recommendation 6 (Targeted financial sanctions - terrorism)",
            ],
            text=(
                "Funds, financial assets and economic resources held by or on behalf of an "
                "individual or entity designated under Section 51A must be frozen without "
                "delay and without prior notice to the party. No funds may be made "
                "available, directly or indirectly, for the benefit of a designated party. "
                "The obligation arises on designation and does not depend on a prior "
                "judicial finding against the party."
            ),
        ),
        RulebookClause(
            clause_id="SANC-002",
            title="Name screening, aliases and match resolution",
            category=RulebookCategory.SANCTIONS,
            sources=[
                "FATF Recommendation 6",
                "RBI Master Direction on KYC - screening against designated lists",
            ],
            text=(
                "Customers and counterparties must be screened against the designated lists "
                "at onboarding and whenever the lists are updated. Screening must account "
                "for known aliases, alternative transliterations and name order variation. "
                "A potential match must be resolved against corroborating identifiers such "
                "as date of birth, identification number or address before it is discounted; "
                "a match discounted on name alone is not a resolved match."
            ),
        ),
        RulebookClause(
            clause_id="SANC-003",
            title="Reporting obligations on a positive designation match",
            category=RulebookCategory.SANCTIONS,
            sources=[
                "UAPA Section 51A read with the relevant MHA procedure",
                "FATF Recommendation 7 (Targeted financial sanctions - proliferation)",
            ],
            text=(
                "A confirmed match against a designated party must be reported to the "
                "designated authority and to the Financial Intelligence Unit immediately, "
                "and the account must not be operated pending instruction. Attempted "
                "transactions by or on behalf of a designated party are themselves "
                "reportable, whether or not the transaction was completed."
            ),
        ),
        # --------------------------------------------------------------------- NETWORK ---
        RulebookClause(
            clause_id="NET-001",
            title="Beneficial ownership and control",
            category=RulebookCategory.NETWORK,
            sources=[
                "PMLA (Maintenance of Records) Rules, 2005 - beneficial ownership provisions",
                "FATF Recommendations 24 and 25 (Transparency of legal persons and arrangements)",
            ],
            text=(
                "The natural person who ultimately owns or controls a customer, or on whose "
                "behalf a transaction is conducted, must be identified and verified. Where "
                "accounts in different names are in substance controlled by one person, the "
                "relationship must be assessed as a whole. Nominee arrangements that "
                "obscure the controlling person are a higher-risk indicator requiring "
                "enhanced due diligence."
            ),
        ),
        RulebookClause(
            clause_id="NET-002",
            title="Accounts linked by shared operating identifiers",
            category=RulebookCategory.NETWORK,
            sources=[
                "RBI Master Direction on KYC - transaction monitoring",
                "FATF Recommendation 20",
            ],
            text=(
                "Unrelated accounts operated from a common device, a common network "
                "address, or registered to a common address without an evident family or "
                "business relationship are to be examined together rather than "
                "individually. Common operation is not by itself evidence of wrongdoing - "
                "shared household and shared business premises are ordinary - but it is a "
                "basis for examining whether the accounts form a single layering structure."
            ),
        ),
        RulebookClause(
            clause_id="NET-003",
            title="Politically exposed persons and close associates",
            category=RulebookCategory.NETWORK,
            sources=[
                "RBI Master Direction on KYC - provisions relating to politically exposed persons",
                "FATF Recommendation 12",
            ],
            text=(
                "Relationships with politically exposed persons, their family members and "
                "close associates require senior management approval, establishment of the "
                "source of funds and the source of wealth, and enhanced ongoing monitoring. "
                "The requirement extends to a customer who becomes politically exposed "
                "during the relationship, not only to one identified as such at onboarding."
            ),
        ),
        # ------------------------------------------------------------ DOCUMENT FORGERY ---
        RulebookClause(
            clause_id="DOC-001",
            title="Officially valid documents for identity and address",
            category=RulebookCategory.DOCUMENT_FORGERY,
            sources=[
                "RBI Master Direction on KYC - definition of Officially Valid Document",
                "FATF Recommendation 10",
            ],
            text=(
                "Identity and address must be established using an officially valid "
                "document, which includes the passport, the driving licence, the Aadhaar "
                "number in the permitted form, the Voter's Identity Card, and the job card "
                "issued under MGNREGA. The Permanent Account Number is required separately "
                "for tax purposes and does not by itself establish address. Verification "
                "must be against the original or an equivalently reliable electronic "
                "verification."
            ),
        ),
        RulebookClause(
            clause_id="DOC-002",
            title="Indications of tampering or fabrication in submitted documents",
            category=RulebookCategory.DOCUMENT_FORGERY,
            sources=[
                "RBI Master Direction on KYC - verification requirements",
                "FATF Recommendation 10 (reliability of identification data)",
            ],
            text=(
                "A document showing inconsistent typography, misaligned fields, "
                "irregularities in the background security pattern, or particulars that do "
                "not agree with the customer's other records must not be accepted as an "
                "officially valid document. The account must not be operated on the "
                "strength of the disputed document pending resolution, and the "
                "discrepancy must be escalated for a documented human decision rather than "
                "resolved by the verifying officer alone."
            ),
        ),
        RulebookClause(
            clause_id="DOC-003",
            title="Expired or invalid identity documents",
            category=RulebookCategory.DOCUMENT_FORGERY,
            sources=["RBI Master Direction on KYC - validity of documents obtained"],
            text=(
                "Where a document submitted for identity or address verification has "
                "expired, the customer must provide a current document within the permitted "
                "period. An account whose only officially valid document has lapsed is to be "
                "treated as having incomplete due diligence, and operations may be "
                "restricted until the record is regularised."
            ),
        ),
        # ---------------------------------------------------------------- CROSS-BORDER ---
        RulebookClause(
            clause_id="XB-001",
            title="Originator and beneficiary information on cross-border transfers",
            category=RulebookCategory.CROSS_BORDER,
            sources=[
                "FATF Recommendation 16 (Wire transfers)",
                "RBI instructions on wire transfers and originator information",
            ],
            text=(
                "Cross-border wire transfers must carry accurate originator information - "
                "name, account number and address or an equivalent identifier - together "
                "with the beneficiary's name and account number, and that information must "
                "remain with the transfer throughout the payment chain. Transfers lacking "
                "required information must be held, rejected or made subject to follow-up, "
                "and repeated transfers with deficient information are reportable."
            ),
        ),
        RulebookClause(
            clause_id="XB-002",
            title="Liberalised Remittance Scheme limits and permissible purposes",
            category=RulebookCategory.CROSS_BORDER,
            sources=[
                "Foreign Exchange Management Act, 1999",
                "RBI Master Direction - Liberalised Remittance Scheme",
            ],
            text=(
                "A resident individual may remit up to the limit prescribed under the "
                "Liberalised Remittance Scheme in a financial year for permitted current "
                "and capital account transactions. Splitting remittances across family "
                "members or across accounts in order to exceed the individual limit is not "
                "permitted. Remittances for prohibited purposes, and remittances whose "
                "stated purpose is inconsistent with the underlying activity, must be "
                "refused and examined."
            ),
        ),
        RulebookClause(
            clause_id="XB-003",
            title="Remittances involving higher-risk jurisdictions and concentrated counterparties",
            category=RulebookCategory.CROSS_BORDER,
            sources=[
                "FATF Recommendation 19 (Higher-risk countries)",
                "RBI Master Direction on KYC - enhanced due diligence",
            ],
            text=(
                "Enhanced due diligence must be applied to business relationships and "
                "transactions involving jurisdictions identified as higher risk. Repeated "
                "outward transfers from otherwise unrelated accounts to a small common set "
                "of overseas beneficiaries, particularly where those beneficiaries are "
                "registered in jurisdictions offering limited beneficial ownership "
                "transparency, require the commercial rationale to be established and "
                "recorded before the transfers are permitted to continue."
            ),
        ),
    ]


def write_markdown(clauses: list[RulebookClause]) -> None:
    """Write one tagged markdown file per clause, plus a combined index.

    Markdown is the authored source of record: a human can read and edit it, and the
    ChromaDB collection is a derived artefact rebuilt from it.
    """
    cfg.RULEBOOK_MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)

    for clause in clauses:
        body = [
            f"# {clause.clause_id} - {clause.title}",
            "",
            f"**Category:** {clause.category.value}",
            "",
            "**Sources:**",
            "",
        ]
        body.extend(f"- {source}" for source in clause.sources)
        body.extend(["", clause.text, ""])
        if clause.metadata:
            body.append("**Machine-readable parameters:**")
            body.append("")
            body.extend(f"- `{key}`: {value}" for key, value in clause.metadata.items())
            body.append("")
        (cfg.RULEBOOK_MARKDOWN_DIR / f"{clause.clause_id}.md").write_text(
            "\n".join(body), encoding="utf-8"
        )

    index = ["# Policy Rulebook", "", f"{len(clauses)} clauses across six categories.", ""]
    for category in RulebookCategory:
        index.append(f"## {category.value}")
        index.append("")
        for clause in clauses:
            if clause.category == category:
                index.append(f"- **{clause.clause_id}** - {clause.title}")
        index.append("")
    (cfg.RULEBOOK_DIR / "README.md").write_text("\n".join(index), encoding="utf-8")


def write_json(clauses: list[RulebookClause]) -> None:
    """Machine-readable clause set.

    This is what Stages 0 and 2 read for the STRUCT-001 threshold lookup - parsing markdown
    for a number that governs routing would be fragile in exactly the way a compliance
    system cannot afford.
    """
    cfg.RULEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    cfg.RULEBOOK_JSON_FILE.write_text(
        json.dumps([clause.model_dump(mode="json") for clause in clauses], indent=2),
        encoding="utf-8",
    )


def embed(clauses: list[RulebookClause]) -> int:
    """Embed one chunk per clause into a persistent ChromaDB collection.

    One chunk per clause, matching the retrieval design in MASTER_PLAN.md Phase 8: clauses
    are short and self-contained, so splitting them would return fragments that cannot be
    cited as a whole clause.

    Embeddings use ChromaDB's bundled local model rather than an API, which keeps the
    pipeline runnable offline once the data is generated and adds no per-query cost. The
    model file downloads once on first use and is cached thereafter.
    """
    cfg.RULEBOOK_CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(cfg.RULEBOOK_CHROMA_DIR))

    # Rebuilt from scratch each run so a re-run never leaves a stale clause behind.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    # Cosine space, not Chroma's default squared-L2. MASTER_PLAN.md Phase 8 specifies a
    # cosine-distance cutoff for Stage 6 retrieval, and the distance metric is baked into the
    # collection at creation time - so it has to be set here, in the data layer, rather than
    # chosen later when the stage is written.
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[clause.clause_id for clause in clauses],
        documents=[f"{clause.title}. {clause.text}" for clause in clauses],
        metadatas=[
            {
                "clause_id": clause.clause_id,
                "title": clause.title,
                "category": clause.category.value,
                # Chroma metadata values must be scalars, so the source list is flattened.
                "sources": " | ".join(clause.sources),
                **{key: value for key, value in clause.metadata.items()},
            }
            for clause in clauses
        ],
    )
    return collection.count()
