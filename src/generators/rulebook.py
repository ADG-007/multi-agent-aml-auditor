"""The rulebook: 24 clauses in 8 categories, as clauses.json, markdown and a ChromaDB index.

The one real part of the data: each clause paraphrases a real source (PMLA Rules, RBI KYC
Master Direction, UAPA 51A, FEMA LRS, FATF). Stage 6 searches it. Stages 0 and 2 read
thresholds from its metadata. The text is fixed, never reworded run to run.
"""

from __future__ import annotations

import json

import chromadb

from src.generators import config as cfg
from src.schemas import RulebookClause, RulebookCategory

#: The collection Stage 6 queries.
COLLECTION_NAME = "rulebook"


def build_clauses() -> list[RulebookClause]:
    """The clauses. Metadata: XB-004 wire threshold, MULE-002 mule floor, STRUCT-001 cash
    threshold and band floor. Stages 0 and 2 both read them, so they always move together."""
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
        RulebookClause(
            clause_id="XB-004",
            title="Reporting of cross-border wire transfers above five lakh rupees",
            category=RulebookCategory.CROSS_BORDER,
            sources=[
                "Prevention of Money-laundering (Maintenance of Records) Rules, 2005 - Rule 3(1)(E)",
                "FIU-IND Cross-Border Wire Transfer Report (CBWTR) guidance",
            ],
            text=(
                "Every cross-border wire transfer of more than five lakh rupees, or its "
                "equivalent in foreign currency, where either the origin or the destination "
                "of the funds is in India, must be recorded and reported to FIU-IND in the "
                "monthly Cross-Border Wire Transfer Report. The threshold applies to each "
                "transfer individually. The purpose code declared for the remittance must be "
                "consistent with the customer's profile; a declared purpose the customer's "
                "circumstances cannot support is a ground for further examination."
            ),
            metadata={"threshold_inr": cfg.CBWTR_THRESHOLD_INR},
        ),
        RulebookClause(
            clause_id="XB-005",
            title="Round-tripping of funds through overseas entities",
            category=RulebookCategory.CROSS_BORDER,
            sources=[
                "Foreign Exchange Management (Overseas Investment) Rules, 2022 - Rule 19(3)",
                "FATF Recommendation 20 (Reporting of suspicious transactions)",
            ],
            text=(
                "Funds remitted from India to an overseas entity that return to India, to the "
                "same customer or to connected parties, from that entity or from an entity "
                "linked to it - by common registered address, ownership or control - must be "
                "treated as potential round-tripping. Overseas investment structures that "
                "route money back into India are permitted only within the limits of the "
                "Overseas Investment Rules. Where remittances out and receipts back are close "
                "in amount and timing and the declared purposes differ, the commercial "
                "rationale must be established, and a suspicious transaction report "
                "considered if it cannot be."
            ),
        ),
        # ----------------------------------------------------------------- MONEY MULES ---
        RulebookClause(
            clause_id="MULE-001",
            title="Operation of bank accounts and money mules",
            category=RulebookCategory.MONEY_MULE,
            sources=[
                "RBI Master Direction on Know Your Customer, 2016 (as amended) - Operation of "
                "Bank Accounts and Money Mules",
                "FATF Recommendation 20 (Reporting of suspicious transactions)",
            ],
            text=(
                "Criminals launder the proceeds of fraud schemes such as phishing, investment "
                "and impersonation scams through deposit accounts belonging to third parties "
                "they recruit or pay, known as money mules. Instructions on the opening of "
                "accounts and the monitoring of transactions must be strictly followed to "
                "minimise the operation of such accounts. The bank must monitor transactions "
                "diligently to identify accounts being operated as money mules and report them "
                "to FIU-IND as suspicious. Where an account opened and operated at the bank is "
                "established to be a money mule and no suspicious transaction report was filed, "
                "the bank is deemed not to have complied with these requirements."
            ),
        ),
        RulebookClause(
            clause_id="MULE-002",
            title="Pass-through of funds from unrelated remitters: suspected money-mule transactions",
            category=RulebookCategory.MONEY_MULE,
            sources=[
                "RBI draft Standard Operating Procedure on Suspected Money Mule Accounts "
                "(proposed Annex III to the KYC Directions)",
                "FATF, Illicit Financial Flows from Cyber-Enabled Fraud (November 2023)",
            ],
            text=(
                "Payments from many unrelated people that arrive in an account and are "
                "forwarded onward within hours, usually to one beneficiary, leaving little "
                "balance behind, are the characteristic pattern of a money-mule account. A "
                "suspected money-mule transaction is one of one thousand rupees or more that "
                "transaction monitoring flags as possible proceeds of mule activity or "
                "cyber-enabled fraud. Further indicators are activity disproportionate to the "
                "customer's declared profile, an account recently opened or long dormant that "
                "suddenly becomes active, and linkage to an account already identified as a "
                "mule, such as a common beneficiary or a common device. No single indicator is "
                "conclusive: a legitimate business receiving many small payments is told apart "
                "by where its money goes and how long it stays. On identification the bank "
                "places a temporary debit hold, informs the account holder and seeks an "
                "explanation, and either lifts the hold or reports the account to law "
                "enforcement, filing a suspicious transaction report with FIU-IND."
            ),
            metadata={"txn_floor_inr": cfg.MULE_TXN_FLOOR_INR},
        ),
        # ----------------------------------------------------- SUSPICIOUS TRANSACTIONS ---
        RulebookClause(
            clause_id="STR-001",
            title="Suspicious transactions on any payment channel",
            category=RulebookCategory.SUSPICIOUS_TRANSACTION,
            sources=[
                "PMLA (Maintenance of Records) Rules, 2005 - Rule 2(1)(g) (definition of "
                "suspicious transaction)",
                "PMLA (Maintenance of Records) Rules, 2005 - Rule 3(1)(D)",
                "FATF Recommendation 20 (Reporting of suspicious transactions)",
            ],
            text=(
                "A suspicious transaction is any transaction, including an attempted one and "
                "whether or not it is made in cash, that gives a person acting in good faith "
                "reasonable grounds to suspect that it involves the proceeds of crime, that "
                "appears to be made in circumstances of unusual or unjustified complexity, that "
                "appears to have no economic rationale or bona fide purpose, or that may involve "
                "the financing of terrorism. Every such transaction must be reported to FIU-IND "
                "as a suspicious transaction report. The obligation has no amount threshold and "
                "applies to every payment channel: electronic transfers, cheques, cards and "
                "cross-border wires as much as cash."
            ),
        ),
        RulebookClause(
            clause_id="STR-002",
            title="Domestic non-cash transfers: no threshold report, examined by pattern",
            category=RulebookCategory.SUSPICIOUS_TRANSACTION,
            sources=[
                "PMLA (Maintenance of Records) Rules, 2005 - Rules 3(1)(A), 3(1)(D) and 3(1)(E)",
                "RBI Master Direction on KYC - ongoing due diligence and transaction monitoring",
                "FATF Recommendation 20",
            ],
            text=(
                "Domestic electronic transfers (NEFT, RTGS, IMPS and UPI), cheques, card "
                "payments and ATM withdrawals carry no amount-based reporting obligation: the "
                "cash transaction report covers cash only and the cross-border wire report "
                "covers cross-border wires only. Keeping such a transfer below a given amount "
                "therefore avoids no report and is not structuring. These channels are examined "
                "by pattern instead: funds that pass through the account rapidly, flows to or "
                "from many unrelated counterparties, layering through a chain of accounts, and "
                "activity inconsistent with the customer's declared profile. Where the activity "
                "has no evident economic rationale, it is reported as a suspicious transaction."
            ),
        ),
    ]


def write_markdown(clauses: list[RulebookClause]) -> None:
    """One markdown file per clause plus an index, for reading (git-ignored)."""
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

    index = ["# Policy Rulebook", "", f"{len(clauses)} clauses across {len({c.category for c in clauses})} categories.", ""]
    for category in RulebookCategory:
        index.append(f"## {category.value}")
        index.append("")
        for clause in clauses:
            if clause.category == category:
                index.append(f"- **{clause.clause_id}** - {clause.title}")
        index.append("")
    (cfg.RULEBOOK_DIR / "README.md").write_text("\n".join(index), encoding="utf-8")


def write_json(clauses: list[RulebookClause]) -> None:
    """clauses.json: the committed source of truth the stages read."""
    cfg.RULEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    cfg.RULEBOOK_JSON_FILE.write_text(
        json.dumps([clause.model_dump(mode="json") for clause in clauses], indent=2),
        encoding="utf-8",
    )


def embed(clauses: list[RulebookClause]) -> int:
    """Embed one chunk per clause into ChromaDB, with its bundled local model (offline,
    downloaded once)."""
    cfg.RULEBOOK_CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(cfg.RULEBOOK_CHROMA_DIR))

    # Rebuilt from scratch each run so a re-run never leaves a stale clause behind.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    # Cosine space (Stage 6's cutoff is a cosine distance). Fixed at creation time.
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


def rebuild_index() -> int:
    """Rebuild the ChromaDB index from clauses.json (the index is not committed: builds differ)."""
    raw = json.loads(cfg.RULEBOOK_JSON_FILE.read_text(encoding="utf-8"))
    return embed([RulebookClause.model_validate(item) for item in raw])


if __name__ == "__main__":
    count = rebuild_index()
    print(f"Rebuilt {cfg.RULEBOOK_CHROMA_DIR} with {count} clauses.")
