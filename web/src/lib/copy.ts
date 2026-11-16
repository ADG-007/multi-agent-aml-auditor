// Fixed words of the site: what each pipeline step does, and a plain-word glossary.
// No account id, no metric and no threshold is written here; numbers come from the server.
import type { Disposition } from "./types";

export interface NodeInfo {
  label: string;
  kind: "python" | "model";
  who: string;
  short: string;
  what: string;
  reads: string;
  never: string;
  doing: string[];
}

export const NODE_INFO: Record<string, NodeInfo> = {
  triage: {
    label: "Triage",
    kind: "python",
    who: "Python rules",
    short: "Decide which checks this account needs",
    what: "Fixed rules look at the account's transactions and pick the checks to run. A quiet account still gets its periodic review.",
    reads: "The account's transactions, the rulebook's thresholds, what this run already found.",
    never: "It never scores anything. It only routes.",
    doing: ["Reading the account's transactions", "Checking cash against the rulebook's limits", "Choosing which checks to route"],
  },
  vision: {
    label: "Vision agent",
    kind: "model",
    who: "Qwen 3.5 4B (local) + Tesseract",
    short: "Read the ID card and the KYC profile",
    what: "Reads the ID card twice (text reader first, then the image model) and compares each field with the bank's record. It also pulls the customer's own claims out of the KYC profile.",
    reads: "The ID card images, the customer record, the KYC narrative.",
    never: "The ID check never touches the score. A flagged card only pauses the case for a human.",
    doing: ["Reading the ID card with the text reader", "Asking the image model to read unclear fields again", "Comparing name, date of birth and ID number with the record", "Pulling claims out of the KYC profile"],
  },
  web_research: {
    label: "Web Research agent",
    kind: "model",
    who: "Qwen 3.5 4B (local)",
    short: "Screen names against sanctions and bad news",
    what: "Matches the holder and everyone they pay against the bank's sanctions and adverse-news list. For a look-alike name the model decides if it is the same person.",
    reads: "The bank's screening list (a local file; nothing goes online), the holder's and payees' names.",
    never: "It never invents a match: a look-alike name with nothing else stays a look-alike.",
    doing: ["Screening the account holder's name", "Screening every counterparty name", "Checking look-alike names against date of birth and aliases"],
  },
  judge: {
    label: "Judge",
    kind: "python",
    who: "Python, no model",
    short: "Turn the findings into one verdict",
    what: "Takes the four scored inputs (transaction pattern, KYC, adverse media, links) and applies fixed rules: veto check, weighted average, thresholds, and a gate that asks for a second signal before escalating.",
    reads: "The four confidences, each between 0 and 1.",
    never: "No model does this maths. The same inputs always give the same verdict.",
    doing: ["Checking for a veto", "Taking the weighted average", "Applying the thresholds and the second-signal gate"],
  },
  admin_gate: {
    label: "Admin gate",
    kind: "python",
    who: "A human, when needed",
    short: "Pause if the ID card looks forged",
    what: "If both reads of the ID card disagree with the record, the case stops here until an admin approves or blocks it.",
    reads: "The ID check result only.",
    never: "It never changes the score. It only pauses.",
    doing: ["Checking if the ID card was flagged"],
  },
  graph_label: {
    label: "Graph agent",
    kind: "model",
    who: "Gemma 4 E4B (local)",
    short: "Name the pattern in the money and the links",
    what: "Python has already measured the cash pattern and the shared devices, IPs and addresses. The model reads those facts and names the pattern in plain words.",
    reads: "The pattern and link facts Python computed, each with an id.",
    never: "The label is never scored. It explains; it does not decide.",
    doing: ["Reading the cash pattern Python measured", "Reading links to other accounts", "Choosing the pattern label"],
  },
  rag: {
    label: "RAG agent",
    kind: "model",
    who: "Gemma 4 E4B + embedding search (local)",
    short: "Find the rulebook clauses that apply",
    what: "Searches the bank's rulebook for the clauses that match the findings, and quotes them.",
    reads: "The findings so far, the rulebook index.",
    never: "It never cites a clause that the search did not return.",
    doing: ["Searching the rulebook", "Reading the closest clauses", "Quoting the clauses that apply"],
  },
  auditor: {
    label: "Auditor agent",
    kind: "model",
    who: "Gemma 4 E4B (local)",
    short: "Write the report from a fixed outline",
    what: "Python builds an outline of facts and clauses. The model turns the outline into readable text. A Python check then compares the text with the outline.",
    reads: "The outline only.",
    never: "It cites only clauses the RAG agent retrieved, and every number it writes must be in the outline.",
    doing: ["Building the outline in Python", "Writing the summary and narrative", "Checking every number and clause against the outline"],
  },
  backup: {
    label: "Second opinion",
    kind: "model",
    who: "Qwen 3.5 9B (local, slower)",
    short: "Re-check the unsure answers, once",
    what: "If a small model was unsure or failed its format twice, the bigger model checks that one answer again. At most one re-check per case.",
    reads: "The one question that was unsure.",
    never: "It never runs when nothing was unsure.",
    doing: ["Checking if any answer was unsure", "Asking the bigger model to check again"],
  },
  report: {
    label: "Report",
    kind: "python",
    who: "Python",
    short: "Put the case file together",
    what: "Assembles the verdict, the evidence, every agent's reasoning trail and the text into one draft for the analyst.",
    reads: "Everything above.",
    never: "It is a draft until a human analyst signs it off.",
    doing: ["Assembling the case file"],
  },
};

export const VERDICT: Record<Disposition, { short: string; tone: string; plain: string }> = {
  CLEAR: { short: "CLEAR", tone: "clear", plain: "Nothing suspicious was found." },
  MONITOR: { short: "MONITOR", tone: "monitor", plain: "One real signal. Keep watching this account." },
  "SUSPICIOUS-ESCALATE": { short: "ESCALATE", tone: "escalate", plain: "Two or more real signals, or a sanctions hit. Send to an analyst." },
  BLOCKED_BY_ADMIN_REVIEW: { short: "BLOCKED", tone: "blocked", plain: "An admin blocked the case after the ID check." },
};

export const INPUT_NAMES: Record<string, string> = {
  TransactionPattern: "Transaction pattern",
  KYC: "KYC mismatch",
  AdverseMedia: "Sanctions and news",
  Graph: "Links to flagged accounts",
};

export const GLOSSARY: Record<string, { term: string; plain: string }> = {
  aml: { term: "Money laundering", plain: "Making money from crime look like it came from honest work, usually by moving it through bank accounts." },
  structuring: { term: "Structuring", plain: "Splitting cash into deposits that each stay just under the amount a bank must report, so no report is filed." },
  cap: { term: "Monthly cash cap", plain: "The total cash in one month above which the bank must file a report. Staying just under it month after month is a warning sign." },
  kyc: { term: "KYC", plain: "Know Your Customer: what the customer told the bank about their job, income and expected activity." },
  str: { term: "STR", plain: "Suspicious Transaction Report: the report a bank files with the regulator about a suspicious account." },
  sanctions: { term: "Sanctions list", plain: "People and companies a bank is not allowed to deal with." },
  mule: { term: "Mule account", plain: "An account that receives money and passes it straight on for someone else." },
  ring: { term: "Ring", plain: "A group of accounts run by the same people, often seen through a shared phone, IP address or home address." },
  confidence: { term: "Confidence", plain: "A number from 0 to 1. Python computes it from the evidence; no model does the maths." },
  veto: { term: "Veto", plain: "A sanctions hit or a strong ring link that escalates the case on its own, whatever the average says." },
  gate: { term: "Second-signal gate", plain: "A case only escalates when at least two checks found something. One signal alone is held at MONITOR." },
  pause: { term: "ID pause", plain: "When both reads of the ID card disagree with the record, the case waits for a human admin." },
  label: { term: "Pattern label", plain: "A plain name for what the money is doing. It explains the case; it is never scored." },
  clause: { term: "Clause", plain: "One numbered rule from the bank's rulebook, such as STRUCT-001." },
  rag: { term: "RAG", plain: "Retrieval: search the rulebook first, then let the model quote only what the search returned." },
  key: { term: "Answer key", plain: "The correct label for every account, known because the data is synthetic. Only the grading script may read it." },
  design: { term: "Design label", plain: "The answer key with the forged-ID channel left out, because by design the ID check pauses a case and never scores it." },
  tuning: { term: "Tuning set", plain: "The 100 accounts every threshold and prompt was set on. A fair test needs accounts the system has not been tuned on." },
  falsealarm: { term: "False alarm", plain: "A clean account that was flagged anyway." },
  replay: { term: "Replay", plain: "Running the case again with the models' saved answers. The result is the same as the first run, without the wait." },
};

export const SPEED_LABEL = { real: "Real time", fast: "10x faster", instant: "Instant" } as const;
