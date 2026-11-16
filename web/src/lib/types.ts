export type Disposition = "CLEAR" | "MONITOR" | "SUSPICIOUS-ESCALATE" | "BLOCKED_BY_ADMIN_REVIEW";

export interface Step {
  step: string;
  finding: string;
  impact: string;
}

export interface Report {
  case_id: string;
  target_account: string;
  triggering_transaction_id: string | null;
  pipeline_status: string;
  stage_status: Record<string, string>;
  confidence: number;
  disposition: Disposition;
  degraded_stages: string[];
  evidence_incomplete: boolean;
  single_signal_capped: boolean;
  policy_gap_identified: boolean;
  forgery_notice: string | null;
  executive_summary: string;
  investigation_narrative: string;
  str_narrative: string | null;
  compliance_conclusion: string;
  triage_findings: { routed_stages: string[]; trigger_reasons: string[] } | null;
  vision_findings: {
    document_type: string;
    extracted_fields: Record<string, string>;
    field_discrepancies: string[];
    forgery_confidence: number;
  } | null;
  rulebook_findings: { retrieved_clause_ids: string[]; policy_violations: string[] } | null;
  judge_inputs: Record<string, number>;
  judge_trail: Step[];
  graph_label: string | null;
  agent_trails: Record<string, Step[]>;
  report_outline: string[];
  is_draft: boolean;
}

export interface Row {
  account: string;
  case_id: string;
  customer_name: string;
  occupation: string;
  verdict: Disposition;
  confidence: number;
  label: string;
  id_pause: boolean;
  inputs: Record<string, number>;
  clauses: string[];
  signed: boolean;
  admin_decision: string | null;
}

export interface RowPage {
  rows: Row[];
  total: number;
  page: number;
  page_size: number;
  labels: string[];
  counts: Record<string, number>;
}

export interface CaseQuery {
  verdicts?: string[];
  label?: string;
  id_pause?: boolean;
  q?: string;
  sort?: string;
  order?: "asc" | "desc";
  page?: number;
  page_size?: number;
}

export interface DocLink {
  type: string;
  file: string;
}

export interface AdminItem {
  account_id: string;
  case_id: string;
  customer_name: string;
  forgery_confidence: number;
  field_discrepancies: string[];
  disposition_if_approved: Disposition;
  batch_decision: string | null;
  decision: string | null;
  documents: DocLink[];
}

export interface CasePage {
  report: Report;
  customer_name: string;
  admin_decision: string | null;
  admin_item: AdminItem | null;
  signed: boolean;
  documents: DocLink[];
  clauses: Record<string, { title: string; text: string }>;
}

export interface Month {
  month: string;
  cash_in: number;
  cash_out: number;
  credits: number;
  debits: number;
  count: number;
}

export interface GraphNode {
  id: string;
  kind: "account" | "device" | "ip" | "address";
  label: string;
  verdict: Disposition | null;
  is_target: boolean;
}

export interface Evidence {
  account: string;
  customer: {
    name: string;
    occupation: string;
    stated_income: number;
    residency: string;
    country: string;
    account_type: string;
    opened: string;
  };
  transactions: number;
  months: Month[];
  cap: { threshold_inr: number; band_floor_inr: number };
  counterparties: { id: string; name: string; total: number; count: number; country: string; is_foreign: boolean }[];
  kyc_narrative: string;
  documents: DocLink[];
  graph: { nodes: GraphNode[]; edges: { source: string; target: string; relation: string }[]; linked_total: number };
}

export interface RunInfo {
  id: string;
  label: string;
  accounts: number | null;
  set: string | null;
  mode: string | null;
  has_scoreboard: boolean;
  modified: number;
}

export interface Scoreboard {
  set: string;
  mode: string;
  accounts: number;
  metrics: Record<string, number> & {
    structurers: Record<string, number>;
    structurers_caught: Record<string, number>;
  };
  confusion: Record<string, Record<string, number>>;
  misses: { account: string; key: string; design: string; got: string }[];
  show_cases: { account: string; title: string; key_label: string; design_label: string; got: Disposition; confidence: number }[];
}

export interface ModelCalls {
  calls: number;
  from_cache: number;
  failed: number;
  seconds: number;
  retried: number;
  seconds_per_call: number | null;
}

export interface RunSummary {
  run: string;
  summary: {
    accounts?: number;
    cases?: number;
    rounds?: number[];
    dispositions?: Record<string, number>;
    admin_queue?: number;
    backup_share?: number;
    warnings?: string[];
    phase_seconds?: Record<string, number>;
    model_calls?: Record<string, ModelCalls>;
  };
  scoreboard: Scoreboard | null;
}

export interface Rules {
  weights: Record<string, number>;
  clear_below: number;
  escalate_from: number;
  veto: number;
  corroboration_min: number;
  solo_min: number;
  id_pause: number;
  id_notice: number;
}

export interface Capabilities {
  presenter: boolean;
  models_online: boolean;
  live_available: boolean;
  replay_available: boolean;
  static: boolean;
  reason: string | null;
  nodes: string[];
  node_weights: Record<string, number>;
  hardware: string;
  rules: Rules;
  recorded?: string;
}

export interface PipelineEvent {
  seq: number;
  type: "queued" | "started" | "node_done" | "paused" | "resumed" | "done" | "error";
  node?: string;
  seconds?: number;
  source?: "live" | "recorded" | "python";
  note?: string | null;
  detail?: string | null;
  calls?: number;
  position?: number;
  evidence?: {
    account_id: string;
    forgery_confidence: number;
    field_discrepancies: string[];
    disposition_if_approved: Disposition;
    score_if_approved: number;
  };
  case?: { account_id: string; case_id: string; disposition: Disposition; confidence: number } | null;
  report?: Report | null;
  message?: string;
}

export interface Health {
  ready: boolean;
  stage: number;
  stages: string[];
  error: string | null;
}
