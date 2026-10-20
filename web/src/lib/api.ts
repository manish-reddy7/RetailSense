"use client";

// Retailer API (src/retailsense/service/app.py), served from the same origin as this site.

export type Inventory = { lead_time: number; coverage: number; service_level: number; case_pack: number };
export type StoreSettings = { name: string; country: string; lat: number; lon: number; inventory: Inventory };
export type FileMeta = { filename: string; rows: number; items?: number; terms?: string[]; uploaded: number; first?: string; last?: string };
export type Store = StoreSettings & { id: number; files: { sales: FileMeta | null; stock: FileMeta | null; trends: FileMeta | null } };
export type Run = {
  id: number;
  store_id: number;
  status: "queued" | "running" | "done" | "failed";
  created: number;
  started: number | null;
  finished: number | null;
  error: string | null;
};

export type ItemResult = {
  item_id: string;
  D: number;
  consensus: number;
  deliberated: boolean;
  fallback: string | null;
  decision: {
    on_hand: number;
    on_order: number;
    action: "RESTOCK" | "HOLD" | "REDUCE";
    order_qty: number;
    reorder_point: number;
    order_up_to: number;
  } | null;
  forecast: { date: string; h: number; yhat: number; lower: number; upper: number }[];
  agents: { agent: string; forecast: number; lower: number; upper: number; confidence: number; key_factors: string[] }[];
  critic: {
    primary_conflict: string;
    evidence_assessment: { agent: string; strength: "strong" | "moderate" | "weak"; reason: string }[];
    missing_or_stale_evidence: string[];
  } | null;
  revisions: { agent_id: string; initial: number; y: number; m: number; held: boolean; rationale: string }[];
  history: { date: string; units: number | null }[];
};

export type Result = {
  made_on: string;
  tau: number;
  agents: string[];
  llm: { provider: string; model: string; calls: number };
  items: ItemResult[];
};

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const json = typeof init.body === "string";
  const r = await fetch(`/api${path}`, {
    ...init,
    credentials: "same-origin",
    headers: json ? { "Content-Type": "application/json" } : undefined,
  });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    // FastAPI: {detail: "message"} or, for a schema error, {detail: [{loc, msg}]}
    const d = body.detail;
    const msg = typeof d === "string" ? d : Array.isArray(d) ? d.map((e) => `${e.loc.at(-1)}: ${e.msg}`).join("; ") : r.statusText;
    throw new ApiError(msg, r.status);
  }
  return body as T;
}

export const post = <T>(path: string, data?: unknown) =>
  api<T>(path, { method: "POST", body: data === undefined ? undefined : JSON.stringify(data) });

export const upload = <T>(path: string, file: File) => {
  const form = new FormData();
  form.append("file", file);
  return api<T>(path, { method: "POST", body: form });
};
