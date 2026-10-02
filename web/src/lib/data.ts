"use client";

import { useQuery } from "@tanstack/react-query";

// Shapes written by `python -m retailsense.web_export` (web/public/data/<dataset>/).

export type Agent = {
  id: string;
  forecast: number;
  lower: number | null;
  upper: number | null;
  confidence: number;
  revised: number | null;
  m: number | null;
  rationale: string | null;
  key_factors: string[];
  limitations: string[];
};

export type Critic = {
  primary_conflict: string;
  conflicting_agents: string[];
  evidence_assessment: { agent: string; strength: "strong" | "moderate" | "weak"; reason: string }[];
  missing_or_stale_evidence: string[];
  questions_for_agents: { agent: string; question: string }[];
};

export type Inventory = {
  on_hand: number;
  on_order: number;
  safety_stock: number;
  reorder_point: number;
  order_up_to: number;
  action: string;
  order_qty: number;
  sold: number;
  lost: number;
};

export type Case = {
  date: string;
  d: number;
  D: number;
  tau: number;
  triggered: boolean;
  consensus: number;
  deliberated: number;
  final: number;
  actual: number;
  fallback: string | null;
  agents: Agent[];
  critic: Critic | null;
  inventory: Inventory | null;
};

export type Experiment = {
  label: string;
  items: string[];
  n_cases: number;
  tau: number;
  deciles: { decile: number; mean_rel_err: number; wape: number; n: number }[];
};

export type Meta = {
  dataset: string;
  label: string;
  store: string;
  exported_at: string;
  model: string;
  prompt_version: string;
  tau_frozen: number | null;
  experiments: Record<string, Experiment>;
  inventory: { case_pack: number };
  live: { lat: number; lon: number; timezone: string; rainy_mm: number };
};

export type Row = Record<string, string | number | null>;
export type Tables = Partial<Record<"validation" | "dryrun" | "final", Record<string, Row[]>>>;

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

const base = (ds: string) => `/data/${ds}`;

export const useDatasets = () =>
  useQuery({ queryKey: ["datasets"], queryFn: () => getJSON<string[]>("/data/index.json") });

export const useMeta = (ds?: string) =>
  useQuery({
    queryKey: ["meta", ds],
    queryFn: () => getJSON<Meta>(`${base(ds!)}/meta.json`),
    enabled: !!ds,
  });

export const useTables = (ds?: string) =>
  useQuery({
    queryKey: ["tables", ds],
    queryFn: () => getJSON<Tables>(`${base(ds!)}/tables.json`),
    enabled: !!ds,
  });

export const useCases = (ds?: string, exp?: string, item?: string) =>
  useQuery({
    queryKey: ["cases", ds, exp, item],
    queryFn: () => getJSON<Case[]>(`${base(ds!)}/runs/${exp}/${encodeURIComponent(item!)}.json`),
    enabled: !!ds && !!exp && !!item,
  });

export const fmt = (x: number | null | undefined, digits = 2) =>
  x === null || x === undefined || Number.isNaN(x) ? "–" : x.toFixed(digits);
