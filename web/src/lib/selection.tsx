"use client";

import { useMemo, useSyncExternalStore } from "react";
import { useCases, useDatasets, useMeta, type Case } from "@/lib/data";

type Picked = { dataset?: string; experiment?: string; item?: string; date?: string };

// The current pick lives in localStorage (a per-viewer convenience), with an in-memory copy so
// the app still works when storage is blocked. Server render always sees an empty pick.
const KEY = "retailsense.selection";
const listeners = new Set<() => void>();
let memory = "{}";

function read(): string {
  try {
    return localStorage.getItem(KEY) ?? memory;
  } catch {
    return memory;
  }
}

function write(p: Partial<Picked>) {
  memory = JSON.stringify({ ...JSON.parse(read()), ...p });
  try {
    localStorage.setItem(KEY, memory);
  } catch {}
  listeners.forEach((l) => l());
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

/** The current selection, with every level defaulted to something that exists. */
export function useSelection() {
  const raw = useSyncExternalStore(subscribe, read, () => "{}");
  const s = useMemo<Picked>(() => JSON.parse(raw), [raw]);
  const datasets = useDatasets();
  const dataset = s.dataset && datasets.data?.includes(s.dataset) ? s.dataset : datasets.data?.[0];
  const meta = useMeta(dataset);
  const exps = Object.keys(meta.data?.experiments ?? {});
  const experiment = s.experiment && exps.includes(s.experiment) ? s.experiment : exps[0];
  const items = experiment ? (meta.data?.experiments[experiment].items ?? []) : [];
  const item = s.item && items.includes(s.item) ? s.item : items[0];
  const cases = useCases(dataset, experiment, item);
  const dates = cases.data?.map((c) => c.date) ?? [];
  const date = s.date && dates.includes(s.date) ? s.date : dates[dates.length - 1];
  const current: Case | undefined = cases.data?.find((c) => c.date === date);
  return {
    set: write,
    datasets: datasets.data ?? [],
    dataset,
    meta: meta.data,
    experiments: exps,
    experiment,
    items,
    item,
    cases: cases.data ?? [],
    dates,
    date,
    current,
    loading: datasets.isPending || meta.isPending || cases.isPending,
    error: datasets.error ?? meta.error ?? cases.error,
  };
}
