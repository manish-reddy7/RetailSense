"use client";

import { ArrowRight, CheckCircle2, CircleDashed, Lock, ShieldCheck } from "lucide-react";
import Link from "next/link";
import { Empty, Kpi, Loading, PageHeader, Problem, Section, cell } from "@/components/common";
import { useTables, type Row } from "@/lib/data";
import { useSelection } from "@/lib/selection";
import { cn } from "@/lib/utils";

type Status = "done" | "open" | "locked";

function Phase({ n, title, status, note, i }: { n: string; title: string; status: Status; note: string; i: number }) {
  const Icon = { done: CheckCircle2, open: CircleDashed, locked: Lock }[status];
  return (
    <li
      data-index={i}
      className={cn(
        "rounded-xl border p-3",
        status === "done" ? "border-good/30 bg-good/5" : status === "open" ? "border-deliberate/40 bg-deliberate/5" : "border-border bg-muted/40",
      )}
    >
      <div className="flex items-center justify-between">
        <span className="num text-xs text-muted-foreground">{n}</span>
        <Icon
          className={cn("size-4", status === "done" ? "text-good" : status === "open" ? "text-deliberate" : "text-muted-foreground")}
          aria-label={status}
        />
      </div>
      <p className="mt-1 text-sm font-medium">{title}</p>
      <p className="text-xs text-muted-foreground">{note}</p>
    </li>
  );
}

function Node({ title, text, tone = "border-border" }: { title: string; text: string; tone?: string }) {
  return (
    <div className={cn("rounded-xl border bg-card px-3 py-2.5", tone)}>
      <p className="text-sm font-medium">{title}</p>
      <p className="text-xs text-muted-foreground">{text}</p>
    </div>
  );
}

const Arrow = () => <ArrowRight className="hidden size-4 shrink-0 text-muted-foreground lg:block" aria-hidden />;

const find = (rows: Row[] | undefined, key: string, name: string) => rows?.find((r) => r[key] === name);

export default function Overview() {
  const s = useSelection();
  const tables = useTables(s.dataset);
  if (s.error) return <Problem error={s.error} />;
  if (!s.meta || tables.isPending) return <Loading />;
  const t = tables.data ?? {};
  const v = t.validation ?? {};
  const h1 = find(v.h1_summary, "Unnamed: 0", "D");
  const inv = v.phase4_inventory;
  const best = inv?.length ? inv.reduce((a, b) => ((a.total_cost as number) <= (b.total_cost as number) ? a : b)) : undefined;
  const exps = s.meta.experiments;
  const deciles = (v.h1_deciles_D ?? []).map((r) => Number(r.mean_rel_err));
  const cases = Object.values(exps).reduce((n, e) => n + e.n_cases, 0);

  const phases: { n: string; title: string; status: Status; note: string }[] = [
    { n: "P1-2", title: "Agents & baselines", status: "done", note: "per-horizon models, context agents" },
    { n: "P3a", title: "H1 on validation", status: v.h1_summary ? "done" : "open", note: "does D predict error?" },
    { n: "P3b", title: "Deliberation pilot", status: exps.phase3b_pilot ? "done" : "open", note: "critic + one revision" },
    { n: "P3b", title: "Full validation arms", status: exps.phase3b || v.phase3b_arms ? "done" : "open", note: "needs LLM credits" },
    { n: "P4", title: "Inventory simulator", status: inv ? "done" : "open", note: "(s, S), 6 systems" },
    { n: "P5", title: "Pipeline dry run", status: t.dryrun ? "done" : "open", note: "stub LLM, validation" },
    { n: "τ", title: "Freeze τ & prompt", status: s.meta.tau_frozen === null ? "open" : "done", note: "before any test data" },
    { n: "P5a", title: "Final test run", status: t.final ? "done" : "locked", note: "touched once" },
  ];

  return (
    <>
      <PageHeader
        eyebrow="Research capstone · results viewer"
        title={
          <>
            Does disagreement tell us <em className="text-brand">when</em> to think harder?
          </>
        }
      >
        Specialised agents forecast daily demand. When they disagree beyond a calibrated threshold τ, an LLM
        critic and one bounded revision round step in, and the forecast drives an (s, S) restocking policy.
      </PageHeader>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Kpi
          index={0}
          tone={s.meta.tau_frozen === null ? "deliberate" : "good"}
          label="Threshold τ"
          value={s.meta.tau_frozen === null ? "open" : s.meta.tau_frozen.toFixed(3)}
          hint="80th percentile of validation D, frozen before test"
        />
        <Kpi
          index={1}
          tone="brand"
          label="H1 · AUROC of D"
          value={h1 ? cell(h1.auroc_worst) : "–"}
          hint={h1 ? `Spearman ρ ${cell(h1.spearman)}; 0.5 = no signal` : "not run"}
          spark={deciles}
        />
        <Kpi
          index={2}
          tone="fast"
          label="Cheapest inventory (val.)"
          value={best ? `$${Math.round(Number(best.total_cost) / 1000)}k` : "–"}
          hint={best ? String(best.system).replaceAll("_", " ") : "Phase 4 not run"}
        />
        <Kpi index={3} tone="muted" label="Stored cases" value={cases.toLocaleString()} hint={`${Object.keys(exps).length} run${Object.keys(exps).length === 1 ? "" : "s"} exported`} />
      </div>

      <Section eyebrow="Progress" title="Where the experiment stands" description="Derived from the exported results; the test split stays locked until τ is frozen.">
        <ol className="grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-8">
          {phases.map((p, i) => (
            <Phase key={p.title} i={i} {...p} />
          ))}
        </ol>
      </Section>

      <Section eyebrow="Method" title="One forecast, two paths" description="The LLM never writes a forecast or an order; code computes every number.">
        <div className="flex flex-col items-stretch gap-3 lg:flex-row lg:items-center">
          <Node title="4 agents" text="Historical · Calendar · Weather · Trend, with 10-90% bands" />
          <Arrow />
          <Node title="Disagreement D" text="confidence-weighted spread of day-1 forecasts" tone="border-brand/40" />
          <Arrow />
          <div className="grid gap-2 lg:min-w-72 lg:flex-[1.3]">
            <div className="flex items-center gap-2 rounded-xl border border-fast/40 bg-fast/5 p-2.5">
              <span className="shrink-0 rounded-md bg-fast/20 px-1.5 text-xs font-medium whitespace-nowrap text-fast">D &lt; τ</span>
              <span className="text-sm">confidence-weighted consensus</span>
            </div>
            <div className="flex items-center gap-2 rounded-xl border border-deliberate/40 bg-deliberate/5 p-2.5">
              <span className="shrink-0 rounded-md bg-deliberate/20 px-1.5 text-xs font-medium whitespace-nowrap text-deliberate">D ≥ τ</span>
              <span className="text-sm">critic report → one bounded revision</span>
            </div>
          </div>
          <Arrow />
          <Node title="(s, S) policy" text="RESTOCK · HOLD · REDUCE in one shared simulator" />
        </div>
      </Section>

      <Section eyebrow="Data" title="Stored runs" description="Open one to browse its cases in Forecast and Explain.">
        {Object.keys(exps).length ? (
          <ul className="grid gap-2 md:grid-cols-2">
            {Object.entries(exps).map(([id, e]) => (
              <li key={id}>
                <Link
                  href="/forecast"
                  onClick={() => s.set({ experiment: id, item: undefined, date: undefined })}
                  className="group flex items-center gap-3 rounded-xl border border-border p-3 transition-colors hover:border-brand/50 hover:bg-accent/40"
                >
                  <div className="min-w-0 flex-1">
                    <p className="num text-xs text-muted-foreground">{id}</p>
                    <p className="truncate text-sm font-medium">{e.label}</p>
                    <p className="text-xs text-muted-foreground">
                      {e.n_cases.toLocaleString()} cases · {e.items.length} items · τ {e.tau.toFixed(3)}
                    </p>
                  </div>
                  <ArrowRight className="size-4 text-muted-foreground transition-transform group-hover:translate-x-0.5 group-hover:text-brand" aria-hidden />
                </Link>
              </li>
            ))}
          </ul>
        ) : (
          <Empty>No stored runs for this dataset yet.</Empty>
        )}
      </Section>

      <p className="flex items-center gap-2 text-xs text-muted-foreground">
        <ShieldCheck className="size-4" aria-hidden />
        Exported {s.meta.exported_at}. Every result is reported, including where RetailSense does not win.
      </p>
    </>
  );
}
