"use client";

import { ArrowRight, BookOpen, CheckCircle2, CircleSlash, CircleDot, ShieldCheck, TriangleAlert } from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";
import { Empty, Kpi, Loading, PageHeader, Problem, Section } from "@/components/common";
import { useCases, useStudies, type Meta } from "@/lib/data";
import { Plume } from "@/components/plume";
import { nice, summarise, type Summary, type Verdict } from "@/lib/results";
import { useSelection } from "@/lib/selection";
import { cn } from "@/lib/utils";

const VERDICT: Record<Verdict, { label: string; icon: typeof CheckCircle2; cls: string }> = {
  supported: { label: "Supported", icon: CheckCircle2, cls: "border-good/40 bg-good/10 text-good" },
  weak: { label: "Weak", icon: CircleDot, cls: "border-deliberate/40 bg-deliberate/10 text-deliberate-ink" },
  "not supported": { label: "Not supported", icon: CircleSlash, cls: "border-border bg-muted text-muted-foreground" },
  "no data": { label: "No data", icon: CircleSlash, cls: "border-border text-muted-foreground" },
};

function Chip({ v }: { v: Verdict }) {
  const { label, icon: Icon, cls } = VERDICT[v];
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium whitespace-nowrap", cls)}>
      <Icon className="size-3.5" aria-hidden /> {label}
    </span>
  );
}

const usd = (x: number) => `$${Math.round(x / 1000).toLocaleString()}k`;
const studyName = (m: Meta) => (m.real_data ? "Rohlik, real sales" : "India, simulated sales");

function Node({ title, text, tone = "border-border" }: { title: string; text: string; tone?: string }) {
  return (
    <div className={cn("rounded-xl border bg-card px-3 py-2.5", tone)}>
      <p className="text-sm font-medium">{title}</p>
      <p className="text-xs text-muted-foreground">{text}</p>
    </div>
  );
}

const Arrow = () => <ArrowRight className="hidden size-4 shrink-0 text-muted-foreground lg:block" aria-hidden />;

function List({ icon: Icon, tone, items }: { icon: typeof ShieldCheck; tone: string; items: ReactNode[] }) {
  return (
    <ul className="grid gap-2.5 text-sm">
      {items.map((x, i) => (
        <li key={i} className="flex gap-2">
          <Icon className={cn("mt-0.5 size-4 shrink-0", tone)} aria-hidden />
          <span>{x}</span>
        </li>
      ))}
    </ul>
  );
}

/** The plume for the item picked in the sidebar, on the frozen test run. */
function Hero({ dataset, item, final }: { dataset?: string; item?: string; final: boolean }) {
  const q = useCases(final ? dataset : undefined, "final_test", item);
  const cases = q.data ?? [];
  if (!final || !item) return null;
  const sent = cases.filter((c) => c.triggered).length;
  return (
    <figure className="grid gap-3">
      {cases.length ? <Plume cases={cases} /> : <div className="h-[340px]" aria-hidden />}
      <figcaption className="measure text-sm text-muted-foreground">
        {item}, every day of the locked test. Each line is one agent&apos;s next-day forecast; the
        shaded band is how far apart they were, and dots are what actually sold.{" "}
        {cases.length > 0 && (
          <>
            On the <span className="font-medium text-deliberate-ink">{sent} amber days</span> of{" "}
            {cases.length} the spread crossed τ and the LLM critic was called. Pick another item in
            the sidebar.
          </>
        )}
      </figcaption>
    </figure>
  );
}

export default function Overview() {
  const s = useSelection();
  const studies = useStudies(s.datasets);
  if (s.error) return <Problem error={s.error} />;
  if (!s.meta || studies.some((q) => q.isPending)) return <Loading />;

  const all = studies
    .map((q) => q.data)
    .filter((d) => d !== undefined)
    .map((d) => ({ ...d, sum: summarise(d.tables) }))
    .filter((d): d is typeof d & { sum: Summary } => d.sum !== undefined);
  const cur = all.find((d) => d.ds === s.dataset);
  const saved = all.map((d) => d.sum.callsSaved);
  const savedRange = saved.length
    ? `${Math.round(100 * Math.min(...saved))}–${Math.round(100 * Math.max(...saved))}%`
    : "–";
  const exps = s.meta.experiments;

  return (
    <>
      <PageHeader title="Does disagreement tell us when to think harder?">
        <p>
          <span className="font-medium text-foreground">Short answer: only a little.</span> Across a simulated kirana
          store and real grocery sales, disagreement between the agents is a weak but real warning sign. The LLM
          critic did not reliably fix the forecasts it flagged, and simpler models were more accurate and cheaper.
          Triggering the LLM only on disagreement still cut LLM calls by {savedRange}.
        </p>
      </PageHeader>

      <Hero dataset={s.dataset} item={s.item} final={"final_test" in exps} />

      {cur ? (
        <>
          <div className="grid grid-cols-1 gap-x-6 gap-y-4 sm:grid-cols-2 xl:grid-cols-4">
            <Kpi index={0} tone="good" label="Most accurate system" value={cur.sum.best.wape.toFixed(3)} hint={`${nice(cur.sum.best.system)}, next-day WAPE`} />
            <Kpi
              index={1}
              tone="deliberate"
              label="RetailSense WAPE"
              value={cur.sum.rs.wape.toFixed(3)}
              hint={`95% CI [${cur.sum.rs.lo.toFixed(3)}, ${cur.sum.rs.hi.toFixed(3)}]`}
            />
            <Kpi index={2} tone="brand" label="LLM calls avoided" value={`${Math.round(100 * cur.sum.callsSaved)}%`} hint={`compared with deliberating every case; ${(100 * cur.sum.triggerRate).toFixed(1)}% of cases triggered`} />
            <Kpi index={3} tone="fast" label="Lowest inventory cost" value={usd(cur.sum.cheapest.cost)} hint={`${nice(cur.sum.cheapest.system)}; RetailSense ${usd(cur.sum.rsCost)}`} />
          </div>

          <Section
            title={`Five hypotheses, one locked test: ${studyName(s.meta)}`}
            description={`${cur.sum.points.toLocaleString()} forecasts on the untouched test split. A hypothesis counts as supported only if its planned test is significant.`}
          >
            <ol className="grid divide-y divide-border border-y border-border">
              {cur.sum.hyps.map((h) => (
                <li key={h.id} className="grid gap-1 py-3 md:grid-cols-[3rem_1fr_auto] md:items-center md:gap-4">
                  <span className="wide text-base font-semibold text-brand">{h.id}</span>
                  <div className="min-w-0">
                    <p className="text-sm font-medium">{h.question}</p>
                    <p className="num text-xs text-muted-foreground">{h.evidence}</p>
                  </div>
                  <Chip v={h.verdict} />
                </li>
              ))}
            </ol>
          </Section>
        </>
      ) : (
        <Empty>No final test run exported for this dataset.</Empty>
      )}

      {all.length > 1 && (
        <Section title="Two studies, same protocol" description="Same code, prompt and LLM. Only the data, the dates and the number of agents differ.">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[36rem] text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-muted-foreground">
                  <th className="py-2 pr-4 font-medium" />
                  {all.map((d) => (
                    <th key={d.ds} className="py-2 pr-4 font-medium">
                      {studyName(d.meta)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="num">
                {(
                  [
                    ["Agents", (d) => d.meta.agents.length],
                    ["Frozen τ", (d) => d.meta.tau_frozen?.toFixed(4) ?? "–"],
                    ["Most accurate", (d) => `${nice(d.sum.best.system)} (${d.sum.best.wape.toFixed(3)})`],
                    ["RetailSense WAPE", (d) => d.sum.rs.wape.toFixed(3)],
                    ["Cheapest inventory", (d) => `${nice(d.sum.cheapest.system)} (${usd(d.sum.cheapest.cost)})`],
                    ["Agents kept forecast after critique", (d) => `${(100 * d.sum.holdRate).toFixed(1)}%`],
                  ] as [string, (d: (typeof all)[number]) => ReactNode][]
                ).map(([label, f]) => (
                  <tr key={label} className="border-b border-border/60">
                    <td className="py-2 pr-4 font-sans text-muted-foreground">{label}</td>
                    {all.map((d) => (
                      <td key={d.ds} className="py-2 pr-4">
                        {f(d)}
                      </td>
                    ))}
                  </tr>
                ))}
                {all[0].sum.hyps.map((h, i) => (
                  <tr key={h.id} className="border-b border-border/60">
                    <td className="py-2 pr-4 font-sans text-muted-foreground">
                      {h.id}. {h.question}
                    </td>
                    {all.map((d) => (
                      <td key={d.ds} className="py-2 pr-4">
                        <Chip v={d.sum.hyps[i].verdict} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      <Section title="One forecast, two paths" description="The LLM never writes a forecast or an order; code computes every number.">
        <div className="flex flex-col items-stretch gap-3 lg:flex-row lg:items-center">
          <Node title={`${s.meta.agents.length} agents`} text={`${s.meta.agents.map(nice).join(", ")}, each with a 10–90% band`} />
          <Arrow />
          <Node title="Disagreement D" text="confidence-weighted spread of day-1 forecasts" tone="border-brand/40" />
          <Arrow />
          <div className="grid gap-2 lg:min-w-72 lg:flex-[1.3]">
            <div className="flex items-center gap-2 rounded-xl border border-fast/40 bg-fast/5 p-2.5">
              <span className="shrink-0 rounded-md bg-fast/20 px-1.5 text-xs font-medium whitespace-nowrap text-fast">D &lt; τ</span>
              <span className="text-sm">confidence-weighted consensus</span>
            </div>
            <div className="flex items-center gap-2 rounded-xl border border-deliberate/40 bg-deliberate/5 p-2.5">
              <span className="shrink-0 rounded-md bg-deliberate/20 px-1.5 text-xs font-medium whitespace-nowrap text-deliberate-ink">D ≥ τ</span>
              <span className="text-sm">LLM critic → one bounded revision, clipped by code</span>
            </div>
          </div>
          <Arrow />
          <Node title="(s, S) policy" text="Restock, hold or reduce, in one simulator shared by every system" />
        </div>
      </Section>

      <div className="grid gap-6 lg:grid-cols-2">
        <Section title="How the test was protected">
          <List
            icon={ShieldCheck}
            tone="text-good"
            items={[
              "τ and the prompt were frozen on validation before the test split was read.",
              "Each dataset's test split was evaluated by one locked run. Interrupted runs resumed with the same config and replayed cached LLM answers.",
              "Every feature records when it became available; tests check that none is newer than the forecast time.",
              "Missing sales stay missing, never zero-filled. An invalid agent is excluded, never counted as zero.",
              "All nine systems go through the same (s, S) inventory simulator.",
            ]}
          />
        </Section>
        <Section title="Limitations">
          <List
            icon={TriangleAlert}
            tone="text-deliberate-ink"
            items={[
              "The agents agree too much (correlation above the 0.95 design gate), so D mostly measures small adjustments.",
              "One model seed; differences below about 0.001 WAPE should not be over-read.",
              "Observed weather stands in for a perfect forecast (tested with a noise ablation).",
              "One small local LLM (Llama 3.1 8B); a stronger critic might behave differently.",
              "India sales are simulated with known drivers; Rohlik is real but has no search-interest data, so no Trend agent.",
            ]}
          />
        </Section>
      </div>

      <Section title="Base paper: MARIOD (Yang et al., Sensors 2025)" description="Same problem, joint forecasting and restocking with specialised agents, approached differently.">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[36rem] text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs text-muted-foreground">
                <th className="py-2 pr-4 font-medium" />
                <th className="py-2 pr-4 font-medium">MARIOD</th>
                <th className="py-2 pr-4 font-medium">RetailSense</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Agents are…", "supply-chain levels (store, DC, HQ)", "evidence sources (history, calendar, weather, interest)"],
                ["Coordination", "learned messages, trained jointly", "disagreement score + one LLM revision"],
                ["Orders", "learned RL policy", "fixed (s, S) formula"],
                ["Evaluation", "70/15/15 split, zero-filled sales, point estimates", "frozen τ, one locked test run, bootstrap CIs"],
                ["Question", "how accurate can a joint deep system be?", "does disagreement tell us when LLM reasoning is worth it?"],
              ].map(([k, a, b]) => (
                <tr key={k} className="border-b border-border/60">
                  <td className="py-2 pr-4 text-muted-foreground">{k}</td>
                  <td className="py-2 pr-4">{a}</td>
                  <td className="py-2 pr-4">{b}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Stored runs" description="Open one to browse its cases in Forecast and Explain.">
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
                      {e.n_cases.toLocaleString()} cases over {e.items.length} items, τ = {e.tau.toFixed(3)}
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
        <BookOpen className="size-4" aria-hidden />
        Exported {s.meta.exported_at}. Every result is reported, including where RetailSense does not win. Full write-up:
        docs/paper and reports/RESULTS.md.
      </p>
    </>
  );
}
