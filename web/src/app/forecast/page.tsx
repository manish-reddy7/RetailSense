"use client";

import { motion } from "framer-motion";
import { MessagesSquare, PackageCheck, PackageMinus, PackagePlus, Zap } from "lucide-react";
import { useMemo } from "react";
import { EChart } from "@/components/echart";
import { Empty, Kpi, Legend, Loading, Meter, PageHeader, Problem, Section } from "@/components/common";
import { fmt } from "@/lib/data";
import { useSelection } from "@/lib/selection";
import { useTokens } from "@/lib/theme";
import { cn } from "@/lib/utils";

const ACTION_ICON = { RESTOCK: PackagePlus, REDUCE: PackageMinus, HOLD: PackageCheck } as const;

export default function ForecastPage() {
  const s = useSelection();
  const t = useTokens();
  const c = s.current;

  const bands = useMemo(() => {
    if (!c || !t) return {};
    const lo = c.agents.map((a) => a.lower ?? a.forecast);
    const line = (y: number, color: string, text: string, type: "dashed" | "solid") => ({
      yAxis: y,
      lineStyle: { color, type, width: 1.5 },
      label: { formatter: text, color, position: "insideEndTop", fontSize: 11 },
    });
    return {
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
      grid: { left: 44, right: 16, top: 28, bottom: 32 },
      xAxis: { type: "category", data: c.agents.map((a) => a.id) },
      yAxis: { type: "value", name: "units / day", min: 0 },
      series: [
        { type: "bar", stack: "b", data: lo, itemStyle: { color: "transparent" }, tooltip: { show: false }, silent: true },
        {
          name: "10-90% band",
          type: "bar",
          stack: "b",
          barWidth: 34,
          data: c.agents.map((a, i) => (a.upper ?? a.forecast) - lo[i]),
          itemStyle: { color: t.brand, opacity: 0.18, borderRadius: 6 },
        },
        {
          name: "initial",
          type: "scatter",
          symbolSize: 13,
          data: c.agents.map((a) => a.forecast),
          itemStyle: { color: t.brand, borderColor: t.card, borderWidth: 2 },
          markLine: {
            symbol: "none",
            silent: true,
            data: [line(c.final, t.deliberate, `final ${fmt(c.final)}`, "dashed"), line(c.actual, t.actual, `actual ${fmt(c.actual, 0)}`, "solid")],
          },
        },
        ...(c.triggered
          ? [{ name: "revised", type: "scatter", symbol: "diamond", symbolSize: 13, data: c.agents.map((a) => a.revised), itemStyle: { color: t.deliberate, borderColor: t.card, borderWidth: 2 } }]
          : []),
      ],
    };
  }, [c, t]);

  const timeline = useMemo(() => {
    if (!t) return {};
    return {
      tooltip: { trigger: "axis" },
      grid: { left: 44, right: 16, top: 16, bottom: 56 },
      xAxis: { type: "category", data: s.cases.map((x) => x.date), boundaryGap: false },
      yAxis: { type: "value", min: 0 },
      dataZoom: [{ type: "inside" }, { type: "slider", height: 16, bottom: 8, borderColor: "transparent" }],
      series: [
        { name: "actual", type: "line", data: s.cases.map((x) => x.actual), itemStyle: { color: t.actual }, lineStyle: { color: t.actual, width: 1.5 }, symbol: "none" },
        { name: "consensus", type: "line", data: s.cases.map((x) => x.consensus), itemStyle: { color: t["chart-4"] }, lineStyle: { color: t["chart-4"], type: "dotted" }, symbol: "none" },
        {
          name: "RetailSense final",
          type: "line",
          data: s.cases.map((x) => x.final),
          itemStyle: { color: t.brand },
          lineStyle: { color: t.brand },
          areaStyle: { color: t.brand, opacity: 0.06 },
          symbol: "none",
        },
        { name: "deliberated", type: "scatter", data: s.cases.map((x) => (x.triggered ? x.final : null)), itemStyle: { color: t.deliberate }, symbolSize: 9 },
      ],
    };
  }, [s.cases, t]);

  if (s.error) return <Problem error={s.error} />;
  if (s.loading) return <Loading />;
  if (!c) return <Empty>No case for this selection.</Empty>;
  const inv = c.inventory;
  const err = Math.abs(c.actual - c.final);
  const ActionIcon = inv ? (ACTION_ICON[inv.action as keyof typeof ACTION_ICON] ?? PackageCheck) : PackageCheck;

  return (
    <>
      <PageHeader
        context={s.meta?.experiments[s.experiment!]?.label}
        title={
          <>
            {s.item} on {c.date}
          </>
        }
        aside={
          <motion.div key={`${s.item}${c.date}`} initial={{ scale: 0.92, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}>
            {c.triggered ? (
              <span className="flex items-center gap-2 rounded-full border border-deliberate/40 bg-deliberate/10 px-3 py-1.5 text-sm font-medium text-deliberate-ink">
                <MessagesSquare className="size-4" aria-hidden /> Deliberated
              </span>
            ) : (
              <span className="flex items-center gap-2 rounded-full border border-fast/40 bg-fast/10 px-3 py-1.5 text-sm font-medium text-fast">
                <Zap className="size-4" aria-hidden /> Fast path
              </span>
            )}
          </motion.div>
        }
      >
        The forecast for this day was made at the end of the previous day, from data available then.
        {c.fallback && <span className="block text-destructive">Fallback: {c.fallback}</span>}
      </PageHeader>

      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        <Kpi index={0} tone="brand" label="Final forecast" value={fmt(c.final)} hint={c.triggered ? `consensus was ${fmt(c.consensus)}` : "confidence-weighted consensus"} />
        <Kpi index={1} tone="muted" label="Actual demand" value={fmt(c.actual, 0)} hint="revealed after the fact" />
        <Kpi index={2} tone={err <= Math.max(1, 0.25 * c.actual) ? "good" : "deliberate"} label="Absolute error" value={fmt(err)} hint={`${c.actual > 0 ? fmt((100 * err) / c.actual, 0) : "–"}% of actual`} />
        <Kpi index={3} tone="fast" label="Stock on hand" value={inv ? fmt(inv.on_hand, 0) : "–"} hint={inv ? `${fmt(inv.on_order, 0)} on order` : "no simulation for this run"} />
      </div>

      <div className="grid gap-4 xl:grid-cols-5">
        <Section className="xl:col-span-3" title="What each agent expects" description="Day-1 forecast with its 10-90% band; diamonds show revised values after deliberation.">
          <EChart option={bands} height={330} />
          <Legend
            items={[
              { label: "initial forecast", className: "bg-brand" },
              { label: "10-90% band", className: "bg-brand/25" },
              { label: "revised / final", className: "bg-deliberate" },
              { label: "actual", className: "bg-actual" },
            ]}
          />
        </Section>
        <div className="grid gap-4 xl:col-span-2">
          <Section title="Disagreement" description="Normalised spread of the agents' forecasts against the threshold.">
            <Meter D={c.D} tau={c.tau} />
          </Section>
          <Section title="Recommended action">
            {inv ? (
              <div className="flex items-center gap-4">
                <span
                  className={cn(
                    "grid size-14 shrink-0 place-items-center rounded-2xl",
                    inv.action === "RESTOCK" ? "bg-brand/12 text-brand" : inv.action === "REDUCE" ? "bg-destructive/12 text-destructive" : "bg-fast/12 text-fast",
                  )}
                >
                  <ActionIcon className="size-7" aria-hidden />
                </span>
                <div>
                  <p className="font-display text-3xl leading-none">
                    {inv.action}
                    {inv.order_qty > 0 && <span className="num text-2xl"> {inv.order_qty.toFixed(0)}</span>}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    stock position <span className="num">{fmt(inv.on_hand + inv.on_order, 0)}</span>, reorder at{" "}
                    <span className="num">{fmt(inv.reorder_point, 1)}</span>, fill up to <span className="num">{fmt(inv.order_up_to, 1)}</span>
                  </p>
                </div>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">No inventory simulation is stored for this run (a validation deliberation sample).</p>
            )}
          </Section>
        </div>
      </div>

      <Section title={`${s.item} over the run`} description="Drag the slider or scroll to zoom. Amber points were deliberated.">
        <EChart option={timeline} height={300} />
        <Legend
          items={[
            { label: "actual", className: "bg-actual" },
            { label: "consensus", className: "bg-chart-4" },
            { label: "RetailSense final", className: "bg-brand" },
            { label: "deliberated", className: "bg-deliberate" },
          ]}
        />
      </Section>
    </>
  );
}
