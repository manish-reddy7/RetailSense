"use client";

import { FlaskConical } from "lucide-react";
import { useState } from "react";
import { DataTable, Empty, Loading, PageHeader, Problem, Section } from "@/components/common";
import { EChart } from "@/components/echart";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useTables, type Row } from "@/lib/data";
import { useSelection } from "@/lib/selection";
import { useTokens, type Tokens } from "@/lib/theme";

type Kind = "final" | "dryrun" | "validation";
const LABEL: Record<Kind, string> = { final: "Final test run", dryrun: "Dry run", validation: "Validation" };
const EXP: Record<Kind, string | undefined> = { final: "final_test", dryrun: "dryrun_val", validation: undefined };
const num = (v: unknown) => (typeof v === "number" ? v : null);
const isRS = (r: Row) => r.system === "retailsense" || r.vs === "retailsense";
const nice = (s: unknown) => String(s).replaceAll("_", " ");

function wapeChart(rows: Row[], t: Tokens) {
  return {
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    grid: { left: 52, right: 16, top: 16, bottom: 72 },
    xAxis: { type: "category", data: rows.map((r) => nice(r.system)), axisLabel: { rotate: 30, interval: 0 } },
    yAxis: { type: "value", name: "WAPE", scale: true },
    series: [
      {
        type: "bar",
        barWidth: "55%",
        data: rows.map((r) => ({ value: num(r.h1_WAPE), itemStyle: { color: isRS(r) ? t.deliberate : t.brand, opacity: isRS(r) ? 1 : 0.75 } })),
      },
      {
        type: "custom",
        name: "95% CI",
        renderItem: (_: unknown, api: { value: (i: number) => number; coord: (p: number[]) => number[] }) => {
          const x = api.value(0);
          const lo = api.coord([x, api.value(1)]);
          const hi = api.coord([x, api.value(2)]);
          const style = { stroke: t.foreground, lineWidth: 1.2, opacity: 0.6 };
          return {
            type: "group",
            children: [
              { type: "line", shape: { x1: lo[0], y1: lo[1], x2: hi[0], y2: hi[1] }, style },
              { type: "line", shape: { x1: lo[0] - 5, y1: lo[1], x2: lo[0] + 5, y2: lo[1] }, style },
              { type: "line", shape: { x1: hi[0] - 5, y1: hi[1], x2: hi[0] + 5, y2: hi[1] }, style },
            ],
          };
        },
        data: rows.map((r, i) => [i, num(r.h1_WAPE_ci_lo), num(r.h1_WAPE_ci_hi)]),
        z: 10,
      },
    ],
  };
}

function costChart(rows: Row[], t: Tokens) {
  const parts: [string, string, string][] = [
    ["holding_cost", "holding", t.fast],
    ["ordering_cost", "ordering", t["chart-5"]],
    ["lost_sales_cost", "lost sales", t.destructive],
  ];
  return {
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    legend: { top: 0, right: 0 },
    grid: { left: 64, right: 16, top: 32, bottom: 72 },
    xAxis: { type: "category", data: rows.map((r) => nice(r.system)), axisLabel: { rotate: 30, interval: 0 } },
    yAxis: { type: "value", name: "$" },
    series: parts.map(([k, name, color], i) => ({
      name,
      type: "bar",
      stack: "cost",
      barWidth: "55%",
      data: rows.map((r) => num(r[k])),
      itemStyle: { color, borderRadius: i === parts.length - 1 ? [4, 4, 0, 0] : 0 },
    })),
  };
}

function curveChart(rows: Row[], metric: "wape" | "cost", t: Tokens) {
  const mean = rows.map((r) => num(r[`${metric}_random`]) ?? 0);
  const sd = rows.map((r) => num(r[`${metric}_random_sd`]) ?? 0);
  return {
    tooltip: { trigger: "axis" },
    legend: { top: 0, right: 0, data: ["selective (highest D first)", "random trigger"] },
    grid: { left: 64, right: 16, top: 36, bottom: 40 },
    xAxis: { type: "category", data: rows.map((r) => `${Math.round(100 * Number(r.trigger_rate))}%`), name: "trigger rate", nameLocation: "middle", nameGap: 26, boundaryGap: false },
    yAxis: { type: "value", scale: true, name: metric === "wape" ? "WAPE (h=1)" : "total cost ($)" },
    series: [
      { type: "line", stack: "band", data: mean.map((m, i) => m - sd[i]), lineStyle: { opacity: 0 }, symbol: "none", tooltip: { show: false } },
      { type: "line", stack: "band", data: sd.map((v) => 2 * v), lineStyle: { opacity: 0 }, areaStyle: { color: t.fast, opacity: 0.15 }, symbol: "none", tooltip: { show: false } },
      { name: "random trigger", type: "line", data: mean, lineStyle: { type: "dashed", color: t.fast }, itemStyle: { color: t.fast } },
      { name: "selective (highest D first)", type: "line", data: rows.map((r) => num(r[`${metric}_selective`])), itemStyle: { color: t.deliberate }, lineStyle: { color: t.deliberate, width: 2.5 } },
    ],
  };
}

function decileChart(rows: { decile: number; mean_rel_err: number; wape: number }[], t: Tokens) {
  return {
    tooltip: { trigger: "axis" },
    legend: { top: 0, left: 0 },
    grid: { left: 52, right: 52, top: 48, bottom: 40 },
    xAxis: { type: "category", data: rows.map((r) => `D${r.decile}`), name: "disagreement decile (low → high)", nameLocation: "middle", nameGap: 26 },
    yAxis: [{ type: "value", name: "mean rel. |error|" }, { type: "value", name: "WAPE", splitLine: { show: false } }],
    series: [
      { name: "mean relative |error|", type: "bar", barWidth: "55%", data: rows.map((r) => r.mean_rel_err), itemStyle: { color: t.brand, opacity: 0.8 } },
      { name: "WAPE", type: "line", yAxisIndex: 1, data: rows.map((r) => r.wape), itemStyle: { color: t.actual }, lineStyle: { color: t.actual } },
    ],
  };
}

function groupedChart(rows: Row[], x: string, series: string, y: string, t: Tokens, name: string) {
  const xs = [...new Set(rows.map((r) => String(r[x])))];
  const ss = [...new Set(rows.map((r) => String(r[series])))];
  return {
    tooltip: { trigger: "axis" },
    legend: { top: 0, type: "scroll" },
    grid: { left: 64, right: 16, top: 40, bottom: 32 },
    xAxis: { type: "category", data: xs },
    yAxis: { type: "value", name, scale: true },
    series: ss.map((s, i) => ({
      name: nice(s),
      type: x === "value" ? "line" : "bar",
      data: xs.map((v) => num(rows.find((r) => String(r[x]) === v && String(r[series]) === s)?.[y])),
      itemStyle: { color: s === "retailsense" ? t.deliberate : t.palette[i % t.palette.length] },
    })),
  };
}

const SHOWN = new Set([
  "table_forecast_metrics", "table_inventory", "table_cost_accuracy", "table_backtest",
  "phase4_inventory", "phase4_sensitivity", "h1_deciles_D", "phase3b_cost_accuracy",
]);

export default function EvaluatePage() {
  const s = useSelection();
  const t = useTokens();
  const tables = useTables(s.dataset);
  const [picked, setPicked] = useState<Kind>();
  const [param, setParam] = useState("lead_time");
  const available = (["final", "dryrun", "validation"] as Kind[]).filter((k) => tables.data?.[k]);
  if (s.error || tables.error) return <Problem error={s.error ?? tables.error} />;
  if (!s.meta || tables.isPending || !t) return <Loading />;
  if (!available.length) return <Empty>No result tables exported for this dataset yet.</Empty>;
  const kind = picked && available.includes(picked) ? picked : available[0];
  const tb = tables.data![kind]!;
  const exp = EXP[kind];
  const deciles = exp ? s.meta.experiments[exp]?.deciles : undefined;
  const valDeciles = tb.h1_deciles_D?.map((r, i) => ({ decile: i + 1, mean_rel_err: Number(r.mean_rel_err), wape: Number(r.wape) }));
  const inv = tb.table_inventory ?? tb.phase4_inventory;
  const curve = tb.table_cost_accuracy ?? tb.phase3b_cost_accuracy;
  const params = [...new Set((tb.phase4_sensitivity ?? []).map((r) => String(r.param)))];

  return (
    <>
      <PageHeader title="How the systems compare">
        Every system runs through the same inventory simulator with the same parameters; only the forecasts differ.
      </PageHeader>

      <div className="sticky top-12 z-10 -my-2 flex items-center gap-3 bg-background/80 py-2 backdrop-blur">
        <Tabs value={kind} onValueChange={(v) => setPicked(v as Kind)}>
          <TabsList>
            {available.map((k) => (
              <TabsTrigger key={k} value={k} className="px-3">
                {LABEL[k]}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </div>

      {kind === "dryrun" && (
        <Alert>
          <FlaskConical aria-hidden />
          <AlertTitle>Pipeline check, not a result</AlertTitle>
          <AlertDescription>
            Validation window with a stub LLM where every agent holds, so systems 7-9 must equal system 6. Baselines and H1
            numbers are real validation numbers.
          </AlertDescription>
        </Alert>
      )}

      {tb.table_forecast_metrics && (
        <Section title="Forecast error, all systems" description="Day-1 WAPE with 95% CI from a bootstrap over items. Amber = RetailSense.">
          <EChart option={wapeChart(tb.table_forecast_metrics, t)} height={340} />
          <div className="mt-4">
            <DataTable rows={tb.table_forecast_metrics} highlight={isRS} />
          </div>
        </Section>
      )}

      {inv && (
        <Section title="What each forecast costs" description="Total cost split into holding, ordering and lost sales.">
          <EChart option={costChart(inv, t)} height={340} />
          <div className="mt-4">
            <DataTable rows={inv} highlight={isRS} />
          </div>
        </Section>
      )}

      <div className="grid gap-4 xl:grid-cols-2">
        {(deciles?.length || valDeciles?.length) && (
          <Section eyebrow="H1" title="Does disagreement predict error?" description="If it does, bars climb from left to right.">
            <EChart option={decileChart((deciles?.length ? deciles : valDeciles)!, t)} height={300} />
          </Section>
        )}
        {curve && (
          <Section eyebrow="H3 and H4" title="Cost-accuracy curve" description="Deliberating the highest-D cases first vs a random trigger at the same rate (band = ±1 sd).">
            <EChart option={curveChart(curve, "wape", t)} height={300} />
          </Section>
        )}
      </div>
      {tb.table_cost_accuracy && (
        <Section eyebrow="H5" title="Inventory cost vs trigger rate">
          <EChart option={curveChart(tb.table_cost_accuracy, "cost", t)} height={300} />
        </Section>
      )}

      {tb.table_backtest && (
        <Section title="Rolling-origin backtest" description="Three earlier 56-day windows, each refit on the days before it.">
          <EChart option={groupedChart(tb.table_backtest, "window", "system", "h1_WAPE", t, "WAPE (h=1)")} height={320} />
          <div className="mt-4">
            <DataTable rows={tb.table_backtest} highlight={isRS} />
          </div>
        </Section>
      )}

      {tb.phase4_sensitivity && (
        <Section title="Inventory cost under other settings" description="One parameter at a time around the defaults.">
          <Tabs value={param} onValueChange={(v) => setParam(String(v))}>
            <TabsList className="mb-2 flex-wrap">
              {params.map((p) => (
                <TabsTrigger key={p} value={p} className="px-3">
                  {nice(p)}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
          <EChart option={groupedChart(tb.phase4_sensitivity.filter((r) => r.param === param), "value", "system", "total_cost", t, "total cost ($)")} height={300} />
        </Section>
      )}

      {Object.entries(tb)
        .filter(([name]) => !SHOWN.has(name))
        .map(([name, rows]) => (
          <Section key={name} title={nice(name.replace(/^table_/, ""))}>
            <DataTable rows={rows} highlight={isRS} />
          </Section>
        ))}
    </>
  );
}
