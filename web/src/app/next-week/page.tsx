"use client";

import { useMemo } from "react";
import { EChart } from "@/components/echart";
import { DataTable, Empty, Kpi, Legend, Loading, PageHeader, Problem, Section } from "@/components/common";
import { fmt, useTables, type Row } from "@/lib/data";
import { useSelection } from "@/lib/selection";
import { useTokens } from "@/lib/theme";

const SYSTEMS = [
  { id: "retailsense", label: "RetailSense", tone: "deliberate", note: "agents + LLM when they disagree" },
  { id: "lgbm_all_features", label: "LightGBM", tone: "brand", note: "most accurate in the test run" },
  { id: "ets", label: "ETS", tone: "fast", note: "cheapest inventory in the test run" },
] as const;

const num = (v?: Row[string]) => Number(v ?? 0);
const BG = { deliberate: "bg-deliberate", brand: "bg-brand", fast: "bg-fast" } as const;

export default function NextWeekPage() {
  const s = useSelection();
  const t = useTokens();
  const tables = useTables(s.dataset);
  const rows = useMemo(() => tables.data?.validation?.future_forecast ?? [], [tables.data]);
  const items = useMemo(() => [...new Set(rows.map((r) => String(r.item_id)))].sort(), [rows]);
  const item = s.item && items.includes(s.item) ? s.item : items[0];

  const chart = useMemo(() => {
    if (!t || !item) return {};
    const of = (sys: string) => rows.filter((r) => r.item_id === item && r.system === sys).sort((a, b) => num(a.h) - num(b.h));
    const dates = of("retailsense").map((r) => String(r.date).slice(5));
    const color = { deliberate: t.deliberate, brand: t.brand, fast: t["chart-4"] };  // ETS: distinct from LightGBM
    return {
      tooltip: { trigger: "axis" },
      grid: { left: 44, right: 16, top: 24, bottom: 32 },
      xAxis: { type: "category", data: dates, boundaryGap: false },
      yAxis: { type: "value", name: "units / day", min: 0 },
      series: SYSTEMS.flatMap((sys) => {
        const r = of(sys.id);
        const c = color[sys.tone];
        return [
          { type: "line", stack: sys.id, data: r.map((x) => num(x.lower)), lineStyle: { opacity: 0 }, symbol: "none", tooltip: { show: false }, silent: true },
          { type: "line", stack: sys.id, data: r.map((x) => num(x.upper) - num(x.lower)), lineStyle: { opacity: 0 }, symbol: "none", areaStyle: { color: c, opacity: 0.1 }, tooltip: { show: false }, silent: true },
          { name: sys.label, type: "line", data: r.map((x) => num(x.forecast)), symbolSize: 6, lineStyle: { color: c, width: 2 }, itemStyle: { color: c } },
        ];
      }),
    };
  }, [rows, item, t]);

  if (s.error) return <Problem error={s.error} />;
  if (tables.isPending) return <Loading />;
  if (!rows.length)
    return (
      <Empty>
        No next-week forecast yet. Run <code>python -m retailsense.forecast_future</code>, then <code>python -m retailsense.web_export</code>.
      </Empty>
    );

  const day1 = rows.filter((r) => num(r.h) === 1);
  const madeOn = String(rows[0].made_on);
  const first = String(rows.find((r) => num(r.h) === 1)?.date);
  const last = String(rows.reduce((a, b) => (num(b.h) > num(a.h) ? b : a)).date);
  const pick = (sys: string, it: string) => day1.find((r) => r.system === sys && r.item_id === it);
  const order = (r?: Row) => (r ? (r.action === "RESTOCK" ? `RESTOCK ${num(r.order_qty)}` : String(r.action)) : "–");
  const list: Row[] = items.map((it) => {
    const rs = pick("retailsense", it);
    return {
      item: it,
      "in stock (RetailSense)": Math.round(num(rs?.on_hand)),
      RetailSense: order(rs),
      LightGBM: order(pick("lgbm_all_features", it)),
      ETS: order(pick("ets", it)),
      D: num(rs?.D),
      "LLM used": ["true", "True"].includes(String(rs?.deliberated)) ? "yes" : "no",
    };
  });
  const sel = rows.filter((r) => r.item_id === item && num(r.h) === 1);

  return (
    <>
      <PageHeader
        context={`Forecast made at the end of ${madeOn}`}
        title={
          <>
            What to order for {first.slice(5)} to {last.slice(5)}
          </>
        }
      >
        Every model is refit on all data up to {madeOn} and forecasts the next 7 days. Each system starts from its own stock at
        the end of the test run and applies the same (s, S) policy. RetailSense was not the most accurate system in testing, so
        all three are shown.
      </PageHeader>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        {SYSTEMS.map((sys, i) => {
          const r = day1.filter((x) => x.system === sys.id);
          const units = r.reduce((a, x) => a + num(x.order_qty), 0);
          const n = r.filter((x) => x.action === "RESTOCK").length;
          return (
            <Kpi key={sys.id} index={i} tone={sys.tone} label={`${sys.label}: order now`} value={`${units.toLocaleString()} units`} hint={`${n} of ${r.length} items to restock. ${sys.note}`} />
          );
        })}
      </div>

      <Section
        title={`${item}, next 7 days`}
        description="Lines are each system's daily forecast; shaded areas are its 10-90% band. Pick another item in the sidebar."
      >
        <Legend items={SYSTEMS.map((x) => ({ label: x.label, className: BG[x.tone] }))} />
        <EChart option={chart} height={300} />
        <div className="mt-3 grid gap-2 text-sm md:grid-cols-3">
          {SYSTEMS.map((sys) => {
            const r = sel.find((x) => x.system === sys.id);
            return (
              <p key={sys.id} className="rounded-lg border border-border p-2.5">
                <span className="font-medium">{sys.label}:</span> {order(r)}
                <span className="block text-xs text-muted-foreground">
                  stock {fmt(num(r?.on_hand), 0)} plus {fmt(num(r?.on_order), 0)} on order; reorder at {fmt(num(r?.reorder_point), 1)},
                  fill up to {fmt(num(r?.order_up_to), 1)}
                </span>
              </p>
            );
          })}
        </div>
      </Section>

      <Section  title="Decision per item" description="RESTOCK n = order n units now (whole case packs); arrives after the lead time.">
        <DataTable rows={list} highlight={(r) => String(r.RetailSense).startsWith("RESTOCK")} highlightClass="bg-good/10 hover:bg-good/15" />
      </Section>
    </>
  );
}
