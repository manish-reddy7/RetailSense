"use client";

import { useMemo } from "react";
import { EChart } from "@/components/echart";
import { type Case } from "@/lib/data";
import { nice } from "@/lib/results";
import { useTokens } from "@/lib/theme";

/** The study's object in one picture, as an ensemble forecaster would draw it: every agent's
 *  next-day forecast as one member of a plume, the spread between them shaded, the days whose
 *  disagreement crossed tau (sent to the LLM critic) marked in amber, actual sales as dots. */
export function Plume({ cases, height = 340 }: { cases: Case[]; height?: number }) {
  const t = useTokens();
  const option = useMemo(() => {
    if (!t || !cases.length) return {};
    const dates = cases.map((c) => c.date.slice(5));
    const agents = cases[0].agents.map((a) => a.id);
    const lo = cases.map((c) => Math.min(...c.agents.map((a) => a.forecast)));
    const hi = cases.map((c, i) => Math.max(...c.agents.map((a) => a.forecast)) - lo[i]);
    const reduce = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const warn = t.deliberate.replace("rgb(", "rgba(").replace(")", t.dark ? ",0.26)" : ",0.2)");
    const runs: [number, number][] = [];
    cases.forEach((c, i) => {
      if (!c.triggered) return;
      const last = runs.at(-1);
      if (last && last[1] === i - 1) last[1] = i;
      else runs.push([i, i]);
    });
    return {
      animation: !reduce,
      animationDuration: 1400,
      grid: { left: 44, right: 92, top: 12, bottom: 28 },
      tooltip: { trigger: "axis", valueFormatter: (v: number) => (typeof v === "number" ? v.toFixed(1) : v) },
      xAxis: { type: "category", data: dates, boundaryGap: false, axisLabel: { interval: 6 } },
      yAxis: { type: "value", name: "units", nameGap: 8, nameTextStyle: { align: "right" } },
      series: [
        { name: "low", type: "line", data: lo, stack: "spread", symbol: "none", lineStyle: { opacity: 0 }, tooltip: { show: false }, silent: true },
        {
          name: "spread between agents",
          type: "line",
          data: hi,
          stack: "spread",
          symbol: "none",
          lineStyle: { opacity: 0 },
          areaStyle: { color: t.fast, opacity: t.dark ? 0.22 : 0.16 },
          tooltip: { show: false },
          silent: true,
          markArea: {
            silent: true,
            itemStyle: { color: warn },
            data: runs.map(([a, b]) => [{ xAxis: dates[a] }, { xAxis: dates[b] }]),
          },
        },
        ...agents.map((id, k) => ({
          name: nice(id),
          type: "line",
          data: cases.map((c) => c.agents.find((a) => a.id === id)?.forecast ?? null),
          symbol: "none",
          lineStyle: { width: 1.5, color: t.fast, opacity: 0.45 + 0.15 * (k % 3) },
          endLabel: { show: true, formatter: nice(id), color: t["muted-foreground"], fontSize: 11 },
          labelLayout: { moveOverlap: "shiftY" },
          emphasis: { focus: "series", lineStyle: { width: 2.5, opacity: 1 } },
        })),
        {
          name: "actual sales",
          type: "scatter",
          data: cases.map((c) => c.actual),
          symbolSize: 5,
          itemStyle: { color: t.actual },
          z: 5,
        },
      ],
    };
  }, [t, cases]);
  return <EChart option={option} height={height} />;
}
