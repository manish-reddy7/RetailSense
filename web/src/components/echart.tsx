"use client";

import * as echarts from "echarts";
import { useEffect, useRef } from "react";
import { chartTheme, useTokens } from "@/lib/theme";

/** ECharts with the app theme (from CSS tokens). Re-initialises when the colour scheme
 * changes, replaces the option on change and follows the container size. */
export function EChart({ option, height = 320 }: { option: object; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const tokens = useTokens();

  useEffect(() => {
    if (!tokens || !ref.current) return;
    const name = tokens.dark ? "rs-dark" : "rs-light";
    echarts.registerTheme(name, chartTheme(tokens));
    const c = echarts.init(ref.current, name, { renderer: "canvas" });
    chart.current = c;
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(ref.current);
    return () => {
      ro.disconnect();
      c.dispose();
      chart.current = null;
    };
  }, [tokens]);

  useEffect(() => {
    chart.current?.setOption(option as echarts.EChartsOption, true);
  }, [option, tokens]);

  return <div ref={ref} style={{ width: "100%", height }} />;
}
