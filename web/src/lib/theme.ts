"use client";

import { useSyncExternalStore } from "react";

// Chart colours come from the CSS tokens in globals.css, so charts and UI always agree and
// follow the light/dark scheme. ECharts cannot parse oklch(), so each token is resolved to rgb
// through a 1x1 canvas.

const NAMES = ["brand", "fast", "deliberate", "actual", "good", "muted-foreground", "foreground", "border", "card", "grid", "chart-4", "chart-5", "destructive"] as const;
export type Tokens = Record<(typeof NAMES)[number], string> & { palette: string[]; dark: boolean };

function resolve(v: string): string {
  const c = document.createElement("canvas");
  c.width = c.height = 1;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#000";
  ctx.fillStyle = v;
  ctx.fillRect(0, 0, 1, 1);
  const [r, g, b, a] = ctx.getImageData(0, 0, 1, 1).data;
  return a === 255 ? `rgb(${r},${g},${b})` : `rgba(${r},${g},${b},${(a / 255).toFixed(3)})`;
}

let cache: { key: string; t: Tokens } | undefined;
const media = () => window.matchMedia("(prefers-color-scheme: dark)");

function read(): Tokens | undefined {
  if (typeof window === "undefined") return undefined;
  const dark = media().matches;
  const key = dark ? "dark" : "light";
  if (cache?.key === key) return cache.t;
  const css = getComputedStyle(document.documentElement);
  const t = Object.fromEntries(NAMES.map((n) => [n, resolve(css.getPropertyValue(`--${n}`).trim())])) as unknown as Tokens;
  t.dark = dark;
  t.palette = [t.brand, t.deliberate, t.fast, t["chart-4"], t.good, t["chart-5"], t.destructive];
  cache = { key, t };
  return t;
}

function subscribe(cb: () => void) {
  const m = media();
  m.addEventListener("change", cb);
  return () => m.removeEventListener("change", cb);
}

/** Current colour tokens (undefined during the static server render). */
export const useTokens = () => useSyncExternalStore(subscribe, read, () => undefined);

/** ECharts theme built from the tokens. */
export function chartTheme(t: Tokens) {
  const axis = {
    axisLine: { show: true, lineStyle: { color: t.border } },
    axisTick: { show: false },
    axisLabel: { color: t["muted-foreground"], fontSize: 11 },
    splitLine: { lineStyle: { color: t.grid } },
    nameTextStyle: { color: t["muted-foreground"], fontSize: 11 },
  };
  return {
    color: t.palette,
    backgroundColor: "transparent",
    textStyle: { fontFamily: "var(--font-geist-sans), system-ui, sans-serif", color: t.foreground },
    legend: { textStyle: { color: t["muted-foreground"], fontSize: 11 }, itemWidth: 14, itemHeight: 8, icon: "roundRect" },
    tooltip: {
      backgroundColor: t.card,
      borderColor: t.border,
      borderWidth: 1,
      textStyle: { color: t.foreground, fontSize: 12 },
      extraCssText: "border-radius:10px;box-shadow:0 8px 24px rgba(0,0,0,.12);",
    },
    categoryAxis: { ...axis, splitLine: { show: false } },
    valueAxis: { ...axis, axisLine: { show: false } },
    line: { symbol: "circle", symbolSize: 5, smooth: false, lineStyle: { width: 2 } },
    bar: { itemStyle: { borderRadius: [4, 4, 0, 0] } },
  };
}
