"use client";

import { useQuery } from "@tanstack/react-query";
import { CloudRain, Radio, WifiOff } from "lucide-react";
import { DataTable, Empty, Loading, PageHeader, Problem, Section } from "@/components/common";
import { EChart } from "@/components/echart";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { fmt } from "@/lib/data";
import { useSelection } from "@/lib/selection";
import { useTokens } from "@/lib/theme";

type Cached<T> = { data: T; fromCache: boolean; savedAt: string };

/** Fetch live; on any failure fall back to the last good response kept in this browser. */
async function withFallback<T>(key: string, get: () => Promise<T>): Promise<Cached<T>> {
  try {
    const data = await get();
    const savedAt = new Date().toISOString();
    try {
      localStorage.setItem(key, JSON.stringify({ data, savedAt }));
    } catch {}
    return { data, fromCache: false, savedAt };
  } catch (e) {
    try {
      const hit = localStorage.getItem(key);
      if (hit) return { ...JSON.parse(hit), fromCache: true };
    } catch {}
    throw e;
  }
}

type Daily = { time: string[]; temperature_2m_max: number[]; temperature_2m_min: number[]; precipitation_sum: number[] };

export default function LivePage() {
  const s = useSelection();
  const t = useTokens();
  const live = s.meta?.live;
  const weather = useQuery({
    queryKey: ["live-weather", live?.lat, live?.lon],
    enabled: !!live,
    retry: 1,
    queryFn: () =>
      withFallback(`rs.live.weather.${live!.lat},${live!.lon}`, async () => {
        const q = new URLSearchParams({
          latitude: String(live!.lat),
          longitude: String(live!.lon),
          daily: "temperature_2m_max,temperature_2m_min,precipitation_sum",
          timezone: live!.timezone,
          forecast_days: "7",
        });
        const r = await fetch(`https://api.open-meteo.com/v1/forecast?${q}`);
        if (!r.ok) throw new Error(`Open-Meteo ${r.status}`);
        return (await r.json()).daily as Daily;
      }),
  });

  if (s.error) return <Problem error={s.error} />;
  if (!s.meta) return <Loading />;
  const w = weather.data?.data;

  return (
    <div className="grid gap-6">
      <PageHeader
        title="What the agents would see today"
        aside={
          <span className="flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5 text-sm font-medium text-muted-foreground">
            <Radio className="size-4" aria-hidden /> Demo, not a live store
          </span>
        }
      >
        Today&apos;s live signals are shown next to a <b>replayed</b> item&apos;s stored forecasts from {s.meta.label.replaceAll(" · ", ", ")}.
        Models are not re-run here, there is no point-of-sale data, and nothing on this page is a forecast for a real
        store today.
      </PageHeader>

      <Section
        title="Today's weather forecast"
        description={`Open-Meteo forecast API for the store's representative city (${fmt(live?.lat, 2)}, ${fmt(live?.lon, 2)}).`}
      >
        {weather.isPending ? (
          <Loading />
        ) : weather.error ? (
          <Alert variant="destructive">
            <WifiOff aria-hidden />
            <AlertTitle>Weather unavailable and nothing cached in this browser</AlertTitle>
            <AlertDescription>{String(weather.error)}</AlertDescription>
          </Alert>
        ) : (
          w && (
            <>
              <p className="mb-2 text-xs text-muted-foreground">
                {weather.data!.fromCache ? `API unavailable: showing cached response from ${weather.data!.savedAt}` : "Fresh from the API"}
              </p>
              <EChart
                height={300}
                option={{
                  tooltip: { trigger: "axis" },
                  legend: { bottom: 0 },
                  grid: { left: 48, right: 48, top: 24, bottom: 56 },
                  xAxis: { type: "category", data: w.time },
                  yAxis: [
                    { type: "value", name: "°C", scale: true },
                    { type: "value", name: "mm", min: 0 },
                  ],
                  series: [
                    { name: "max °C", type: "line", data: w.temperature_2m_max, itemStyle: { color: t?.destructive }, lineStyle: { color: t?.destructive } },
                    { name: "min °C", type: "line", data: w.temperature_2m_min, itemStyle: { color: t?.brand }, lineStyle: { color: t?.brand } },
                    { name: "precipitation", type: "bar", yAxisIndex: 1, barWidth: "40%", data: w.precipitation_sum, itemStyle: { color: t?.fast, opacity: 0.55 } },
                  ],
                }}
              />
              <div className="mt-2 flex flex-wrap gap-2">
                {w.time.map((d, i) =>
                  w.precipitation_sum[i] >= (live?.rainy_mm ?? 1) ? (
                    <Badge key={d} variant="secondary">
                      <CloudRain aria-hidden /> rainy-day flag {d}
                    </Badge>
                  ) : null,
                )}
              </div>
            </>
          )
        )}
      </Section>

      <Section title={`Replayed item ${s.item ?? ""}`} description={`Stored agent forecasts for ${s.date ?? "–"} (change them in the sidebar).`}>
        {s.current ? (
          <DataTable
            rows={s.current.agents.map((a) => ({
              agent: a.id,
              forecast: a.forecast,
              lower: a.lower,
              upper: a.upper,
              confidence: a.confidence,
            }))}
          />
        ) : (
          <Empty>No stored case selected.</Empty>
        )}
      </Section>
    </div>
  );
}
