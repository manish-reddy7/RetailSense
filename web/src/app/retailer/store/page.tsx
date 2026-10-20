"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Download, FileSpreadsheet, Loader2, Play, Upload } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useMemo, useRef, useState } from "react";
import { Empty, Kpi, Loading, Meter, PageHeader, Problem, Section } from "@/components/common";
import { EChart } from "@/components/echart";
import { FormError, StoreForm } from "@/components/retailer";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api, post, upload, type FileMeta, type ItemResult, type Result, type Run, type Store } from "@/lib/api";
import { fmt } from "@/lib/data";
import { useTokens } from "@/lib/theme";
import { cn } from "@/lib/utils";

const ACTION_TONE = { RESTOCK: "bg-brand/10 text-brand", HOLD: "bg-muted text-muted-foreground", REDUCE: "bg-deliberate/15 text-deliberate-ink" };
// Agent evidence text is shared with the LLM prompt, so long decimals are shortened here only.
const tidy = (s: string) => s.replace(/-?\d+\.\d{3,}/g, (m) => String(+Number(m).toFixed(2)));
const when = (t: number | null) => (t ? new Date(t * 1000).toLocaleString() : "–");

export default function StorePage() {
  return (
    <Suspense fallback={<Loading />}>
      <StoreView />
    </Suspense>
  );
}

function DeleteStore({ id, name }: { id: number; name: string }) {
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const remove = async () => {
    if (!window.confirm(`Delete "${name}" with all its uploads and runs? This cannot be undone.`)) return;
    setBusy(true);
    try {
      await api(`/stores/${id}`, { method: "DELETE" });
      window.location.href = "/retailer/";
    } catch (e) {
      setError(e);
      setBusy(false);
    }
  };
  return (
    <div className="mt-6 grid gap-2 border-t border-border pt-4">
      <Button variant="destructive" className="w-fit" disabled={busy} onClick={remove}>
        Delete store
      </Button>
      <FormError error={error} />
    </div>
  );
}

function StoreView() {
  const id = Number(useSearchParams().get("id"));
  const store = useQuery({ queryKey: ["store", id], queryFn: () => api<Store>(`/stores/${id}`), enabled: id > 0, staleTime: 0 });
  const runs = useQuery({
    queryKey: ["runs", id],
    queryFn: () => api<Run[]>(`/stores/${id}/runs`),
    enabled: !!store.data,
    staleTime: 0,
    refetchInterval: (q) => (q.state.data?.some((r) => r.status === "queued" || r.status === "running") ? 3000 : false),
  });
  const latest = runs.data?.[0];
  const done = runs.data?.find((r) => r.status === "done");
  const result = useQuery({
    queryKey: ["result", id, done?.id],
    queryFn: () => api<Result>(`/stores/${id}/runs/${done!.id}/result`),
    enabled: !!done,
  });

  if (store.error) return <Problem error={store.error} />;
  if (store.isPending) return <Loading />;
  const s = store.data;

  return (
    <>
      <PageHeader
        title={s.name}
        aside={
          <Link href="/retailer/" className="text-sm text-muted-foreground hover:text-foreground">
            ← All stores
          </Link>
        }
      >
        <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm sm:grid-cols-4">
          {[
            ["Location", `${s.country}, ${s.lat.toFixed(2)}, ${s.lon.toFixed(2)}`],
            ["Lead time", `${s.inventory.lead_time} days`],
            ["Orders cover", `${s.inventory.coverage} days`],
            ["Service level", `${Math.round(s.inventory.service_level * 100)}%, case pack ${s.inventory.case_pack}`],
          ].map(([k, v]) => (
            <div key={k}>
              <dt className="text-muted-foreground">{k}</dt>
              <dd className="num">{v}</dd>
            </div>
          ))}
        </dl>
      </PageHeader>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <UploadCard
          step={1}
          title="Sales history"
          what="date, item_id, units_sold, price"
          template="/templates/sales_template.csv"
          meta={s.files.sales}
          path={`/stores/${id}/sales`}
        />
        <UploadCard
          step={2}
          title="Current stock (optional)"
          what="item_id, on_hand, on_order"
          template="/templates/stock_template.csv"
          meta={s.files.stock}
          path={`/stores/${id}/stock`}
        />
        <UploadCard
          step={3}
          title="Search interest (optional)"
          what="Week, term: (region), ..."
          note="A weekly Google Trends download (Interest over time, 1-5 search terms, at least 52 weeks). With it, runs add the Trend agent."
          template="/templates/trends_template.csv"
          meta={s.files.trends}
          path={`/stores/${id}/trends`}
          removable
        />
        <RunCard store={s} latest={latest} />
      </div>

      {result.data ? (
        <Results result={result.data} run={done!} />
      ) : (
        done && result.isPending && <Loading />
      )}
      {result.error && <Problem error={result.error} />}

      <details className="rounded-xl border border-border bg-card p-5 [&[open]>summary]:mb-4">
        <summary className="cursor-pointer text-lg">Store settings</summary>
        <StoreForm
          initial={s}
          submitLabel="Save settings"
          onSubmit={async (v) => {
            await api(`/stores/${id}`, { method: "PUT", body: JSON.stringify(v) });
            await store.refetch();
          }}
        />
        <DeleteStore id={id} name={s.name} />
      </details>

      {(runs.data?.length ?? 0) > 0 && (
        <Section title="Runs" description="Each run uses the files and settings as they were when it started.">
          <ul className="grid gap-1 text-sm">
            {runs.data!.map((r) => (
              <li key={r.id} className="flex flex-wrap items-center gap-x-3 gap-y-0.5 rounded-lg px-2 py-1.5 even:bg-muted/40">
                <span className="num text-muted-foreground">#{r.id}</span>
                <StatusBadge status={r.status} />
                <span className="text-muted-foreground">{when(r.created)}</span>
                {r.error && <span className="text-destructive">{r.error}</span>}
              </li>
            ))}
          </ul>
        </Section>
      )}
    </>
  );
}

function StatusBadge({ status }: { status: Run["status"] }) {
  const tone = { queued: "bg-muted text-muted-foreground", running: "bg-fast/15 text-fast", done: "bg-good/15 text-good", failed: "bg-destructive/10 text-destructive" };
  return <span className={cn("rounded-full px-2 py-0.5 text-xs font-medium", tone[status])}>{status}</span>;
}

function StepCard({ step, title, children }: { step: number; title: string; children: React.ReactNode }) {
  return (
    <section className="flex min-w-0 flex-col gap-3 border-t-2 border-foreground/80 pt-4">
      <h2 className="flex items-baseline gap-2.5 text-lg">
        <span className="wide text-2xl leading-none font-semibold text-brand">{step}</span>
        {title}
      </h2>
      {children}
    </section>
  );
}

function UploadCard({
  step,
  title,
  what,
  note,
  template,
  meta,
  path,
  removable,
}: {
  step: number;
  title: string;
  what: string;
  note?: string;
  template: string;
  meta: FileMeta | null;
  path: string;
  removable?: boolean;
}) {
  const client = useQueryClient();
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const send = async (file?: File | "remove") => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      if (file === "remove") await api(path, { method: "DELETE" });
      else await upload(path, file);
      await client.invalidateQueries({ queryKey: ["store"] });
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  };

  return (
    <StepCard step={step} title={title}>
      <p className="text-sm text-muted-foreground">
        CSV with <code className="text-foreground">{what}</code>.{" "}
        <a href={template} download className="inline-flex items-center gap-1 text-brand hover:underline">
          <FileSpreadsheet className="size-3.5" aria-hidden /> template
        </a>
      </p>
      {note && <p className="text-sm text-muted-foreground">{note}</p>}
      {meta && (
        <p className="flex items-start gap-1.5 text-sm">
          <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-good" aria-hidden />
          <span>
            <span className="font-medium">{meta.filename}</span>
            <span className="block text-muted-foreground">
              {meta.terms ? meta.terms.join(", ") : `${meta.items} items`}, {meta.rows.toLocaleString()} {meta.terms ? "weeks" : "rows"}
              {meta.first && `, ${meta.first} to ${meta.last}`}
            </span>
          </span>
        </p>
      )}
      <FormError error={error} />
      <input ref={input} type="file" accept=".csv,text/csv" className="hidden" onChange={(e) => send(e.target.files?.[0])} />
      <Button variant={meta ? "outline" : "default"} size="lg" className="mt-auto" disabled={busy} onClick={() => input.current?.click()}>
        {busy ? <Loader2 className="animate-spin" aria-hidden /> : <Upload aria-hidden />} {meta ? "Replace file" : "Upload CSV"}
      </Button>
      {removable && meta && (
        <Button variant="ghost" size="sm" disabled={busy} onClick={() => send("remove")}>
          Remove
        </Button>
      )}
    </StepCard>
  );
}

function RunCard({ store, latest }: { store: Store; latest?: Run }) {
  const client = useQueryClient();
  const [error, setError] = useState<unknown>(null);
  const active = latest && (latest.status === "queued" || latest.status === "running");
  const log = useQuery({
    queryKey: ["log", latest?.id],
    queryFn: () => api<{ lines: string[] }>(`/stores/${store.id}/runs/${latest!.id}/log`),
    enabled: !!active,
    refetchInterval: 3000,
    staleTime: 0,
  });

  const start = async () => {
    setError(null);
    try {
      await post(`/stores/${store.id}/runs`);
      await client.invalidateQueries({ queryKey: ["runs", store.id] });
    } catch (e) {
      setError(e);
    }
  };

  return (
    <StepCard step={4} title="Forecast next week">
      <p className="text-sm text-muted-foreground">
        Trains the agents on this store&apos;s history, then forecasts the 7 days after the last sales day. Takes a few minutes
        per store.
      </p>
      {active && (
        <div className="grid gap-1.5 text-sm">
          <span className="flex items-center gap-2">
            <Loader2 className="size-4 animate-spin text-fast" aria-hidden /> {latest.status === "queued" ? "Waiting for a free worker" : "Running"}
          </span>
          {log.data?.lines.at(-1) && <span className="num truncate text-xs text-muted-foreground">{log.data.lines.at(-1)}</span>}
        </div>
      )}
      {latest?.status === "failed" && <FormError error={`Last run failed: ${latest.error}`} />}
      <FormError error={error} />
      <Button size="lg" className="mt-auto" disabled={!store.files.sales || !!active} onClick={start}>
        <Play aria-hidden /> {active ? "Running…" : "Run forecast"}
      </Button>
    </StepCard>
  );
}

function ordersCsv(result: Result) {
  const rows = [["item_id", "action", "order_qty", "on_hand", "on_order", "forecast_7d", "deliberated"]];
  for (const it of result.items) {
    const d = it.decision;
    const week = it.forecast.reduce((a, f) => a + f.yhat, 0);
    rows.push([it.item_id, d?.action ?? "", String(d?.order_qty ?? ""), String(d?.on_hand ?? ""), String(d?.on_order ?? ""), week.toFixed(1), String(it.deliberated)]);
  }
  const csv = rows.map((r) => r.map((c) => (/[",\n]/.test(c) ? `"${c.replaceAll('"', '""')}"` : c)).join(",")).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = `orders_${result.made_on}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function Results({ result, run }: { result: Result; run: Run }) {
  const items = useMemo(
    () => [...result.items].sort((a, b) => (b.decision?.order_qty ?? -1) - (a.decision?.order_qty ?? -1) || b.D - a.D),
    [result.items],
  );
  const [selected, setSelected] = useState<string>();
  const item = items.find((i) => i.item_id === selected) ?? items[0];
  const restock = items.filter((i) => i.decision?.action === "RESTOCK");
  const units = restock.reduce((a, i) => a + (i.decision?.order_qty ?? 0), 0);
  const deliberated = items.filter((i) => i.deliberated).length;
  const first = items[0]?.forecast[0]?.date;
  const last = items[0]?.forecast.at(-1)?.date;

  return (
    <>
      <Section
        title={`Plan for ${first} to ${last}`}
        description={`Run #${run.id}, finished ${when(run.finished)}, from sales up to ${result.made_on}. Agents: ${result.agents.join(", ")}. Critic: ${result.llm.model} (${result.llm.provider}).`}
      >
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Kpi label="Items" value={items.length} index={0} />
          <Kpi label="To restock" value={restock.length} hint={`${units.toLocaleString()} units`} tone="brand" index={1} />
          <Kpi label="Reviewed by the critic" value={deliberated} hint="agents disagreed" tone="deliberate" index={2} />
          <Kpi label="Disagreement threshold τ" value={result.tau.toFixed(3)} hint="from this store's last 86 days" tone="muted" index={3} />
        </div>
      </Section>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
        <Section
          title="Orders"
          description="Click an item for its forecast and the agents' reasoning."
          className="xl:self-start"
        >
          <div className="mb-3">
            <Button variant="outline" onClick={() => ordersCsv(result)}>
              <Download aria-hidden /> Download orders CSV
            </Button>
          </div>
          <div className="max-h-[36rem] overflow-auto rounded-lg border border-border">
            <Table>
              <TableHeader className="sticky top-0 bg-muted">
                <TableRow>
                  <TableHead className="eyebrow">Item</TableHead>
                  <TableHead className="eyebrow">Action</TableHead>
                  <TableHead className="eyebrow text-right">Order</TableHead>
                  <TableHead className="eyebrow text-right">On hand</TableHead>
                  <TableHead className="eyebrow text-right">Next 7 days</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {items.map((i) => (
                  <TableRow
                    key={i.item_id}
                    onClick={() => setSelected(i.item_id)}
                    className={cn("cursor-pointer", i.item_id === item?.item_id && "bg-brand/5")}
                    aria-selected={i.item_id === item?.item_id}
                  >
                    <TableCell className="whitespace-nowrap">
                      <span className="flex items-center gap-2">
                        {i.item_id}
                        {i.deliberated && <Badge variant="outline" className="border-deliberate/40 text-deliberate-ink">reviewed</Badge>}
                      </span>
                    </TableCell>
                    <TableCell>
                      {i.decision ? (
                        <span className={cn("rounded-full px-2 py-0.5 text-xs font-medium", ACTION_TONE[i.decision.action])}>{i.decision.action}</span>
                      ) : (
                        <span className="text-xs text-muted-foreground">no stock given</span>
                      )}
                    </TableCell>
                    <TableCell className="num text-right">{i.decision?.order_qty ? i.decision.order_qty.toLocaleString() : "–"}</TableCell>
                    <TableCell className="num text-right">{i.decision ? fmt(i.decision.on_hand, 0) : "–"}</TableCell>
                    <TableCell className="num text-right">{fmt(i.forecast.reduce((a, f) => a + f.yhat, 0), 0)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </Section>
        {item ? <ItemDetail item={item} tau={result.tau} /> : <Empty>No items.</Empty>}
      </div>
    </>
  );
}

function ItemDetail({ item, tau }: { item: ItemResult; tau: number }) {
  const t = useTokens();
  const chart = useMemo(() => {
    if (!t) return {};
    const dates = [...item.history.map((h) => h.date), ...item.forecast.map((f) => f.date)].map((d) => d.slice(5));
    const pad = item.history.map(() => null);
    return {
      tooltip: { trigger: "axis" },
      grid: { left: 40, right: 12, top: 16, bottom: 28 },
      xAxis: { type: "category", data: dates, boundaryGap: false },
      yAxis: { type: "value", name: "units / day", min: 0 },
      series: [
        { name: "Sold", type: "line", data: [...item.history.map((h) => h.units), ...item.forecast.map(() => null)], symbol: "none", lineStyle: { color: t.actual, width: 1.5 } },
        { type: "line", stack: "band", data: [...pad, ...item.forecast.map((f) => f.lower)], lineStyle: { opacity: 0 }, symbol: "none", tooltip: { show: false }, silent: true },
        { type: "line", stack: "band", data: [...pad, ...item.forecast.map((f) => f.upper - f.lower)], lineStyle: { opacity: 0 }, symbol: "none", areaStyle: { color: t.brand, opacity: 0.12 }, tooltip: { show: false }, silent: true },
        { name: "Forecast", type: "line", data: [...pad, ...item.forecast.map((f) => f.yhat)], symbolSize: 6, lineStyle: { color: t.brand, width: 2 }, itemStyle: { color: t.brand } },
      ],
    };
  }, [item, t]);
  const d = item.decision;
  const revised = Object.fromEntries(item.revisions.map((r) => [r.agent_id, r]));

  return (
    <div className="grid min-w-0 gap-6 xl:self-start">
      <Section  title={item.item_id} description="Last 28 days sold, and the forecast with its 10-90% range.">
        <EChart option={chart} height={260} />
        {d && (
          <p className="mt-3 text-sm text-muted-foreground">
            Stock {fmt(d.on_hand, 0)} on hand + {fmt(d.on_order, 0)} on order. Reorder when at or below{" "}
            <span className="num text-foreground">{fmt(d.reorder_point, 1)}</span>, then order up to{" "}
            <span className="num text-foreground">{fmt(d.order_up_to, 1)}</span> in packs:{" "}
            <span className="font-medium text-foreground">
              {d.action}
              {d.order_qty ? ` ${d.order_qty.toLocaleString()} units` : ""}
            </span>
            .
          </p>
        )}
      </Section>

      <Section  title="Do the agents agree?">
        <Meter D={item.D} tau={tau} />
        <div className="mt-4 overflow-x-auto rounded-lg border border-border">
          <Table>
            <TableHeader className="bg-muted/50">
              <TableRow>
                <TableHead className="eyebrow">Agent</TableHead>
                <TableHead className="eyebrow text-right">Tomorrow</TableHead>
                <TableHead className="eyebrow text-right">Range</TableHead>
                <TableHead className="eyebrow">Main evidence</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {item.agents.map((a) => (
                <TableRow key={a.agent}>
                  <TableCell className="capitalize">{a.agent}</TableCell>
                  <TableCell className="num text-right">
                    {fmt(a.forecast, 1)}
                    {revised[a.agent] && !revised[a.agent].held && <span className="block text-xs text-deliberate-ink">→ {fmt(revised[a.agent].y, 1)}</span>}
                  </TableCell>
                  <TableCell className="num text-right whitespace-nowrap text-muted-foreground">
                    {fmt(a.lower, 1)}–{fmt(a.upper, 1)}
                  </TableCell>
                  <TableCell className="min-w-48 text-xs whitespace-normal text-muted-foreground">{a.key_factors.map(tidy).join("; ")}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </Section>

      {item.critic ? (
        <Section  title="Why they disagree" description="The critic judges evidence only; it cannot write a forecast or an order.">
          <p className="rounded-lg bg-deliberate/10 p-3 text-sm">{item.critic.primary_conflict}</p>
          <ul className="mt-3 grid gap-2 text-sm">
            {item.critic.evidence_assessment.map((e) => (
              <li key={e.agent} className="flex gap-2">
                <span className="w-20 shrink-0 capitalize">{e.agent}</span>
                <span className="w-20 shrink-0 text-xs font-medium first-letter:uppercase">{e.strength}</span>
                <span className="text-muted-foreground">{e.reason}</span>
              </li>
            ))}
          </ul>
          {item.revisions.length > 0 && (
            <ul className="mt-4 grid gap-2 border-t border-border pt-3 text-sm">
              {item.revisions.map((r) => (
                <li key={r.agent_id}>
                  <span className="capitalize">{r.agent_id}</span>{" "}
                  <span className={cn("rounded-full px-1.5 text-[10px] font-medium", r.held ? "bg-muted text-muted-foreground" : "bg-deliberate/15 text-deliberate-ink")}>
                    {r.held ? "held" : "moved"}
                  </span>
                  <span className="block text-muted-foreground">{r.rationale}</span>
                </li>
              ))}
            </ul>
          )}
        </Section>
      ) : (
        item.fallback && <Empty>The critic could not review this item ({item.fallback}); the agents&apos; consensus was used.</Empty>
      )}
    </div>
  );
}
