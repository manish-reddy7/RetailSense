"use client";

import { motion } from "framer-motion";
import { AlertTriangle, Inbox } from "lucide-react";
import type { ReactNode } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { Row } from "@/lib/data";
import { cn } from "@/lib/utils";

export function cell(v: unknown): string {
  if (v === null || v === undefined || v === "") return "–";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "–";
    if (Number.isInteger(v)) return v.toLocaleString();
    const a = Math.abs(v);
    return a >= 100 ? v.toLocaleString(undefined, { maximumFractionDigits: 1 }) : a >= 1 ? v.toFixed(3) : v.toFixed(4);
  }
  return String(v);
}

/** Page title. `context` is an optional line naming the data shown (an item, a date); it is
 *  not a section label. */
export function PageHeader({
  context,
  title,
  children,
  aside,
}: {
  context?: ReactNode;
  title: ReactNode;
  children?: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <header className="flex flex-wrap items-end gap-x-8 gap-y-4 pt-2 pb-2">
      <div className="min-w-0 flex-1">
        {context && <p className="text-sm text-muted-foreground">{context}</p>}
        <h1 className="mt-1 text-[1.75rem] break-words sm:text-[2.25rem] md:text-[2.75rem]">{title}</h1>
        {children && <div className="measure mt-3 text-muted-foreground">{children}</div>}
      </div>
      {aside}
    </header>
  );
}

/** Tiny inline trend line for KPI tiles. */
function Spark({ values, color = "var(--brand)" }: { values: number[]; color?: string }) {
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pts = values
    .map((v, i) => `${(i / (values.length - 1)) * 100},${28 - ((v - min) / (max - min || 1)) * 24}`)
    .join(" ");
  return (
    <svg viewBox="0 0 100 30" className="h-8 w-24" preserveAspectRatio="none" aria-hidden>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export function Kpi({
  label,
  value,
  hint,
  tone,
  spark,
  index = 0,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "brand" | "fast" | "deliberate" | "good" | "muted";
  spark?: number[];
  index?: number;
}) {
  const rule = {
    brand: "border-brand",
    fast: "border-fast",
    deliberate: "border-deliberate",
    good: "border-good",
    muted: "border-border",
  }[tone ?? "muted"];
  return (
    <div data-index={index} className={cn("min-w-0 border-t-2 pt-3", rule)}>
      <p className="text-sm text-muted-foreground">{label}</p>
      <div className="mt-1 flex items-end justify-between gap-2">
        <p className="num wide text-[1.75rem] leading-none font-semibold">{value}</p>
        {spark && <Spark values={spark} />}
      </div>
      {hint && <p className="mt-2 text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

/** Disagreement D on a scale with the frozen/provisional threshold tau marked. */
export function Meter({ D, tau }: { D: number; tau: number }) {
  const max = Math.max(D, tau) * 1.6 || 1;
  const pct = (x: number) => `${Math.min(100, (x / max) * 100)}%`;
  const over = D >= tau;
  return (
    <div className="grid gap-3">
      <div className="flex items-baseline justify-between">
        <p className="num text-4xl">{D.toFixed(3)}</p>
        <p className={cn("text-sm font-medium", over ? "text-deliberate-ink" : "text-fast")}>
          {over ? "above τ: deliberate" : "below τ: fast path"}
        </p>
      </div>
      <div className="relative h-3 rounded-full bg-muted">
        <div className="absolute inset-y-0 left-0 rounded-l-full bg-fast/25" style={{ width: pct(tau) }} />
        <div className="absolute inset-y-0 right-0 rounded-r-full bg-deliberate/20" style={{ left: pct(tau) }} />
        <motion.div
          className={cn("absolute top-1/2 size-4 -translate-x-1/2 -translate-y-1/2 rounded-full ring-4 ring-card", over ? "bg-deliberate" : "bg-fast")}
          initial={{ left: "0%" }}
          animate={{ left: pct(D) }}
          transition={{ type: "spring", stiffness: 120, damping: 18 }}
        />
        <div className="absolute -top-1.5 -bottom-1.5 w-px bg-foreground/60" style={{ left: pct(tau) }} />
      </div>
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>0, agents agree</span>
        <span className="num">τ = {tau.toFixed(3)}</span>
        <span>agents disagree</span>
      </div>
    </div>
  );
}

export function Section({
  title,
  eyebrow,
  description,
  children,
  className,
}: {
  title: string;
  eyebrow?: string;
  description?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("min-w-0 border-t border-border pt-5", className)}>
      <div className="mb-4">
        <h2 className="text-[1.375rem]">
          {eyebrow && <span className="text-muted-foreground">{eyebrow}. </span>}
          {title}
        </h2>
        {description && <p className="measure mt-1.5 text-sm text-muted-foreground">{description}</p>}
      </div>
      {children}
    </section>
  );
}

export function Legend({ items }: { items: { label: string; className: string }[] }) {
  return (
    <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
      {items.map((i) => (
        <span key={i.label} className="flex items-center gap-1.5">
          <span className={cn("size-2.5 rounded-sm", i.className)} aria-hidden />
          {i.label}
        </span>
      ))}
    </div>
  );
}

/** Any CSV-derived table. `highlight` marks one row (e.g. RetailSense). */
export function DataTable({
  rows,
  highlight,
  highlightClass = "bg-deliberate/10 font-medium hover:bg-deliberate/15",
  columns,
}: {
  rows: Row[];
  highlight?: (r: Row) => boolean;
  /** amber by default: the highlighted row is the deliberating system */
  highlightClass?: string;
  columns?: string[];
}) {
  if (!rows?.length) return <Empty>No rows.</Empty>;
  const cols = columns ?? Object.keys(rows[0]);
  const numeric = (c: string) => typeof rows[0][c] === "number";
  return (
    <div className="overflow-x-auto rounded-md border border-border bg-card">
      <Table>
        <TableHeader className="bg-muted/50">
          <TableRow>
            {cols.map((c) => (
              <TableHead key={c} className={cn("eyebrow h-9 whitespace-nowrap", numeric(c) && "text-right")}>
                {c.startsWith("Unnamed") ? "" : c.replaceAll("_", " ")}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((r, i) => (
            <TableRow key={i} className={highlight?.(r) ? highlightClass : undefined}>
              {cols.map((c) => (
                <TableCell key={c} className={numeric(c) ? "num text-right" : "whitespace-nowrap"}>
                  {cell(r[c])}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
      <Inbox className="size-4 shrink-0" aria-hidden />
      {children}
    </div>
  );
}

export function Loading() {
  return (
    <div className="grid gap-4">
      <Skeleton className="h-28" />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-24" />
        ))}
      </div>
      <Skeleton className="h-72" />
    </div>
  );
}

export function Problem({ error }: { error: unknown }) {
  return (
    <Alert variant="destructive">
      <AlertTriangle aria-hidden />
      <AlertTitle>Could not load stored results</AlertTitle>
      <AlertDescription>
        {String(error)}. Run <code>python -m retailsense.web_export</code> for this dataset, then reload.
      </AlertDescription>
    </Alert>
  );
}
