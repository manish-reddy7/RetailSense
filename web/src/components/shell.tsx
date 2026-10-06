"use client";

import { motion } from "framer-motion";
import {
  BarChart3,
  CalendarClock,
  CalendarDays,
  Database,
  FlaskConical,
  LayoutDashboard,
  MessagesSquare,
  Package,
  Radio,
  TrendingUp,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import { useSelection } from "@/lib/selection";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/forecast", label: "Forecast", icon: TrendingUp },
  { href: "/next-week", label: "Next week", icon: CalendarClock },
  { href: "/explain", label: "Explain", icon: MessagesSquare },
  { href: "/evaluate", label: "Evaluate", icon: BarChart3 },
  { href: "/live", label: "Live demo", icon: Radio },
];

function Picker({
  icon: Icon,
  label,
  value,
  options,
  onChange,
  format,
}: {
  icon: typeof Database;
  label: string;
  value?: string;
  options: string[];
  onChange: (v: string) => void;
  format?: (v: string) => string;
}) {
  return (
    <label className="grid min-w-0 gap-1">
      <span className="eyebrow flex items-center gap-1.5">
        <Icon className="size-3" aria-hidden /> {label}
      </span>
      <select
        className="h-9 w-full min-w-0 rounded-lg border border-input bg-card px-2.5 text-sm text-foreground transition-colors hover:border-foreground/30 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none disabled:opacity-50"
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value)}
        disabled={!options.length}
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {format ? format(o) : o}
          </option>
        ))}
      </select>
    </label>
  );
}

function Logo() {
  return (
    <Link href="/" className="flex items-center gap-2.5 px-5 pt-6 pb-5">
      <span className="grid size-9 place-items-center rounded-xl bg-brand text-primary-foreground shadow-sm">
        <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
          <path d="M4 17c3-6 5-6 8-2s5 4 8-4" strokeLinecap="round" />
          <circle cx="12" cy="15" r="1.6" fill="currentColor" stroke="none" />
        </svg>
      </span>
      <span>
        <span className="block font-display text-2xl leading-none">RetailSense</span>
        <span className="text-[11px] text-muted-foreground">disagreement-triggered forecasting</span>
      </span>
    </Link>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const s = useSelection();
  return (
    <div className="flex min-h-dvh flex-col md:flex-row">
      <aside className="border-b border-sidebar-border bg-sidebar md:sticky md:top-0 md:flex md:h-dvh md:w-72 md:shrink-0 md:flex-col md:border-r md:border-b-0">
        <Logo />
        <nav className="flex gap-1 overflow-x-auto px-3 pb-3 md:flex-col md:overflow-visible">
          {NAV.map(({ href, label, icon: Icon }) => {
            const active = path === href;
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "relative flex shrink-0 items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition-colors",
                  active ? "font-medium text-foreground" : "text-muted-foreground hover:text-foreground",
                )}
              >
                {active && (
                  <motion.span
                    layoutId="nav-active"
                    className="absolute inset-0 rounded-lg border border-sidebar-border bg-sidebar-accent shadow-xs"
                    transition={{ type: "spring", stiffness: 500, damping: 40 }}
                  />
                )}
                <Icon className={cn("relative size-4", active && "text-brand")} aria-hidden />
                <span className="relative">{label}</span>
              </Link>
            );
          })}
        </nav>
        <div className="mx-3 mb-4 grid grid-cols-2 gap-3 rounded-xl border border-sidebar-border bg-sidebar-accent/60 p-3 md:mt-auto md:grid-cols-1">
          <p className="eyebrow col-span-full">Viewing</p>
          <Picker
            icon={Database}
            label="Dataset"
            value={s.dataset}
            options={s.datasets}
            onChange={(dataset) => s.set({ dataset, experiment: undefined, item: undefined, date: undefined })}
          />
          <Picker
            icon={FlaskConical}
            label="Run"
            value={s.experiment}
            options={s.experiments}
            format={(e) => s.meta?.experiments[e]?.label ?? e}
            onChange={(experiment) => s.set({ experiment, item: undefined, date: undefined })}
          />
          <Picker icon={Package} label="Item" value={s.item} options={s.items} onChange={(item) => s.set({ item, date: undefined })} />
          <Picker icon={CalendarDays} label="Target date" value={s.date} options={s.dates} onChange={(date) => s.set({ date })} />
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex flex-wrap items-center gap-2 border-b border-border bg-background/80 px-4 py-2.5 backdrop-blur md:sticky md:top-0 md:z-10 md:px-10">
          {s.meta && (
            <>
              <span className="rounded-full bg-secondary px-2.5 py-0.5 text-[11px] font-semibold tracking-wide uppercase">
                {s.meta.dataset} · {s.meta.store}
              </span>
              <span
                className={cn(
                  "rounded-full border px-2 py-0.5 text-[11px] font-medium",
                  s.meta.real_data ? "border-good/40 text-good" : "border-brand/40 text-brand",
                )}
              >
                {s.meta.real_data ? "real sales" : "simulated sales"}
              </span>
            </>
          )}
          <span className="min-w-0 truncate text-xs text-muted-foreground">{s.meta?.label}</span>
          {s.meta && (
            <span className="ml-auto flex flex-wrap gap-1.5 text-[11px]">
              <span className="num rounded-md border border-border px-1.5 py-0.5 text-muted-foreground">{s.meta.model}</span>
              <span className="num rounded-md border border-border px-1.5 py-0.5 text-muted-foreground">prompt {s.meta.prompt_version}</span>
              <span
                className={cn(
                  "num rounded-md border px-1.5 py-0.5",
                  s.meta.tau_frozen === null ? "border-deliberate/40 text-deliberate" : "border-good/40 text-good",
                )}
              >
                τ {s.meta.tau_frozen === null ? "not frozen" : s.meta.tau_frozen.toFixed(4)}
              </span>
            </span>
          )}
        </div>
        <motion.main
          key={path}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.22 }}
          className="min-w-0 flex-1 px-4 py-6 md:px-10"
        >
          <div className="mx-auto grid max-w-7xl gap-6">{children}</div>
        </motion.main>
      </div>
    </div>
  );
}
