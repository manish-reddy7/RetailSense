"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronRight, FileSpreadsheet, Plus, Store as StoreIcon } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Loading, PageHeader, Problem, Section } from "@/components/common";
import { DEFAULT_STORE, StoreForm } from "@/components/retailer";
import { Button } from "@/components/ui/button";
import { api, post, type Store } from "@/lib/api";

const STEPS = [
  ["Add your store", "Its town sets the local weather and public holidays the agents use."],
  ["Upload sales history", "At least 400 days, one row per item per day. Optionally your current stock."],
  ["Get next week's plan", "A 7-day forecast per item and what to order: restock, hold or reduce."],
];

export default function StoresPage() {
  const router = useRouter();
  const client = useQueryClient();
  const stores = useQuery({ queryKey: ["stores"], queryFn: () => api<Store[]>("/stores"), staleTime: 0 });
  const [adding, setAdding] = useState(false);

  if (stores.error) return <Problem error={stores.error} />;
  if (stores.isPending) return <Loading />;
  const list = stores.data;

  return (
    <>
      <PageHeader
        title="Plan next week's restocking"
        aside={
          list.length > 0 &&
          !adding && (
            <Button size="lg" onClick={() => setAdding(true)}>
              <Plus aria-hidden /> Add a store
            </Button>
          )
        }
      >
        Forecasting agents read your sales history, the local weather, the holiday calendar and, if you add it, search interest. When they disagree on an
        item, an AI critic reviews the evidence and the agents get one chance to revise. The order quantities come from a fixed
        stocking rule, never from the AI. Everything runs on this computer; your data does not leave it.
      </PageHeader>

      {list.length > 0 && (
        <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {list.map((s) => (
            <li key={s.id}>
              <Link
                href={`/retailer/store/?id=${s.id}`}
                className="group flex items-center gap-3 rounded-xl border border-border bg-card p-4 transition-colors hover:border-brand/40"
              >
                <span className="grid size-10 place-items-center rounded-lg bg-brand/10 text-brand">
                  <StoreIcon className="size-5" aria-hidden />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate font-medium">{s.name}</span>
                  <span className="block text-xs text-muted-foreground">
                    {s.files.sales ? `${s.files.sales.items} items, sales up to ${s.files.sales.last}` : "no sales uploaded yet"}
                  </span>
                </span>
                <ChevronRight className="size-4 text-muted-foreground transition-transform group-hover:translate-x-0.5" aria-hidden />
              </Link>
            </li>
          ))}
        </ul>
      )}

      <div className="grid gap-6 lg:grid-cols-[1fr_22rem]">
        {(adding || !list.length) && (
          <Section title="Add a store" className="lg:order-first">
            <StoreForm
              initial={DEFAULT_STORE}
              submitLabel="Add store"
              onSubmit={async (s) => {
                const store = await post<Store>("/stores", s);
                await client.invalidateQueries({ queryKey: ["stores"] });
                router.push(`/retailer/store/?id=${store.id}`);
              }}
            />
          </Section>
        )}
        <Section title="How it works" className={adding || !list.length ? "" : "lg:col-span-2"}>
          <ol className="grid gap-4">
            {STEPS.map(([title, text], i) => (
              <li key={title} className="flex gap-3">
                <span className="num grid size-7 shrink-0 place-items-center rounded-full bg-brand text-sm text-primary-foreground">{i + 1}</span>
                <span>
                  <span className="font-medium">{title}</span>
                  <span className="block text-sm text-muted-foreground">{text}</span>
                </span>
              </li>
            ))}
          </ol>
          <div className="mt-5 flex flex-wrap gap-2">
            <a className="flex items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-sm hover:bg-muted" href="/templates/sales_template.csv" download>
              <FileSpreadsheet className="size-4 text-brand" aria-hidden /> Sales file template
            </a>
            <a className="flex items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-sm hover:bg-muted" href="/templates/stock_template.csv" download>
              <FileSpreadsheet className="size-4 text-brand" aria-hidden /> Stock file template
            </a>
            <a className="flex items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-sm hover:bg-muted" href="/templates/trends_template.csv" download>
              <FileSpreadsheet className="size-4 text-brand" aria-hidden /> Search interest template
            </a>
          </div>
          <p className="mt-4 text-xs text-muted-foreground">
            This is a research prototype. In our own tests on two datasets it was not more accurate than simpler forecasting
            methods; see the research results before relying on it.
          </p>
        </Section>
      </div>
    </>
  );
}
