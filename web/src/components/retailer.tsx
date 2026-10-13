"use client";

import { MapPin, Search } from "lucide-react";
import { useState, type ComponentProps, type FormEvent, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import type { StoreSettings } from "@/lib/api";
import { cn } from "@/lib/utils";

export const inputClass =
  "h-9 w-full min-w-0 rounded-lg border border-input bg-card px-2.5 text-sm text-foreground transition-colors hover:border-foreground/30 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none disabled:opacity-50";

export function Field({ label, hint, className, ...props }: { label: string; hint?: ReactNode } & ComponentProps<"input">) {
  return (
    <label className={cn("grid min-w-0 gap-1", className)}>
      <span className="eyebrow">{label}</span>
      <input className={inputClass} {...props} />
      {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
    </label>
  );
}

export function FormError({ error }: { error: unknown }) {
  if (!error) return null;
  return (
    <p role="alert" className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
      {error instanceof Error ? error.message : String(error)}
    </p>
  );
}

export const DEFAULT_STORE: StoreSettings = {
  name: "",
  country: "",
  lat: NaN,
  lon: NaN,
  inventory: { lead_time: 3, coverage: 7, service_level: 0.95, case_pack: 6 },
};

type Place = { name: string; admin1?: string; country: string; country_code: string; latitude: number; longitude: number };

/** Store name, location (weather + public holidays) and restocking rules. */
export function StoreForm({
  initial,
  submitLabel,
  onSubmit,
}: {
  initial: StoreSettings;
  submitLabel: string;
  onSubmit: (s: StoreSettings) => Promise<unknown>;
}) {
  const [s, setS] = useState(initial);
  const [city, setCity] = useState("");
  const [places, setPlaces] = useState<Place[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const inv = (k: keyof StoreSettings["inventory"], v: string) => setS({ ...s, inventory: { ...s.inventory, [k]: Number(v) } });

  const find = async () => {
    setError(null);
    const r = await fetch(`https://geocoding-api.open-meteo.com/v1/search?count=5&name=${encodeURIComponent(city)}`);
    const found: Place[] = (await r.json()).results ?? [];
    setPlaces(found);
    if (!found.length) setError(new Error(`No place called "${city}" found`));
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await onSubmit(s);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  const located = Number.isFinite(s.lat) && Number.isFinite(s.lon) && s.country.length === 2;
  return (
    <form onSubmit={submit} className="grid gap-5">
      <Field label="Store name" required maxLength={100} value={s.name} onChange={(e) => setS({ ...s, name: e.target.value })} />

      <fieldset className="grid gap-2">
        <legend className="eyebrow mb-1">Location: local weather and public holidays</legend>
        <div className="flex gap-2">
          <input
            className={inputClass}
            placeholder="Town or city"
            value={city}
            onChange={(e) => setCity(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                if (city.trim()) find();
              }
            }}
          />
          <Button type="button" variant="outline" size="lg" onClick={find} disabled={!city.trim()}>
            <Search aria-hidden /> Find
          </Button>
        </div>
        {places.length > 0 && (
          <ul className="grid gap-1">
            {places.map((p) => (
              <li key={`${p.latitude},${p.longitude}`}>
                <button
                  type="button"
                  className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-sm hover:bg-muted"
                  onClick={() => {
                    setS({ ...s, country: p.country_code, lat: p.latitude, lon: p.longitude });
                    setPlaces([]);
                    setCity(`${p.name}, ${p.country}`);
                  }}
                >
                  <MapPin className="size-3.5 text-muted-foreground" aria-hidden />
                  {p.name}
                  <span className="text-muted-foreground">
                    {[p.admin1, p.country].filter(Boolean).join(", ")}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
        <p className={cn("num text-xs", located ? "text-muted-foreground" : "text-deliberate-ink")}>
          {located ? `${s.country}, ${s.lat.toFixed(2)}, ${s.lon.toFixed(2)}` : "Find your town to set the location."}
        </p>
      </fieldset>

      <fieldset className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <legend className="eyebrow mb-1">Restocking rules</legend>
        <Field label="Lead time (days)" type="number" min={1} max={60} required value={s.inventory.lead_time} onChange={(e) => inv("lead_time", e.target.value)} hint="Order to delivery" />
        <Field label="Order covers (days)" type="number" min={1} max={60} required value={s.inventory.coverage} onChange={(e) => inv("coverage", e.target.value)} hint="Days between orders" />
        <label className="grid min-w-0 gap-1">
          <span className="eyebrow">Service level</span>
          <select className={inputClass} value={s.inventory.service_level} onChange={(e) => inv("service_level", e.target.value)}>
            <option value={0.9}>90%</option>
            <option value={0.95}>95%</option>
            <option value={0.99}>99%</option>
          </select>
          <span className="text-xs text-muted-foreground">Chance of not running out</span>
        </label>
        <Field label="Case pack" type="number" min={1} max={1000} required value={s.inventory.case_pack} onChange={(e) => inv("case_pack", e.target.value)} hint="Orders round up to this" />
      </fieldset>

      <FormError error={error} />
      <div>
        <Button type="submit" size="lg" disabled={busy || !located || !s.name.trim()}>
          {busy ? "Saving…" : submitLabel}
        </Button>
      </div>
    </form>
  );
}
