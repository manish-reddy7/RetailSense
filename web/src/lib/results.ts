import type { Row, Tables } from "@/lib/data";

// Reads the final-run tables (reports/<dataset>/final/) into one summary. Verdicts follow the
// paper's rules: a hypothesis is "supported" only when its pre-registered test is significant.

export type Verdict = "supported" | "weak" | "not supported" | "no data";

export type Hypothesis = { id: string; question: string; verdict: Verdict; evidence: string };

export type Summary = {
  points: number;
  triggerRate: number;
  best: { system: string; wape: number };
  rs: { wape: number; lo: number; hi: number };
  cheapest: { system: string; cost: number };
  rsCost: number;
  callsSaved: number; // share of always-on LLM calls avoided
  rho: number;
  auroc: number;
  holdRate: number;
  hyps: Hypothesis[];
};

const n = (v: unknown) => (typeof v === "number" ? v : Number(v));
const pct = (x: number, d = 0) => `${(100 * x).toFixed(d)}%`;
const p = (x: number) => (x < 0.001 ? "p < 0.001" : `p = ${x.toFixed(x < 0.01 ? 3 : 2)}`);
export const nice = (s: unknown) => String(s).replaceAll("_", " ");

function ci(v: unknown): [number, number] {
  const m = String(v).match(/-?[\d.]+/g) ?? [];
  return [Number(m[0]), Number(m[1])];
}

export function summarise(t: Tables): Summary | undefined {
  const f = t.final;
  if (!f?.table_forecast_metrics || !f.table_mechanism) return undefined;
  const mech = Object.fromEntries(f.table_mechanism.map((r) => [String(r["Unnamed: 0"]), r["0"]]));
  const fm = f.table_forecast_metrics;
  const sys = (rows: Row[] | undefined, key: string, name: string) => rows?.find((r) => r[key] === name);
  const rs = sys(fm, "system", "retailsense")!;
  const best = fm.reduce((a, b) => (n(a.h1_WAPE) <= n(b.h1_WAPE) ? a : b));
  const inv = f.table_inventory ?? [];
  const cheap = inv.reduce((a, b) => (n(a.total_cost) <= n(b.total_cost) ? a : b), inv[0]);
  const rsCost = n(sys(inv, "system", "retailsense")?.total_cost);
  const calls = (s: string) => n(sys(f.table_latency_llm, "system", s)?.llm_calls_per_1000);
  const callsSaved = 1 - calls("retailsense") / calls("always_on");
  const sig = (s: string) => sys(f.table_significance, "vs", s);
  const vsAlways = sig("always_on");
  const vsRandom = sig("random_trigger");

  const rho = n(mech.H1_spearman);
  const [rhoLo, rhoHi] = ci(mech.H1_spearman_ci);
  const auroc = n(mech.H1_auroc_worst);
  const improved = n(mech.H2_improved_share);
  const worsened = n(mech.H2_worsened_share);
  const h2p = n(mech.H2_wilcoxon_p);
  const h4p = n(vsRandom?.DM_p);
  const h4d = n(vsRandom?.WAPE_diff);
  const h3p = n(vsAlways?.DM_p);

  const hyps: Hypothesis[] = [
    {
      id: "H1",
      question: "Does disagreement D predict forecast error?",
      verdict: rhoLo > 0 ? (rho < 0.3 ? "weak" : "supported") : "not supported",
      evidence: `Spearman ρ ${rho.toFixed(3)} [${rhoLo.toFixed(3)}, ${rhoHi.toFixed(3)}], AUROC ${auroc.toFixed(3)} (0.5 = no signal)`,
    },
    {
      id: "H2",
      question: "Does deliberation improve the triggered forecasts?",
      verdict: h2p < 0.05 && improved > worsened ? "supported" : "not supported",
      evidence: `${n(mech.H2_triggered_deliberated)} triggered: ${pct(improved, 1)} improved, ${pct(worsened, 1)} worsened, Wilcoxon ${p(h2p)}`,
    },
    {
      id: "H3",
      question: "Does selective deliberation match always-on at lower LLM cost?",
      verdict: callsSaved > 0 && h3p >= 0.05 ? "supported" : "not supported",
      evidence: `${pct(callsSaved)} fewer LLM calls than always-on; WAPE difference not significant (DM ${p(h3p)})`,
    },
    {
      id: "H4",
      question: "Is disagreement a better trigger than chance?",
      verdict: h4p < 0.05 && h4d < 0 ? "supported" : "not supported",
      evidence: `Selective − random WAPE ${h4d >= 0 ? "+" : ""}${h4d.toFixed(4)}, DM ${p(h4p)}`,
    },
    {
      id: "H5",
      question: "Does RetailSense give the lowest inventory cost?",
      verdict: cheap?.system === "retailsense" ? "supported" : "not supported",
      evidence: `Cheapest: ${nice(cheap?.system)} $${Math.round(n(cheap?.total_cost)).toLocaleString()}; RetailSense $${Math.round(rsCost).toLocaleString()}`,
    },
  ];

  return {
    points: n(mech.points),
    triggerRate: n(mech.trigger_rate),
    best: { system: String(best.system), wape: n(best.h1_WAPE) },
    rs: { wape: n(rs.h1_WAPE), lo: n(rs.h1_WAPE_ci_lo), hi: n(rs.h1_WAPE_ci_hi) },
    cheapest: { system: String(cheap?.system), cost: n(cheap?.total_cost) },
    rsCost,
    callsSaved,
    rho,
    auroc,
    holdRate: n(mech.hold_rate),
    hyps,
  };
}
