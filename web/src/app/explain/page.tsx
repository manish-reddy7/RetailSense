"use client";

import { CalendarDays, CloudSun, History, Info, Search } from "lucide-react";
import { Empty, Loading, PageHeader, Problem, Section } from "@/components/common";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { fmt } from "@/lib/data";
import { useSelection } from "@/lib/selection";
import { cn } from "@/lib/utils";

const ICON = { historical: History, calendar: CalendarDays, weather: CloudSun, trend: Search } as const;
const STRENGTH = {
  strong: "border-good/40 bg-good/10 text-good",
  moderate: "border-brand/40 bg-brand/10 text-brand",
  weak: "border-border bg-muted text-muted-foreground",
} as const;

function Chip({ children, className }: { children: React.ReactNode; className?: string }) {
  return <span className={cn("num rounded-md border border-border bg-muted/50 px-2 py-1 text-xs", className)}>{children}</span>;
}

export default function ExplainPage() {
  const s = useSelection();
  const c = s.current;
  if (s.error) return <Problem error={s.error} />;
  if (s.loading) return <Loading />;
  if (!c) return <Empty>No case for this selection.</Empty>;
  const inv = c.inventory;
  const pack = s.meta?.inventory.case_pack ?? 1;
  const ip = inv ? inv.on_hand + inv.on_order : 0;

  return (
    <>
      <PageHeader
        eyebrow={`Explain · ${s.item} · ${c.date}`}
        title={
          <>
            Why <span className="num text-brand">{fmt(c.final)}</span> units?
          </>
        }
      >
        Each agent sees one kind of evidence. When they disagree, the critic judges that evidence and each agent may
        revise once, inside its own band. Code computes the final number.
      </PageHeader>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {c.agents.map((a, i) => {
          const Icon = ICON[a.id as keyof typeof ICON] ?? History;
          return (
            <div key={a.id} data-index={i} className="grid content-start gap-3 rounded-xl border border-border bg-card p-4">
              <div className="flex items-center gap-2">
                <span className="grid size-8 place-items-center rounded-lg bg-brand/10 text-brand">
                  <Icon className="size-4" aria-hidden />
                </span>
                <p className="font-medium capitalize">{a.id}</p>
                <p className="num ml-auto text-lg">{fmt(a.forecast)}</p>
              </div>
              <p className="num text-xs text-muted-foreground">
                band {fmt(a.lower)}–{fmt(a.upper)} · confidence {fmt(a.confidence)}
              </p>
              <div className="flex flex-wrap gap-1.5">
                {a.key_factors.map((k) => (
                  <Chip key={k}>{k.replace(/(\d+\.\d{3})\d+/g, "$1")}</Chip>
                ))}
              </div>
              {a.limitations.length > 0 && <p className="text-xs text-deliberate">Limitation: {a.limitations.join("; ")}</p>}
            </div>
          );
        })}
      </div>

      <Section eyebrow="Critic" title="Conflict report" description="The critic judges evidence only; its output schema has no numeric field.">
        {c.critic ? (
          <div className="grid gap-5">
            {!c.triggered && (
              <Alert>
                <Info aria-hidden />
                <AlertTitle>Not used by RetailSense for this case</AlertTitle>
                <AlertDescription>
                  D was below τ, so RetailSense took the fast path. This report comes from the always-on comparison arm.
                </AlertDescription>
              </Alert>
            )}
            <blockquote className="border-l-2 border-deliberate pl-4 font-display text-2xl leading-snug">
              {c.critic.primary_conflict}
            </blockquote>
            <div className="grid gap-2 md:grid-cols-2">
              {c.critic.evidence_assessment.map((e) => (
                <div key={e.agent} className="rounded-xl border border-border p-3">
                  <div className="mb-1 flex items-center gap-2">
                    <span className="text-sm font-medium capitalize">{e.agent}</span>
                    <span className={cn("rounded-full border px-2 py-0.5 text-[11px] font-medium capitalize", STRENGTH[e.strength])}>
                      {e.strength} evidence
                    </span>
                  </div>
                  <p className="text-sm text-muted-foreground">{e.reason}</p>
                </div>
              ))}
            </div>
            {c.critic.questions_for_agents.length > 0 && (
              <div className="grid gap-2">
                <p className="eyebrow">Questions to the agents</p>
                {c.critic.questions_for_agents.map((q) => (
                  <p key={q.agent} className="text-sm">
                    <span className="mr-2 rounded bg-muted px-1.5 py-0.5 text-xs capitalize">{q.agent}</span>
                    {q.question}
                  </p>
                ))}
              </div>
            )}
          </div>
        ) : (
          <Empty>No critic report (fast path{c.fallback ? `, fallback: ${c.fallback}` : ""}).</Empty>
        )}
      </Section>

      <Section
        eyebrow="Revision"
        title="Initial vs revised"
        description={`Consensus ${fmt(c.consensus)} → deliberated ${fmt(c.deliberated)}. Code clips m to [0.25, 2] and each value to the agent's own band.`}
      >
        <div className="overflow-x-auto rounded-lg border border-border">
          <Table>
            <TableHeader className="bg-muted/50">
              <TableRow>
                <TableHead className="eyebrow">Agent</TableHead>
                <TableHead className="eyebrow text-right">Initial</TableHead>
                <TableHead className="eyebrow text-right">Revised</TableHead>
                <TableHead className="eyebrow text-right">m</TableHead>
                <TableHead className="eyebrow">Rationale</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {c.agents.map((a) => {
                const held = a.revised !== null && Math.abs(a.revised - a.forecast) < 1e-9 && a.m === 1;
                return (
                  <TableRow key={a.id}>
                    <TableCell className="capitalize">
                      <span className="flex items-center gap-2">
                        {a.id}
                        {a.revised !== null && (
                          <span className={cn("rounded-full px-1.5 text-[10px] font-medium", held ? "bg-muted text-muted-foreground" : "bg-deliberate/15 text-deliberate")}>
                            {held ? "held" : "moved"}
                          </span>
                        )}
                      </span>
                    </TableCell>
                    <TableCell className="num text-right">{fmt(a.forecast)}</TableCell>
                    <TableCell className="num text-right">{fmt(a.revised)}</TableCell>
                    <TableCell className="num text-right">{fmt(a.m)}</TableCell>
                    <TableCell className="min-w-80 text-sm whitespace-normal text-muted-foreground">{a.rationale ?? "–"}</TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      </Section>

      <Section eyebrow="Policy" title="Why this action">
        {inv ? (
          <div className="grid gap-4">
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <Chip>IP = {fmt(inv.on_hand, 0)} + {fmt(inv.on_order, 0)} = {fmt(ip, 0)}</Chip>
              <span className="text-muted-foreground">{ip <= inv.reorder_point ? "≤" : ">"}</span>
              <Chip>s = {fmt(inv.reorder_point, 1)}</Chip>
              <span className="text-muted-foreground">·</span>
              <Chip>S = {fmt(inv.order_up_to, 1)}</Chip>
              <span className="text-muted-foreground">·</span>
              <Chip>safety stock {fmt(inv.safety_stock, 1)}</Chip>
            </div>
            <p className="text-sm">
              {inv.action === "RESTOCK" ? (
                <>
                  The inventory position is at or below the reorder point, so the policy orders S − IP rounded up to case packs of{" "}
                  {pack}: <b className="num">{fmt(inv.order_qty, 0)} units</b>.
                </>
              ) : (
                <>The inventory position is above the reorder point, so no order is placed ({inv.action}).</>
              )}
            </p>
          </div>
        ) : (
          <Empty>No inventory simulation is stored for this run.</Empty>
        )}
      </Section>
    </>
  );
}
