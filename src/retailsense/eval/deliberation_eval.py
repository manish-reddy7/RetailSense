"""Phase 3b on validation (blueprint 7.2-7.7, 9.3): no deliberation, always-on, random trigger at
the selective rate, and selective (D >= tau); plus the cost-accuracy curve.

The LLM runs ONCE per sampled case (always-on). tau is not in any prompt, so every arm and every
point of the curve is a subset of the same cached deliberation results - a paired comparison.
Sample: stratified by D decile over all validation points (blueprint 7.7). Test split never read.

Run: python -m retailsense.eval.deliberation_eval --pilot 50
     python -m retailsense.eval.deliberation_eval            (full sample)
     python -m retailsense.eval.deliberation_eval --freeze   (full sample, then freeze tau)
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from retailsense.agents.schema import AgentForecast  # noqa: E402
from retailsense.config import ROOT, ExperimentCfg, load_experiment  # noqa: E402
from retailsense.eval.h1_analysis import load_points  # noqa: E402
from retailsense.eval.metrics import mae, rmse, wape  # noqa: E402
from retailsense.orchestrator.graph import build_graph, run_case  # noqa: E402
from retailsense.orchestrator.llm import LLM  # noqa: E402
from retailsense.store.db import connect, log_case, log_evaluation  # noqa: E402
from retailsense.store.tracking import start_run  # noqa: E402


def stratified_sample(pts: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    dec = pd.qcut(pts.D.rank(method="first"), 10, labels=False)
    return (
        pts.groupby(dec, group_keys=False)
        .apply(lambda g: g.sample(min(n // 10, len(g)), random_state=seed))
        .reset_index(drop=True)
    )


def build_cases(cfg: ExperimentCfg, sample: pd.DataFrame, tag: str = "phase2") -> list[dict]:
    d = cfg.deliberation
    af = pd.read_parquet(cfg.path("processed") / f"{tag}_agent_forecasts.parquet")
    af = af[af.h == 1].assign(d=lambda x: x.origin + 1).set_index(["item_id", "d"]).sort_index()
    feats = pd.read_parquet(
        cfg.path("processed") / "features.parquet", columns=["item_id", "d", "h", "date", "sales"]
    )
    h1 = feats[feats.h == 1].astype({"item_id": str}).set_index(["item_id", "d"]).sort_index()
    label = cfg.dataset
    cases = []
    for item, tgt in zip(sample.item_id, sample.d, strict=True):
        origin = tgt - 1
        rows = af.loc[(item, tgt)]
        hist = h1.loc[item].loc[origin - d.history_days + 1 : origin, "sales"]
        agents = [
            AgentForecast(
                agent_id=r.agent,
                item_id=item,
                store_id=cfg.data.store,
                prediction_time=h1.loc[(item, origin), "date"].date(),
                horizon_days=[1],
                forecast=[r.forecast],
                lower=[r.lower],
                upper=[r.upper],
                confidence=r.confidence,
                key_factors=json.loads(r.key_factors),
                limitations=json.loads(r.limitations),
            )
            for r in rows.itertuples()
        ]
        cases.append(
            {
                "dataset": label,
                "item_id": item,
                "store_id": cfg.data.store,
                "origin": origin,
                "history_end_d": int(hist.index.max()),
                "target_date": h1.loc[(item, tgt), "date"].date(),
                "history": [float(x) for x in hist],
                "agents": agents,
            }
        )
    return cases


def run_all(
    cfg: ExperimentCfg, cases: list[dict], tau: float, trigger: bool, llm: LLM | None = None
) -> list[dict]:
    graph = build_graph(cfg, llm or LLM(cfg))
    out = [None] * len(cases)

    def one(i):
        out[i] = run_case(graph, cases[i], tau, trigger=trigger)

    with ThreadPoolExecutor(cfg.deliberation.workers) as pool:
        for k, _ in enumerate(pool.map(one, range(len(cases))), 1):
            if trigger and k % 50 == 0:
                done = [s for s in out if s is not None]
                print(
                    f"  {k}/{len(cases)} cases, LLM calls so far "
                    f"{sum(s['llm_calls'] for s in done)}",
                    flush=True,
                )
    return out


def arm_metrics(name, y, yhat, trig, calls, lat) -> dict:
    return {
        "system": name,
        "WAPE": wape(y, yhat),
        "MAE": mae(y, yhat),
        "RMSE": rmse(y, yhat),
        "trigger_rate": float(np.mean(trig)),
        "llm_calls": float(np.sum(calls)),
        "llm_calls_per_1000": 1000 * float(np.mean(calls)),
        "mean_latency": float(np.mean(lat)),
    }


def evaluate(
    cfg: ExperimentCfg, sample: pd.DataFrame, deli: list[dict], fast: list[dict], tau: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    y = sample.y.to_numpy()
    cons = np.array([s["consensus"] for s in fast])
    final = np.array([s["final"] for s in deli])
    D = sample.D.to_numpy()
    calls = np.array([s["llm_attempts"] for s in deli], float)  # cost of deliberating each case
    lat_fast = np.array([sum(s["timings"].values()) for s in fast])
    lat_deli = np.array([sum(s["timings"].values()) for s in deli])
    rng = np.random.default_rng(cfg.seed)

    def pick(trig):
        return (
            np.where(trig, final, cons),
            np.where(trig, calls, 0),
            np.where(trig, lat_deli, lat_fast),
        )

    sel = D >= tau
    rows = []
    for name, trig in [
        ("no_deliberation", np.zeros_like(sel)),
        ("always_on", np.ones_like(sel)),
        ("selective", sel),
    ]:
        yh, c, lat = pick(trig)
        rows.append(arm_metrics(name, y, yh, trig, c, lat))
    k = int(sel.sum())
    draws = []
    for _ in range(cfg.deliberation.random_trigger_draws):
        trig = np.zeros(len(y), bool)
        trig[rng.choice(len(y), k, replace=False)] = True
        yh, c, lat = pick(trig)
        draws.append(arm_metrics("random", y, yh, trig, c, lat))
    rnd = pd.DataFrame(draws).drop(columns="system").mean().to_dict()
    rnd["WAPE_sd"] = float(pd.DataFrame(draws).WAPE.std())
    rows.append({"system": "random_same_rate", **rnd})
    arms = pd.DataFrame(rows).set_index("system")

    rates = np.round(np.arange(0, 1.0001, 0.05), 2)
    order = np.argsort(-D)
    curve = []
    for r in rates:
        n = int(round(r * len(y)))
        trig = np.zeros(len(y), bool)
        trig[order[:n]] = True
        rand = []
        for _ in range(cfg.deliberation.random_trigger_draws):
            t2 = np.zeros(len(y), bool)
            t2[rng.choice(len(y), n, replace=False)] = True
            rand.append(wape(y, np.where(t2, final, cons)))
        curve.append(
            {
                "trigger_rate": r,
                "wape_selective": wape(y, np.where(trig, final, cons)),
                "wape_random": float(np.mean(rand)),
                "wape_random_sd": float(np.std(rand)),
                "llm_calls_per_1000": 1000 * calls[trig].sum() / len(y),
            }
        )
    curve = pd.DataFrame(curve).set_index("trigger_rate")

    ok = np.array([s["deliberated"] for s in deli])
    before, after = np.abs(y - cons), np.abs(y - final)
    t = sel & ok
    revs = [r for s in deli for r in (s.get("revisions") or [])]
    h2 = {
        "triggered_cases": int(t.sum()),
        "improved_share": float(np.mean(after[t] < before[t] - 1e-9)),
        "worsened_share": float(np.mean(after[t] > before[t] + 1e-9)),
        "mean_abs_err_before": float(before[t].mean()),
        "mean_abs_err_after": float(after[t].mean()),
        "wilcoxon_p": float(wilcoxon(before[t], after[t]).pvalue) if t.sum() > 10 else np.nan,
        "fallback_rate": float(1 - ok.mean()),
        "hold_rate": float(np.mean([r["held"] for r in revs])) if revs else np.nan,
        **{
            f"hold_rate_{a}": float(np.mean([r["held"] for r in revs if r["agent_id"] == a]))
            for a in sorted({r["agent_id"] for r in revs})
        },
        "clip_rate": float(np.mean([r["clipped"] for r in revs])) if revs else np.nan,
        "mean_latency_fast_ms": float(lat_fast.mean()),
        "mean_latency_deliberation_ms": float(lat_deli.mean()),
    }
    return arms, curve, h2


def figure(curve: pd.DataFrame, tau_rate: float, path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve.index, curve.wape_selective, marker="o", label="selective (highest D first)")
    ax.plot(curve.index, curve.wape_random, ls="--", label="random trigger (mean)")
    ax.fill_between(
        curve.index,
        curve.wape_random - curve.wape_random_sd,
        curve.wape_random + curve.wape_random_sd,
        alpha=0.2,
    )
    ax.axvline(tau_rate, color="grey", lw=0.8, ls=":", label=f"tau (rate {tau_rate:.0%})")
    ax.set(xlabel="trigger rate (share of forecasts deliberated)", ylabel="WAPE (h=1)", title=title)
    per = curve.llm_calls_per_1000.iloc[-1]  # calls per 1,000 forecasts at 100 % trigger
    sec = ax.secondary_xaxis("top", functions=(lambda r: r * per, lambda c: c / per))
    sec.set_xlabel("LLM calls per 1,000 forecasts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def freeze(tau: float) -> None:
    """Write tau into the active config file. After this, tau and prompt_version never change."""
    import os  # noqa: PLC0415

    p = ROOT / os.environ.get("RETAILSENSE_CONFIG", "configs/experiment.yaml")
    s = p.read_text(encoding="utf-8")
    s, n = re.subn(
        r"  tau: null .*",
        f"  tau: {tau:.6f}                # FROZEN {date.today()} "
        "(q-tau_quantile of validation D); never change",
        s,
    )
    if n != 1:
        raise RuntimeError("tau is already frozen or missing in config")
    s = s.replace(
        "  prompt_version: v1\n", f"  prompt_version: v1        # FROZEN {date.today()}\n"
    )
    p.write_text(s, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    cfg = load_experiment()
    d = cfg.deliberation
    pts = load_points(cfg)
    tau = d.tau if d.tau is not None else float(pts.D.quantile(cfg.disagreement.tau_quantile))
    sample = stratified_sample(pts, d.sample_size, cfg.seed)
    if args.pilot:
        sample = sample.sample(args.pilot, random_state=cfg.seed).reset_index(drop=True)
    cases = build_cases(cfg, sample)
    print(
        f"{len(cases)} validation cases ({'pilot' if args.pilot else 'stratified sample'}), "
        f"tau = {tau:.4f}, model {d.model}, prompt {d.prompt_version}",
        flush=True,
    )
    fast = run_all(cfg, cases, tau, trigger=False)
    deli = run_all(cfg, cases, tau, trigger=True)

    tag = "phase3b_pilot" if args.pilot else "phase3b"
    con = connect(cfg.path("db"))
    for i, (s, y) in enumerate(zip(deli, sample.y, strict=True)):
        log_case(con, f"{tag}-{i}", tag, "always_on", s, float(y), d.prompt_version, d.model)
    con.commit()

    tokens_in = sum(s["tokens_in"] for s in deli)
    tokens_out = sum(s["tokens_out"] for s in deli)
    arms, curve, h2 = evaluate(cfg, sample, deli, fast, tau)
    log_evaluation(con, tag, [{"system": k, **v} for k, v in arms.to_dict("index").items()])
    con.commit()

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"\ntokens this run (0 if cached): in {tokens_in:,}, out {tokens_out:,}")
        print(
            f"per deliberated case: {tokens_in / len(cases):,.0f} in, "
            f"{tokens_out / len(cases):,.0f} out"
        )
        print("\n== arms ==\n", arms.round(4))
        print("\n== H2 / mechanism ==\n", pd.Series(h2).round(4).to_string())
        if args.pilot:
            shown = sorted(range(len(deli)), key=lambda i: -deli[i]["D"])[:3]
            for i in shown:
                s = deli[i]
                print(
                    f"\n==== critic report: {s['case']['item_id']} for "
                    f"{s['case']['target_date']}, D = {s['D']:.3f}, actual = {sample.y[i]:.0f}"
                )
                print(
                    json.dumps(s["critic"].model_dump(), indent=2) if s["critic"] else s["fallback"]
                )
                for r in s.get("revisions") or []:
                    print(
                        f"  {r['agent_id']:10s} {r['initial']:7.2f} -> {r['y']:7.2f}  "
                        f"m={r['m']:.2f}{' HELD' if r['held'] else ''}  {r['rationale']}"
                    )
                print(f"  consensus {s['consensus']:.2f} -> final {s['final']:.2f}")
            return

    out = ROOT / "reports" / cfg.dataset
    figs = ROOT / "reports" / "figures" / cfg.dataset
    arms.to_csv(out / "phase3b_arms.csv")
    curve.to_csv(out / "phase3b_cost_accuracy.csv")
    pd.Series(h2).to_csv(out / "phase3b_mechanism.csv")
    label = cfg.dataset
    figure(
        curve,
        float((sample.D >= tau).mean()),
        figs / "cost_accuracy.png",
        f"Cost-accuracy on validation sample - {label}",
    )
    print("\n== cost-accuracy curve ==\n", curve.round(4))
    with start_run(cfg, "phase3b", "deliberation_validation"):
        mlflow.log_metrics(
            {
                "tau": tau,
                "sample_size": len(cases),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                **{
                    f"{a}.{m}": float(v)
                    for a, r in arms.iterrows()
                    for m, v in r.items()
                    if pd.notna(v)
                },
                **{k: float(v) for k, v in h2.items() if pd.notna(v)},
            }
        )
        mlflow.set_tags({"model": d.model, "prompt_version": d.prompt_version})
        mlflow.log_artifact(str(figs / "cost_accuracy.png"))
        mlflow.log_text(curve.round(4).to_string(), "cost_accuracy.txt")
    if args.freeze:
        freeze(tau)
        print(f"\nFROZEN tau = {tau:.6f} and prompt_version {d.prompt_version} in config")


if __name__ == "__main__":
    main()
