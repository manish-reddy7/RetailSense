"""Phase 5a (blueprint 9): the single test run of all 9 systems, rolling-origin backtests,
ablations (9.5), statistics (9.4) and every table/figure of 9.6.

--final    touches the test split. Refuses unless tau is frozen in config, and refuses a second
           run under a different config (reruns with the same frozen config reproduce from cache).
--dry-run  the same code on the VALIDATION window with a stub LLM where every agent holds, so
           systems 7-9 must equal system 6 exactly. Separate cache, DB and report folders.
--offline  LLM answers from the disk cache only (reproducibility check).

Models are fit on train only (blueprint 5: "Train: fit all models"); test origins continue from
validation with the same models. Backtest windows refit every model on the days before them.

Run: python -m retailsense.eval.run_experiment --final
"""

import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from retailsense.config import ROOT, load_experiment, load_inventory  # noqa: E402
from retailsense.data.load import load_calendar, load_prices  # noqa: E402
from retailsense.eval.deliberation_eval import build_cases, run_all  # noqa: E402
from retailsense.eval.h1_analysis import bootstrap_ci, load_points, signal_stats  # noqa: E402
from retailsense.eval.inventory_eval import run as simulate  # noqa: E402
from retailsense.eval.inventory_eval import system_forecasts  # noqa: E402
from retailsense.eval.metrics import all_metrics, wape  # noqa: E402
from retailsense.eval.run_experiment import run_validation  # noqa: E402
from retailsense.eval.stats import diebold_mariano, paired_bootstrap, wilcoxon_paired  # noqa: E402
from retailsense.orchestrator.disagreement import score_table  # noqa: E402
from retailsense.orchestrator.llm import LLM, hold_transport, offline_transport  # noqa: E402
from retailsense.store.db import connect, log_case, log_evaluation  # noqa: E402
from retailsense.store.tracking import cfg_path, file_hash, start_run  # noqa: E402

SYSTEMS = {  # blueprint 9.1
    1: "seasonal_naive",
    2: "ets",
    3: "lgbm_all_features",
    4: "historical",
    5: "multi_agent_mean",
    6: "multi_agent_confidence",
    7: "random_trigger",
    8: "always_on",
    9: "retailsense",
}
OK = ["item_id", "origin"]


# ---------------------------------------------------------------- deliberation -> systems 7-9
def deliberate(cfg, llm, pts: pd.DataFrame, tag: str, tau: float):
    """Fast path and forced deliberation for each point. Returns a per-case table, the revision
    rows and the raw graph states (for logging and the case study)."""
    cases = build_cases(cfg, pts, tag)
    fast = run_all(cfg, cases, tau, trigger=False, llm=llm)
    deli = run_all(cfg, cases, tau, trigger=True, llm=llm)
    res = pd.DataFrame(
        {
            "item_id": pts.item_id.to_numpy(),
            "origin": pts.d.to_numpy() - 1,
            "consensus": [s["consensus"] for s in fast],
            "final": [s["final"] for s in deli],
            "deliberated": [s["deliberated"] for s in deli],
            "fallback": [s.get("fallback") for s in deli],
            "lat_fast": [sum(s["timings"].values()) for s in fast],
            "lat_deli": [sum(s["timings"].values()) for s in deli],
            "attempts": [s["llm_attempts"] for s in deli],
            "tokens_in": [s["tokens_in"] for s in deli],
            "tokens_out": [s["tokens_out"] for s in deli],
        }
    )
    if not np.allclose(res.consensus, pts.consensus):
        raise RuntimeError("graph consensus differs from the scored consensus")
    rev = pd.DataFrame(
        [
            {"item_id": c["item_id"], "origin": c["origin"], "agent": r["agent_id"], **r}
            for c, s in zip(cases, deli, strict=True)
            for r in s.get("revisions") or []
        ],
        columns=["item_id", "origin", "agent", "m", "y", "initial", "w", "held"],
    )
    return res, rev[["item_id", "origin", "agent", "m", "y", "initial", "held"]], deli


def deliberated_system(af: pd.DataFrame, rev: pd.DataFrame, keys, values: bool = True):
    """Confidence-weighted consensus, except on deliberated (item, origin) in `keys`: weights
    c*m, and at h=1 the revised (code-clipped) value; h2-7 use the new weights on the original
    forecasts. Bands use the same weights (blueprint 7.6). values=False: weights-only ablation."""
    r = rev.merge(pd.DataFrame(list(keys), columns=OK), on=OK) if len(keys) else rev.iloc[:0]
    a = af.merge(r[[*OK, "agent", "m", "y"]], on=[*OK, "agent"], how="left")
    w = a.confidence * a.m.fillna(1.0)
    w = w / w.groupby([a.item_id, a.origin, a.h]).transform("sum")
    f = np.where(values & (a.h == 1) & a.y.notna(), a.y, a.forecast)
    out = a[[*OK, "h"]].assign(yhat=w * f, lower=w * a.lower, upper=w * a.upper)
    return out.groupby([*OK, "h"], as_index=False)[["yhat", "lower", "upper"]].sum()


def keys_of(res: pd.DataFrame, mask) -> set:
    return set(zip(res.item_id[mask], res.origin[mask], strict=True))


# ---------------------------------------------------------------- scoring
def scored(fc: pd.DataFrame, actual: pd.Series, score_from: int, last: int) -> pd.DataFrame:
    """Forecasts from window origins with targets inside the window, joined to actuals."""
    e = fc.assign(d=fc.origin + fc.h)
    e = e[(e.origin >= score_from) & (e.d <= last)]
    e = e.assign(y=actual.reindex(pd.MultiIndex.from_arrays([e.item_id, e.d])).to_numpy())
    if e.y.isna().any():
        raise RuntimeError("missing actuals in the scored window")
    return e.sort_values([*OK, "h"]).reset_index(drop=True)


def forecast_table(evals: dict, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for num, name in SYSTEMS.items():
        if name not in evals:
            continue
        e = evals[name]
        h1 = e[e.h == 1]
        err = (h1.y - h1.yhat).abs()
        ci = paired_bootstrap(h1.item_id, err, 0 * err, h1.y, n_boot, seed)
        rows.append(
            {
                "#": num,
                "system": name,
                **{f"h1_{k}": v for k, v in all_metrics(h1.y, h1.yhat, h1.lower, h1.upper).items()},
                "h1_WAPE_ci_lo": ci["ci_lo"],
                "h1_WAPE_ci_hi": ci["ci_hi"],
                **{f"h1-7_{k}": v for k, v in all_metrics(e.y, e.yhat, e.lower, e.upper).items()},
            }
        )
    return pd.DataFrame(rows).set_index("system")


def compare(evals: dict, logs: dict, n_boot: int, seed: int, ref: str) -> pd.DataFrame:
    """RetailSense (ref) vs every other system: DM, paired bootstrap, Wilcoxon (blueprint 9.4).
    Negative differences mean `ref` is better."""
    a = evals[ref][lambda x: x.h == 1]
    cost = {
        k: v.groupby("item_id")[["holding_cost", "ordering_cost", "lost_sales_cost"]]
        .sum()
        .sum(axis=1)
        for k, v in logs.items()
    }
    rows = []
    for name, e in evals.items():
        if name == ref:
            continue
        b = e[e.h == 1]
        if not (a[[*OK]].to_numpy() == b[[*OK]].to_numpy()).all():
            raise RuntimeError(f"{name} is not aligned with {ref}")
        ea, eb = (a.y - a.yhat).abs().to_numpy(), (b.y - b.yhat).abs().to_numpy()
        daily = pd.DataFrame({"d": a.d.to_numpy(), "a": ea, "b": eb}).groupby("d").mean()
        dm = diebold_mariano(daily.a, daily.b)
        per_item = [
            diebold_mariano(g.a, g.b)
            for _, g in pd.DataFrame({"i": a.item_id.to_numpy(), "a": ea, "b": eb}).groupby("i")
        ]
        wa = (
            pd.Series(ea).groupby(a.item_id.to_numpy()).sum()
            / a.y.groupby(a.item_id.to_numpy()).sum().clip(lower=1).to_numpy()
        )
        wb = (
            pd.Series(eb).groupby(a.item_id.to_numpy()).sum()
            / a.y.groupby(a.item_id.to_numpy()).sum().clip(lower=1).to_numpy()
        )
        bw = paired_bootstrap(a.item_id, ea, eb, a.y, n_boot, seed)
        ca, cb = cost[ref], cost[name].reindex(cost[ref].index)
        bc = paired_bootstrap(ca.index, ca, cb, np.ones(len(ca)), n_boot, seed)
        wx = wilcoxon_paired(wa, wb)
        rows.append(
            {
                "vs": name,
                "WAPE_diff": bw["diff"],
                "WAPE_diff_ci_lo": bw["ci_lo"],
                "WAPE_diff_ci_hi": bw["ci_hi"],
                "DM_stat": dm["dm"],
                "DM_p": dm["p"],
                "items_ref_better_p05": sum(r["p"] < 0.05 and r["dm"] < 0 for r in per_item),
                "items_ref_worse_p05": sum(r["p"] < 0.05 and r["dm"] > 0 for r in per_item),
                "wilcoxon_item_WAPE_p": wx["p"],
                "median_item_WAPE_diff": wx["median_diff"],
                "cost_diff_per_item": bc["diff"],
                "cost_diff_ci_lo": bc["ci_lo"],
                "cost_diff_ci_hi": bc["ci_hi"],
            }
        )
    return pd.DataFrame(rows).set_index("vs")


# ---------------------------------------------------------------- one window
class Window:
    """Everything that is repeated for the test window and each backtest window."""

    def __init__(self, cfg, inv, llm, tag, fit_end, score_from, last, tau, full, noise=None):
        self.cfg, self.inv, self.tag, self.tau = cfg, inv, tag, tau
        self.score_from, self.last = score_from, last
        proc = cfg.path("processed")
        if not (proc / f"{tag}_agent_forecasts.parquet").exists():
            print(f"[{tag}] fitting models on days <= {fit_end} ...", flush=True)
            run_validation(
                cfg, 2, fit_end, score_from, last, tag, log=False, weather_noise_sd=noise
            )
        self.af = pd.read_parquet(proc / f"{tag}_agent_forecasts.parquet").astype({"item_id": str})
        self.pts = load_points(cfg, tag, score_from + 1, last)
        feats = pd.read_parquet(proc / "features.parquet", columns=["item_id", "d", "h", "sales"])
        sales = feats[feats.h == 1].drop(columns="h").astype({"item_id": str})
        self.actual = sales.set_index(["item_id", "d"]).sales
        self.fcs = system_forecasts(cfg, inv, sales, tag, fit_end)

        rng = np.random.default_rng(cfg.seed)
        n = len(self.pts)
        self.sel = self.pts.D.to_numpy() >= tau
        self.rand = np.zeros(n, bool)
        self.rand[rng.choice(n, int(self.sel.sum()), replace=False)] = True
        need = np.ones(n, bool) if full else (self.sel | self.rand)
        print(f"[{tag}] deliberating {need.sum()} of {n} points (tau {tau:.4f}) ...", flush=True)
        res, self.rev, deli = deliberate(cfg, llm, self.pts[need].reset_index(drop=True), tag, tau)
        self.res = (
            self.pts[["item_id", "d"]]
            .assign(origin=lambda x: x.d - 1)
            .merge(res, on=OK, how="left")
        )
        self.states = dict(zip(zip(res.item_id, res.origin, strict=True), deli, strict=True))
        ok = self.res.deliberated.fillna(False).to_numpy(bool)
        arms = {"random_trigger": self.rand, "retailsense": self.sel}
        if full:
            arms["always_on"] = np.ones(n, bool)
        for name, trig in arms.items():
            self.fcs[name] = deliberated_system(self.af, self.rev, keys_of(self.res, trig & ok))
        self.ok = ok
        self._check_final(keys_of(self.res, self.sel & ok))
        self.evals = {k: scored(v, self.actual, score_from, last) for k, v in self.fcs.items()}

        days = list(range(score_from + 1, last + 1))
        self.days = days
        win = sales[sales.d.isin(days)]
        self.demand = {i: g.set_index("d").sales.to_dict() for i, g in win.groupby("item_id")}
        cal = load_calendar(cfg).assign(d=lambda c: c.d.str.removeprefix("d_").astype(int))
        pr = load_prices(cfg, sorted(self.demand)).merge(cal[["d", "wm_yr_wk"]], on="wm_yr_wk")
        pr = pr[pr.d.isin(days)]
        self.price = {i: g.set_index("d").sell_price.to_dict() for i, g in pr.groupby("item_id")}

    def _check_final(self, keys):
        """The system table's h=1 value must be exactly the graph's code-computed final."""
        f = deliberated_system(self.af, self.rev, keys)
        f = f[f.h == 1].merge(self.res[[*OK, "final", "deliberated"]], on=OK)
        f = f[f.set_index(OK).index.isin(list(keys))]
        if not np.allclose(f.yhat, f.final):
            raise RuntimeError("systems 7-9 h=1 forecast differs from the deliberated final")

    def systems(self):
        return {k: v for k, v in SYSTEMS.items() if v in self.fcs}

    def inventory(self, inv=None, systems=None, fcs=None):
        return simulate(
            self.cfg,
            inv or self.inv,
            fcs or self.fcs,
            self.demand,
            self.price,
            self.days,
            systems or self.systems(),
            self.cfg.data.test_end,
        )

    def h1_mechanism(self) -> dict:
        p, cfg = self.pts, self.cfg
        worst = (p.rel_err >= p.rel_err.quantile(1 - cfg.analysis.worst_share)).to_numpy()
        st = signal_stats(p, "D", worst)
        ci = bootstrap_ci(p, "D", cfg.analysis.worst_share, cfg.analysis.bootstrap, cfg.seed)
        r = self.res
        t = self.sel & self.ok
        before = (p.y - r.consensus).abs().to_numpy()
        after = (p.y - r.final).abs().to_numpy()
        return {
            "points": len(p),
            "H1_spearman": st["spearman"],
            "H1_spearman_ci": ci["spearman_ci"],
            "H1_auroc_worst": st["auroc_worst"],
            "H1_auroc_ci": ci["auroc_ci"],
            "trigger_rate": float(self.sel.mean()),
            "H2_triggered_deliberated": int(t.sum()),
            "H2_fallback_rate": float(1 - self.ok[self.sel].mean()) if self.sel.any() else np.nan,
            "H2_improved_share": float(np.mean(after[t] < before[t] - 1e-9)),
            "H2_worsened_share": float(np.mean(after[t] > before[t] + 1e-9)),
            "H2_mean_abs_err_before": float(before[t].mean()),
            "H2_mean_abs_err_after": float(after[t].mean()),
            "H2_wilcoxon_p": float(wilcoxon(before[t], after[t]).pvalue)
            if (np.abs(before[t] - after[t]) > 1e-9).sum() > 10
            else np.nan,
            "hold_rate": float(self.rev.held.mean()) if len(self.rev) else np.nan,
        }


# ---------------------------------------------------------------- test-only analyses
def cost_accuracy(w: Window, cfg) -> pd.DataFrame:
    """WAPE (h=1) and total inventory cost vs trigger rate; selective = highest D first."""
    y, cons, fin = w.pts.y.to_numpy(), w.res.consensus.to_numpy(), w.res.final.to_numpy()
    fin = np.where(w.ok, fin, cons)
    order = np.argsort(-w.pts.D.to_numpy())
    n, rng = len(y), np.random.default_rng(cfg.seed)
    step = cfg.final.curve_step
    rows = []
    for r in np.round(np.arange(0, 1 + 1e-9, step), 4):
        k = int(round(r * n))
        sel = np.zeros(n, bool)
        sel[order[:k]] = True
        rand_masks = []
        for _ in range(cfg.deliberation.random_trigger_draws):
            m = np.zeros(n, bool)
            m[rng.choice(n, k, replace=False)] = True
            rand_masks.append(m)
        fcs = {"sel": deliberated_system(w.af, w.rev, keys_of(w.res, sel & w.ok))}
        for j, m in enumerate(rand_masks[: cfg.final.curve_draws]):
            fcs[f"rand{j}"] = deliberated_system(w.af, w.rev, keys_of(w.res, m & w.ok))
        table, _ = w.inventory(systems=dict(enumerate(fcs)), fcs=fcs)
        rand_cost = table.drop(index="sel").total_cost
        rand_wape = [wape(y, np.where(m, fin, cons)) for m in rand_masks]
        rows.append(
            {
                "trigger_rate": r,
                "wape_selective": wape(y, np.where(sel, fin, cons)),
                "wape_random": float(np.mean(rand_wape)),
                "wape_random_sd": float(np.std(rand_wape)),
                "cost_selective": float(table.loc["sel", "total_cost"]),
                "cost_random": float(rand_cost.mean()),
                "cost_random_sd": float(rand_cost.std()),
                "llm_calls_per_1000": 1000 * float(w.res.attempts.to_numpy()[sel].sum()) / n,
            }
        )
        print(f"  curve rate {r:.2f} done", flush=True)
    return pd.DataFrame(rows).set_index("trigger_rate")


def ablations(w: Window, cfg, val: pd.DataFrame, noisy: pd.DataFrame | None) -> pd.DataFrame:
    """Blueprint 9.5 on the test window. Every arm reuses the one always-on deliberation per
    point (tau is not in any prompt), so the comparisons are paired. Rows that change the
    agents themselves (drop-one, weather noise) report the fast path and the trigger only:
    their deliberation would need the critic to see a different agent set."""
    p, r = w.pts, w.res
    y, cons = p.y.to_numpy(), r.consensus.to_numpy()
    fin = np.where(w.ok, r.final.to_numpy(), cons)
    q = cfg.disagreement.tau_quantile
    worst = (p.rel_err >= p.rel_err.quantile(1 - cfg.analysis.worst_share)).to_numpy()
    calls = r.attempts.to_numpy()
    rows = []

    def arm(ablation, variant, trig, yhat=None, signal=None, tau=None):
        yh = np.where(trig, fin, cons) if yhat is None else yhat
        rows.append(
            {
                "ablation": ablation,
                "variant": variant,
                "tau": tau,
                "trigger_rate": float(np.mean(trig)),
                "WAPE_h1": wape(y, yh),
                "llm_calls_per_1000": 1000 * float(calls[trig].sum()) / len(y),
                "auroc_worst": signal_stats(p.assign(s=signal), "s", worst)["auroc_worst"]
                if signal is not None
                else np.nan,
            }
        )

    arm("baseline", "no deliberation", np.zeros(len(y), bool))
    arm("tau", f"calibrated (q{q:.2f} of validation D)", w.sel, tau=w.tau, signal=p.D)
    ft = cfg.final.fixed_tau
    arm("tau", f"fixed tau = {ft}", p.D.to_numpy() >= ft, tau=ft, signal=p.D)
    for v in ("D", "D_range", "D_std"):
        t = float(val[v].quantile(q))
        arm("normalisation", v, p[v].to_numpy() >= t, tau=t, signal=p[v])
    ks = keys_of(r, w.sel & w.ok)
    wo = deliberated_system(w.af, w.rev, ks, values=False)
    wo = p[["item_id", "d"]].assign(origin=p.d - 1).merge(wo[wo.h == 1], on=OK)
    arm("revision", "weights + revised values", w.sel, tau=w.tau)
    arm("revision", "weights only", w.sel, yhat=wo.yhat.to_numpy(), tau=w.tau)

    agents = [c.removeprefix("f_") for c in p if c.startswith("f_")]
    for drop in [a for a in agents if a != "historical"]:
        keep = [a for a in agents if a != drop]

        def table(df, keep=keep):
            f = df[[f"f_{a}" for a in keep]].set_axis(keep, axis=1)
            c = df[[f"c_{a}" for a in keep]].set_axis(keep, axis=1)
            return score_table(f, c, cfg.disagreement.floor)

        tv, tt = table(val), table(p)
        t = float(tv.D.quantile(q))
        arm(
            "drop agent",
            f"without {drop} (fast path)",
            tt.D.to_numpy() >= t,
            yhat=tt.consensus.to_numpy(),
            signal=tt.D,
            tau=t,
        )
    if noisy is not None:
        n1 = noisy[noisy.h == 1].astype({"item_id": str}).assign(d=lambda x: x.origin + 1)
        f = p[[f"f_{a}" for a in agents]].set_axis(agents, axis=1).copy()
        f["weather"] = (
            p[["item_id", "d"]].merge(n1, on=["item_id", "d"], how="left").forecast.to_numpy()
        )
        c = p[[f"c_{a}" for a in agents]].set_axis(agents, axis=1)
        tt = score_table(f, c, cfg.disagreement.floor)
        trig = tt.D.to_numpy() >= w.tau
        arm(
            "weather noise",
            f"sd {cfg.final.weather_noise_sd} (fast path)",
            trig,
            yhat=tt.consensus.to_numpy(),
            signal=tt.D,
            tau=w.tau,
        )
        rows[-1]["trigger_jaccard_vs_clean"] = float(
            (trig & w.sel).sum() / max((trig | w.sel).sum(), 1)
        )
    return pd.DataFrame(rows)


def latency_table(w: Window) -> pd.DataFrame:
    r, n = w.res, len(w.res)
    rows = []
    for name, trig in (
        ("no_deliberation", np.zeros(n, bool)),
        ("retailsense", w.sel),
        ("random_trigger", w.rand),
        ("always_on", np.ones(n, bool)),
    ):
        lat = np.where(trig, r.lat_deli, r.lat_fast)
        rows.append(
            {
                "system": name,
                "trigger_rate": float(trig.mean()),
                "llm_calls_per_1000": 1000 * float(r.attempts[trig].sum()) / n,
                "tokens_in_per_1000": 1000 * float(r.tokens_in[trig].sum()) / n,
                "tokens_out_per_1000": 1000 * float(r.tokens_out[trig].sum()) / n,
                "mean_latency_ms": float(lat.mean()),
                "p95_latency_ms": float(np.percentile(lat, 95)),
            }
        )
    return pd.DataFrame(rows).set_index("system")


# ---------------------------------------------------------------- figures
def fig_deciles(p: pd.DataFrame, path, title):
    dec = pd.qcut(p.D.rank(method="first"), 10, labels=False) + 1
    g = p.groupby(dec).agg(rel=("rel_err", "mean"))
    g["wape"] = p.groupby(dec).apply(lambda x: wape(x.y, x.consensus))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(g.index, g.rel, color="tab:blue", alpha=0.7, label="mean relative |error|")
    ax.plot(g.index, g.wape, "o-", color="tab:red", label="WAPE")
    ax.set(xlabel="D decile (1 = lowest disagreement)", ylabel="error of consensus", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_curve(c: pd.DataFrame, rate: float, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, lab in ((axes[0], "wape", "WAPE (h=1)"), (axes[1], "cost", "total cost ($)")):
        ax.plot(c.index, c[f"{col}_selective"], "o-", label="selective (highest D first)")
        ax.plot(c.index, c[f"{col}_random"], "--", label="random trigger (mean)")
        ax.fill_between(
            c.index,
            c[f"{col}_random"] - c[f"{col}_random_sd"],
            c[f"{col}_random"] + c[f"{col}_random_sd"],
            alpha=0.2,
        )
        ax.axvline(rate, color="grey", ls=":", lw=0.8, label=f"frozen tau ({rate:.0%})")
        ax.set(xlabel="trigger rate", ylabel=lab)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_before_after(w: Window, path, title):
    t = w.sel & w.ok
    b = (w.pts.y - w.res.consensus).abs()[t]
    a = (w.pts.y - w.res.final).abs()[t]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(b, a, s=8, alpha=0.5)
    m = float(max(b.max(), a.max(), 1))
    ax.plot([0, m], [0, m], color="grey", lw=0.8)
    ax.set(
        xlabel="|error| before (consensus)",
        ylabel="|error| after deliberation",
        title=f"{title}\n{int((a < b - 1e-9).sum())} improved, {int((a > b + 1e-9).sum())} "
        f"worsened of {int(t.sum())}",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_case(w: Window, logs: dict, path, title) -> dict:
    """Case study chosen by a fixed rule - the highest-D triggered, deliberated point - never by
    outcome."""
    idx = int(np.argmax(np.where(w.sel & w.ok, w.pts.D, -np.inf)))
    p, r = w.pts.iloc[idx], w.res.iloc[idx]
    s = w.states[(r.item_id, r.origin)]
    inv = logs["retailsense"]
    day = inv[(inv.item_id == r.item_id) & (inv.d == r.origin)]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, a in enumerate(s["agents"]):
        rv = next((x for x in s.get("revisions") or [] if x["agent_id"] == a.agent_id), None)
        ax.errorbar(
            i,
            a.forecast[0],
            yerr=[[a.forecast[0] - a.lower[0]], [a.upper[0] - a.forecast[0]]],
            fmt="o",
            capsize=4,
            color="tab:blue",
            label="initial (10-90% band)" if i == 0 else None,
        )
        if rv:
            ax.plot(i + 0.15, rv["y"], "s", color="tab:orange", label="revised" if i == 0 else None)
    k = len(s["agents"])
    ax.axhline(r.consensus, color="tab:blue", ls=":", label=f"consensus {r.consensus:.2f}")
    ax.axhline(r.final, color="tab:orange", ls="--", label=f"final {r.final:.2f}")
    ax.axhline(p.y, color="black", lw=1.2, label=f"actual {p.y:.0f}")
    ax.set_xticks(range(k), [a.agent_id for a in s["agents"]])
    act = day.iloc[0] if len(day) else None
    txt = (
        f"D = {p.D:.3f}   action on day {r.origin}: "
        f"{act.action + ' ' + str(int(act.order_qty)) if act is not None else 'n/a'}\n"
        + textwrap.fill("Critic: " + s["critic"].primary_conflict, 110)
    )
    ax.set_title(f"{title}: {r.item_id}, target d_{int(p.d)}", fontsize=10)
    fig.text(0.01, 0.01, txt, fontsize=7.5, va="bottom")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {
        "item_id": r.item_id,
        "d": int(p.d),
        "D": float(p.D),
        "actual": float(p.y),
        "consensus": float(r.consensus),
        "final": float(r.final),
        "critic": s["critic"].model_dump(),
        "revisions": s.get("revisions"),
        "action": None if act is None else {"action": act.action, "qty": int(act.order_qty)},
    }


# ---------------------------------------------------------------- main
def main(dry_run: bool = False, offline: bool = False) -> None:
    cfg, inv = load_experiment(), load_inventory()
    d, fin = cfg.data, cfg.final
    if fin is None:
        raise RuntimeError("config has no `final` block")
    kind = "dryrun" if dry_run else "final"
    out = ROOT / "reports" / cfg.dataset / kind
    figs = ROOT / "reports" / "figures" / cfg.dataset / kind
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    label = cfg.dataset
    conf_hash = file_hash(cfg_path(), ROOT / "configs" / "inventory.yaml")

    if dry_run:
        paths = cfg.paths.model_copy(
            update={
                "cache": cfg.paths.cache / "dryrun",
                "db": cfg.paths.db.with_name(f"{cfg.paths.db.stem}_dryrun.db"),
            }
        )
        cfg = cfg.model_copy(update={"paths": paths})
        llm = LLM(cfg, transport=hold_transport(cfg))
        val_pts = load_points(cfg)
        tau = float(val_pts.D.quantile(cfg.disagreement.tau_quantile))  # provisional, not frozen
        main_win = ("dryrun_val", d.train_end, d.train_end, d.val_end)
        print(f"DRY RUN on validation with a hold-stub LLM; provisional tau {tau:.4f}")
    else:
        tau = cfg.deliberation.tau
        if tau is None:
            raise RuntimeError("tau is not frozen - run deliberation_eval --freeze first")
        lock = out / "RUN_LOCK.json"
        if lock.exists() and json.loads(lock.read_text())["config_hash"] != conf_hash:
            raise RuntimeError("the test split was already evaluated under a different config")
        lock.write_text(json.dumps({"config_hash": conf_hash, "tau": tau}))
        llm = LLM(cfg, transport=offline_transport(cfg) if offline else None)
        val_pts = load_points(cfg)
        main_win = ("final_test", d.train_end, d.val_end, d.test_end)
        print(
            f"FINAL TEST RUN: tau {tau:.6f}, prompt {cfg.deliberation.prompt_version}, "
            f"model {cfg.deliberation.model}, config {conf_hash}"
        )

    tag, fit_end, score_from, last = main_win
    w = Window(cfg, inv, llm, tag, fit_end, score_from, last, tau, True, fin.weather_noise_sd)
    B, seed = cfg.analysis.bootstrap, cfg.seed
    ftab = forecast_table(w.evals, B, seed)
    itab, logs = w.inventory()
    cmp_ = compare(w.evals, logs, B, seed, "retailsense")
    mech = w.h1_mechanism()
    lat = latency_table(w)
    noisy_p = cfg.path("processed") / f"{tag}_weather_noisy.parquet"
    abl = ablations(w, cfg, val_pts, pd.read_parquet(noisy_p) if noisy_p.exists() else None)
    lead = []
    for L in fin.lead_times:
        t, _ = w.inventory(inv=inv.model_copy(update={"lead_time": L}))
        lead += [{"lead_time": L, "system": s, "total_cost": c} for s, c in t.total_cost.items()]
    lead = pd.DataFrame(lead).pivot(index="system", columns="lead_time", values="total_cost")
    print("cost-accuracy curve ...", flush=True)
    curve = cost_accuracy(w, cfg)

    back = []
    for k, (a, b) in enumerate(fin.backtest_windows, 1):
        bw = Window(cfg, inv, llm, f"backtest{k}", a - 1, a - 1, b, tau, False)
        bt, blogs = bw.inventory()
        bf = forecast_table(bw.evals, B, seed)
        bc = compare(bw.evals, blogs, B, seed, "retailsense")
        bm = bw.h1_mechanism()
        for s in bt.index:
            back.append(
                {
                    "window": f"d_{a}-d_{b}",
                    "system": s,
                    "h1_WAPE": bf.loc[s, "h1_WAPE"],
                    "total_cost": bt.loc[s, "total_cost"],
                    "fill_rate": bt.loc[s, "fill_rate"],
                    "WAPE_diff_vs_retailsense": np.nan
                    if s == "retailsense"
                    else -bc.loc[s, "WAPE_diff"],
                    "H1_spearman": bm["H1_spearman"],
                    "H1_auroc": bm["H1_auroc_worst"],
                    "trigger_rate": bm["trigger_rate"],
                }
            )
    back = pd.DataFrame(back)

    # ---- write everything (blueprint 9.6)
    ftab.to_csv(out / "table_forecast_metrics.csv")
    itab.to_csv(out / "table_inventory.csv")
    cmp_.to_csv(out / "table_significance.csv")
    pd.Series(mech).to_csv(out / "table_mechanism.csv")
    lat.to_csv(out / "table_latency_llm.csv")
    abl.to_csv(out / "table_ablations.csv", index=False)
    lead.to_csv(out / "table_lead_time.csv")
    curve.to_csv(out / "table_cost_accuracy.csv")
    back.to_csv(out / "table_backtest.csv", index=False)
    rate = float(w.sel.mean())
    fig_deciles(w.pts, figs / "h1_deciles.png", f"H1: D decile vs consensus error - {label} {kind}")
    fig_curve(curve, rate, figs / "cost_accuracy.png", f"Cost-accuracy - {label} {kind}")
    fig_before_after(w, figs / "before_after.png", f"Triggered cases - {label} {kind}")
    case = fig_case(w, logs, figs / "case_study.png", f"Case study ({label} {kind})")
    (out / "case_study.json").write_text(json.dumps(case, indent=2, default=str))

    con = connect(cfg.path("db"))
    con.execute("DELETE FROM forecast_runs WHERE experiment_id = ?", (tag,))
    for (item, origin), s in w.states.items():
        y = float(w.actual.loc[(item, origin + 1)])
        log_case(
            con,
            f"{tag}-{item}-{origin + 1}",
            tag,
            "always_on",
            s,
            y,
            cfg.deliberation.prompt_version,
            cfg.deliberation.model,
        )
    con.execute("DELETE FROM inventory_state WHERE run_id LIKE ?", (f"{tag}-%",))
    for name, log in logs.items():
        con.executemany(
            "INSERT INTO inventory_state VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    f"{tag}-{name}-{r.item_id}-{r.d}",
                    r.on_hand,
                    r.on_order,
                    r.safety_stock,
                    r.reorder_point,
                    r.order_up_to,
                    r.action,
                    r.order_qty,
                    r.sold,
                    r.lost,
                )
                for r in log.itertuples()
            ],
        )
    ev = []
    for s in ftab.index:
        row = {k.removeprefix("h1_"): v for k, v in ftab.loc[s].items() if k.startswith("h1_")}
        row |= itab.loc[s].to_dict()
        if s in lat.index:
            row |= {
                "trigger_rate": lat.loc[s, "trigger_rate"],
                "llm_calls": lat.loc[s, "llm_calls_per_1000"],
                "mean_latency": lat.loc[s, "mean_latency_ms"],
            }
        ev.append({"system": s, **row})
    log_evaluation(con, tag, ev)
    con.commit()

    with start_run(cfg, "phase5_dryrun" if dry_run else "phase5", tag):
        mlflow.set_tags(
            {
                "final_test_run": str(not dry_run),
                "tau": tau,
                "prompt_version": cfg.deliberation.prompt_version,
                "model": cfg.deliberation.model,
            }
        )
        mlflow.log_metrics(
            {
                f"{s}.{m}": float(v)
                for s, r in ftab.iterrows()
                for m, v in r.items()
                if pd.notna(v) and m != "#"
            }
        )
        mlflow.log_metrics(
            {f"{s}.inv.{m}": float(v) for s, r in itab.iterrows() for m, v in r.items() if m != "#"}
        )
        mlflow.log_metrics(
            {k: float(v) for k, v in mech.items() if isinstance(v, (int, float)) and pd.notna(v)}
        )
        mlflow.log_artifacts(str(out), "tables")
        mlflow.log_artifacts(str(figs), "figures")

    if dry_run:  # with every agent holding, deliberation must change nothing
        for s in ("random_trigger", "always_on", "retailsense"):
            if not np.isclose(
                ftab.loc[s, "h1-7_WAPE"], ftab.loc["multi_agent_confidence", "h1-7_WAPE"]
            ):
                raise AssertionError(f"dry run: {s} differs from confidence consensus")
            if not np.isclose(
                itab.loc[s, "total_cost"], itab.loc["multi_agent_confidence", "total_cost"]
            ):
                raise AssertionError(f"dry run: {s} inventory differs from confidence consensus")
        print("DRY RUN CHECK PASSED: systems 7-9 == system 6 when every agent holds")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(ftab.round(4))
        print(itab.round(2))
        print(cmp_.round(4))
        print(pd.Series(mech))
        print(abl.round(4))
        print(back.round(4))
