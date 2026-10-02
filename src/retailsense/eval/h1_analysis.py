"""H1 on validation (blueprint 7.2): does disagreement D predict consensus error?

Uses the saved per-agent h=1 forecasts and confidences from `run_experiment --phase 2`.
Validation targets only (d <= val_end); the test split is never read.
Writes tables to reports/<dataset>/ and figures to reports/figures/<dataset>/, and logs to MLflow.

Run: python -m retailsense.eval.h1_analysis
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from retailsense.config import ROOT, ExperimentCfg, load_experiment  # noqa: E402
from retailsense.eval.metrics import wape  # noqa: E402
from retailsense.orchestrator.disagreement import score_table  # noqa: E402
from retailsense.store.tracking import start_run  # noqa: E402


def load_points(
    cfg: ExperimentCfg, tag: str = "phase2", first: int | None = None, last: int | None = None
) -> pd.DataFrame:
    """One row per (item, d) at h=1 in [first, last] (default: validation): agent forecasts,
    consensus, D, D_range, error."""
    d = cfg.data
    first = d.train_end + 1 if first is None else first
    last = d.val_end if last is None else last
    af = pd.read_parquet(cfg.path("processed") / f"{tag}_agent_forecasts.parquet")
    af = af[af.h == 1].assign(d=lambda x: x.origin + 1)
    af = af[(af.d >= first) & (af.d <= last)]
    y = af.pivot(index=["item_id", "d"], columns="agent", values="forecast")
    c = af.pivot(index=["item_id", "d"], columns="agent", values="confidence")
    hist = af[af.agent == "historical"].set_index(["item_id", "d"])
    feats = pd.read_parquet(
        cfg.path("processed") / "features.parquet", columns=["item_id", "d", "h", "sales"]
    )
    actual = feats[(feats.h == 1) & feats.d.between(first, last)]
    actual = actual.astype({"item_id": str}).set_index(["item_id", "d"]).sales
    pts = (
        score_table(y, c, cfg.disagreement.floor).join(y.add_prefix("f_")).join(c.add_prefix("c_"))
    )
    pts["y"] = actual.reindex(pts.index).to_numpy()
    if pts.y.isna().any():
        raise RuntimeError("missing actuals for some points")
    pts["abs_err"] = (pts.y - pts.consensus).abs()
    pts["rel_err"] = pts.abs_err / pts.y.clip(lower=1)
    # reference signal: the Historical agent's own relative band width (no disagreement needed)
    pts["hist_band"] = ((hist.upper - hist.lower) / hist.forecast.clip(lower=1)).reindex(pts.index)
    pts["low_volume"] = 1 / pts.consensus.clip(lower=1)  # reference: is D just "small item"?
    return pts.reset_index()


def signal_stats(pts: pd.DataFrame, signal: str, worst: np.ndarray) -> dict:
    return {
        "spearman": spearmanr(pts[signal], pts.rel_err).statistic,
        "auroc_worst": roc_auc_score(worst, pts[signal]),
    }


def bootstrap_ci(pts: pd.DataFrame, signal: str, share: float, n: int, seed: int) -> dict:
    """95% CIs from resampling items (points of one item stay together)."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in pts.groupby("item_id")}
    items = np.array(list(groups))
    rho, auc = [], []
    for _ in range(n):
        s = pd.concat([groups[i] for i in rng.choice(items, len(items))], ignore_index=True)
        w = s.rel_err >= s.rel_err.quantile(1 - share)
        st = signal_stats(s, signal, w.to_numpy())
        rho.append(st["spearman"])
        auc.append(st["auroc_worst"])
    q = lambda v: np.quantile(v, [0.025, 0.975]).round(4).tolist()  # noqa: E731
    return {"spearman_ci": q(rho), "auroc_ci": q(auc)}


def by_tertile(pts: pd.DataFrame, share: float) -> pd.DataFrame:
    """H1 within item-volume tertiles (blueprint 5.4), so volume cannot drive the result."""
    vol = pts.groupby("item_id").y.sum()
    t = pd.qcut(vol.rank(method="first"), 3, labels=["low", "mid", "high"])
    rows = {}
    for name, g in pts.groupby(pts.item_id.map(t), observed=True):
        w = (g.rel_err >= g.rel_err.quantile(1 - share)).to_numpy()
        rows[name] = {
            "n": len(g),
            **{f"D_{k}": v for k, v in signal_stats(g, "D", w).items()},
            **{f"band_{k}": v for k, v in signal_stats(g, "hist_band", w).items()},
        }
    return pd.DataFrame(rows).T


def added_value(pts: pd.DataFrame, worst: np.ndarray) -> pd.DataFrame:
    """Does D add to the Historical band width? AUROC of a logistic model on ranks
    (2-3 inputs on thousands of points, so in-sample fit is a fair first look)."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    out = {}
    for name, cols in {
        "band": ["hist_band"],
        "band + D": ["hist_band", "D"],
        "band + volume": ["hist_band", "low_volume"],
        "band + volume + D": ["hist_band", "low_volume", "D"],
    }.items():
        x = pts[cols].rank(pct=True).to_numpy()
        prob = LogisticRegression().fit(x, worst).predict_proba(x)[:, 1]
        out[name] = {"auroc_worst": roc_auc_score(worst, prob)}
    return pd.DataFrame(out).T


def deciles(pts: pd.DataFrame, signal: str) -> pd.DataFrame:
    dec = pd.qcut(pts[signal].rank(method="first"), 10, labels=range(1, 11))
    return pts.groupby(dec, observed=True).agg(
        n=("rel_err", "size"),
        signal_mean=(signal, "mean"),
        mean_rel_err=("rel_err", "mean"),
        wape=("abs_err", lambda s: s.sum() / max(pts.loc[s.index, "y"].sum(), 1)),
    )


def drop_one(pts: pd.DataFrame, agents: list[str], cfg: ExperimentCfg, worst) -> pd.DataFrame:
    """D recomputed without each agent (consensus error kept fixed, so only the signal changes)."""
    rows = []
    for a in agents:
        keep = [x for x in agents if x != a]
        y = pts[[f"f_{x}" for x in keep]].set_axis(keep, axis=1)
        conf = pts[[f"c_{x}" for x in keep]].set_axis(keep, axis=1)
        d = score_table(y, conf, cfg.disagreement.floor).D
        rows.append(
            {
                "dropped": a,
                "spearman": spearmanr(d, pts.rel_err).statistic,
                "auroc_worst": roc_auc_score(worst, d),
                "corr_with_full_D": spearmanr(d, pts.D).statistic,
            }
        )
    return pd.DataFrame(rows).set_index("dropped")


def sweep(pts: pd.DataFrame, rates: list[float], worst: np.ndarray) -> pd.DataFrame:
    rows = []
    for r in rates:
        tau = float(pts.D.quantile(1 - r))
        trig = (pts.D >= tau).to_numpy()
        rows.append(
            {
                "trigger_rate_target": r,
                "tau": tau,
                "trigger_rate": trig.mean(),
                "mean_rel_err_triggered": pts.rel_err[trig].mean(),
                "mean_rel_err_not": pts.rel_err[~trig].mean(),
                "wape_triggered": wape(pts.y[trig], pts.consensus[trig]),
                "wape_not": wape(pts.y[~trig], pts.consensus[~trig]),
                "recall_worst": worst[trig].sum() / worst.sum(),
                "recall_if_random": trig.mean(),
            }
        )
    return pd.DataFrame(rows).set_index("trigger_rate_target")


def figures(dec: dict[str, pd.DataFrame], sw: pd.DataFrame, out, title: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, t in dec.items():
        ax[0].plot(t.index.astype(int), t.mean_rel_err, marker="o", label=name)
    ax[0].set(
        xlabel="decile of signal (1 = lowest)",
        ylabel="mean relative consensus error",
        title="Error by signal decile",
    )
    ax[0].legend()
    ax[1].plot(sw.trigger_rate, sw.recall_worst, marker="o", label="D trigger")
    ax[1].plot(sw.trigger_rate, sw.recall_if_random, ls="--", label="random trigger")
    ax[1].set(
        xlabel="trigger rate", ylabel="share of worst-20% errors caught", title="Trigger-rate sweep"
    )
    ax[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / "h1_deciles_and_sweep.png", dpi=130)
    plt.close(fig)


def main() -> None:
    cfg = load_experiment()
    a = cfg.analysis
    pts = load_points(cfg)
    agents = [c.removeprefix("f_") for c in pts.columns if c.startswith("f_")]
    worst = (pts.rel_err >= pts.rel_err.quantile(1 - a.worst_share)).to_numpy()

    signals = {
        "D": "D",
        "D_range": "D_range",
        "historical band width": "hist_band",
        "low volume": "low_volume",
    }
    summary = pd.DataFrame(
        {
            name: {
                **signal_stats(pts, col, worst),
                **bootstrap_ci(pts, col, a.worst_share, a.bootstrap, cfg.seed),
            }
            for name, col in signals.items()
        }
    ).T
    dec = {name: deciles(pts, col) for name, col in signals.items()}
    drops = drop_one(pts, agents, cfg, worst)
    sw = sweep(pts, a.trigger_rates, worst)
    tau = float(pts.D.quantile(cfg.disagreement.tau_quantile))

    rep = ROOT / "reports" / cfg.dataset
    figs = ROOT / "reports" / "figures" / cfg.dataset
    rep.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    label = cfg.dataset
    figures(dec, sw, figs, f"H1 on validation - {label}, agents: {', '.join(agents)}")
    tables = {
        "h1_summary": summary,
        "h1_deciles_D": dec["D"],
        "h1_drop_one": drops,
        "h1_by_volume_tertile": by_tertile(pts, a.worst_share),
        "h1_added_value": added_value(pts, worst),
        "h1_trigger_sweep": sw,
    }
    for k, t in tables.items():
        t.to_csv(rep / f"{k}.csv")

    with start_run(cfg, "phase3a", "h1_validation"):
        mlflow.log_metrics(
            {
                "n_points": len(pts),
                "tau_proposed": tau,
                "consensus_wape": wape(pts.y, pts.consensus),
                **{
                    f"{s}.{m}".replace(" ", "_"): float(summary.loc[s, m])
                    for s in summary.index
                    for m in ("spearman", "auroc_worst")
                },
            }
        )
        for k, t in tables.items():
            mlflow.log_text(t.round(4).to_string(), f"{k}.txt")
        mlflow.log_artifact(str(figs / "h1_deciles_and_sweep.png"))

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"== {label}: {len(pts)} validation points, agents {agents} ==")
        print(
            f"consensus WAPE {wape(pts.y, pts.consensus):.4f}; "
            f"proposed tau (q{cfg.disagreement.tau_quantile:.2f} of D) = {tau:.4f}"
        )
        for k, t in tables.items():
            print(f"\n== {k} ==\n{t.round(4)}")


if __name__ == "__main__":
    main()
