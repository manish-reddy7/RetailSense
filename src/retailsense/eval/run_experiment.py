"""Experiment runner.
Phase 1: baselines + Historical + Calendar. Phase 2: + Weather, Trend, all-features control.
Final (Phase 5a): all 9 systems on the test split, backtests, ablations - see eval/final.py.

Run: python -m retailsense.eval.run_experiment --phase 2
     python -m retailsense.eval.run_experiment --final      (touches test: frozen config only)
"""

import argparse
import json

import mlflow
import numpy as np
import pandas as pd

from retailsense.agents.calendar import calendar_agent
from retailsense.agents.confidence import trailing_confidence
from retailsense.agents.historical import FEATURES, HistoricalAgent
from retailsense.agents.schema import AgentForecast
from retailsense.agents.trend import trend_agent
from retailsense.agents.weather import weather_agent
from retailsense.config import ExperimentCfg, load_experiment
from retailsense.data.weather import add_forecast_noise
from retailsense.eval.baselines import ets, origin_grid, seasonal_naive
from retailsense.eval.metrics import all_metrics, wape
from retailsense.store.tracking import start_run

KEYS = ["item_id", "d"]
FKEYS = ["item_id", "d", "h"]  # feature rows are per (item, target day, horizon)


def build_agent_forecasts(
    agent_id: str,
    preds: pd.DataFrame,
    conf: pd.DataFrame,
    feats: pd.DataFrame,
    origins,
    horizons: list[int],
    key_factors,
) -> list[tuple[int, AgentForecast]]:
    """One schema-validated AgentForecast per (item, origin), covering every horizon.
    preds must contain every (item, origin + h, h)."""
    p = preds.merge(feats, on=FKEYS, suffixes=("", "_feat"))
    p = p.assign(origin=p.d - p.h).set_index(["item_id", "origin", "h"]).sort_index()
    c = conf.set_index(["item_id", "origin"])
    out = []
    for item in p.index.get_level_values(0).unique():
        rows = p.loc[item]
        for o in origins:
            hs = [h for h in horizons if (o, h) in rows.index]  # fewer at the end of the data
            r = rows.loc[[(o, h) for h in hs]]
            first = r.iloc[0]
            conf_o = c.loc[(item, o)]
            lims = ["confidence from department-level fallback"] if conf_o.fallback else []
            if pd.isna(first.price):
                lims.append("no closed-week price available")
            out.append(
                (
                    o,
                    AgentForecast(
                        agent_id=agent_id,
                        item_id=str(item),
                        store_id=feats.attrs["store"],
                        prediction_time=(first.date - pd.Timedelta(days=horizons[0])).date(),
                        horizon_days=hs,
                        forecast=r.forecast.tolist(),
                        lower=r.lower.tolist(),
                        upper=r.upper.tolist(),
                        confidence=float(conf_o.confidence),
                        key_factors=key_factors(first),
                        limitations=lims,
                    ),
                )
            )
    return out


def flatten(forecasts: list[tuple[int, AgentForecast]]) -> pd.DataFrame:
    rows = [
        (f.item_id, o, h, o + h, y, lo, hi)
        for o, f in forecasts
        for h, y, lo, hi in zip(f.horizon_days, f.forecast, f.lower, f.upper, strict=True)
    ]
    return pd.DataFrame(rows, columns=["item_id", "origin", "h", "d", "yhat", "lower", "upper"])


def metrics_table(evals: dict[str, pd.DataFrame], tertile: pd.Series) -> pd.DataFrame:
    rows = []
    for system, e in evals.items():
        lo, hi = (e.lower, e.upper) if "lower" in e else (None, None)
        h1 = e[e.h == 1]
        row = {"system": system}
        row |= {
            f"h1_{k}": v
            for k, v in all_metrics(
                h1.y, h1.yhat, *((h1.lower, h1.upper) if lo is not None else ())
            ).items()
        }
        row |= {f"h1-7_{k}": v for k, v in all_metrics(e.y, e.yhat, lo, hi).items()}
        for t, g in h1.groupby(h1.item_id.map(tertile)):
            row[f"h1_WAPE_{t}"] = wape(g.y, g.yhat)
        rows.append(row)
    return pd.DataFrame(rows).set_index("system")


def slice_table(evals: dict[str, pd.DataFrame], ctx: pd.DataFrame, agents) -> pd.DataFrame:
    """h=1 WAPE of each agent vs Historical on evidence slices where the agent should matter."""
    base = evals["historical"][lambda e: e.h == 1].merge(ctx, on=KEYS)
    slices = {
        "calendar": {
            "event day": base.event_type.notna(),
            "SNAP day": base.snap == 1,
            "weekend": base.wday.isin([1, 2]),  # calendar wday 1 = Saturday, 2 = Sunday
        },
        "weather": {
            "hot day": base.hot_day == 1,
            "rainy day": base.rainy_day == 1,
            "normal day": (base.hot_day == 0) & (base.rainy_day == 0),
        },
        "trend": {},
    }
    rows = []
    for a in agents:
        e = evals[f"historical+{a}"][lambda x: x.h == 1].merge(ctx[KEYS], on=KEYS)
        adj = np.abs(e.yhat.to_numpy() - base.yhat.to_numpy())
        named = {**slices[a], "top-10% |adjustment|": adj >= np.quantile(adj, 0.9), "all": adj >= 0}
        for name, mask in named.items():
            m = np.asarray(mask, bool)
            rows.append(
                {
                    "agent": a,
                    "slice": name,
                    "n": int(m.sum()),
                    "WAPE_historical": wape(base.y[m], base.yhat[m]),
                    "WAPE_agent": wape(e.y[m], e.yhat[m]),
                }
            )
    t = pd.DataFrame(rows)
    t["delta"] = t.WAPE_agent - t.WAPE_historical
    return t.set_index(["agent", "slice"])


def confidence_check(forecasts: dict, actual: pd.DataFrame) -> pd.DataFrame:
    """Does higher confidence go with lower error? Per agent over (item, origin) at h=1, plus
    how often the most confident agent is also the most accurate one on that point."""
    y = actual.set_index(KEYS).sales
    rows, parts = [], []
    for a, fs in forecasts.items():
        c = pd.DataFrame(
            [(f.item_id, o + f.horizon_days[0], f.confidence, f.forecast[0]) for o, f in fs],
            columns=["item_id", "d", "conf", "yhat"],
        )
        c["y"] = y.loc[list(zip(c.item_id, c.d, strict=True))].to_numpy()
        c["err"] = (c.y - c.yhat).abs()
        q = pd.qcut(c.conf.rank(method="first"), 5, labels=False)
        rows.append(
            {
                "agent": a,
                "spearman(conf, rel_err)": c.conf.corr(
                    c.err / c.y.clip(lower=1), method="spearman"
                ),
                **{f"WAPE_conf_q{i + 1}": wape(g.y, g.yhat) for i, g in c.groupby(q)},
            }
        )
        parts.append(c.assign(agent=a))
    pp = pd.concat(parts, ignore_index=True)
    most_conf = pp.loc[pp.groupby(["item_id", "d"]).conf.idxmax()].set_index(["item_id", "d"])
    min_err = pp.groupby(["item_id", "d"]).err.min()
    hit = (most_conf.err <= min_err.loc[most_conf.index]).mean()
    out = pd.DataFrame(rows).set_index("agent")
    out.loc["most-confident agent is most accurate (share)"] = np.nan
    out.loc["most-confident agent is most accurate (share)", "spearman(conf, rel_err)"] = hit
    return out


def run_validation(
    cfg: ExperimentCfg,
    phase: int,
    fit_end: int | None = None,
    score_from: int | None = None,
    last: int | None = None,
    tag: str | None = None,
    log: bool = True,
    weather_noise_sd: float | None = None,
):
    """Fit on days <= fit_end, forecast from every origin in [fit_end, last), score targets in
    (score_from, last]. Defaults are the validation run: fit on train, score validation.
    The final test run keeps fit_end = train_end (blueprint 5: train fits all models) and
    scores (val_end, test_end]; a backtest window moves fit_end back before the window."""
    d = cfg.data
    fit_end = d.train_end if fit_end is None else fit_end
    last = d.val_end if last is None else last
    score_from = fit_end if score_from is None else score_from
    tag = tag or f"phase{phase}"
    horizons = cfg.agents.horizons
    feats = pd.read_parquet(cfg.path("processed") / "features.parquet")
    feats.attrs["store"] = d.store
    # a day with no sales record (item not offered) is unknown, so it is never a training target
    train = feats[(feats.d <= fit_end) & feats.launched & feats.sales.notna()]
    origins = list(range(fit_end, last))  # forecast made at end of day p
    # Forecast every horizon from every origin (targets past `last` feed the inventory
    # simulator; they use data up to the origin only). Only targets in the window are scored.
    grid = origin_grid(origins, horizons, min(last + max(horizons), d.test_end))
    # Agents forecast every horizon from every origin; only targets inside validation are scored.
    origin = feats.d - feats.h
    pred_rows = feats[(origin >= origins[0]) & (origin <= origins[-1])]
    pred_rows = pred_rows.drop(columns="sales").reset_index(drop=True)
    h1 = feats[feats.h == horizons[0]]
    actual = h1.loc[h1.d <= last, KEYS + ["sales"]].astype({"item_id": str})

    residual = {"calendar": calendar_agent(cfg)}
    if phase >= 2:
        residual["weather"] = weather_agent(cfg)
        trend = trend_agent(cfg)
        missing = [f for f in trend.features[1:] if f not in feats]
        if missing:
            print(f"Trend agent SKIPPED - {len(missing)} interest columns missing (no data yet)")
        else:
            residual["trend"] = trend

    print("fitting Historical agent ...")
    hist = HistoricalAgent(cfg).fit(train)
    hist_pred = hist.predict(pred_rows)
    print("out-of-fold baseline ...")
    oof = hist.oof_baseline(train)
    oof.to_parquet(cfg.path("processed") / "oof_baseline.parquet", index=False)

    def track(prior: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
        """The agent's h=1 out-of-sample forecasts (the disagreement horizon) with actuals."""
        prior = prior[prior.h == horizons[0]].drop(columns="h")
        val = val.loc[val.h == horizons[0], KEYS + ["forecast"]]
        t = pd.concat([prior, val.rename(columns={"forecast": "yhat"})])
        t = t.astype({"item_id": str})
        return t.merge(actual.rename(columns={"sales": "y"}), on=KEYS)

    preds = {"historical": hist_pred}
    tracks = {"historical": track(oof[FKEYS].assign(yhat=oof.baseline), hist_pred)}
    key_factors = {"historical": hist.key_factors}
    for name, agent in residual.items():
        print(f"fitting {name} agent ...")
        agent.fit(oof)
        preds[name] = agent.apply(hist_pred, pred_rows)
        ao = agent.oof_adjustment(oof)
        tracks[name] = track(
            ao[FKEYS].assign(yhat=np.clip(ao.baseline + ao.adjustment, 0, None)), preds[name]
        )
        key_factors[name] = lambda r, ag=agent: ag.key_factors(r, r.adjustment)
    if weather_noise_sd and "weather" in residual:  # ablation: weather forecast error
        h1_train = train[train.h == horizons[0]].drop_duplicates("date")
        scale = {c: float(h1_train[c].std()) for c in ("tmax", "tmin", "precip")}
        noisy = add_forecast_noise(pred_rows, scale, weather_noise_sd, cfg, cfg.seed)
        wn = residual["weather"].apply(hist_pred, noisy)
        wn.assign(origin=wn.d - wn.h)[
            ["item_id", "origin", "h", "forecast", "lower", "upper"]
        ].to_parquet(cfg.path("processed") / f"{tag}_weather_noisy.parquet", index=False)
    a = cfg.agents
    conf = {
        k: trailing_confidence(v, origins, a.confidence_window, a.confidence_min_days)
        for k, v in tracks.items()
    }

    print("building AgentForecast objects ...")
    forecasts = {
        k: build_agent_forecasts(k, preds[k], conf[k], pred_rows, origins, horizons, key_factors[k])
        for k in preds
    }

    print("baselines: seasonal naive, ETS ...")
    item_sales = actual[actual.d <= origins[-1]]  # baselines never receive later sales
    evals = {"seasonal_naive": seasonal_naive(item_sales, grid), "ets": ets(item_sales, grid, cfg)}
    for k, f in forecasts.items():
        name = k if k == "historical" else f"historical+{k}"
        evals[name] = grid.merge(flatten(f), on=["origin", "h", "d"])
    control = None
    if phase >= 2:
        print("control: one LightGBM with all features ...")
        all_feats = FEATURES + [f for ag in residual.values() for f in ag.features[1:]]
        control = HistoricalAgent(cfg, all_feats).fit(train)
        cp = control.predict(pred_rows).astype({"item_id": str})
        cp = cp.assign(origin=cp.d - cp.h).rename(columns={"forecast": "yhat"})
        cols = ["item_id", "origin", "h", "d", "yhat", "lower", "upper"]
        evals["lgbm_all_features"] = grid.merge(cp[cols], on=["origin", "h", "d"])
    pd.concat([v.assign(system=k) for k, v in evals.items()]).to_parquet(
        cfg.path("processed") / f"{tag}_all_forecasts.parquet", index=False
    )
    evals = {k: v[(v.d > score_from) & (v.d <= last)] for k, v in evals.items()}
    evals = {k: v.merge(actual.rename(columns={"sales": "y"}), on=KEYS) for k, v in evals.items()}
    sizes = {k: len(v) for k, v in evals.items()}
    if len(set(sizes.values())) != 1:
        raise RuntimeError(f"systems scored on different grids: {sizes}")
    evals = {
        k: v.sort_values(["item_id", "origin", "h"]).reset_index(drop=True)
        for k, v in evals.items()
    }

    vol = train[train.h == horizons[0]].groupby("item_id", observed=True).sales.sum()
    vol = vol.rename(index=str)
    tertile = pd.qcut(vol.rank(method="first"), 3, labels=["low", "mid", "high"])
    table = metrics_table(evals, tertile)

    pd.concat([v.assign(system=k) for k, v in evals.items()]).to_parquet(
        cfg.path("processed") / f"{tag}_scored_predictions.parquet", index=False
    )
    agent_rows = [
        (
            k,
            f.item_id,
            o,
            h,
            y,
            lo,
            hi,
            f.confidence,
            json.dumps(f.key_factors),
            json.dumps(f.limitations),
        )
        for k, fs in forecasts.items()
        for o, f in fs
        for h, y, lo, hi in zip(f.horizon_days, f.forecast, f.lower, f.upper, strict=True)
    ]
    cols = [
        "agent",
        "item_id",
        "origin",
        "h",
        "forecast",
        "lower",
        "upper",
        "confidence",
        "key_factors",
        "limitations",
    ]
    pd.DataFrame(agent_rows, columns=cols).to_parquet(
        cfg.path("processed") / f"{tag}_agent_forecasts.parquet", index=False
    )
    extra = {}
    if phase >= 2:
        ctx_cols = ["event_type", "snap", "wday", "hot_day", "rainy_day"]
        ctx = h1.loc[h1.d.between(score_from + 1, last), KEYS + ctx_cols].astype({"item_id": str})
        ctx = evals["historical"].loc[evals["historical"].h == 1, KEYS].merge(ctx, on=KEYS)
        extra["slices"] = slice_table(evals, ctx, residual)
        extra["confidence"] = confidence_check(forecasts, actual)
        agents_h1 = pd.DataFrame(
            {
                k: v.loc[v.h == 1, "yhat"].to_numpy()
                for k, v in evals.items()
                if k == "historical" or k.startswith("historical+")
            }
        )
        extra["corr_forecast_h1"] = agents_h1.corr()
        extra["corr_adjustment_h1"] = (
            agents_h1.drop(columns="historical").sub(agents_h1.historical, axis=0).corr()
        )

    if not log:
        return table, extra
    with start_run(cfg, tag, f"{tag}_validation"):
        mlflow.log_metrics(
            {
                f"{s}.{m}".replace("+", "_"): v
                for s, r in table.iterrows()
                for m, v in r.items()
                if pd.notna(v)
            }
        )
        mlflow.log_metrics(
            {
                **{
                    f"historical.rounds_{name}_h{h}": m.num_trees()
                    for name, ms in (("point", hist.point), ("q_lo", hist.lo), ("q_hi", hist.hi))
                    for h, m in ms.items()
                },
                **{f"{k}.rounds": ag.model.num_trees() for k, ag in residual.items()},
                **{
                    f"{k}.mean_abs_adjustment": float(np.abs(preds[k].adjustment).mean())
                    for k in residual
                },
                "historical.band_widened_share": float(hist_pred.band_widened.mean()),
                "n_eval_points": len(evals["historical"]),
                **{
                    f"{k}.confidence_fallback_share": float(v.fallback.mean())
                    for k, v in conf.items()
                },
                **{f"{k}.mean_confidence": float(v.confidence.mean()) for k, v in conf.items()},
            }
        )
        mlflow.log_text(table.to_string(float_format="%.4f"), f"{tag}_metrics.txt")
        mlflow.log_text(hist.importance.to_string(), "historical_importance.txt")
        for k, ag in residual.items():
            mlflow.log_text(ag.importance.to_string(), f"{k}_importance.txt")
        if control is not None:
            mlflow.log_text(control.importance.to_string(), "control_importance.txt")
        for k, v in extra.items():
            mlflow.log_text(v.to_string(float_format="%.4f"), f"{tag}_{k}.txt")
    return table, extra


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--phase", type=int, choices=[1, 2])
    mode.add_argument("--final", action="store_true", help="Phase 5a: the one test run")
    mode.add_argument("--dry-run", action="store_true", help="Phase 5a code on validation")
    ap.add_argument("--offline", action="store_true", help="LLM answers from cache only")
    args = ap.parse_args()
    if args.final or args.dry_run:
        from retailsense.eval.final import main as final_main  # noqa: PLC0415

        return final_main(dry_run=args.dry_run, offline=args.offline)
    table, extra = run_validation(load_experiment(), args.phase)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.round(4).T)
        for k, v in extra.items():
            print(f"\n== {k} ==\n{v.round(4)}")


if __name__ == "__main__":
    main()
