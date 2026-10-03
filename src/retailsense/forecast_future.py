"""Next-week forecast and restocking decision, made after the experiment.

The experiment is finished, so every model is refit on ALL data up to the last day (test_end,
the test split included). Three systems forecast the 7 days after it:
  retailsense        agents + disagreement routing + LLM critic/revision when D >= tau
  lgbm_all_features  the most accurate system in the test run
  ets                the cheapest system in the test run
Each system's (s, S) decision starts from ITS OWN stock at the end of the frozen test run
(`inventory_state`), so the order quantities are what each system would order next.

Weather for the new days is the Open-Meteo archive (observed weather used as a perfect forecast,
as in the experiment); festivals come from the same `holidays` calendar as the data. The frozen
config file is not changed: the longer weather range lives only in an in-memory config copy.

Run: python -m retailsense.forecast_future   -> reports/<dataset>/future_forecast.csv
"""

import json
import sqlite3

import numpy as np
import pandas as pd
import yaml

from retailsense.agents.calendar import calendar_agent
from retailsense.agents.confidence import trailing_confidence
from retailsense.agents.historical import FEATURES, HistoricalAgent
from retailsense.agents.schema import AgentForecast
from retailsense.agents.trend import trend_agent
from retailsense.agents.weather import weather_agent
from retailsense.config import ROOT, load_experiment, load_inventory
from retailsense.data import weather as wx
from retailsense.data.features import build_features, join_context, to_long
from retailsense.data.india_generator import build_calendar
from retailsense.data.load import load_calendar, load_prices, load_sales, load_subset_items
from retailsense.eval.baselines import ets, origin_grid
from retailsense.eval.deliberation_eval import run_all
from retailsense.eval.final import deliberated_system
from retailsense.eval.run_experiment import build_agent_forecasts
from retailsense.inventory.policy import policy
from retailsense.orchestrator.disagreement import score_table

FKEYS = ["item_id", "d", "h"]


def extended_calendar(cfg, last: int) -> pd.DataFrame:
    """calendar.csv plus the days after it, from the generator's own festival calendar."""
    cal = load_calendar(cfg)
    g = yaml.safe_load((ROOT / "configs/india_generator.yaml").read_text(encoding="utf-8"))
    ext = build_calendar(
        g, pd.date_range(g["start"], periods=last), sorted({s for s, _ in g["stores"].values()})
    )
    ext = ext[ext.d.str.removeprefix("d_").astype(int) > len(cal)]
    return pd.concat([cal, ext.assign(date=pd.to_datetime(ext.date))], ignore_index=True)


def feature_frame(cfg, last: int) -> pd.DataFrame:
    """The experiment's feature table, extended to target days up to `last` (sales stay NaN)."""
    cal = extended_calendar(cfg, last)
    end = cal.loc[cal.d == f"d_{last}", "date"].iloc[0]
    wcfg = cfg.model_copy(
        update={"weather": cfg.weather.model_copy(update={"fetch_end": f"{end:%Y-%m-%d}"})}
    )
    weather = wx.weather_features(wx.fetch(wcfg), wcfg)
    items = load_subset_items(cfg)
    wide = load_sales(cfg)
    wide = wide[wide.item_id.isin(items)]
    df = build_features(
        to_long(wide, cfg.data.test_end),
        cal,
        load_prices(cfg, items),
        cfg.data.state,
        last,
        items,
        cfg.agents.horizons,
    )
    trends = pd.read_parquet(cfg.path("processed") / "trends.parquet")
    df = join_context(df, weather, {"trends": trends})
    df.attrs["store"] = cfg.data.store
    return df


def agent_table(cfg, df: pd.DataFrame, origin: int):
    """Fit every agent on days <= origin and forecast h = 1..7 from it. Returns the agent rows
    (same columns as the experiment's *_agent_forecasts.parquet) and the all-features control."""
    hs = cfg.agents.horizons
    train = df[(df.d <= origin) & df.launched & df.sales.notna()]
    rows = df[(df.d > origin) & (df.d - df.h == origin)].drop(columns="sales")
    rows = rows.reset_index(drop=True)
    rows.attrs["store"] = cfg.data.store
    actual = df.loc[(df.h == hs[0]) & (df.d <= origin), ["item_id", "d", "sales"]]
    actual = actual.astype({"item_id": str}).rename(columns={"sales": "y"})

    def track(oof_rows: pd.DataFrame) -> pd.DataFrame:
        t = oof_rows[oof_rows.h == hs[0]].drop(columns="h").astype({"item_id": str})
        return t.merge(actual, on=["item_id", "d"])

    print("fitting Historical agent on all data ...", flush=True)
    hist = HistoricalAgent(cfg).fit(train)
    hp = hist.predict(rows)
    oof = hist.oof_baseline(train)
    preds = {"historical": hp}
    tracks = {"historical": track(oof[FKEYS].assign(yhat=oof.baseline))}
    factors = {"historical": hist.key_factors}
    residual = {"calendar": calendar_agent(cfg), "weather": weather_agent(cfg)}
    residual["trend"] = trend_agent(cfg)
    for name, agent in residual.items():
        print(f"fitting {name} agent ...", flush=True)
        agent.fit(oof)
        preds[name] = agent.apply(hp, rows)
        ao = agent.oof_adjustment(oof)
        tracks[name] = track(ao[FKEYS].assign(yhat=np.clip(ao.baseline + ao.adjustment, 0, None)))
        factors[name] = lambda r, ag=agent: ag.key_factors(r, r.adjustment)
    a = cfg.agents
    out = []
    for k in preds:
        conf = trailing_confidence(tracks[k], [origin], a.confidence_window, a.confidence_min_days)
        for o, f in build_agent_forecasts(k, preds[k], conf, rows, [origin], hs, factors[k]):
            for h, y, lo, hi in zip(f.horizon_days, f.forecast, f.lower, f.upper, strict=True):
                out.append(
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
                )
    af = pd.DataFrame(
        out,
        columns=[
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
        ],
    )
    print("fitting the all-features LightGBM ...", flush=True)
    all_feats = FEATURES + [f for ag in residual.values() for f in ag.features[1:]]
    cp = HistoricalAgent(cfg, all_feats).fit(train).predict(rows).astype({"item_id": str})
    lgbm = cp.assign(origin=origin).rename(columns={"forecast": "yhat"})
    return af, lgbm[["item_id", "origin", "h", "yhat", "lower", "upper"]], actual


def retailsense(cfg, af: pd.DataFrame, df: pd.DataFrame, origin: int):
    """Route every item by D vs the frozen tau, exactly as the system does (no forced arms)."""
    h1 = af[af.h == 1].assign(d=origin + 1)
    y = h1.pivot(index=["item_id", "d"], columns="agent", values="forecast")
    c = h1.pivot(index=["item_id", "d"], columns="agent", values="confidence")
    pts = score_table(y, c, cfg.disagreement.floor).reset_index()
    hist = df[df.h == 1].astype({"item_id": str}).set_index(["item_id", "d"]).sales
    target = df.loc[df.d == origin + 1, "date"].iloc[0].date()
    pred_time = df.loc[df.d == origin, "date"].iloc[0].date()
    n = cfg.deliberation.history_days
    cases = []
    for item in pts.item_id:
        rows = h1[h1.item_id == item]
        past = hist.loc[item].loc[origin - n + 1 : origin]
        cases.append(
            {
                "dataset": cfg.dataset,
                "item_id": item,
                "store_id": cfg.data.store,
                "origin": origin,
                "history_end_d": int(past.index.max()),
                "target_date": target,
                "history": [float(x) for x in past],
                "agents": [
                    AgentForecast(
                        agent_id=r.agent,
                        item_id=item,
                        store_id=cfg.data.store,
                        prediction_time=pred_time,
                        horizon_days=[1],
                        forecast=[r.forecast],
                        lower=[r.lower],
                        upper=[r.upper],
                        confidence=r.confidence,
                        key_factors=json.loads(r.key_factors),
                        limitations=json.loads(r.limitations),
                    )
                    for r in rows.itertuples()
                ],
            }
        )
    tau = cfg.deliberation.tau
    print(f"routing {len(cases)} items (tau {tau:.4f}); D >= tau goes to the LLM ...", flush=True)
    states = run_all(cfg, cases, tau, trigger=None)
    rev = pd.DataFrame(
        [
            {"item_id": c["item_id"], "origin": origin, "agent": r["agent_id"], **r}
            for c, s in zip(cases, states, strict=True)
            for r in s.get("revisions") or []
        ],
        columns=["item_id", "origin", "agent", "m", "y", "initial", "w", "held"],
    )
    done = [s["case"]["item_id"] for s in states if s["deliberated"]]
    fc = deliberated_system(af, rev, {(i, origin) for i in done})
    pts["deliberated"] = pts.item_id.isin(done)
    return fc, pts[["item_id", "D", "consensus", "deliberated"]]


def start_stock(cfg, system: str, items, day: int) -> pd.DataFrame:
    """Each system's on-hand and on-order stock at the end of the frozen test run."""
    con = sqlite3.connect(cfg.path("db"))
    ids = [f"final_test-{system}-{i}-{day}" for i in items]
    marks = ",".join("?" * len(ids))
    q = f"SELECT run_id, on_hand, on_order FROM inventory_state WHERE run_id IN ({marks})"
    st = pd.read_sql_query(q, con, params=ids)
    con.close()
    if len(st) != len(ids):
        raise RuntimeError(f"{system}: final-run stock missing for {len(ids) - len(st)} items")
    return st.assign(
        item_id=st.run_id.str.removeprefix(f"final_test-{system}-").str.rsplit("-", n=1).str[0]
    ).drop(columns="run_id")


def decide(fc: pd.DataFrame, stock: pd.DataFrame, inv) -> pd.DataFrame:
    rows = []
    for item, g in fc.sort_values("h").groupby("item_id"):
        s = stock.set_index("item_id").loc[item]
        dec = policy(
            s.on_hand,
            s.on_order,
            g.yhat.clip(lower=0).tolist(),
            g.lower.clip(lower=0).tolist(),
            g.upper.clip(lower=0).tolist(),
            inv,
        )
        rows.append(
            {
                "item_id": item,
                "on_hand": s.on_hand,
                "on_order": s.on_order,
                "action": dec.action,
                "order_qty": dec.qty,
                "reorder_point": dec.s,
                "order_up_to": dec.S,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg, inv = load_experiment(), load_inventory()
    if cfg.deliberation.tau is None:
        raise RuntimeError("tau is not frozen; run the experiment first")
    origin = cfg.data.test_end
    last = origin + max(cfg.agents.horizons)
    df = feature_frame(cfg, last)
    af, lgbm, actual = agent_table(cfg, df, origin)
    rs, pts = retailsense(cfg, af, df, origin)
    print("ETS ...", flush=True)
    sales = actual.rename(columns={"y": "sales"})
    e = ets(sales, origin_grid([origin], cfg.agents.horizons, last), cfg).astype({"item_id": str})
    fcs = {"retailsense": rs, "lgbm_all_features": lgbm, "ets": e}

    dates = df.drop_duplicates("d").set_index("d").date
    items = sorted(af.item_id.unique())
    out = []
    for name, fc in fcs.items():
        fc = fc[["item_id", "h", "yhat", "lower", "upper"]].astype({"item_id": str})
        dec = decide(fc, start_stock(cfg, name, items, origin), inv)
        out.append(
            fc.assign(system=name, date=(origin + fc.h).map(dates).dt.strftime("%Y-%m-%d"))
            .merge(dec, on="item_id")
            .merge(pts, on="item_id")
        )
    res = pd.concat(out, ignore_index=True).rename(columns={"yhat": "forecast"})
    res.insert(1, "made_on", f"{dates[origin]:%Y-%m-%d}")
    res = res[
        [
            "system",
            "made_on",
            "item_id",
            "date",
            "h",
            "forecast",
            "lower",
            "upper",
            "on_hand",
            "on_order",
            "action",
            "order_qty",
            "reorder_point",
            "order_up_to",
            "D",
            "consensus",
            "deliberated",
        ]
    ].sort_values(["system", "item_id", "h"])
    path = ROOT / "reports" / cfg.dataset / "future_forecast.csv"
    res.to_csv(path, index=False, float_format="%.3f")
    print(f"wrote {path}: {res.item_id.nunique()} items x {res.h.nunique()} days x 3 systems")
    summary = res[res.h == 1].pivot(index="item_id", columns="system", values="order_qty")
    print("units to order now:", summary.sum().to_dict())
    print(f"RetailSense deliberated {int(pts.deliberated.sum())} of {len(pts)} items")


if __name__ == "__main__":
    main()
