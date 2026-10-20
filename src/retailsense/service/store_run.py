"""Retailer run: a store's uploaded sales -> RetailSense forecast and restocking decision for the
next 7 days. The forecasting system is the experiment's, unchanged: Historical, Calendar and
Weather agents, D against tau, LLM critic + one bounded revision, (s, S) policy. What live use
forces to differ:
- tau = the tau_quantile of D over the store's last `val_days` days, from agents fitted on the
  days before them (the experiment's validation procedure, run on the retailer's own data).
- Weather: Open-Meteo archive (observed) for history; days the archive does not reach yet come
  from the Open-Meteo forecast API, i.e. the forecast issued today (blueprint 1.5 live mode).
- Holidays: the `holidays` library for the store's country, every holiday typed National.
- Trend agent only when the store uploaded a Google Trends download (trends.csv); its search
  terms replace the experiment's keywords. Without one the run has three agents.

Input  <run_dir>/store.json, sales.csv, stock.csv, trends.csv (copied there when queued)
Output <run_dir>/result.json; raw/ and processed/ hold the intermediate files.
A data problem the retailer must fix (DataError) exits with a line starting "ERROR:".

Run: python -m retailsense.service.store_run <run_dir>
"""

import csv
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import holidays
import pandas as pd

from retailsense.config import (
    DataCfg,
    ExperimentCfg,
    InterestCfg,
    Paths,
    ServiceCfg,
    load_experiment,
    load_inventory,
    load_service,
)
from retailsense.data import weather as wx
from retailsense.data.features import add_split, build_features, join_context, to_long
from retailsense.data.load import load_calendar, load_prices, load_sales
from retailsense.data.trends import WEEKS, trends_table
from retailsense.eval.run_experiment import run_validation
from retailsense.forecast_future import agent_table, decide, retailsense
from retailsense.orchestrator.disagreement import score_table
from retailsense.web_export import _clean

SALES_COLS = ["date", "item_id", "units_sold", "price"]
STOCK_COLS = ["item_id", "on_hand", "on_order"]
STATE = "STORE"  # the weather city key and calendar column for the one store of a run
HISTORY_DAYS = 28  # recent sales shown next to each item's forecast
TRENDS_MIN_WEEKS = WEEKS  # the Trend agent's 52-week features need this much history


class DataError(ValueError):
    """A problem in the retailer's files, shown to them as is."""


def _bad_rows(mask: pd.Series, what: str) -> None:
    if mask.any():
        rows = ", ".join(str(i + 2) for i in mask[mask].index[:5])  # +2: header, 1-based
        raise DataError(f"{what} (CSV rows {rows}{' ...' if mask.sum() > 5 else ''})")


def _read(src, cols: list[str], text: list[str]) -> pd.DataFrame:
    try:
        df = pd.read_csv(src, dtype={c: str for c in text})
    except Exception as e:  # noqa: BLE001 - any parse failure is the retailer's file
        raise DataError(f"not a readable CSV file: {e}") from None
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataError(f"missing columns {missing}; the file needs {cols}")
    return df[cols].reset_index(drop=True)


def read_sales(src, svc: ServiceCfg) -> pd.DataFrame:
    """date, item_id, units_sold, price: one row per item for every day it was on sale, 0 when
    it sold nothing. A day with no row means not on sale, and stays missing (never zero)."""
    df = _read(src, SALES_COLS, ["item_id"])
    df["item_id"] = df.item_id.str.strip()
    df["date"] = pd.to_datetime(df.date, format="%Y-%m-%d", errors="coerce")
    for c in ("units_sold", "price"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    _bad_rows(df.isna().any(axis=1), "empty or unreadable values; dates must be YYYY-MM-DD")
    _bad_rows(df.units_sold < 0, "negative units_sold")
    _bad_rows(df.price <= 0, "price must be above 0")
    _bad_rows(df.duplicated(["date", "item_id"]), "more than one row for the same item and date")
    _bad_rows(df.date.dt.date > date.today(), "dates in the future")
    days = (df.date.max() - df.date.min()).days + 1
    if days < svc.min_days:
        raise DataError(f"{days} days of sales; at least {svc.min_days} are needed")
    if df.item_id.nunique() > svc.max_items:
        raise DataError(f"{df.item_id.nunique()} items; at most {svc.max_items} per store")
    return df.sort_values(["item_id", "date"]).reset_index(drop=True)


def read_stock(src) -> pd.DataFrame:
    """item_id, on_hand, on_order: the stock position at the end of the last sales day."""
    df = _read(src, STOCK_COLS, ["item_id"])
    df["item_id"] = df.item_id.str.strip()
    for c in ("on_hand", "on_order"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    _bad_rows(df.isna().any(axis=1), "empty or unreadable values")
    _bad_rows((df.on_hand < 0) | (df.on_order < 0), "negative stock")
    _bad_rows(df.duplicated("item_id"), "the same item twice")
    return df


def read_trends(src) -> pd.DataFrame:
    """A Google Trends "Interest over time" download (trends.google.com, up to 5 search terms):
    `Week,<term>: (<region>),...` then one row per week. Returns `date` (week start) plus one
    column per term, named by a safe key; values stay as Google wrote them ("<1" included).
    The Trend agent's features need 52 weeks of history, so a shorter file is refused."""
    raw = src.read() if hasattr(src, "read") else Path(src).read_bytes()
    try:
        lines = raw.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError:
        raise DataError("not a text CSV file") from None
    rows = list(csv.reader(lines))
    periods = ("Week", "Month", "Day")
    head = next((i for i, r in enumerate(rows) if r and r[0].strip() in periods), None)
    if head is None:
        raise DataError("not a Google Trends download: no 'Week' header line")
    if rows[head][0].strip() != "Week":
        raise DataError("the Trends file must be weekly: download a window shorter than 5 years")
    terms = [c.split(":")[0].strip() for c in rows[head][1:]]
    keys = [re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_") for t in terms]
    if not terms or not all(keys) or len(set(keys)) != len(keys):
        raise DataError("search terms must be distinct and contain letters or digits")
    body = [r for r in rows[head + 1 :] if r and r[0].strip()]
    df = pd.DataFrame([r[: len(keys) + 1] for r in body], columns=["date", *keys])
    df["date"] = pd.to_datetime(df.date.str.strip(), format="%Y-%m-%d", errors="coerce")
    vals = df[keys].apply(lambda c: pd.to_numeric(c.str.strip().replace("<1", "0.5")))
    _bad_rows(df.date.isna() | vals.isna().any(axis=1), "unreadable week or value")
    _bad_rows(df.date.diff().dropna() != pd.Timedelta(days=7), "weeks must be 7 days apart")
    _bad_rows(df.date.dt.date > date.today(), "weeks in the future")
    if len(df) < TRENDS_MIN_WEEKS:
        raise DataError(f"{len(df)} weeks of search interest; at least {TRENDS_MIN_WEEKS} needed")
    df[keys] = df[keys].apply(lambda c: c.str.strip())
    df.attrs["terms"] = dict(zip(keys, terms, strict=True))
    return df


def write_trends(trends: pd.DataFrame, out: Path) -> None:
    """One file per term in the layout data/trends.py reads (<key>__<first>_<last>.csv)."""
    out.mkdir(parents=True, exist_ok=True)
    span = f"{trends.date.min():%Y-%m-%d}_{trends.date.max():%Y-%m-%d}"
    for key in trends.columns.drop("date"):
        rows = [f"{d:%Y-%m-%d},{v}" for d, v in zip(trends.date, trends[key], strict=True)]
        (out / f"{key}__{span}.csv").write_text(
            "\n".join(["Category: All categories", "", f"Week,{key}: (upload)", *rows]) + "\n",
            encoding="utf-8",
        )


def write_raw(sales: pd.DataFrame, store: dict, out: Path, horizon: int) -> tuple[int, date]:
    """The raw file layout the pipeline reads (as data/rohlik.py writes it). d_1 is the Saturday
    on or before the first sale (weeks run Sat-Fri); days n+1..n+horizon are the future, with
    no sales. Only items sold on the last day are forecast. Returns (n, date of d_1)."""
    out.mkdir(parents=True, exist_ok=True)
    first, last = sales.date.min(), sales.date.max()
    d0 = first - pd.Timedelta(days=(first.weekday() - 5) % 7)
    n, total = (last - d0).days + 1, (last - d0).days + 1 + horizon
    sales = sales[sales.item_id.isin(sales.item_id[sales.date == last])]
    sales = sales.assign(d=(sales.date - d0).dt.days + 1)
    wide = sales.pivot(index="item_id", columns="d", values="units_sold")
    wide = wide.reindex(columns=range(1, total + 1))
    wide.columns = [f"d_{c}" for c in wide.columns]
    ids = pd.DataFrame(
        {
            "id": [f"{i}_{store['code']}" for i in wide.index],
            "item_id": wide.index,
            "dept_id": "ALL",
            "cat_id": "ALL",
            "store_id": store["code"],
            "state_id": STATE,
        }
    )
    pd.concat([ids, wide.reset_index(drop=True)], axis=1).to_csv(
        out / "sales_train_evaluation.csv", index=False
    )

    dates = pd.date_range(d0, periods=total)
    hol = holidays.country_holidays(store["country"], years=range(d0.year, dates[-1].year + 1))
    names = [hol.get(x.date()) for x in dates]
    k = range(total)
    pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": [11001 + i // 7 for i in k],
            "weekday": dates.day_name(),
            "wday": [i % 7 + 1 for i in k],
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{i + 1}" for i in k],
            "event_name_1": names,
            # ponytail: every public holiday typed National; map religious/cultural per country
            # if the Calendar agent needs the distinction
            "event_type_1": ["National" if x else None for x in names],
            "event_name_2": None,
            "event_type_2": None,
            f"snap_{STATE}": 0,
        }
    ).to_csv(out / "calendar.csv", index=False)

    p = sales.assign(wm_yr_wk=11001 + (sales.d - 1) // 7)
    p = p.groupby(["item_id", "wm_yr_wk"], as_index=False).price.mean()
    p.insert(0, "store_id", store["code"])
    p.rename(columns={"price": "sell_price"}).to_csv(out / "sell_prices.csv", index=False)
    return n, d0.date()


def store_config(svc: ServiceCfg, store: dict, run: Path, n: int, d0: date) -> ExperimentCfg:
    tpl = load_experiment(svc.template)
    v, horizon = svc.val_days, max(tpl.agents.horizons)
    clim_start = pd.Timestamp(d0) - pd.DateOffset(years=tpl.weather.clim_years)
    return tpl.model_copy(
        update={
            "dataset": f"store_{store['id']}",
            "paths": Paths(
                raw=run / "raw",
                processed=run / "processed",
                cache=tpl.paths.cache,
                db=run / "run.db",
                subset=run / "subset.yaml",
            ),
            "data": DataCfg(
                store=store["code"],
                state=STATE,
                dept="ALL",
                top_n=0,
                rank_end=n - v,
                train_end=n - v,
                val_end=n,
                test_end=n + horizon,
            ),
            "weather": tpl.weather.model_copy(
                update={
                    "city": {STATE: (store["lat"], store["lon"])},
                    "timezone": "auto",
                    "fetch_start": f"{clim_start:%Y-%m-%d}",
                }
            ),
            "deliberation": tpl.deliberation.model_copy(update={**svc.llm, "tau": None}),
            "final": None,
            "source": None,
        }
    )


def weather(cfg: ExperimentCfg, svc: ServiceCfg, end: date) -> pd.DataFrame:
    """Archive observations up to `archive_lag_days` ago, the forecast issued today after that.
    Days neither reaches stay missing."""
    today = date.today()
    a_end = min(end, today - timedelta(days=svc.archive_lag_days))

    def get(start: date, stop: date, url: str) -> pd.DataFrame:
        w = cfg.weather.model_copy(update={"fetch_start": str(start), "fetch_end": str(stop)})
        return wx.fetch(cfg.model_copy(update={"weather": w}), url=url).set_index("date")

    obs = get(date.fromisoformat(cfg.weather.fetch_start), a_end, wx.URL)
    f_end = min(end, today + timedelta(days=svc.forecast_days))
    if f_end > a_end:
        obs = obs.combine_first(get(a_end + timedelta(days=1), f_end, wx.FORECAST_URL))
    return wx.weather_features(obs.reset_index(), cfg)


def with_trends(cfg: ExperimentCfg, trends: pd.DataFrame, run: Path) -> ExperimentCfg:
    """The store's own search terms in place of the experiment's keywords."""
    write_trends(trends, run / "raw" / "trends")
    interest = InterestCfg(
        trends_dir=run / "raw" / "trends",
        trends_geo="upload",
        keywords={k: k for k in trends.columns.drop("date")},
        trends_lt1=cfg.interest.trends_lt1,
    )
    return cfg.model_copy(update={"interest": interest})


def features(cfg: ExperimentCfg, svc: ServiceCfg, d0: date, trends: bool) -> pd.DataFrame:
    """features.parquet for the store, as data/features.py builds it; search interest joins as
    of the prediction date only when the store uploaded it."""
    wide = load_sales(cfg)
    items = wide.item_id.tolist()
    d = cfg.data
    df = build_features(
        to_long(wide, d.test_end),
        load_calendar(cfg),
        load_prices(cfg, items),
        d.state,
        d.test_end,
        items,
        cfg.agents.horizons,
    )
    end = d0 + timedelta(days=d.test_end - 1)
    interest = {"trends": trends_table(cfg)} if trends else {}
    df = add_split(join_context(df, weather(cfg, svc, end), interest), cfg)
    cfg.path("processed").mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.path("processed") / "features.parquet", index=False)
    return df


def calibrate_tau(cfg: ExperimentCfg) -> float:
    """Fit on the days before the validation window, forecast every validation day, and take
    the tau_quantile of D (blueprint 7.2)."""
    run_validation(cfg, phase=2, log=False, tag="calib")
    af = pd.read_parquet(cfg.path("processed") / "calib_agent_forecasts.parquet")
    h1 = af[af.h == 1].assign(d=lambda x: x.origin + 1)
    h1 = h1[(h1.d > cfg.data.train_end) & (h1.d <= cfg.data.val_end)]
    y = h1.pivot(index=["item_id", "d"], columns="agent", values="forecast")
    c = h1.pivot(index=["item_id", "d"], columns="agent", values="confidence")
    pts = score_table(y, c, cfg.disagreement.floor)
    return float(pts.D.quantile(cfg.disagreement.tau_quantile))


def _critic(state: dict) -> dict | None:
    c = state.get("critic")
    return c.model_dump() if c is not None else None


def result(
    cfg: ExperimentCfg,
    df: pd.DataFrame,
    af: pd.DataFrame,
    fc: pd.DataFrame,
    pts: pd.DataFrame,
    states: list[dict],
    dec: pd.DataFrame,
    tau: float,
    origin: int,
) -> dict:
    dates = df.drop_duplicates("d").set_index("d").date
    hist = df[df.h == 1].astype({"item_id": str}).set_index(["item_id", "d"]).sales
    by_state = {s["case"]["item_id"]: s for s in states}
    dec = dec.set_index("item_id")
    items = []
    for p in pts.itertuples():
        st = by_state[p.item_id]
        f = fc[fc.item_id == p.item_id].sort_values("h")
        a = af[(af.item_id == p.item_id) & (af.h == 1)]
        past = hist.loc[p.item_id].loc[origin - HISTORY_DAYS + 1 : origin]
        d = dec.loc[p.item_id].to_dict() if p.item_id in dec.index else None
        items.append(
            {
                "item_id": p.item_id,
                "D": p.D,
                "consensus": p.consensus,
                "deliberated": bool(p.deliberated),
                "fallback": st.get("fallback"),
                "decision": d,
                "forecast": [
                    {
                        "date": f"{dates[origin + r.h]:%Y-%m-%d}",
                        "h": int(r.h),
                        "yhat": r.yhat,
                        "lower": r.lower,
                        "upper": r.upper,
                    }
                    for r in f.itertuples()
                ],
                "agents": [
                    {
                        "agent": r.agent,
                        "forecast": r.forecast,
                        "lower": r.lower,
                        "upper": r.upper,
                        "confidence": r.confidence,
                        "key_factors": json.loads(r.key_factors),
                    }
                    for r in a.itertuples()
                ],
                "critic": _critic(st),
                "revisions": st.get("revisions") or [],
                "history": [
                    {"date": f"{dates[k]:%Y-%m-%d}", "units": None if pd.isna(v) else float(v)}
                    for k, v in past.items()
                ],
            }
        )
    d = cfg.deliberation
    return {
        "made_on": f"{dates[origin]:%Y-%m-%d}",
        "tau": tau,
        "agents": sorted(af.agent.unique()),
        "llm": {
            "provider": d.provider,
            "model": d.model,
            "calls": int(sum(s["llm_calls"] for s in states)),
        },
        "items": items,
    }


def main(run: Path) -> None:
    svc = load_service()
    store = json.loads((run / "store.json").read_text(encoding="utf-8"))
    sales = read_sales(run / "sales.csv", svc)
    stock = read_stock(run / "stock.csv") if (run / "stock.csv").exists() else None
    tpl_h = max(load_experiment(svc.template).agents.horizons)
    n, d0 = write_raw(sales, store, run / "raw", tpl_h)
    cfg = store_config(svc, store, run, n, d0)
    trends = read_trends(run / "trends.csv") if (run / "trends.csv").exists() else None
    if trends is not None:
        if trends.date.max() < sales.date.min() or trends.date.min() > sales.date.max():
            raise DataError("the search-interest weeks do not overlap the sales dates")
        cfg = with_trends(cfg, trends, run)
        print(f"search interest: {', '.join(trends.attrs['terms'].values())}", flush=True)
    print(f"{n} days from {d0}; building features ...", flush=True)
    df = features(cfg, svc, d0, trends is not None)
    print("calibrating tau on the last validation days ...", flush=True)
    tau = calibrate_tau(cfg)
    print(f"tau = {tau:.4f}; fitting every agent on all days ...", flush=True)
    af, _, _ = agent_table(cfg, df, n)
    cfg = cfg.model_copy(update={"deliberation": cfg.deliberation.model_copy(update={"tau": tau})})
    fc, pts, states = retailsense(cfg, af, df, n)
    inv = load_inventory().model_copy(update=store["inventory"])  # validated by the API
    dec = pd.DataFrame(columns=["item_id"])
    if stock is not None:
        have = fc[fc.item_id.isin(stock.item_id)]
        dec = decide(have[["item_id", "h", "yhat", "lower", "upper"]], stock, inv)
    out = result(cfg, df, af, fc, pts, states, dec, tau, n)
    (run / "result.json").write_text(json.dumps(_clean(out), allow_nan=False), encoding="utf-8")
    print(f"done: {len(out['items'])} items, {int(pts.deliberated.sum())} deliberated", flush=True)


if __name__ == "__main__":
    try:
        main(Path(sys.argv[1]))
    except DataError as e:
        print(f"ERROR: {e}", flush=True)
        sys.exit(1)
