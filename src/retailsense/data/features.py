"""Long feature table (blueprint 5.1-5.3).

One row per (item, target day t, horizon h); one model per horizon. The prediction time is
p = t - h, and every sales- and price-derived value comes from data at or before p. Calendar
features are scheduled in advance; weather is observed weather on day t used as a perfect
forecast (blueprint 4.2 limitation). Those are the exceptions. `available_at` is the latest
availability date of any other feature on the row (sales, price, search interest).

Run: python -m retailsense.data.features
"""

import numpy as np
import pandas as pd

from retailsense.config import ExperimentCfg, load_experiment
from retailsense.data.load import (
    day_cols,
    load_calendar,
    load_prices,
    load_sales,
    load_subset_items,
)

ROLL_WINDOWS = (7, 28, 56)
SALES_FEATURES = [
    "lag_h",
    "lag_7",
    "lag_14",
    "lag_28",
    *[f"roll_mean_{w}" for w in ROLL_WINDOWS],
    *[f"roll_std_{w}" for w in ROLL_WINDOWS],
    "zero_share_28",
]
PRICE_FEATURES = ["price", "price_change_1w", "price_rel_52w_max"]
ITEM_FEATURES = ["item_id"]
CALENDAR_FEATURES = [
    "wday",
    "month",
    "weekofyear",
    "mday",
    "event_type",
    "days_to_event",
    "days_since_event",
    "snap",
]
EVENT_TYPES = ["Cultural", "National", "Religious", "Sporting"]


def to_long(wide: pd.DataFrame, last_day: int) -> pd.DataFrame:
    long = wide.melt(
        id_vars="item_id", value_vars=day_cols(1, last_day), var_name="d", value_name="sales"
    )
    long["d"] = long.d.str.removeprefix("d_").astype(int)
    return long


def calendar_features(calendar: pd.DataFrame, state: str) -> pd.DataFrame:
    cal = calendar.assign(d=calendar.d.str.removeprefix("d_").astype(int))
    out = pd.DataFrame(
        {
            "d": cal.d,
            "date": cal.date,
            "wday": cal.wday,
            "month": cal.month,
            "weekofyear": cal.date.dt.isocalendar().week.astype(int).to_numpy(),
            "mday": cal.date.dt.day,
            "event_type": pd.Categorical(cal.event_type_1, categories=EVENT_TYPES),
            "snap": cal[f"snap_{state}"],
        }
    )
    ev = np.sort(cal.date[cal.event_name_1.notna()].to_numpy())
    dates = cal.date.to_numpy()
    nxt = np.searchsorted(ev, dates, side="left")  # first event on/after the day
    prv = nxt - 1  # last event strictly before the day
    day = np.timedelta64(1, "D")
    out["days_to_event"] = np.where(
        nxt < len(ev), (ev[np.minimum(nxt, len(ev) - 1)] - dates) / day, np.nan
    )
    out["days_since_event"] = np.where(prv >= 0, (dates - ev[np.maximum(prv, 0)]) / day, np.nan)
    return out


def weekly_price_features(prices: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """One row per (item, closed week). `week_end` is the last day of the (Sat-Fri) week."""
    week_end = calendar.groupby("wm_yr_wk").date.max().rename("week_end")
    p = prices.merge(week_end, left_on="wm_yr_wk", right_index=True)
    p = p.sort_values(["item_id", "week_end"]).reset_index(drop=True)
    g = p.groupby("item_id").sell_price
    p["price_change_1w"] = p.sell_price / g.shift(1) - 1
    p["price_rel_52w_max"] = p.sell_price / g.transform(lambda s: s.rolling(52, 1).max())
    return p.rename(columns={"sell_price": "price"})[
        ["item_id", "week_end", *PRICE_FEATURES]
    ]


def build_features(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    state: str,
    last_day: int,
    items: list[str],
    horizons: list[int],
) -> pd.DataFrame:
    """sales: long (item_id, d, sales). Rows exist for every item and day 1..last_day;
    sales after the data cut-off are simply NaN, so truncated inputs give the same features."""
    cal = calendar_features(calendar, state)
    cal = cal[cal.d <= last_day]
    grid = pd.MultiIndex.from_product([items, cal.d], names=["item_id", "d"]).to_frame(
        index=False
    )
    df = grid.merge(sales, on=["item_id", "d"], how="left").merge(cal, on="d")
    df = df.sort_values(["item_id", "d"]).reset_index(drop=True)
    wk = weekly_price_features(prices, calendar).rename(columns={"week_end": "price_available_at"})
    df = pd.concat([_horizon_block(df, wk, h) for h in horizons], ignore_index=True)
    df["item_id"] = pd.Categorical(df.item_id, categories=items)
    return df


def _horizon_block(df: pd.DataFrame, wk: pd.DataFrame, h: int) -> pd.DataFrame:
    """Sales/price features for horizon h: every source date is <= t - h (the prediction time)."""
    df = df.assign(h=h)
    g = df.groupby("item_id", sort=False).sales
    df["lag_h"] = g.shift(h)
    for lag in (7, 14, 28):
        df[f"lag_{lag}"] = g.shift(lag)
    shifted = df.groupby("item_id", sort=False).lag_h  # rolling windows only on the lag-h series
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = shifted.transform(lambda s, w=w: s.rolling(w, w).mean())
        df[f"roll_std_{w}"] = shifted.transform(lambda s, w=w: s.rolling(w, w).std())
    is_zero = df.lag_h.eq(0).where(df.lag_h.notna()).astype(float)
    df["zero_share_28"] = is_zero.groupby(df.item_id).transform(lambda s: s.rolling(28, 28).mean())
    df["launched"] = df.lag_h.fillna(0).groupby(df.item_id).cumsum() > 0

    df["sales_available_at"] = df.date - pd.Timedelta(days=h)
    df = pd.merge_asof(
        df.sort_values("sales_available_at"),
        wk.sort_values("price_available_at"),
        left_on="sales_available_at",
        right_on="price_available_at",
        by="item_id",
        direction="backward",
    )
    df = df.sort_values(["item_id", "d"]).reset_index(drop=True)
    df["available_at"] = df[["sales_available_at", "price_available_at"]].max(axis=1)
    return df


def join_context(
    df: pd.DataFrame, weather: pd.DataFrame | None, interest: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Weather joins on the TARGET date (observed weather as a perfect forecast, blueprint 4.2).
    Each interest signal joins as of the prediction date t - h: the latest row whose
    available_at <= t - h. Its availability is folded into the row's available_at."""
    if weather is not None:
        df = df.merge(weather, on="date", how="left")
    df = df.assign(pred_date=df.date - pd.to_timedelta(df.h, unit="D")).sort_values("pred_date")
    for name, sig in interest.items():
        sig = sig.rename(columns={"available_at": f"{name}_available_at"})
        df = pd.merge_asof(
            df,
            sig.drop(columns="week_start", errors="ignore").sort_values(f"{name}_available_at"),
            left_on="pred_date",
            right_on=f"{name}_available_at",
            direction="backward",
        )
        df["available_at"] = df[["available_at", f"{name}_available_at"]].max(axis=1)
    return df.drop(columns="pred_date").sort_values(["item_id", "h", "d"]).reset_index(drop=True)


def add_split(df: pd.DataFrame, cfg: ExperimentCfg) -> pd.DataFrame:
    d = cfg.data
    df["split"] = pd.cut(
        df.d, [0, d.train_end, d.val_end, d.test_end], labels=["train", "val", "test"]
    ).astype(str)
    return df


def main() -> None:
    cfg = load_experiment()
    items = load_subset_items(cfg)
    wide = load_sales(cfg)
    wide = wide[wide.item_id.isin(items)]
    df = build_features(
        to_long(wide, cfg.data.test_end),
        load_calendar(cfg),
        load_prices(cfg, items),
        cfg.data.state,
        cfg.data.test_end,
        items,
        cfg.agents.horizons,
    )
    proc = cfg.path("processed")
    signals = {n: proc / f"{n}.parquet" for n in ("weather", "trends")}
    for n, p in signals.items():
        print(f"{n}: {'joined' if p.exists() else 'MISSING - columns absent'}")
    df = join_context(
        df,
        pd.read_parquet(signals["weather"]) if signals["weather"].exists() else None,
        {n: pd.read_parquet(p) for n, p in signals.items() if n != "weather" and p.exists()},
    )
    df = add_split(df, cfg)
    out = cfg.path("processed") / "features.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df):,} rows, {df.item_id.nunique()} items")
    print(df[df.h == df.h.min()].groupby("split").size().to_string())


if __name__ == "__main__":
    main()
