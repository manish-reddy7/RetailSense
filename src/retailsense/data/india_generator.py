"""India kirana-chain dataset generator (simulation study, fixed seed).

Writes the shared raw file layout (sales_train_evaluation.csv, calendar.csv, sell_prices.csv) so
the same pipeline runs unchanged, plus Trends-format interest CSVs, ground_truth.parquet (every
demand multiplier per row: which driver moved demand on which day) and a README describing how the
data was generated.
Weather is REAL (Open-Meteo archive for each store's city); festivals are the real Indian calendar.

Run: python -m retailsense.data.india_generator
"""

import holidays
import numpy as np
import pandas as pd
import yaml

from retailsense.config import ROOT, load_experiment
from retailsense.data.weather import fetch, weather_features

WEEKDAYS = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"]  # wday 1 = Saturday
NATIONAL = ("Republic Day", "Independence Day", "Gandhi")


def build_calendar(g: dict, dates: pd.DatetimeIndex, states: list[str]) -> pd.DataFrame:
    hol = holidays.India(years=range(dates[0].year, dates[-1].year + 1))
    names = [hol.get(d.date(), "").split(";")[0] or None for d in dates]
    fest = set(g["festivals"]["names"])
    kind = [
        None if n is None
        else "Religious" if any(f in n for f in fest)
        else "National" if any(k in n for k in NATIONAL)
        else "Cultural"
        for n in names
    ]
    i = np.arange(len(dates))
    cal = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "wm_yr_wk": 11901 + i // 7,
        "weekday": dates.day_name(),
        "wday": i % 7 + 1,
        "month": dates.month,
        "year": dates.year,
        "d": [f"d_{k + 1}" for k in i],
        "event_name_1": names,
        "event_type_1": kind,
        "event_name_2": None,
        "event_type_2": None,
    })
    for s in states:
        cal[f"snap_{s}"] = 0  # no SNAP programme in India; column kept for the shared file layout
    return cal


def festival_mult(g: dict, dates: pd.DatetimeIndex, names: list, affinity: dict) -> np.ndarray:
    f = g["festivals"]
    m = np.ones(len(dates))
    for t, n in enumerate(names):
        key = next((v for k, v in f["names"].items() if isinstance(n, str) and k in n), None)
        if key is None:
            continue
        m[t] *= f["day_of_mult"]
        a = affinity.get(key, 0.0)
        for k in range(1, f["pre_days"] + 1):  # ramp up to the day before the festival
            if t - k >= 0:
                m[t - k] *= 1 + a * (f["pre_days"] - k + 1) / f["pre_days"]
    return m


def shock_series(g: dict, rng, n: int) -> np.ndarray:
    s = g["shocks"]
    m = np.ones(n)
    for _ in range(int(s["per_year"] * n / 365)):
        start = rng.integers(0, n)
        length = rng.integers(s["length"][0], s["length"][1] + 1)
        m[start : start + length] *= rng.uniform(*s["mult"])
    return m


def trends_csvs(g: dict, keyword: str, daily: pd.Series, out) -> None:
    """Weekly (Sun-Sat) Google-style CSVs, each window scaled 0-100 to its own maximum."""
    weekly = daily.resample("W-SAT").mean()
    weekly.index = weekly.index - pd.Timedelta(days=6)  # label by week start (Sunday)
    for start, end in g["trends"]["windows"]:
        w = weekly[(weekly.index >= start) & (weekly.index + pd.Timedelta(days=6) <= end)]
        v = (100 * w / w.max()).round()
        rows = "".join(f"{d:%Y-%m-%d},{'<1' if x < 1 else int(x)}\n" for d, x in v.items())
        path = out / f"{keyword.replace(' ', '_')}__{start}_{end}.csv"
        path.write_text(
            f"Category: All categories\n\nWeek,{keyword}: (India)\n{rows}", encoding="utf-8"
        )


def generate(g: dict) -> None:
    exp = load_experiment(ROOT / "configs/experiment_india.yaml")
    rng = np.random.default_rng(g["seed"])
    out = ROOT / g["out_dir"]
    (out / "trends").mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(g["start"], periods=g["n_days"], freq="D")
    n, t_years = len(dates), np.arange(g["n_days"]) / 365
    states = sorted({s for s, _ in g["stores"].values()})
    cal = build_calendar(g, dates, states)
    wx = {
        s: weather_features(fetch(exp, s), exp).set_index("date").reindex(dates)
        for s in states
    }

    # interest shocks per keyword over the whole Trends span (so window A has history too)
    span = pd.date_range(g["trends"]["windows"][0][0], dates[-1], freq="D")
    shocks = {kw: pd.Series(shock_series(g, rng, len(span)), index=span)
              for kw in {a[6] for a in g["archetypes"].values() if a[6]}}
    lead = g["shocks"]["lead_days"]
    for kw, s in shocks.items():
        interest = g["trends"]["base"] * s.shift(-lead).fillna(1.0)  # interest leads demand
        interest *= np.exp(rng.normal(0, g["trends"]["noise"], len(s)))
        trends_csvs(g, kw, interest, out / "trends")

    w, sales, prices, truth = g["weather"], [], [], []
    wd = np.array([g["weekday"][WEEKDAYS[k % 7]] for k in range(n)])
    payday = np.isin(dates.day, g["payday"]["days"])
    for name, (cat, base, price, heat, rain, aff, kw) in g["archetypes"].items():
        fest = festival_mult(g, dates, cal.event_name_1.tolist(), aff)
        for v in range(g["variants"]):
            item = f"{name}_{v + 1}"
            scale, p0 = rng.uniform(*g["variant_scale"]), price * rng.uniform(0.85, 1.15)
            for store, (state, s_scale) in g["stores"].items():
                x = wx[state]
                weather = np.exp(
                    heat * (w["heat_level"] * (x.tmax - w["heat_ref"])
                            + w["heat_anom"] * x.tmax_anom.fillna(0))
                    + w["rain"] * rain * x.rainy_day.fillna(0)
                ).to_numpy() * np.where(x.precip >= w["heavy_rain_mm"], w["heavy_rain_footfall"], 1)
                weeks = n // 7 + 1
                promo_w = rng.random(weeks) < g["promo"]["week_prob"]
                wprice = (p0 * (1 + g["inflation_per_year"]) ** (np.arange(weeks) * 7 / 365)
                          * (1 + rng.normal(0, g["price_noise"], weeks))
                          * np.where(promo_w, 1 - g["promo"]["discount"], 1))
                promo = np.repeat(np.where(promo_w, 1 + g["promo"]["elasticity"]
                                           * g["promo"]["discount"], 1.0), 7)[:n]
                pay = np.where(payday & (cat in g["payday"]["cats"]), g["payday"]["mult"], 1.0)
                shock = shocks[kw].reindex(dates).to_numpy() if kw else np.ones(n)
                trend = (1 + g["growth_per_year"]) ** t_years
                lam = base * scale * s_scale * wd * trend * weather * fest * pay * promo * shock
                k = g["dispersion"]
                y = rng.poisson(rng.gamma(k, lam / k))
                so = np.zeros(n, bool)
                for t in np.flatnonzero(rng.random(n) < g["stockout"]["start_prob"]):
                    so[t : t + rng.integers(g["stockout"]["length"][0],
                                            g["stockout"]["length"][1] + 1)] = True
                y = np.where(so, 0, y)
                sales.append([f"{item}_{store}_evaluation", item, g["dept"], cat, store, state,
                              *y.tolist()])
                prices += [(store, item, 11901 + k_, round(float(p), 2))
                           for k_, p in enumerate(wprice) if k_ * 7 < n]
                truth.append(pd.DataFrame({
                    "store_id": store, "item_id": item, "d": np.arange(1, n + 1), "lam": lam,
                    "weekday": wd, "trend": trend, "weather": weather, "festival": fest,
                    "payday": pay, "promo": promo, "shock": shock, "stockout": so,
                }))

    cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    pd.DataFrame(sales, columns=cols + [f"d_{k}" for k in range(1, n + 1)]).to_csv(
        out / "sales_train_evaluation.csv", index=False
    )
    cal.to_csv(out / "calendar.csv", index=False)
    pd.DataFrame(prices, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"]).to_csv(
        out / "sell_prices.csv", index=False
    )
    pd.concat(truth, ignore_index=True).to_parquet(out / "ground_truth.parquet", index=False)
    (out / "README.md").write_text(README.format(start=dates[0].date(), end=dates[-1].date(),
                                                 n_items=len(sales) // len(g["stores"]),
                                                 stores=", ".join(g["stores"])), encoding="utf-8")
    span = f"{dates[0].date()}..{dates[-1].date()}"
    print(f"wrote {out}: {len(sales)} item-store series x {n} days ({span})")


README = """# India kirana-chain dataset (generated)

Generated by `python -m retailsense.data.india_generator` from `configs/india_generator.yaml`
(fixed seed) for a controlled simulation study.

- Stores: {stores}; {n_items} items; daily {start} .. {end} (d_ day indices).
- Weather is real (Open-Meteo archive, one city per store); festivals are the real Indian
  holiday calendar (`holidays` library). Sales, prices and the Trends index are generated.
- `ground_truth.parquet` logs every demand multiplier (weekday, trend, weather, festival, payday,
  promo, interest shock, stock-out) per store, item and day, so we know which evidence source
  actually drove demand on each day.
- `trends/` holds a generated interest index per keyword, in Google Trends CSV format, that
  rises a few days before the matching demand shocks.
"""


def main() -> None:
    generate(yaml.safe_load((ROOT / "configs/india_generator.yaml").read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
