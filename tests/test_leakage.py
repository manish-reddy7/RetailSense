"""Leakage suite. The project's defence: if any of these fail, nothing downstream is valid."""

import numpy as np
import pandas as pd
import pytest

from retailsense.agents.confidence import trailing_confidence
from retailsense.agents.historical import expanding_blocks, expanding_oof
from retailsense.config import load_experiment
from retailsense.data.features import (
    CALENDAR_FEATURES,
    PRICE_FEATURES,
    SALES_FEATURES,
    build_features,
    join_context,
    weekly_price_features,
)
from retailsense.data.trends import read_trends_csv, slug, stitch, weekly_features
from retailsense.data.weather import weather_features
from retailsense.eval.baselines import ets, origin_grid, seasonal_naive
from tests.conftest import ITEMS, N_DAYS

FEATURES = SALES_FEATURES + PRICE_FEATURES + CALENDAR_FEATURES + ["launched", "available_at"]
HORIZONS = list(range(1, 8))
PARQUET = load_experiment().path("processed") / "features.parquet"
needs_data = pytest.mark.skipif(not PARQUET.exists(), reason="features.parquet not built")


def build(calendar, sales, prices):
    return build_features(sales, calendar, prices, "MH", N_DAYS, ITEMS, HORIZONS)


def poison_after(calendar, sales, prices, p):
    """Replace every sales value after day p and every price of a week not closed by p."""
    s = sales.copy()
    s.loc[s.d > p, "sales"] = 10**6
    week_end = calendar.groupby("wm_yr_wk").date.max()
    open_weeks = week_end[week_end > calendar.date.iloc[p - 1]].index
    pr = prices.copy()
    pr.loc[pr.wm_yr_wk.isin(open_weeks), "sell_price"] = 999.0
    return s, pr


# Seven consecutive cut-offs cover every weekday, so every position relative to the week close
# (Friday) is exercised; plus early and final cut-offs.
@pytest.mark.parametrize("p", [70, *range(343, 350), N_DAYS - 7])
def test_no_future_features(synth, p):
    """Every row whose prediction time d - h is <= p must not change when everything after p is
    poisoned (covers each horizon's targets p+1..p+h)."""
    calendar, sales, prices = synth
    clean = build(calendar, sales, prices)
    dirty = build(calendar, *poison_after(calendar, sales, prices, p))
    sel = lambda df: df[df.d - df.h <= p][FEATURES].reset_index(drop=True)  # noqa: E731
    pd.testing.assert_frame_equal(sel(clean), sel(dirty))


def test_available_at_precedes_every_prediction_time(synth):
    df = build(*synth)
    # prediction time for target date t at horizon h is t - h
    assert (df.available_at <= df.date - pd.to_timedelta(df.h, unit="D")).all()


def test_rolling_windows_on_shifted_series(synth):
    df = build(*synth)
    for h in (1, 3, 7):
        s = df[(df.item_id == ITEMS[0]) & (df.h == h)].set_index("d")
        y = s.sales
        for t in (100, 250, N_DAYS):
            e = t - h  # last usable day
            assert s.loc[t, "lag_h"] == y.loc[e] and s.loc[t, "lag_7"] == y.loc[t - 7]
            assert s.loc[t, "roll_mean_7"] == pytest.approx(y.loc[e - 6 : e].mean())
            assert s.loc[t, "roll_std_28"] == pytest.approx(y.loc[e - 27 : e].std())
            assert s.loc[t, "roll_mean_56"] == pytest.approx(y.loc[e - 55 : e].mean())
            assert s.loc[t, "zero_share_28"] == pytest.approx((y.loc[e - 27 : e] == 0).mean())


def test_price_from_closed_week_only(synth):
    calendar, _, prices = synth
    df = build(*synth)
    p = df.dropna(subset=["price_available_at"])
    assert (p.price_available_at <= p.date - pd.to_timedelta(p.h, unit="D")).all()
    wk = weekly_price_features(prices, calendar).set_index(["item_id", "week_end"]).price
    r = df.dropna(subset=["price_available_at"]).sample(50, random_state=0)
    assert (
        r.price.to_numpy()
        == wk.loc[list(zip(r.item_id.astype(str), r.price_available_at, strict=True))].to_numpy()
    ).all()


def test_seasonal_naive_rejects_long_horizons(synth):
    _, sales, _ = synth
    with pytest.raises(ValueError):
        seasonal_naive(sales, origin_grid([300], [8], 400))


def test_baselines_ignore_sales_after_origin(synth, cfg):
    _, sales, _ = synth
    small = cfg.model_copy(update={"ets": cfg.ets.model_copy(update={"fit_window": 200})})
    grid = origin_grid(range(380, 390), range(1, 8), 396)
    s = sales[sales.item_id == ITEMS[0]]
    poisoned = s.assign(sales=np.where(s.d > 389, 10**6, s.sales))
    for fn in (seasonal_naive, lambda x, g: ets(x, g, small)):
        pd.testing.assert_frame_equal(fn(s, grid), fn(poisoned, grid))


def test_oof_models_train_strictly_before_their_block():
    df = pd.DataFrame({"d": np.arange(1, 601), "x": 0.0})
    seen = []

    def fit(train):
        seen.append(train.d.max())
        return None

    out = expanding_oof(df, 5, fit, lambda m, b: np.zeros(len(b)))
    blocks = expanding_blocks(1, 600, 5)
    assert [s - 1 for s, _ in blocks] == seen
    assert (out.d > out.fold_train_end).all()
    assert out.d.is_unique and out.d.max() == 600


def test_confidence_uses_only_revealed_days():
    rng = np.random.default_rng(1)
    t = pd.DataFrame(
        [(it, d, rng.poisson(5), rng.poisson(5)) for it in "AB" for d in range(1, 101)],
        columns=["item_id", "d", "y", "yhat"],
    )
    p = 60
    poisoned = t.assign(y=np.where(t.d > p, 10**6, t.y), yhat=np.where(t.d > p, 0, t.yhat))
    a = trailing_confidence(t, [p], 28, 14)
    b = trailing_confidence(poisoned, [p], 28, 14)
    pd.testing.assert_frame_equal(a, b)


def test_splits_match_claude_md(cfg):
    d = cfg.data
    assert (d.train_end, d.val_end, d.test_end) == (1799, 1885, 1941)
    assert d.rank_end <= d.train_end  # subset chosen without val/test sales


@needs_data
def test_real_features_leakage_and_splits():
    df = pd.read_parquet(PARQUET, columns=["d", "h", "date", "split", "available_at"])
    assert (df.available_at <= df.date - pd.to_timedelta(df.h, unit="D")).all()
    assert sorted(df.h.unique()) == HORIZONS
    by = df.drop_duplicates("d").set_index("d")
    for split, lo, hi, start, end in [
        ("train", 1, 1799, "2019-01-05", "2023-12-08"),
        ("val", 1800, 1885, "2023-12-09", "2024-03-03"),
        ("test", 1886, 1941, "2024-03-04", "2024-04-28"),
    ]:
        days = by[by.split == split].index
        assert (days.min(), days.max(), len(days)) == (lo, hi, hi - lo + 1)
        assert by.loc[lo, "date"] == pd.Timestamp(start) and by.loc[hi, "date"] == pd.Timestamp(end)


# ---- external signals (Phase 2) ---------------------------------------------------------------


@pytest.fixture(scope="module")
def obs():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2001-01-01", "2016-05-22", freq="D")
    doy = dates.dayofyear.to_numpy()
    return pd.DataFrame(
        {
            "date": dates,
            "tmax": 22 + 6 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 3, len(dates)),
            "tmin": 12 + 5 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 2, len(dates)),
            "precip": rng.exponential(0.5, len(dates)) * (rng.random(len(dates)) < 0.1),
        }
    )


@pytest.mark.parametrize("t", ["2012-07-04", "2015-12-31", "2016-03-01"])
def test_weather_uses_only_past_observations(obs, cfg, t):
    t = pd.Timestamp(t)
    clean = weather_features(obs, cfg).set_index("date").loc[t]
    after = obs.assign(tmax=np.where(obs.date > t, 999.0, obs.tmax))
    same_year = obs.date.dt.year == t.year
    keep = (obs.date >= t - pd.Timedelta(days=cfg.weather.change_days)) & (obs.date <= t)
    this_year = obs.assign(tmax=np.where(same_year & ~keep, -999.0, obs.tmax))
    for poisoned in (after, this_year):  # climatology must come from previous years only
        pd.testing.assert_series_equal(
            clean, weather_features(poisoned, cfg).set_index("date").loc[t]
        )


def test_weather_climatology_is_past_years_same_season(obs, cfg):
    t = pd.Timestamp("2014-08-10")
    w = cfg.weather
    past = obs[(obs.date.dt.year < t.year) & (obs.date.dt.year >= t.year - w.clim_years)]
    dist = (past.date.dt.dayofyear - t.dayofyear).abs()
    clim = past.tmax[np.minimum(dist, 365 - dist) <= w.clim_half_window].mean()
    f = weather_features(obs, cfg).set_index("date").loc[t]
    assert f.tmax_anom == pytest.approx(obs.set_index("date").tmax[t] - clim)


def test_interest_joined_as_of_prediction_date():
    """A signal row is joined only if available_at <= t - h, and available_at is updated."""
    df = pd.DataFrame(
        {
            "item_id": "A",
            "d": np.repeat(np.arange(1, 31), 2),
            "h": np.tile([1, 7], 30),
            "date": np.repeat(pd.date_range("2016-01-01", periods=30), 2),
            "available_at": pd.Timestamp("2015-12-01"),
        }
    )
    sig = pd.DataFrame(
        {
            "week_start": pd.date_range("2015-11-29", periods=6, freq="7D"),
            "tr_x_rel52": np.arange(6.0),
        }
    )
    sig["available_at"] = sig.week_start + pd.Timedelta(days=7)
    out = join_context(df, None, {"trends": sig})
    pred = out.date - pd.to_timedelta(out.h, unit="D")
    assert (out.trends_available_at.dropna() <= pred[out.trends_available_at.notna()]).all()
    assert (out.available_at <= pred).all()
    # the joined week is the LATEST one available: the next week must not yet be available
    nxt = out.trends_available_at + pd.Timedelta(days=7)
    exists = nxt <= sig.available_at.max()
    assert (nxt[exists] > pred[exists]).all()


def test_trends_week_available_day_after_it_ends(tmp_path, cfg):
    kw = next(iter(cfg.interest.keywords))
    weeks = pd.date_range("2014-01-05", periods=120, freq="7D")  # Sundays
    (tmp_path / f"{slug(kw)}__a.csv").write_text(
        "Category: All categories\n\nWeek,x: (California)\n"
        + "".join(f"{w:%Y-%m-%d},{i % 50 + 1}\n" for i, w in enumerate(weeks)),
        encoding="utf-8",
    )
    s = read_trends_csv(tmp_path / f"{slug(kw)}__a.csv", 0.5)
    f = weekly_features(s, "x").reset_index(names="week_start")
    f["available_at"] = f.week_start + pd.Timedelta(days=7)
    week_end = f.week_start + pd.Timedelta(days=6)  # Saturday
    assert (f.available_at == week_end + pd.Timedelta(days=1)).all()


def test_trends_csv_parsing_and_lt1(tmp_path):
    p = tmp_path / "k.csv"
    p.write_text(
        "Category: All categories\n\nWeek,k: (California)\n2015-01-04,<1\n2015-01-11,40\n",
        encoding="utf-8",
    )
    s = read_trends_csv(p, 0.5)
    assert s.tolist() == [0.5, 40.0] and s.index[0] == pd.Timestamp("2015-01-04")


def test_trends_stitch_recovers_common_scale():
    idx = pd.date_range("2011-01-02", periods=300, freq="7D")
    truth = pd.Series(np.linspace(10, 80, 300), index=idx)
    a, b = truth.iloc[:200], truth.iloc[150:] * 0.37  # second window on another scale
    pd.testing.assert_series_equal(stitch([b, a]), truth, check_freq=False)


def test_trends_features_are_scale_invariant():
    """Window normalisation (which can depend on future weeks) must not change any feature."""
    rng = np.random.default_rng(5)
    idx = pd.date_range("2011-01-02", periods=300, freq="7D")
    s = pd.Series(rng.uniform(5, 100, 300), index=idx)
    pd.testing.assert_frame_equal(
        weekly_features(s, "x"), weekly_features(s * 0.123, "x"), check_exact=False, rtol=1e-9
    )
