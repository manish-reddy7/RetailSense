"""Statistical baselines (blueprint 9.1, systems 1-2) on the shared origin x horizon grid."""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing

from retailsense.config import ExperimentCfg


def origin_grid(origins, horizons, last_target: int) -> pd.DataFrame:
    """(origin, h, d) with d = origin + h, keeping only targets <= last_target.
    The forecast for (origin, h) may use data up to day `origin` only."""
    g = pd.DataFrame([(p, h) for p in origins for h in horizons], columns=["origin", "h"])
    g["d"] = g.origin + g.h
    return g[g.d <= last_target].reset_index(drop=True)


def seasonal_naive(sales: pd.DataFrame, grid: pd.DataFrame, season: int = 7) -> pd.DataFrame:
    """Same weekday last week: yhat(d) = y(d - season).
    Needs h <= season so that d - season <= origin."""
    if (grid.h > season).any():
        raise ValueError("seasonal naive with h > season would read unrevealed sales")
    src = sales.rename(columns={"sales": "yhat"}).assign(d=lambda x: x.d + season)
    out = grid.merge(sales[["item_id"]].drop_duplicates(), how="cross").merge(
        src, on=["item_id", "d"], how="left"
    )
    return out[["item_id", "origin", "h", "d", "yhat"]]


def _ets_fit(y: np.ndarray, season: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ExponentialSmoothing(y, trend=False, seasonal=season).fit(disp=False)


def ets_item(y: pd.Series, origins: list[int], horizon: int, cfg: ExperimentCfg) -> pd.DataFrame:
    """ETS(A,N,A) with weekly seasonality for one item. y is indexed by day.
    Parameters are refit every `refit_every` origins on the trailing `fit_window` days;
    between refits the newly revealed days are appended with parameters held fixed.
    (append, not extend: extend re-estimates the concentrated scale from the new days only,
    which collapses the interval.)"""
    e = cfg.ets
    rows = []
    res, anchor = None, None
    for i, p in enumerate(origins):
        if i % e.refit_every == 0:
            res, anchor = _ets_fit(y.loc[p - e.fit_window + 1 : p].to_numpy(float), e.season), p
            cur = res
        else:
            cur = res.append(y.loc[anchor + 1 : p].to_numpy(float), refit=False)
        fc = cur.get_forecast(horizon).summary_frame(alpha=e.band_alpha)
        for h in range(1, horizon + 1):
            m, lo, hi = fc.iloc[h - 1][["mean", "mean_ci_lower", "mean_ci_upper"]]
            rows.append((p, h, p + h, max(m, 0.0), max(lo, 0.0), max(hi, 0.0)))
    return pd.DataFrame(rows, columns=["origin", "h", "d", "yhat", "lower", "upper"])


def ets(sales: pd.DataFrame, grid: pd.DataFrame, cfg: ExperimentCfg) -> pd.DataFrame:
    origins = sorted(grid.origin.unique())
    horizon = int(grid.h.max())
    parts = []
    for item, s in sales.groupby("item_id", observed=True):
        y = s.set_index("d").sales.loc[: max(origins)]  # never pass sales after the last origin
        parts.append(ets_item(y, origins, horizon, cfg).assign(item_id=item))
    out = pd.concat(parts, ignore_index=True)
    return grid.merge(out, on=["origin", "h", "d"])[
        ["item_id", "origin", "h", "d", "yhat", "lower", "upper"]
    ]
