"""Weather signal (blueprint 4.2, 5.2): Open-Meteo archive for the state's representative city.

The archive is OBSERVED weather. It is used as a perfect forecast for the target day (archived
forecasts do not reach 2011-2016) - a stated limitation, tested later with added noise.
Features for target day t use observations on days <= t only; the climatology uses previous
years only.

Run: python -m retailsense.data.weather
"""

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from retailsense.config import ExperimentCfg, load_experiment

URL = "https://archive-api.open-meteo.com/v1/archive"
COLS = {"temperature_2m_max": "tmax", "temperature_2m_min": "tmin", "precipitation_sum": "precip"}
WEATHER_FEATURES = [
    "tmax",
    "tmin",
    "precip",
    "tmax_anom",
    "tmin_anom",
    "precip_anom",
    "hot_day",
    "rainy_day",
    "tmax_chg",
    "tmin_chg",
]


def fetch(cfg: ExperimentCfg, state: str | None = None) -> pd.DataFrame:
    """Daily observations (date, tmax, tmin, precip) for the state's city (default: the
    experiment's state). Responses are cached on disk forever."""
    w = cfg.weather
    lat, lon = w.city[state or cfg.data.state]
    session = requests_cache.CachedSession(cfg.path("cache") / "openmeteo", expire_after=-1)
    client = openmeteo_requests.Client(session=retry(session, retries=5, backoff_factor=0.5))
    r = client.weather_api(
        URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": w.fetch_start,
            "end_date": w.fetch_end,
            "daily": w.vars,
            "timezone": w.timezone,
        },
    )[0]
    daily = r.Daily()
    dates = pd.date_range(
        pd.to_datetime(daily.Time() + r.UtcOffsetSeconds(), unit="s"),
        periods=daily.Variables(0).ValuesAsNumpy().size,
        freq="D",
    )
    out = pd.DataFrame({"date": dates.normalize()})
    for i, v in enumerate(w.vars):
        out[COLS[v]] = daily.Variables(i).ValuesAsNumpy()
    return out


def weather_features(obs: pd.DataFrame, cfg: ExperimentCfg) -> pd.DataFrame:
    """One row per observed date. Missing observations stay NaN (never filled)."""
    w = cfg.weather
    obs = obs.set_index("date").asfreq("D").reset_index()  # gaps become NaN rows, not skipped
    doy = obs.date.dt.dayofyear.to_numpy()
    year = obs.date.dt.year.to_numpy()
    out = obs.copy()
    for col in COLS.values():
        vals = obs[col].to_numpy()
        clim = np.full(len(obs), np.nan)
        for y in np.unique(year):
            past = (year < y) & (year >= y - w.clim_years) & ~np.isnan(vals)
            for d in np.unique(doy[year == y]):
                dist = np.abs(doy[past] - d)
                near = np.minimum(dist, 365 - dist) <= w.clim_half_window
                if near.any():
                    clim[(year == y) & (doy == d)] = vals[past][near].mean()
        out[f"{col}_anom"] = vals - clim
    out["hot_day"] = (out.tmax_anom >= w.hot_anom_c).astype(float).where(out.tmax_anom.notna())
    out["rainy_day"] = (out.precip >= w.rainy_mm).astype(float).where(out.precip.notna())
    out["tmax_chg"] = out.tmax.diff(w.change_days)  # rows are consecutive days after asfreq
    out["tmin_chg"] = out.tmin.diff(w.change_days)
    return out[["date", *WEATHER_FEATURES]]


def add_forecast_noise(
    rows: pd.DataFrame, scale: dict[str, float], sd: float, cfg: ExperimentCfg, seed: int
) -> pd.DataFrame:
    """Ablation (blueprint 9.5): pretend the target-day weather was forecast with error.
    One Gaussian error per target date and variable (sd x that variable's daily SD in `scale`),
    applied to the value, its anomaly and its day-over-day change; flags are recomputed.
    Missing stays missing. Precipitation is clipped at 0."""
    w = cfg.weather
    dates = rows.date.drop_duplicates().sort_values()
    rng = np.random.default_rng(seed)
    err = pd.DataFrame(
        {c: rng.normal(0, sd * scale[c], len(dates)) for c in ("tmax", "tmin", "precip")},
        index=dates.to_numpy(),
    ).reindex(rows.date.to_numpy())
    out = rows.copy()
    for c in ("tmax", "tmin"):
        e = err[c].to_numpy()
        out[c], out[f"{c}_anom"], out[f"{c}_chg"] = (
            out[c] + e,
            out[f"{c}_anom"] + e,
            out[f"{c}_chg"] + e,
        )
    p = np.clip(out.precip + err.precip.to_numpy(), 0, None)
    out["precip_anom"], out["precip"] = out.precip_anom + (p - out.precip), p
    out["hot_day"] = (out.tmax_anom >= w.hot_anom_c).astype(float).where(out.tmax_anom.notna())
    out["rainy_day"] = (out.precip >= w.rainy_mm).astype(float).where(out.precip.notna())
    return out


def main() -> None:
    cfg = load_experiment()
    feats = weather_features(fetch(cfg), cfg)
    path = cfg.path("processed") / "weather.parquet"
    feats.to_parquet(path, index=False)
    print(
        f"wrote {path}: {len(feats):,} days, "
        f"{feats.date.min():%Y-%m-%d}..{feats.date.max():%Y-%m-%d}"
    )
    print(feats.drop(columns="date").isna().mean().round(4).to_string())


if __name__ == "__main__":
    main()
