"""Historical Agent (blueprint 6.2): LightGBM Tweedie point model + quantile models (0.1, 0.9).

Evidence: sales history, price, item. One model per horizon h (features lag >= h).
Also provides the shared LightGBM fitting helpers and the expanding-fold OOF baseline (6.3).
"""

from collections.abc import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd

from retailsense.config import ExperimentCfg
from retailsense.data.features import ITEM_FEATURES, PRICE_FEATURES, SALES_FEATURES

FEATURES = SALES_FEATURES + PRICE_FEATURES + ITEM_FEATURES


def lgb_params(cfg: ExperimentCfg, block: str, objective: str, **extra) -> dict:
    return {
        **getattr(cfg.lgbm, block),
        "objective": objective,
        "seed": cfg.seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": cfg.lgbm.num_threads,
        "verbose": -1,
        **extra,
    }


def fit_early_stopped(
    df: pd.DataFrame, features: list[str], target: str, params: dict, cfg: ExperimentCfg
) -> lgb.Booster:
    """Pick the round count on the last `early_stop_days` days of df, then refit on all of df.
    Never sees any data after df.d.max()."""
    cut = df.d.max() - cfg.lgbm.early_stop_days
    fit, hold = df[df.d <= cut], df[df.d > cut]
    probe = lgb.train(
        params,
        lgb.Dataset(fit[features], fit[target]),
        num_boost_round=cfg.lgbm.max_rounds,
        valid_sets=[lgb.Dataset(hold[features], hold[target])],
        callbacks=[lgb.early_stopping(cfg.lgbm.early_stopping_rounds, verbose=False)],
    )
    rounds = max(probe.best_iteration, 1)
    return lgb.train(params, lgb.Dataset(df[features], df[target]), num_boost_round=rounds)


def expanding_blocks(first_day: int, last_day: int, n_folds: int) -> list[tuple[int, int]]:
    """Split [first_day, last_day] into n_folds + 1 contiguous chunks; return the last n_folds
    as (start, end). A model for block k is trained only on days < start_k."""
    size = (last_day - first_day + 1) // (n_folds + 1)
    starts = [first_day + k * size for k in range(1, n_folds + 1)]
    ends = [s - 1 for s in starts[1:]] + [last_day]
    return list(zip(starts, ends, strict=True))


def expanding_oof(
    df: pd.DataFrame,
    n_folds: int,
    fit: Callable[[pd.DataFrame], object],
    predict: Callable[[object, pd.DataFrame], np.ndarray],
) -> pd.DataFrame:
    """Out-of-fold predictions: each block is predicted by a model fit on strictly earlier days."""
    parts = []
    for k, (start, end) in enumerate(expanding_blocks(df.d.min(), df.d.max(), n_folds), 1):
        model = fit(df[df.d < start])
        block = df[(df.d >= start) & (df.d <= end)]
        parts.append(block.assign(oof_pred=predict(model, block), fold=k, fold_train_end=start - 1))
    return pd.concat(parts, ignore_index=True)


class HistoricalAgent:
    agent_id = "historical"

    def __init__(self, cfg: ExperimentCfg, features: list[str] = FEATURES):
        """features: the Historical agent's own evidence; the all-features control passes more."""
        self.cfg = cfg
        self.features = features
        self.lo_q, self.hi_q = cfg.lgbm.quantiles

    def _point_params(self) -> dict:
        return lgb_params(self.cfg, "historical", "tweedie")

    def fit(self, train: pd.DataFrame) -> "HistoricalAgent":
        """One (point, lower, upper) model triple per horizon h, each on that horizon's rows."""
        c = self.cfg
        lo_p = lgb_params(c, "historical", "quantile", alpha=self.lo_q)
        hi_p = lgb_params(c, "historical", "quantile", alpha=self.hi_q)
        self.point, self.lo, self.hi = {}, {}, {}
        for h, g in train.groupby("h"):
            self.point[h] = fit_early_stopped(g, self.features, "sales", self._point_params(), c)
            self.lo[h] = fit_early_stopped(g, self.features, "sales", lo_p, c)
            self.hi[h] = fit_early_stopped(g, self.features, "sales", hi_p, c)
        m = self.point[min(self.point)]  # key factors come from the disagreement horizon
        self.importance = pd.Series(
            m.feature_importance("gain"), index=m.feature_name()
        ).sort_values(ascending=False)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per (target day, h): forecast, lower, upper, in df's row order. Quantile crossings are
        resolved by widening the band to include the point forecast; everything is clipped at 0."""
        parts = []
        for h, g in df.groupby("h"):
            x = g[self.features]
            f = np.clip(self.point[h].predict(x), 0, None)
            lo = np.clip(self.lo[h].predict(x), 0, None)
            hi = np.clip(self.hi[h].predict(x), 0, None)
            out = g[["item_id", "d", "h"]].copy()
            out["forecast"], out["lower"], out["upper"] = f, np.minimum(lo, f), np.maximum(hi, f)
            out["band_widened"] = (lo > f) | (hi < f)
            parts.append(out)
        return pd.concat(parts).loc[df.index]

    def oof_baseline(self, train: pd.DataFrame) -> pd.DataFrame:
        """Expanding-fold out-of-fold point forecasts per horizon over the training period (6.3)."""
        params = self._point_params()
        oof = pd.concat(
            expanding_oof(
                g,
                self.cfg.agents.oof_folds,
                lambda d: fit_early_stopped(d, self.features, "sales", params, self.cfg),
                lambda m, d: np.clip(m.predict(d[self.features]), 0, None),
            )
            for _, g in train.groupby("h")
        )
        return oof.rename(columns={"oof_pred": "baseline"}).reset_index(drop=True)

    def key_factors(self, row: pd.Series) -> list[str]:
        top = [f for f in self.importance.index if f != "item_id"][: self.cfg.lgbm.key_factors]
        return [f"{f} = {row[f]:.3g}" if pd.notna(row[f]) else f"{f} missing" for f in top]
