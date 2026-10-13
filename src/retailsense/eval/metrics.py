"""Forecast metrics (blueprint 9.2). Denominators are floored at 1 (blueprint 5.4)."""

import numpy as np


def _arr(*xs):
    return [np.asarray(x, dtype=float) for x in xs]


def mae(y, yhat) -> float:
    y, yhat = _arr(y, yhat)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat) -> float:
    y, yhat = _arr(y, yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def wape(y, yhat, floor: float = 1.0) -> float:
    y, yhat = _arr(y, yhat)
    return float(np.sum(np.abs(y - yhat)) / max(np.sum(np.abs(y)), floor))


def smape(y, yhat) -> float:
    """Mean of 2|y - yhat| / (|y| + |yhat|), skipping points where y = yhat = 0."""
    y, yhat = _arr(y, yhat)
    denom = np.abs(y) + np.abs(yhat)
    keep = denom > 0
    return float(np.mean(2 * np.abs(y - yhat)[keep] / denom[keep])) if keep.any() else np.nan


def coverage(y, lower, upper) -> float:
    y, lower, upper = _arr(y, lower, upper)
    return float(np.mean((y >= lower) & (y <= upper)))


def all_metrics(y, yhat, lower=None, upper=None) -> dict[str, float]:
    out = {
        "MAE": mae(y, yhat),
        "RMSE": rmse(y, yhat),
        "WAPE": wape(y, yhat),
        "sMAPE": smape(y, yhat),
    }
    out["coverage"] = coverage(y, lower, upper) if lower is not None else np.nan
    return out
