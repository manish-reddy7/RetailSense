"""Disagreement score (blueprint 7.1, Appendix A.1).

w_i = c_i / sum(c);  y_bar = sum(w_i y_i);  D = sqrt(sum(w_i (y_i - y_bar)^2)) / max(y_bar, floor)
D_range = (max y_i - min y_i) / max(y_bar, floor)   (sensitivity check)
D_std = sqrt(sum(w_i (y_i - y_bar)^2))                (unnormalised ablation, blueprint 9.5)
"""

import numpy as np
import pandas as pd


def consensus(y, c) -> tuple[float, np.ndarray]:
    w = np.asarray(c, float) / np.sum(c)
    return float(np.dot(w, np.asarray(y, float))), w


def disagreement(y, c, floor: float = 1.0) -> float:
    y = np.asarray(y, float)
    y_bar, w = consensus(y, c)
    return float(np.sqrt(np.dot(w, (y - y_bar) ** 2)) / max(y_bar, floor))


def d_range(y, c, floor: float = 1.0) -> float:
    y = np.asarray(y, float)
    return float((y.max() - y.min()) / max(consensus(y, c)[0], floor))


def route(d: float, tau: float) -> str:
    return "deliberate" if d >= tau else "fast"


def score_table(y: pd.DataFrame, c: pd.DataFrame, floor: float = 1.0) -> pd.DataFrame:
    """Vectorised over rows: y and c are (points x agents) with the same index and columns.
    Returns consensus, D, D_range and D_std per point."""
    if not (y.index.equals(c.index) and y.columns.equals(c.columns)):
        raise ValueError("forecast and confidence tables must align")
    yv, cv = y.to_numpy(float), c.to_numpy(float)
    if np.isnan(yv).any() or np.isnan(cv).any() or (cv <= 0).any():
        raise ValueError("every agent needs a forecast and a positive confidence")
    w = cv / cv.sum(axis=1, keepdims=True)
    y_bar = (w * yv).sum(axis=1)
    denom = np.maximum(y_bar, floor)
    std = np.sqrt((w * (yv - y_bar[:, None]) ** 2).sum(axis=1))
    return pd.DataFrame(
        {
            "consensus": y_bar,
            "D": std / denom,
            "D_range": (yv.max(axis=1) - yv.min(axis=1)) / denom,
            "D_std": std,
        },
        index=y.index,
    )
