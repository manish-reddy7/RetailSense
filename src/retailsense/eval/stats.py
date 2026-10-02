"""Significance tests for Phase 5 (blueprint 9): Diebold-Mariano, paired bootstrap over items,
Wilcoxon signed-rank. Pure functions of error arrays; they never load data themselves."""

import numpy as np
from scipy.stats import t, wilcoxon


def diebold_mariano(e1, e2, h: int = 1, power: int = 1) -> dict:
    """DM test (Harvey-Leybourne-Newbold small-sample correction) on loss |e|^power.
    d = loss1 - loss2: a negative statistic means model 1 is more accurate."""
    d = np.abs(np.asarray(e1, float)) ** power - np.abs(np.asarray(e2, float)) ** power
    n = len(d)
    dc = d - d.mean()
    gamma = [dc @ dc / n] + [dc[k:] @ dc[:-k] / n for k in range(1, h)]  # autocov lags < h
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return {"dm": np.nan, "p": np.nan, "mean_diff": float(d.mean())}
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = hln * d.mean() / np.sqrt(var)
    return {"dm": float(stat), "p": float(2 * t.sf(abs(stat), n - 1)), "mean_diff": float(d.mean())}


def paired_bootstrap(items, abs_err_a, abs_err_b, actual, n: int = 1000, seed: int = 0) -> dict:
    """WAPE(a) - WAPE(b) with a 95% CI from resampling ITEMS (whole series) with replacement.
    Negative = a is better. Resampling items keeps each item's days together."""
    g = np.unique(np.asarray(items), return_inverse=True)[1]
    k = g.max() + 1
    sa, sb, sy = (np.bincount(g, np.asarray(x, float), k) for x in (abs_err_a, abs_err_b, actual))
    diff = sa.sum() / sy.sum() - sb.sum() / sy.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n, k))
    boot = (sa[idx].sum(1) - sb[idx].sum(1)) / sy[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "diff": float(diff),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_gt0": float((boot >= 0).mean()),
    }


def wilcoxon_paired(a, b) -> dict:
    """Wilcoxon signed-rank on paired per-unit scores (e.g. per-item WAPE)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return {"stat": np.nan, "p": 1.0, "median_diff": 0.0}
    r = wilcoxon(a, b)
    return {
        "stat": float(r.statistic),
        "p": float(r.pvalue),
        "median_diff": float(np.median(a - b)),
    }
