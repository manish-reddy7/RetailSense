import numpy as np
import pandas as pd
import pytest

from retailsense.eval.h1_analysis import deciles, signal_stats, sweep


def pts(perfect: bool):
    rng = np.random.default_rng(0)
    e = rng.uniform(0, 1, 1000)
    d = e if perfect else rng.uniform(0, 1, 1000)
    y = np.full(1000, 10.0)
    return pd.DataFrame({"D": d, "rel_err": e, "y": y, "consensus": y * (1 + e), "abs_err": 10 * e})


def test_perfect_signal_catches_all_worst_errors():
    p = pts(True)
    worst = (p.rel_err >= p.rel_err.quantile(0.8)).to_numpy()
    assert signal_stats(p, "D", worst)["auroc_worst"] == pytest.approx(1.0)
    sw = sweep(p, [0.2], worst)
    assert sw.recall_worst.iloc[0] == pytest.approx(1.0) and sw.trigger_rate.iloc[0] == 0.2
    assert deciles(p, "D").mean_rel_err.is_monotonic_increasing


def test_random_signal_is_near_chance():
    p = pts(False)
    worst = (p.rel_err >= p.rel_err.quantile(0.8)).to_numpy()
    assert abs(signal_stats(p, "D", worst)["auroc_worst"] - 0.5) < 0.06
    assert abs(sweep(p, [0.3], worst).recall_worst.iloc[0] - 0.3) < 0.08
