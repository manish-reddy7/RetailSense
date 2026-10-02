import numpy as np
import pandas as pd
import pytest

from retailsense.agents.calendar import calendar_agent
from retailsense.agents.confidence import trailing_confidence
from retailsense.agents.historical import HistoricalAgent
from retailsense.data.features import build_features
from tests.conftest import ITEMS, N_DAYS


@pytest.fixture(scope="module")
def fitted(synth, cfg):
    small = cfg.model_copy(
        update={
            "lgbm": cfg.lgbm.model_copy(update={"max_rounds": 60, "early_stopping_rounds": 10}),
        }
    )
    calendar, sales, prices = synth
    df = build_features(sales, calendar, prices, "MH", N_DAYS, ITEMS, [1, 2])
    train, test = df[(df.d <= 380) & df.launched], df[df.d > 380].drop(columns="sales")
    hist = HistoricalAgent(small).fit(train)
    oof = hist.oof_baseline(train)
    cal = calendar_agent(small).fit(oof)
    hp = hist.predict(test.reset_index(drop=True))
    return hist, cal, oof, hp, cal.apply(hp, test.reset_index(drop=True)), test


def _valid_band(p):
    v = p[["forecast", "lower", "upper"]].to_numpy()
    assert np.isfinite(v).all() and (v >= 0).all()
    assert (p.lower <= p.forecast).all() and (p.forecast <= p.upper).all()


def test_historical_output_valid(fitted):
    _valid_band(fitted[3])


def test_calendar_output_valid_and_shifted_band(fitted):
    _, _, _, hp, cp, _ = fitted
    _valid_band(cp)
    unclipped = hp.lower + cp.adjustment > 0
    assert np.allclose((cp.lower - hp.lower)[unclipped], cp.adjustment[unclipped])


def test_oof_covers_training_tail_out_of_sample(fitted):
    oof = fitted[2]
    assert (oof.d > oof.fold_train_end).all() and oof.baseline.notna().all()


def test_key_factors_are_strings(fitted):
    hist, cal, _, _, cp, test = fitted
    row = test.iloc[0]
    assert all(isinstance(k, str) for k in hist.key_factors(row))
    assert cal.key_factors(row, cp.adjustment.iloc[0])[0].startswith("calendar adjustment")


def test_confidence_formula_and_fallback():
    t = pd.DataFrame(
        {
            "item_id": ["A"] * 28 + ["B"] * 5,
            "d": list(range(1, 29)) + list(range(24, 29)),
            "y": [10] * 33,
            "yhat": [8] * 28 + [10] * 5,
        }
    )
    c = trailing_confidence(t, [28], 28, 14).set_index("item_id")
    assert c.loc["A", "confidence"] == pytest.approx(1 / (1 + 0.2)) and not c.loc["A", "fallback"]
    dept = (28 * 2) / (33 * 10)
    assert c.loc["B", "confidence"] == pytest.approx(1 / (1 + dept)) and c.loc["B", "fallback"]
