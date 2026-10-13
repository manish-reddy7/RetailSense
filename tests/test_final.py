import numpy as np
import pandas as pd
import pytest

from retailsense.config import load_experiment
from retailsense.data.weather import add_forecast_noise
from retailsense.eval.final import deliberated_system

AF = pd.DataFrame(
    {
        "agent": ["historical", "calendar"] * 2,
        "item_id": "A",
        "origin": 10,
        "h": [1, 1, 2, 2],
        "forecast": [10.0, 20.0, 12.0, 24.0],
        "lower": [5.0, 15.0, 6.0, 18.0],
        "upper": [15.0, 25.0, 18.0, 30.0],
        "confidence": 0.5,
    }
)
REV = pd.DataFrame(
    {
        "item_id": "A",
        "origin": 10,
        "agent": ["historical", "calendar"],
        "m": [2.0, 1.0],
        "y": [12.0, 20.0],
        "initial": [10.0, 20.0],
        "held": [False, True],
    }
)


def test_untriggered_is_confidence_consensus():
    out = deliberated_system(AF, REV, set()).set_index("h")
    assert out.yhat.tolist() == [15.0, 18.0] and out.lower[1] == 10.0


def test_triggered_uses_revised_h1_and_new_weights_everywhere():
    out = deliberated_system(AF, REV, {("A", 10)}).set_index("h")
    w = np.array([2, 1]) / 3  # c*m, normalised
    assert out.yhat[1] == pytest.approx(w @ [12.0, 20.0])  # revised values at h=1
    assert out.yhat[2] == pytest.approx(w @ [12.0, 24.0])  # original values, new weights at h2
    assert out.upper[2] == pytest.approx(w @ [18.0, 30.0])
    wo = deliberated_system(AF, REV, {("A", 10)}, values=False).set_index("h")
    assert wo.yhat[1] == pytest.approx(w @ [10.0, 20.0])  # weights-only ablation


def test_weather_noise_keeps_missing_and_is_consistent():
    cfg = load_experiment()
    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(["2016-04-01", "2016-04-01", "2016-04-02"]),
            "tmax": [20.0, 20.0, np.nan],
            "tmin": [10.0, 10.0, 11.0],
            "precip": [0.0, 0.0, 5.0],
            "tmax_anom": [1.0, 1.0, np.nan],
            "tmin_anom": [0.0, 0.0, 0.5],
            "precip_anom": [-1.0, -1.0, 3.0],
            "tmax_chg": [2.0, 2.0, np.nan],
            "tmin_chg": [0.0, 0.0, 1.0],
            "hot_day": [0.0, 0.0, np.nan],
            "rainy_day": [0.0, 0.0, 1.0],
        }
    )
    scale = {"tmax": 3.0, "tmin": 3.0, "precip": 2.0}
    n = add_forecast_noise(rows, scale, 1.0, cfg, seed=0)
    assert np.isnan(n.tmax[2]) and np.isnan(n.hot_day[2])  # missing stays missing
    assert n.tmax[0] == n.tmax[1]  # one error per date, shared by all items
    assert n.tmax[0] - 20.0 == pytest.approx(n.tmax_anom[0] - 1.0)  # anomaly moves with value
    assert (n.precip >= 0).all()
    same = add_forecast_noise(rows, scale, 0.0, cfg, seed=0)
    pd.testing.assert_frame_equal(same[["tmax", "tmin_anom"]], rows[["tmax", "tmin_anom"]])
