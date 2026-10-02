from datetime import date

import pytest
from pydantic import ValidationError

from retailsense.agents.schema import AgentForecast


def make(**kw):
    base = dict(
        agent_id="historical",
        item_id="milk_1",
        store_id="MUM_1",
        prediction_time=date(2016, 1, 1),
        horizon_days=[1, 2, 3],
        forecast=[10.0, 11.0, 12.0],
        lower=[5.0, 6.0, 7.0],
        upper=[15.0, 16.0, 17.0],
        confidence=0.7,
        key_factors=["lag_7 = 10"],
        limitations=[],
    )
    return AgentForecast(**{**base, **kw})


def test_valid():
    assert make().forecast == [10.0, 11.0, 12.0]


@pytest.mark.parametrize(
    "kw",
    [
        {"forecast": [10.0, 11.0]},  # length mismatch
        {"upper": [15.0, 16.0]},
        {"forecast": [20.0, 11.0, 12.0]},  # above upper
        {"forecast": [1.0, 11.0, 12.0]},  # below lower
        {"lower": [-1.0, 6.0, 7.0]},  # negative
        {"forecast": [float("nan"), 11.0, 12.0]},
        {"upper": [float("inf"), 16.0, 17.0]},
        {"horizon_days": [1, 1, 2]},
        {"horizon_days": [0, 1, 2]},
        {"horizon_days": [], "forecast": [], "lower": [], "upper": []},
        {"confidence": 1.5},
        {"confidence": -0.1},
        {"agent_id": "critic"},
        {"extra_field": 1},
    ],
)
def test_invalid(kw):
    with pytest.raises(ValidationError):
        make(**kw)


def test_zero_forecast_allowed():
    assert make(lower=[0, 0, 0], forecast=[0, 0, 0], upper=[0, 1, 2]).forecast[0] == 0
