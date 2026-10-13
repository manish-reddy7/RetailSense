import numpy as np
import pandas as pd
import yaml

from retailsense.config import ROOT
from retailsense.data.features import EVENT_TYPES
from retailsense.data.india_generator import build_calendar, festival_mult, shock_series

G = yaml.safe_load((ROOT / "configs/india_generator.yaml").read_text(encoding="utf-8"))
DATES = pd.date_range(G["start"], periods=400, freq="D")


def test_calendar_matches_raw_layout():
    cal = build_calendar(G, DATES, ["MH"])
    assert set(cal.event_type_1.dropna()) <= set(EVENT_TYPES)
    assert cal.wday.iloc[0] == 1 and pd.Timestamp(cal.date.iloc[0]).day_name() == "Saturday"
    assert "Diwali (Deepavali)" in set(cal.event_name_1)


def test_festival_ramp_peaks_day_before_and_quiet_on_day():
    names = [None] * 20
    names[10] = "Diwali (Deepavali)"
    m = festival_mult(G, pd.date_range("2020-01-01", periods=20), names, {"diwali": 3.0})
    pre = G["festivals"]["pre_days"]
    assert m[9] == 4.0 and m[10] == G["festivals"]["day_of_mult"]
    assert np.all(np.diff(m[10 - pre : 10]) > 0) and m[10 - pre - 1] == 1.0


def test_generation_is_reproducible():
    a = shock_series(G, np.random.default_rng(G["seed"]), 1000)
    b = shock_series(G, np.random.default_rng(G["seed"]), 1000)
    assert np.array_equal(a, b) and a.max() > 1
