import pandas as pd
import pytest

from retailsense.data.load import load_calendar
from retailsense.forecast_future import extended_calendar


def test_extended_calendar_continues_the_data_calendar(cfg):
    if not (cfg.path("raw") / "calendar.csv").exists():
        pytest.skip("raw India data not generated")
    cal = load_calendar(cfg)
    ext = extended_calendar(cfg, len(cal) + 7)
    pd.testing.assert_frame_equal(ext.iloc[: len(cal)], cal, check_dtype=False)  # past untouched
    assert len(ext) == len(cal) + 7
    assert (ext.date.diff().dropna() == pd.Timedelta(days=1)).all()
    assert ext.d.tolist() == [f"d_{i}" for i in range(1, len(ext) + 1)]
    assert (ext.wday == (ext.index % 7) + 1).all()  # Sat-Fri weeks continue
