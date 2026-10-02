import numpy as np
import pandas as pd
import pytest

from retailsense.config import load_experiment

N_DAYS = 420
ITEMS = ["milk_1", "atta_1", "sweets_1"]


@pytest.fixture(scope="session")
def cfg():
    return load_experiment()


@pytest.fixture(scope="session")
def synth():
    """Small dataset in the raw file layout: calendar, long sales, weekly prices."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2011-01-29", periods=N_DAYS, freq="D")  # day 1 is a Saturday
    ev = np.where(rng.random(N_DAYS) < 0.03, "Sporting", None)
    calendar = pd.DataFrame(
        {
            "date": dates,
            "wm_yr_wk": 11101 + np.arange(N_DAYS) // 7,
            "wday": np.arange(N_DAYS) % 7 + 1,
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{i}" for i in range(1, N_DAYS + 1)],
            "event_name_1": np.where(ev == None, None, "Ev"),  # noqa: E711
            "event_type_1": ev,
            "snap_MH": (dates.day <= 10).astype(int),
        }
    )
    sales = pd.DataFrame(
        [
            (it, d, int(rng.poisson(5 + 3 * (d % 7 == 0) + k)))
            for k, it in enumerate(ITEMS)
            for d in range(1, N_DAYS + 1)
        ],
        columns=["item_id", "d", "sales"],
    )
    sales.loc[(sales.item_id == ITEMS[2]) & (sales.d < 40), "sales"] = 0  # late launch
    weeks = calendar.wm_yr_wk.unique()
    prices = pd.DataFrame(
        [("MUM_1", it, w, round(2 + rng.integers(0, 3) * 0.25, 2)) for it in ITEMS for w in weeks],
        columns=["store_id", "item_id", "wm_yr_wk", "sell_price"],
    )
    return calendar, sales, prices
