import math

import pytest

from retailsense.eval.metrics import all_metrics, coverage, mae, rmse, smape, wape

Y, F = [10, 0, 5, 0], [8, 2, 5, 0]


def test_mae():
    assert mae(Y, F) == pytest.approx(1.0)


def test_rmse():
    assert rmse(Y, F) == pytest.approx(math.sqrt(2.0))


def test_wape():
    assert wape(Y, F) == pytest.approx(4 / 15)


def test_wape_floors_zero_denominator():
    assert wape([0, 0], [1, 2]) == pytest.approx(3.0)


def test_smape_skips_double_zero():
    # points: 2*2/18, 2*2/2, 0/10 ; the (0,0) point is skipped
    assert smape(Y, F) == pytest.approx((4 / 18 + 2 + 0) / 3)


def test_smape_all_zero_is_nan():
    assert math.isnan(smape([0, 0], [0, 0]))


def test_coverage_inclusive_bounds():
    assert coverage([1, 5, 9, 12], [1, 1, 1, 1], [9, 9, 9, 9]) == 0.75


def test_all_metrics_without_band():
    m = all_metrics(Y, F)
    assert set(m) == {"MAE", "RMSE", "WAPE", "sMAPE", "coverage"} and math.isnan(m["coverage"])


def test_perfect_forecast():
    m = all_metrics(Y, Y, Y, Y)
    assert m["MAE"] == m["RMSE"] == m["WAPE"] == m["sMAPE"] == 0 and m["coverage"] == 1
