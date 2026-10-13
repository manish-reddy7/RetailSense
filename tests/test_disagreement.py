import math

import numpy as np
import pandas as pd
import pytest

from retailsense.orchestrator.disagreement import (
    consensus,
    d_range,
    disagreement,
    route,
    score_table,
)


def test_hand_computed_case():
    # y = [10, 20], c = [0.75, 0.25] -> w = [0.75, 0.25], y_bar = 12.5
    # var = 0.75 * 2.5^2 + 0.25 * 7.5^2 = 4.6875 + 14.0625 = 18.75 -> D = sqrt(18.75) / 12.5
    y_bar, w = consensus([10, 20], [0.75, 0.25])
    assert y_bar == 12.5 and np.allclose(w, [0.75, 0.25])
    assert disagreement([10, 20], [0.75, 0.25]) == pytest.approx(math.sqrt(18.75) / 12.5)
    assert d_range([10, 20], [0.75, 0.25]) == pytest.approx(10 / 12.5)


def test_equal_confidence_three_agents():
    # y = [8, 10, 12], equal weights: y_bar = 10, var = (4 + 0 + 4) / 3 -> D = sqrt(8/3) / 10
    assert disagreement([8, 10, 12], [0.5, 0.5, 0.5]) == pytest.approx(math.sqrt(8 / 3) / 10)


def test_identical_forecasts():
    assert disagreement([7.0, 7.0, 7.0], [0.2, 0.5, 0.9]) == 0.0
    assert d_range([7.0, 7.0, 7.0], [0.2, 0.5, 0.9]) == 0.0


def test_scale_invariance():
    y, c = [12.0, 15.0, 9.0], [0.6, 0.3, 0.8]
    for k in (2.0, 10.0, 137.5):
        assert disagreement([k * v for v in y], c) == pytest.approx(disagreement(y, c))


def test_confidence_scale_does_not_matter():
    y = [3.0, 6.0, 5.0]
    assert disagreement(y, [0.2, 0.4, 0.6]) == pytest.approx(disagreement(y, [1, 2, 3]))


def test_floor_for_near_zero_forecasts():
    # y_bar = 0.25 < floor 1, so the denominator is 1 and D stays bounded
    assert disagreement([0.0, 0.5], [1, 1]) == pytest.approx(0.25)


def test_route_boundary():
    assert route(0.3, 0.3) == "deliberate" and route(0.2999, 0.3) == "fast"


def test_score_table_matches_scalar_version():
    rng = np.random.default_rng(0)
    y = pd.DataFrame(rng.gamma(2, 5, (50, 3)), columns=list("abc"))
    c = pd.DataFrame(rng.uniform(0.2, 0.9, (50, 3)), columns=list("abc"))
    t = score_table(y, c)
    for i in range(50):
        assert t.D.iloc[i] == pytest.approx(disagreement(y.iloc[i], c.iloc[i]))
        assert t.D_range.iloc[i] == pytest.approx(d_range(y.iloc[i], c.iloc[i]))
        assert t.consensus.iloc[i] == pytest.approx(consensus(y.iloc[i], c.iloc[i])[0])


@pytest.mark.parametrize("bad", ["nan_forecast", "zero_conf", "misaligned"])
def test_score_table_rejects_missing_agents(bad):
    y = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    c = pd.DataFrame({"a": [0.5, 0.5], "b": [0.5, 0.5]})
    if bad == "nan_forecast":
        y.iloc[0, 0] = np.nan  # a failed agent must be excluded explicitly, never silently
    elif bad == "zero_conf":
        c.iloc[0, 0] = 0.0
    else:
        c = c[["b", "a"]]
    with pytest.raises(ValueError):
        score_table(y, c)
