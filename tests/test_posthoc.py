import pandas as pd
import pytest

from retailsense.eval.posthoc import quadrant_table, tail_rates


def test_tail_rates():
    e = pd.DataFrame({"y": [0, 5, 5, 20], "lower": [1, 1, 1, 1], "upper": [10, 10, 10, 10]})
    r = tail_rates(e)
    assert r["below_lower"] == pytest.approx(0.25)
    assert r["above_upper"] == pytest.approx(0.25)
    assert r["coverage"] == pytest.approx(0.5)


def test_quadrant_table():
    # 2 high-D points, 2 low-D; the worst error (rel_err 1.0) is a low-D shared miss
    pts = pd.DataFrame({
        "D": [0.5, 0.5, 0.0, 0.0],
        "rel_err": [0.1, 0.2, 1.0, 0.05],
        "y": [10, 10, 10, 10],
        "f_a": [9, 12, 0, 10],
        "f_b": [11, 8, 0, 10],
    })
    t = quadrant_table(pts, tau=0.3, share=0.25)
    assert t.loc[("low_D", "high_err"), "n"] == 1
    assert t.loc[("low_D", "high_err"), "share_of_worst"] == 1.0
    assert t.loc[("low_D", "high_err"), "same_sign"] == 1.0
    assert t.loc[("high_D", "low_err"), "same_sign"] == 0.0
    assert t.n.sum() == 4


def test_item_aware_factory_adds_item_identity():
    from retailsense.agents.calendar import calendar_agent
    from retailsense.config import load_experiment
    from retailsense.eval.posthoc import item_aware_factory

    cfg = load_experiment()
    frozen = calendar_agent(cfg).features
    aware = item_aware_factory(calendar_agent)(cfg).features
    assert aware == [*frozen, "item_id"] and aware[0] == "baseline"


def test_router_cv_finds_a_planted_signal():
    import numpy as np
    import pandas as pd

    from retailsense.eval.posthoc import router_cv

    rng = np.random.default_rng(0)
    n = 1500
    s = pd.DataFrame({
        "d": rng.integers(1, 87, n),
        "D": rng.random(n),
        "D_range": rng.random(n),
        "hist_band": rng.random(n),
        "low_volume": rng.random(n),
        "y": rng.integers(5, 50, n).astype(float),
    })
    s["consensus"] = s.y + rng.normal(0, 3, n)
    # deliberation fixes the error exactly when hist_band is high, and hurts otherwise
    good = s.hist_band > 0.8
    s["final_forecast"] = np.where(good, s.y, s.consensus + rng.normal(0, 3, n))
    s["gain"] = (s.y - s.consensus).abs() - (s.y - s.final_forecast).abs()
    t = router_cv(s, [0.2], folds=5, seed=0, draws=50).loc[0.2]
    assert t.oracle <= t.learned < t.random_mean - 3 * t.random_sd


def test_holm_matches_hand_computation():
    import pandas as pd

    from retailsense.eval.posthoc import holm

    adj = holm(pd.Series({"a": 0.01, "b": 0.04, "c": 0.03}))
    # sorted 0.01, 0.03, 0.04 -> x3, x2, x1 = 0.03, 0.06, 0.04 -> monotone 0.03, 0.06, 0.06
    assert adj.round(10).to_dict() == {"a": 0.03, "b": 0.06, "c": 0.06}


def test_equivalence_row_flags_identical_systems():
    import numpy as np

    from retailsense.eval.posthoc import equivalence_row

    rng = np.random.default_rng(0)
    items = np.repeat(np.arange(30), 20)
    y = rng.integers(5, 50, items.size).astype(float)
    e = np.abs(rng.normal(0, 5, items.size))
    r = equivalence_row(items, e, e + 1e-6, y, 0.3, 0.001, 200, 0)
    assert all(r[k] for k in r if k.startswith("equivalent_at"))
    far = equivalence_row(items, e, e + 5, y, 0.3, 0.001, 200, 0)
    assert not any(far[k] for k in far if k.startswith("equivalent_at"))


def test_schema_in_prompt_strips_fence_and_adds_schema():
    from pydantic import BaseModel

    from retailsense.eval.posthoc import schema_in_prompt

    class S(BaseModel):
        a: int

    seen = {}

    def transport(system, user, schema):
        seen["system"] = system
        return '```json\n{"a": 1}\n```', 1, 1, "m", 0.0

    text, *_ = schema_in_prompt(transport)("SYS", "U", S)
    assert S.model_validate_json(text).a == 1
    assert seen["system"].startswith("SYS") and '"a"' in seen["system"]


def test_attribution_table_detects_a_critic_that_names_the_driver():
    import pandas as pd

    from retailsense.eval.posthoc import attribution_table

    rows, drivers = [], []
    for d in range(200):
        drv = ["weather", "calendar", "trend"][d % 3]
        drivers.append({"item_id": "x", "d": d, "driver": drv, "driver_size": 0.5})
        for a in ("weather", "calendar", "trend"):
            rows.append({"item_id": "x", "d": d, "agent": a,
                         "strength": "strong" if a == drv else "weak",
                         "m": 1.5 if a == drv else 1.0})
    t = attribution_table(pd.DataFrame(rows), pd.DataFrame(drivers))
    assert t.strong_share_driver_agent == 1 and t.strong_share_other_agents == 0
    assert t.driver_is_most_upweighted == 1 and t.m_ci_lo > 0
