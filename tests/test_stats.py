import numpy as np

from retailsense.eval.stats import diebold_mariano, paired_bootstrap, wilcoxon_paired

RNG = np.random.default_rng(0)


def test_dm_detects_better_model_and_is_symmetric():
    e_good, e_bad = RNG.normal(0, 1, 500), RNG.normal(0, 2, 500)
    r = diebold_mariano(e_good, e_bad, h=7)
    assert r["dm"] < 0 and r["p"] < 0.01
    assert diebold_mariano(e_bad, e_good, h=7)["dm"] == -r["dm"]


def test_dm_no_difference_on_identical_distributions():
    assert diebold_mariano(RNG.normal(0, 1, 500), RNG.normal(0, 1, 500))["p"] > 0.01


def test_dm_identical_errors_is_undefined_not_significant():
    e = RNG.normal(0, 1, 50)
    assert np.isnan(diebold_mariano(e, e)["p"])


def test_bootstrap_ci_and_item_resampling():
    items = np.repeat(np.arange(40), 30)
    y = RNG.poisson(10, items.size) + 1.0
    a, b = np.abs(RNG.normal(0, 2, items.size)), np.abs(RNG.normal(0, 3, items.size))
    r = paired_bootstrap(items, a, b, y, n=1000, seed=1)
    assert r["ci_lo"] <= r["diff"] <= r["ci_hi"] < 0
    assert r == paired_bootstrap(items, a, b, y, n=1000, seed=1)  # reproducible
    same = paired_bootstrap(items.astype(str), a, a, y, n=200)
    assert same["diff"] == same["ci_lo"] == same["ci_hi"] == 0


def test_wilcoxon():
    a = RNG.normal(1, 0.1, 60)
    assert wilcoxon_paired(a - 0.2, a)["p"] < 0.001
    assert wilcoxon_paired(a, a)["p"] == 1.0
