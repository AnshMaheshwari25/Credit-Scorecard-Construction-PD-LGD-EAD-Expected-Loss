import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src import metrics as M
from src.feature_selection import fit_logit, logit_summary, compute_vif
from src.scorecard import Scorecard
from src.woe_binning import WoEBinner


def small_problem(n=30000, seed=1):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n), rng.normal(size=n)
    cat = rng.choice(list("xyz"), size=n)
    lo = -3 + 0.8 * a + 0.5 * b + 0.4 * (cat == "z")
    y = (rng.random(n) < 1 / (1 + np.exp(-lo))).astype(int)
    return pd.DataFrame({"a": a, "b": b, "cat": cat}), y


@pytest.fixture(scope="module")
def card():
    X, y = small_problem()
    binner = WoEBinner(max_bins=5).fit(X, y)
    W = binner.transform(X)
    model = fit_logit(W, y)
    return X, y, Scorecard(binner, model, list(X.columns), base_score=600, base_odds=50, pdo=20)


def test_score_identity_offset_plus_factor_log_odds(card):
    X, y, sc = card
    p = sc.predict_pd(X)
    exact = sc.score(X, exact=True)
    expected = sc.offset + sc.factor * np.log((1 - p) / p)
    assert np.allclose(exact, expected, atol=1e-8)


def test_pdo_property_every_pdo_points_doubles_odds(card):
    _, _, sc = card
    s = np.array([600.0, 620.0])
    pd_ = sc.pd_from_score(s)
    odds = (1 - pd_) / pd_
    assert odds[0] == pytest.approx(50.0, rel=1e-9)          # base score <-> base odds
    assert odds[1] / odds[0] == pytest.approx(2.0, rel=1e-9)  # +PDO points <-> odds x2


def test_pd_score_roundtrip_and_rounding_error(card):
    X, _, sc = card
    p = sc.predict_pd(X)
    assert np.allclose(sc.pd_from_score(sc.score_from_pd(p)), p, atol=1e-9)
    err = (sc.score(X) - sc.score(X, exact=True)).abs().max()
    assert err <= 0.5 * len(sc.features) + 1e-9


def test_points_fall_as_bin_risk_rises_and_reason_codes(card):
    X, y, sc = card
    assert all(c < 0 for c in sc.coefs.values())                  # WoE = ln(good/bad) => negative coefficients
    for _, g in sc.points_table.groupby("variable"):
        pts = g.sort_values("bad_rate")["points"].to_numpy()
        assert (np.diff(pts) <= 1e-9).all()                       # riskier bin => fewer points
    assert roc_auc_score(y, -sc.score(X)) > 0.65
    rc = sc.reason_codes(X.head(3), n=2)
    assert rc.shape == (3, 2) and rc.notna().all().all()


def test_metrics_basic():
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    p = np.array([.1, .2, .15, .8, .6, .3, .4, .05, .5, .9])
    ev = M.evaluate_binary(y, p)
    assert ev["auroc"] == pytest.approx(roc_auc_score(y, p))
    assert ev["gini"] == pytest.approx(2 * ev["auroc"] - 1)
    assert 0 <= ev["ks"] <= 1
    assert M.psi(p, p) == pytest.approx(0, abs=1e-9)
    shifted = M.psi(np.random.default_rng(0).normal(size=5000), np.random.default_rng(1).normal(1, 1, 5000))
    assert shifted > 0.25


def test_logit_summary_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    X, y = small_problem(8000)
    W = WoEBinner(max_bins=4).fit(X, y).transform(X)
    m = fit_logit(W, y, C=1e8)
    ours = logit_summary(m, W, list(W.columns))
    ref = sm.Logit(y, sm.add_constant(W.to_numpy())).fit(disp=0)
    assert np.allclose(ours["coef"].to_numpy(), ref.params, atol=2e-3)
    assert np.allclose(ours["std_err"].to_numpy(), ref.bse, rtol=2e-2)


def test_vif_detects_collinearity():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    X = pd.DataFrame({"a": a, "b": a + 0.05 * rng.normal(size=5000), "c": rng.normal(size=5000)})
    v = compute_vif(X)
    assert v["a"] > 10 and v["b"] > 10 and v["c"] < 1.5
