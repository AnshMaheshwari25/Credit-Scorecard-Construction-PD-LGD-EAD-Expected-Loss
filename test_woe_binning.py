import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.woe_binning import ColumnSelector, WoEBinner, cross_validated_iv


def make_data(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x_sig = rng.normal(size=n)
    x_u = rng.normal(size=n)                      # U-shaped relation: non-monotone
    x_noise = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c", "d", "rare"], size=n, p=[0.35, 0.3, 0.2, 0.149, 0.001])
    lo = -3.5 + 0.9 * x_sig + 0.5 * (x_u ** 2 - 1) + 0.6 * (cat == "d")
    y = (rng.random(n) < 1 / (1 + np.exp(-lo))).astype(int)
    x_sig_m = x_sig.copy()
    x_sig_m[rng.random(n) < 0.1] = np.nan
    return pd.DataFrame({"sig": x_sig_m, "u": x_u, "noise": x_noise, "cat": cat}), y


def test_woe_sign_and_monotone_bins():
    X, y = make_data()
    b = WoEBinner(max_bins=6, min_bin_pct=0.05).fit(X, y)
    t = b.binning_table("sig")
    body = t[t["bin"] != "Missing"]
    d = np.diff(body["bad_rate"].to_numpy())
    assert np.all(d >= -1e-12) or np.all(d <= 1e-12)                         # monotone bad rate across bins
    # WoE = ln(good/bad): the riskier the bin, the LOWER its WoE (strictly opposite ordering)
    assert (np.sign(np.diff(body["woe"])) == -np.sign(np.diff(body["bad_rate"]))).all()
    assert body["bad_rate"].is_monotonic_increasing                           # higher 'sig' raises PD in make_data


def test_iv_ranking_and_nonnegative():
    X, y = make_data()
    b = WoEBinner().fit(X, y)
    assert (b.iv_ >= -1e-12).all()
    assert b.iv_["sig"] > b.iv_["noise"]
    assert b.iv_["noise"] < 0.02


def test_oof_iv_near_zero_for_noise_and_positive_for_signal():
    X, y = make_data(40000)
    oof = cross_validated_iv(X, y, dict(max_bins=6, min_bin_pct=0.05), n_splits=3)
    assert oof["noise"] < 0.01
    assert oof["sig"] > 0.05


def test_monotonic_exception_allows_u_shape():
    X, y = make_data(40000)
    mono = WoEBinner(max_bins=6, monotonic=True).fit(X[["u"]], y)
    free = WoEBinner(max_bins=6, monotonic=True, monotonic_exceptions=["u"]).fit(X[["u"]], y)
    assert free.iv_["u"] > mono.iv_["u"]                       # U-shape is captured only when allowed


def test_missing_gets_own_bin_and_unseen_category_is_safe():
    X, y = make_data()
    b = WoEBinner(min_missing_count=50).fit(X, y)
    assert "Missing" in b.binning_table("sig")["bin"].values
    new = pd.DataFrame({"sig": [np.nan, 0.0], "u": [0.0, np.nan], "noise": [0.0, 0.0], "cat": ["zzz", None]})
    W = b.transform(new)
    assert W.shape == (2, 4) and not W.isna().any().any()


def test_rare_levels_are_pooled():
    X, y = make_data()
    b = WoEBinner(min_level_pct=0.01).fit(X, y)
    labels = " ".join(b.binning_table("cat")["bin"])
    assert "OTHER (pooled)" in labels and "rare" not in labels.replace("OTHER (pooled)", "")


def test_transform_keeps_index_and_columns():
    X, y = make_data(5000)
    X.index = np.arange(1000, 1000 + len(X))
    b = WoEBinner().fit(X, y)
    W = b.transform(X.iloc[10:20])
    assert list(W.index) == list(X.index[10:20]) and list(W.columns) == list(X.columns)


def test_works_in_cv_pipeline_and_clone():
    X, y = make_data(20000)
    pipe = Pipeline([("woe", WoEBinner()), ("sel", ColumnSelector(["sig", "cat"])),
                     ("lr", LogisticRegression(C=1e6, max_iter=1000))])
    auc = cross_val_score(clone(pipe), X, y, scoring="roc_auc", cv=3)
    assert auc.mean() > 0.65
