"""Variable selection for a WoE logistic-regression scorecard.

Pipeline (applied to WoE-encoded training data, ``y = 1`` = bad):

1. **IV screen**       - drop features whose *out-of-fold* Information Value is below ``iv_min`` (in-sample IV is
   biased upwards by supervised binning); *flag* implausibly strong ones (leakage / circularity check).
2. **Correlation**     - walking down the IV ranking, drop a feature if |corr| with an already kept one > ``corr_max``.
3. **Backward elimination** on a (practically) unpenalised logistic regression:
   drop wrong-signed coefficients first (with WoE = ln(good/bad) every coefficient must be **negative**),
   then the least significant coefficient while p > ``p_max``, then the worst VIF while VIF > ``vif_max``.

Coefficient standard errors / p-values come from the Fisher information matrix computed directly
(``logit_summary``), so only scikit-learn / NumPy / SciPy are required. (``tests/`` cross-checks them
against statsmodels when it is installed.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression


def fit_logit(X, y, C: float = 1e6, max_iter: int = 2000) -> LogisticRegression:
    """(Practically) unpenalised logistic regression. Large C avoids the deprecated ``penalty`` argument."""
    model = LogisticRegression(C=C, max_iter=max_iter, tol=1e-8)
    model.fit(np.asarray(X, dtype=float), np.asarray(y).astype(int))
    return model


def logit_summary(model: LogisticRegression, X, names) -> pd.DataFrame:
    """Coefficient table with Wald standard errors, z-scores and p-values (unpenalised-MLE approximation)."""
    X = np.asarray(X, dtype=float)
    Xd = np.column_stack([np.ones(len(X)), X])
    p = model.predict_proba(X)[:, 1]
    w = p * (1.0 - p)
    hessian = (Xd * w[:, None]).T @ Xd
    cov = np.linalg.pinv(hessian)
    beta = np.r_[model.intercept_[0], model.coef_[0]]
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = beta / se
    return pd.DataFrame({"coef": beta, "std_err": se, "z": z, "p_value": 2 * norm.sf(np.abs(z))},
                        index=["Intercept"] + list(names))


def compute_vif(X: pd.DataFrame, max_rows: int = 100_000, seed: int = 0) -> pd.Series:
    """Variance inflation factors from the inverse correlation matrix."""
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed)
    if X.shape[1] == 1:
        return pd.Series([1.0], index=X.columns, name="vif")
    corr = np.corrcoef(X.to_numpy(dtype=float), rowvar=False)
    return pd.Series(np.diag(np.linalg.pinv(corr)), index=X.columns, name="vif")


def correlation_filter(woe: pd.DataFrame, ordered: list[str], corr_max: float):
    """Keep the higher-IV feature of any pair with |corr| > corr_max (``ordered`` is sorted by IV desc)."""
    corr = woe[ordered].corr().abs()
    kept: list[str] = []
    dropped = []
    for f in ordered:
        clash = [(k, corr.loc[f, k]) for k in kept if corr.loc[f, k] > corr_max]
        if clash:
            k, c = max(clash, key=lambda t: t[1])
            dropped.append((f, f"|corr| {c:.2f} with higher-IV feature {k}"))
        else:
            kept.append(f)
    return kept, dropped


def backward_eliminate(woe: pd.DataFrame, y, candidates: list[str], p_max: float = 0.05,
                       vif_max: float = 5.0, C: float = 1e6, expected_sign: int = -1):
    """Iteratively drop wrong-signed, insignificant and collinear features. Returns (model, features, log)."""
    feats = list(candidates)
    log = []
    while True:
        model = fit_logit(woe[feats], y, C)
        summ = logit_summary(model, woe[feats], feats).drop(index="Intercept")
        wrong = summ[np.sign(summ["coef"]) != expected_sign]
        if len(wrong) and len(feats) > 1:
            f = wrong["p_value"].idxmax()
            log.append((f, f"wrong sign (coef {summ.loc[f, 'coef']:+.3f})"))
            feats.remove(f)
            continue
        if summ["p_value"].max() > p_max and len(feats) > 1:
            f = summ["p_value"].idxmax()
            log.append((f, f"p-value {summ.loc[f, 'p_value']:.3f} > {p_max}"))
            feats.remove(f)
            continue
        vif = compute_vif(woe[feats])
        if vif.max() > vif_max and len(feats) > 1:
            f = vif.idxmax()
            log.append((f, f"VIF {vif[f]:.1f} > {vif_max}"))
            feats.remove(f)
            continue
        return model, feats, log


def select_features(woe: pd.DataFrame, y, iv: pd.Series, sel_cfg: dict, C: float = 1e6,
                    oof_iv: pd.Series | None = None) -> dict:
    """Run the full selection cascade. Returns a dict with ``features``, ``model``, ``log`` and diagnostics.

    ``iv`` is the in-sample IV; if ``oof_iv`` (cross-validated IV) is supplied it is used for the IV screen and
    for ordering the correlation filter, because in-sample IV is biased upwards by supervised binning.
    """
    rank = iv if oof_iv is None else oof_iv.reindex(iv.index)
    rank = rank.sort_values(ascending=False)
    label = "IV" if oof_iv is None else "out-of-fold IV"
    rows = []
    keep = rank[rank >= sel_cfg["iv_min"]].index.tolist()
    rows += [("1. IV screen", f, f"{label} {rank[f]:.4f} < {sel_cfg['iv_min']} (in-sample IV {iv[f]:.4f})")
             for f in rank.index if f not in keep]
    flagged = [f for f in keep if iv[f] > sel_cfg["iv_suspicious"]]
    rows += [("flag (kept)", f, f"IV {iv[f]:.3f} > {sel_cfg['iv_suspicious']}: check for leakage / circularity")
             for f in flagged]
    kept, dropped = correlation_filter(woe, keep, sel_cfg["corr_max"])
    rows += [("2. correlation", f, why) for f, why in dropped]
    model, feats, log = backward_eliminate(woe, y, kept, sel_cfg["p_max"], sel_cfg["vif_max"], C)
    rows += [("3. backward elimination", f, why) for f, why in log]
    summary = logit_summary(model, woe[feats], feats)
    return {
        "features": feats,
        "model": model,
        "log": pd.DataFrame(rows, columns=["stage", "feature", "reason"]),
        "summary": summary,
        "vif": compute_vif(woe[feats]),
        "oof_iv": rank,
        "n_candidates": [len(iv), len(keep), len(kept), len(feats)],
    }
