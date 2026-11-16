"""Discrimination, calibration, stability and cut-off-strategy metrics for PD scorecards.

All functions take ``y`` with 1 = bad (default) and ``p`` = predicted probability of default (or a score where stated).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score, roc_curve)


# --------------------------------------------------------------------------------------------------------
# Discrimination
# --------------------------------------------------------------------------------------------------------
def ks_statistic(y, p) -> float:
    """Kolmogorov-Smirnov: max separation between the cumulative bad and good score distributions."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def evaluate_binary(y, p) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("y and p must be non-empty one-dimensional arrays of equal length")
    if pd.isna(y).any() or not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain only non-missing 0/1 labels")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("p must contain finite probabilities between 0 and 1")
    y = y.astype(int)
    if np.unique(y).size != 2:
        raise ValueError("Binary evaluation requires both target classes")
    auc = roc_auc_score(y, p)
    return {
        "auroc": float(auc), "gini": float(2 * auc - 1), "ks": ks_statistic(y, p),
        "pr_auc": float(average_precision_score(y, p)),  # PR-AUC of the *bad* class (the rare one)
        "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, np.clip(p, 1e-9, 1 - 1e-9))),
        "bad_rate": float(y.mean()), "mean_pd": float(p.mean()), "n": int(len(y)), "n_bad": int(y.sum()),
    }


def bootstrap_ci(y, p, metric=roc_auc_score, n_boot: int = 200, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for a ranking metric (default AUROC)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        vals.append(metric(y[i], p[i]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


# --------------------------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------------------------
def decile_table(y, p, n: int = 10) -> pd.DataFrame:
    """Risk deciles (1 = highest predicted PD): predicted vs observed bad rate, gains and lift."""
    d = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(p, dtype=float)})
    d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False), n, labels=range(1, n + 1))
    g = d.groupby("decile", observed=True).agg(count=("y", "size"), bads=("y", "sum"), avg_pd=("p", "mean"),
                                               bad_rate=("y", "mean")).reset_index()
    g["cum_bad_capture_%"] = 100 * g["bads"].cumsum() / g["bads"].sum()
    g["lift"] = g["bad_rate"] / d["y"].mean()
    return g


def hosmer_lemeshow(y, p, g: int = 10) -> tuple[float, float]:
    """Hosmer-Lemeshow chi-square (g-2 d.o.f.). Very sensitive for large samples: read with the decile table."""
    t = decile_table(y, p, g)
    exp_bad = t["avg_pd"] * t["count"]
    exp_good = (1 - t["avg_pd"]) * t["count"]
    stat = (((t["bads"] - exp_bad) ** 2 / exp_bad) + (((t["count"] - t["bads"]) - exp_good) ** 2 / exp_good)).sum()
    return float(stat), float(chi2.sf(stat, g - 2))


# --------------------------------------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------------------------------------
def psi(expected, actual, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index of a continuous variable / score, using quantile bins of ``expected``.

    Rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant shift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def csi_table(train_bins: pd.DataFrame, new_bins: pd.DataFrame, eps: float = 1e-4) -> pd.DataFrame:
    """Characteristic Stability Index per scorecard variable from integer bin indices."""
    rows = []
    for c in train_bins.columns:
        k = int(max(train_bins[c].max(), new_bins[c].max())) + 1
        e = np.bincount(train_bins[c], minlength=k) / len(train_bins)
        a = np.bincount(new_bins[c], minlength=k) / len(new_bins)
        e, a = np.clip(e, eps, None), np.clip(a, eps, None)
        rows.append((c, float(np.sum((a - e) * np.log(a / e)))))
    return pd.DataFrame(rows, columns=["variable", "csi"]).sort_values("csi", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------------------------------------
# Strategy / cut-off analysis (scores: higher = safer)
# --------------------------------------------------------------------------------------------------------
def score_band_table(score, y, width: int = 25) -> pd.DataFrame:
    s = pd.Series(np.asarray(score, dtype=float))
    d = pd.DataFrame({"score": s, "y": np.asarray(y).astype(int)})
    lo = np.floor(d["score"].min() / width) * width
    hi = np.ceil(d["score"].max() / width) * width
    d["band"] = pd.cut(d["score"], np.arange(lo, hi + width, width), include_lowest=True)
    g = d.groupby("band", observed=True).agg(count=("y", "size"), bads=("y", "sum"), bad_rate=("y", "mean")).reset_index()
    g["share_%"] = 100 * g["count"] / g["count"].sum()
    g["band"] = g["band"].astype(str)
    return g


def strategy_table(score, y, approval_rates=(1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5)) -> pd.DataFrame:
    """Approve everyone scoring >= cut-off. For each approval rate: the cut-off and its portfolio bad rates."""
    score = np.asarray(score, dtype=float)
    y = np.asarray(y).astype(int)
    rows = []
    for a in approval_rates:
        cut = np.quantile(score, 1 - a) if a < 1 else score.min()
        ok = score >= cut
        rows.append({"target_approval_rate": a, "cutoff_score": cut, "approval_rate": ok.mean(),
                     "approved_bad_rate": y[ok].mean() if ok.any() else np.nan,
                     "rejected_bad_rate": y[~ok].mean() if (~ok).any() else np.nan,
                     "bads_rejected_%": 100 * y[~ok].sum() / max(y.sum(), 1)})
    return pd.DataFrame(rows)


def ks_cutoff(score, y) -> float:
    """Score cut-off at the KS maximum (approve if score > cut-off)."""
    fpr, tpr, thr = roc_curve(np.asarray(y).astype(int), -np.asarray(score, dtype=float))
    return float(-thr[int(np.argmax(tpr - fpr))])


# --------------------------------------------------------------------------------------------------------
# Regression metrics (LGD / EAD)
# --------------------------------------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    from scipy.stats import spearmanr
    rho = (spearmanr(y_true, y_pred).statistic if (np.ptp(y_pred) > 1e-12 and np.ptp(y_true) > 0)
           else np.nan)
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))), "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)), "spearman": float(rho), "mean_actual": float(y_true.mean()),
            "mean_pred": float(y_pred.mean()), "n": int(len(y_true))}
