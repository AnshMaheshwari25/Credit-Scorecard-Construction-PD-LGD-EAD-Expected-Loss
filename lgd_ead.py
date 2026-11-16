"""LGD and EAD models, fitted on charged-off loans.

Definitions (for a charged-off loan)
------------------------------------
* ``EAD_at_default = funded_amnt - total_rec_prncp``   principal still outstanding at charge-off
* ``CCF = EAD_at_default / funded_amnt``                exposure factor in [0, 1]. A term loan has no undrawn
  commitment, so this is the share of the original principal not yet repaid when the borrower defaults.
* ``recovery_rate = (recoveries - collection_recovery_fee) / EAD_at_default``  clipped to [0, 1]
* ``LGD = 1 - recovery_rate``

Recovery rates are zero-inflated (many defaults recover nothing) and bounded, so linear regression alone is a poor
fit. The primary model is therefore **two-stage**: (1) logistic regression for P(recovery > 0), (2) ridge regression
for the recovery rate given recovery > 0; expected recovery = P(>0) x E[rate | >0]. A single-stage ridge, a gradient
boosting regressor and the sample mean are reported as benchmarks.

.. warning::
   In a snapshot where defaults are recent (e.g. the 2015 file, observed ~Jan-2016) workouts are incomplete: realised
   recoveries understate eventual recoveries, so LGD is biased **upwards**. See README (limitations).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .metrics import regression_metrics


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=0.02))])
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)], sparse_threshold=0.0)


class TwoStageRecovery(BaseEstimator, RegressorMixin):
    """Hurdle model for a zero-inflated recovery rate in [0, 1]."""

    def __init__(self, classifier=None, regressor=None):
        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        pos = y > 0
        clf = self.classifier if self.classifier is not None else LogisticRegression(C=1.0, max_iter=2000)
        reg = self.regressor if self.regressor is not None else Ridge(alpha=10.0)
        self.p_const_ = float(pos.mean())
        if 0 < pos.sum() < len(y):
            self.clf_ = clone(clf).fit(X, pos.astype(int))
        else:
            self.clf_ = None  # degenerate: everything / nothing recovers
        if pos.sum() >= 5:
            self.reg_ = clone(reg).fit(X[pos], y[pos])
            self.mean_pos_ = float(y[pos].mean())
        else:
            self.reg_ = None
            self.mean_pos_ = float(y[pos].mean()) if pos.any() else 0.0
        return self

    def predict_components(self, X):
        p = self.clf_.predict_proba(X)[:, 1] if self.clf_ is not None else np.full(len(X), self.p_const_)
        m = np.clip(self.reg_.predict(X), 0, 1) if self.reg_ is not None else np.full(len(X), self.mean_pos_)
        return p, m

    def predict(self, X):
        p, m = self.predict_components(X)
        return np.clip(p * m, 0, 1)


def charged_off_sample(df: pd.DataFrame, min_ead: float = 1.0) -> pd.DataFrame:
    """Charged-off loans with a valid exposure, recovery rate and CCF."""
    co = df[df["is_charged_off"] & (df["ead_at_default"] > min_ead)]
    return co[co["recovery_rate"].notna() & co["ccf"].notna()].copy()


@dataclass
class LossModels:
    numeric: list
    categorical: list
    recovery: Pipeline          # predicts the expected recovery rate (LGD = 1 - recovery)
    ccf: Pipeline               # predicts the exposure factor
    lgd_comparison: pd.DataFrame
    ead_comparison: pd.DataFrame
    stage1_auc: float | None
    n_train: int
    n_test: int
    selection: pd.DataFrame     # CV check of each primary model against the naive mean (training data only)

    def _x(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.numeric + self.categorical]

    def predict_lgd(self, df: pd.DataFrame) -> np.ndarray:
        return 1.0 - np.clip(self.recovery.predict(self._x(df)), 0, 1)

    def predict_ccf(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.ccf.predict(self._x(df)), 0, 1)

    def predict_ead(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_ccf(df) * df["funded_amnt"].to_numpy(dtype=float)


def _recovery_candidates(num, cat, seed):
    pre = lambda: make_preprocessor(num, cat)  # noqa: E731
    return {
        "Mean (naive benchmark)": Pipeline([("pre", pre()), ("m", DummyRegressor(strategy="mean"))]),
        "Single-stage ridge": Pipeline([("pre", pre()), ("m", Ridge(alpha=10.0))]),
        "Two-stage (logit + ridge)": Pipeline([("pre", pre()), ("m", TwoStageRecovery())]),
        "Gradient boosting": Pipeline([("pre", pre()), ("m", HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=20, random_state=seed))]),
    }


def _ccf_candidates(num, cat, seed):
    pre = lambda: make_preprocessor(num, cat)  # noqa: E731
    return {
        "Mean (naive benchmark)": Pipeline([("pre", pre()), ("m", DummyRegressor(strategy="mean"))]),
        "Ridge": Pipeline([("pre", pre()), ("m", Ridge(alpha=10.0))]),
        "Gradient boosting": Pipeline([("pre", pre()), ("m", HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=20, random_state=seed))]),
    }


def _cv_rmse(pipe, X, y, seed: int, folds: int = 5) -> float:
    """Cross-validated RMSE on the training sample (predictions clipped to [0, 1])."""
    y = np.asarray(y, dtype=float)
    pred = cross_val_predict(clone(pipe), X, y, cv=KFold(folds, shuffle=True, random_state=seed))
    return float(np.sqrt(np.mean((y - np.clip(pred, 0, 1)) ** 2)))


def fit_loss_models(train: pd.DataFrame, test: pd.DataFrame, numeric: list[str], categorical: list[str],
                    min_ead: float = 1.0, seed: int = 42, min_n: int = 40) -> LossModels:
    """Fit LGD (two-stage) and EAD/CCF models on charged-off loans of the training split; compare on the test split."""
    tr, te = charged_off_sample(train, min_ead), charged_off_sample(test, min_ead)
    if len(tr) < min_n or len(te) < 10:
        raise ValueError(f"Too few charged-off loans for LGD/EAD modelling (train={len(tr)}, test={len(te)}). "
                         "Use a larger / more seasoned sample (e.g. the 2007-2014 file).")
    cols = numeric + categorical
    Xtr, Xte = tr[cols], te[cols]

    # ---- recovery rate / LGD -------------------------------------------------------------------------------
    rows, fitted = [], {}
    for name, pipe in _recovery_candidates(numeric, categorical, seed).items():
        pipe.fit(Xtr, tr["recovery_rate"])
        pred_rr = np.clip(pipe.predict(Xte), 0, 1)
        m = regression_metrics(te["lgd"], 1 - pred_rr)  # LGD scale
        rows.append({"model": name, **{k: m[k] for k in ("rmse", "mae", "r2", "spearman", "mean_pred")}})
        fitted[name] = pipe
    lgd_cmp = pd.DataFrame(rows)
    lgd_cmp["mean_actual"] = te["lgd"].mean()
    two = fitted["Two-stage (logit + ridge)"]
    p1, _ = two.named_steps["m"].predict_components(two.named_steps["pre"].transform(Xte))
    pos_te = (te["recovery_rate"] > 0).astype(int)
    auc1 = float(roc_auc_score(pos_te, p1)) if pos_te.nunique() == 2 else None

    # ---- exposure factor / EAD -------------------------------------------------------------------------------
    rows2, fitted2 = [], {}
    for name, pipe in _ccf_candidates(numeric, categorical, seed).items():
        pipe.fit(Xtr, tr["ccf"])
        pred = np.clip(pipe.predict(Xte), 0, 1)
        m = regression_metrics(te["ccf"], pred)
        mae_usd = float(np.mean(np.abs(pred * te["funded_amnt"] - te["ead_at_default"])))
        rows2.append({"model": name, **{k: m[k] for k in ("rmse", "mae", "r2", "spearman", "mean_pred")},
                      "ead_mae_usd": mae_usd})
        fitted2[name] = pipe
    ead_cmp = pd.DataFrame(rows2)
    ead_cmp["mean_actual"] = te["ccf"].mean()

    # ---- governance rule: do not deploy a model that cannot beat the naive mean in cross-validation (train only) ----
    naive_name = "Mean (naive benchmark)"
    cands_r = _recovery_candidates(numeric, categorical, seed)
    cands_c = _ccf_candidates(numeric, categorical, seed)
    sel_rows, recovery, ccf = [], two, fitted2["Ridge"]
    for comp, prim_name, cands, target, fit_models in (
            ("LGD (recovery rate)", "Two-stage (logit + ridge)", cands_r, "recovery_rate", fitted),
            ("EAD (CCF)", "Ridge", cands_c, "ccf", fitted2)):
        cv_p = _cv_rmse(cands[prim_name], Xtr, tr[target], seed)
        cv_n = _cv_rmse(cands[naive_name], Xtr, tr[target], seed)
        use = cv_p < cv_n
        sel_rows.append({"component": comp, "primary_model": prim_name, "cv_rmse_primary": cv_p,
                         "cv_rmse_naive_mean": cv_n, "deployed": prim_name if use else "Mean (fallback: no skill)"})
        if not use:
            if comp.startswith("LGD"):
                recovery = fit_models[naive_name]
            else:
                ccf = fit_models[naive_name]
    return LossModels(numeric, categorical, recovery=recovery, ccf=ccf, lgd_comparison=lgd_cmp,
                      ead_comparison=ead_cmp, stage1_auc=auc1, n_train=len(tr), n_test=len(te),
                      selection=pd.DataFrame(sel_rows))
