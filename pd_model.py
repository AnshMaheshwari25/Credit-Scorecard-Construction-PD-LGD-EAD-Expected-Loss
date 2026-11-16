"""PD model: WoE binning -> variable selection -> logistic regression -> scorecard, plus benchmarks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .feature_selection import select_features
from .scorecard import Scorecard
from .woe_binning import ColumnSelector, WoEBinner, cross_validated_iv


def _woe_params(cfg: dict) -> dict:
    w = cfg["woe"]
    return {k: w[k] for k in ("max_bins", "min_bin_pct", "min_level_pct", "min_missing_count")}


@dataclass
class PDModel:
    name: str
    columns: list
    binner: WoEBinner
    features: list
    scorecard: Scorecard
    selection: dict

    @property
    def model(self) -> LogisticRegression:
        return self.scorecard.model

    def predict_pd(self, X: pd.DataFrame) -> pd.Series:
        return self.scorecard.predict_pd(X)

    def score(self, X: pd.DataFrame, exact: bool = False) -> pd.Series:
        return self.scorecard.score(X, exact=exact)


def fit_pd_model(train: pd.DataFrame, columns: list[str], cfg: dict, name: str = "standalone",
                 target: str = "bad") -> PDModel:
    """Fit binning, select variables and build the scorecard on the training sample only."""
    y = train[target].astype(int).to_numpy()
    binner = WoEBinner(**_woe_params(cfg)).fit(train[columns], y)
    woe = binner.transform(train[columns])
    folds = cfg["selection"].get("oof_iv_folds", 3)
    oof = (cross_validated_iv(train[columns], y, _woe_params(cfg), n_splits=folds,
                              seed=cfg["sample"]["random_state"]) if folds else None)
    sel = select_features(woe, y, binner.iv_, cfg["selection"], C=cfg["model"]["C"], oof_iv=oof)
    sc = Scorecard(binner, sel["model"], sel["features"], **cfg["scorecard"], name=name)
    return PDModel(name=name, columns=list(columns), binner=binner, features=sel["features"], scorecard=sc,
                   selection=sel)


def make_cv_pipeline(features: list[str], cfg: dict) -> Pipeline:
    """Pipeline that re-learns the bins inside every CV fold (no leakage from the validation fold)."""
    return Pipeline([
        ("woe", WoEBinner(**_woe_params(cfg))),
        ("select", ColumnSelector(features)),
        ("lr", LogisticRegression(C=cfg["model"]["C"], max_iter=2000)),
    ])


def cross_validate_pd(train: pd.DataFrame, pdm: PDModel, cfg: dict, target: str = "bad") -> np.ndarray:
    """Repeated stratified CV AUROC of the full binning + LR pipeline (variable list fixed from the training fit)."""
    cv = RepeatedStratifiedKFold(n_splits=cfg["model"]["cv_splits"], n_repeats=cfg["model"]["cv_repeats"],
                                 random_state=cfg["sample"]["random_state"])
    pipe = make_cv_pipeline(pdm.features, cfg)
    return cross_val_score(pipe, train[pdm.columns], train[target].astype(int), scoring="roc_auc", cv=cv)


def fit_gbm_benchmark(train: pd.DataFrame, columns: list[str], cfg: dict, target: str = "bad") -> Pipeline:
    """Gradient-boosting benchmark on the raw (un-binned) features: an upper-bound check for the scorecard."""
    num = [c for c in columns if pd.api.types.is_numeric_dtype(train[c])]
    cat = [c for c in columns if c not in num]
    h = cfg["benchmark"]["hgb"]
    pre = ColumnTransformer(
        [("num", "passthrough", num),
         ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                encoded_missing_value=-1), cat)],
        verbose_feature_names_out=False)
    gbm = HistGradientBoostingClassifier(
        categorical_features=[False] * len(num) + [True] * len(cat), max_iter=h["max_iter"],
        learning_rate=h["learning_rate"], max_leaf_nodes=h["max_leaf_nodes"],
        min_samples_leaf=h["min_samples_leaf"], l2_regularization=h["l2_regularization"],
        early_stopping=True, random_state=cfg["sample"]["random_state"])
    pipe = Pipeline([("prep", pre), ("gbm", gbm)])
    pipe.fit(train[columns], train[target].astype(int))
    pipe.feature_names_ = num + cat
    return pipe
