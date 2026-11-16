"""Weight-of-Evidence (WoE) / Information-Value (IV) binning as a scikit-learn transformer.

Why not hand-picked bins?  Hard-coding cut-offs after eyeballing WoE plots (as in most tutorials) does not
generalise: it cannot be re-fitted inside cross-validation folds, silently breaks when the data scale changes
(e.g. a percent vs fraction column), and cannot be audited. ``WoEBinner`` learns the bins from the *training*
data only:

* numeric features  -> decision-tree splits on the single feature (``max_bins`` leaves, each holding at least
  ``min_bin_pct`` of the sample); the number of leaves is reduced until the bad rate is monotone across bins
  (a standard scorecard requirement), unless the feature is listed in ``monotonic_exceptions``.
* categorical features -> levels rarer than ``min_level_pct`` are pooled into ``OTHER (pooled)``; levels are then ordered by
  bad rate and contiguous levels are grouped with the same tree logic (so groups are monotone by construction).
* missing values get their own bin (with WoE = 0 if there are fewer than ``min_missing_count`` of them).

Conventions
-----------
``y = 1`` marks a *bad* (default). ``WoE = ln(%good / %bad)``, so a **higher WoE means a safer bin**.
Counts are smoothed by ``smoothing`` (default 0.5) to avoid infinite WoE in empty cells.
The last bin of every feature is always the "Missing / unseen" bin, so ``transform`` never fails on new data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted


POOL = "OTHER (pooled)"  # label for rare categorical levels


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"


def _is_monotone(bad_rate: np.ndarray, tol: float = 1e-12) -> bool:
    d = np.diff(bad_rate)
    return bool(np.all(d >= -tol) or np.all(d <= tol))


class WoEBinner(BaseEstimator, TransformerMixin):
    """Learn WoE bins on a DataFrame of features and a 0/1 bad flag; ``transform`` returns WoE values."""

    def __init__(self, max_bins: int = 6, min_bin_pct: float = 0.05, min_level_pct: float = 0.01,
                 min_missing_count: int = 100, smoothing: float = 0.5, monotonic: bool = True,
                 monotonic_exceptions=None, random_state: int = 0):
        self.max_bins = max_bins
        self.min_bin_pct = min_bin_pct
        self.min_level_pct = min_level_pct
        self.min_missing_count = min_missing_count
        self.smoothing = smoothing
        self.monotonic = monotonic
        self.monotonic_exceptions = monotonic_exceptions
        self.random_state = random_state

    # ------------------------------------------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------------------------------------------
    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y).astype(int)
        if len(X) != len(y):
            raise ValueError("X and y have different lengths")
        self.columns_ = list(X.columns)
        self.n_bad_ = int(y.sum())
        self.n_good_ = int(len(y) - y.sum())
        self.n_obs_ = len(y)
        self._specs: dict[str, dict] = {}
        self.tables_: dict[str, pd.DataFrame] = {}
        exceptions = set(self.monotonic_exceptions or [])
        for col in self.columns_:
            if pd.api.types.is_numeric_dtype(X[col]):
                spec = self._fit_numeric(X[col], y, col, enforce=self.monotonic and col not in exceptions)
            else:
                spec = self._fit_categorical(X[col], y, col)
            self._specs[col] = spec
            self.tables_[col] = spec["table"]
        self.iv_ = pd.Series({c: float(t["iv"].sum()) for c, t in self.tables_.items()},
                             name="iv").sort_values(ascending=False)
        return self

    def _tree_thresholds(self, v: np.ndarray, y: np.ndarray, n_total: int, enforce: bool) -> np.ndarray:
        """Split points of a depth-limited tree on one feature; shrink until bad rate is monotone."""
        if v.size == 0 or np.unique(v).size < 2:
            return np.array([])
        min_leaf = max(int(self.min_bin_pct * n_total), 30)
        thresholds = np.array([])
        for k in range(self.max_bins, 1, -1):
            tree = DecisionTreeClassifier(max_leaf_nodes=k, min_samples_leaf=min_leaf,
                                          random_state=self.random_state)
            tree.fit(v.reshape(-1, 1), y)
            t = tree.tree_
            thr = np.sort(t.threshold[t.feature >= 0])
            if thr.size == 0:
                return thr
            if not enforce or k == 2:
                return thr
            idx = np.searchsorted(thr, v, side="left")
            cnt = np.bincount(idx, minlength=thr.size + 1)
            bad = np.bincount(idx, weights=y, minlength=thr.size + 1)
            if _is_monotone(bad / np.maximum(cnt, 1)):
                return thr
            thresholds = thr
        return thresholds

    def _make_table(self, bins: np.ndarray, y: np.ndarray, labels: list[str], neutral_last: bool,
                    force_neutral_last: bool) -> pd.DataFrame:
        k = len(labels)
        cnt = np.bincount(bins, minlength=k).astype(float)
        bad = np.bincount(bins, weights=y, minlength=k).astype(float)
        good = cnt - bad
        s = self.smoothing
        dist_good = (good + s) / (self.n_good_ + s * k)
        dist_bad = (bad + s) / (self.n_bad_ + s * k)
        woe = np.log(dist_good / dist_bad)
        if neutral_last and (force_neutral_last or cnt[-1] == 0):
            woe[-1] = 0.0
        iv = (dist_good - dist_bad) * woe
        return pd.DataFrame({
            "bin_id": np.arange(k), "bin": labels, "count": cnt.astype(int), "share": cnt / max(cnt.sum(), 1),
            "bad": bad.astype(int), "good": good.astype(int), "bad_rate": bad / np.maximum(cnt, 1),
            "woe": woe, "iv": iv,
        })

    def _fit_numeric(self, s: pd.Series, y: np.ndarray, col: str, enforce: bool) -> dict:
        x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        miss = np.isnan(x)
        thr = self._tree_thresholds(x[~miss], y[~miss], len(x), enforce)
        nb = thr.size + 1
        if thr.size == 0:
            labels = ["All values"]
        else:
            labels = [f"<= {_fmt(thr[0])}"]
            labels += [f"({_fmt(thr[i - 1])}, {_fmt(thr[i])}]" for i in range(1, thr.size)]
            labels += [f"> {_fmt(thr[-1])}"]
        labels.append("Missing")
        bins = np.where(miss, nb, np.searchsorted(thr, np.where(miss, 0.0, x), side="left"))
        table = self._make_table(bins, y, labels, neutral_last=True,
                                 force_neutral_last=(miss.sum() < self.min_missing_count))
        return {"kind": "num", "thresholds": thr, "n_bins": nb, "woe": table["woe"].to_numpy(), "table": table}

    @staticmethod
    def _clean_cat(s: pd.Series) -> pd.Series:
        s = s.astype("object")
        return s.where(s.notna(), "MISSING").astype(str)

    def _fit_categorical(self, s: pd.Series, y: np.ndarray, col: str) -> dict:
        s = self._clean_cat(s)
        n = len(s)
        counts = s.value_counts()
        rare = set(counts[counts < self.min_level_pct * n].index)
        pooled = s.where(~s.isin(rare), POOL)
        frame = pd.DataFrame({"lvl": pooled.to_numpy(), "y": y})
        stat = frame.groupby("lvl")["y"].agg(["sum", "count"])
        stat["rate"] = stat["sum"] / stat["count"]
        order = stat.sort_values("rate", kind="mergesort").index.tolist()
        rank = {lvl: i for i, lvl in enumerate(order)}
        r = pooled.map(rank).to_numpy(dtype=float)
        thr = self._tree_thresholds(r, y, n, enforce=False)  # contiguous groups of a rate-ordered list
        grp_of = {lvl: int(np.searchsorted(thr, rank[lvl], side="left")) for lvl in order}
        ng = thr.size + 1
        members = [[lvl for lvl in order if grp_of[lvl] == g] for g in range(ng)]
        labels = [", ".join(m)[:90] + ("..." if len(", ".join(m)) > 90 else "") for m in members]
        labels.append("Unseen / missing")
        bins = pooled.map(grp_of).to_numpy(dtype=int)
        table = self._make_table(bins, y, labels, neutral_last=True, force_neutral_last=True)
        level_to_bin = {}
        for lvl in s.unique():
            key = POOL if lvl in rare else lvl
            if key in grp_of:
                level_to_bin[lvl] = grp_of[key]
        other_bin = grp_of.get(POOL)
        return {"kind": "cat", "level_to_bin": level_to_bin, "other_bin": other_bin, "n_bins": ng,
                "woe": table["woe"].to_numpy(), "table": table}

    # ------------------------------------------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------------------------------------------
    def _bin_index(self, col: str, s: pd.Series) -> np.ndarray:
        spec = self._specs[col]
        if spec["kind"] == "num":
            x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            miss = np.isnan(x)
            idx = np.searchsorted(spec["thresholds"], np.where(miss, 0.0, x), side="left")
            return np.where(miss, spec["n_bins"], idx).astype(int)
        cleaned = self._clean_cat(s)
        mapped = cleaned.map(spec["level_to_bin"])
        fallback = spec["other_bin"] if spec["other_bin"] is not None else spec["n_bins"]
        # known level -> its group; unseen level -> OTHER group if one exists, else the neutral 'unseen' bin
        out = mapped.fillna(fallback).to_numpy(dtype=int)
        return out

    def transform_bins(self, X, columns=None) -> pd.DataFrame:
        """Integer bin index per feature (the last index of each feature is its Missing/Unseen bin)."""
        check_is_fitted(self, "columns_")
        X = pd.DataFrame(X)
        columns = self.columns_ if columns is None else list(columns)
        return pd.DataFrame({c: self._bin_index(c, X[c]) for c in columns}, index=X.index)

    def transform(self, X) -> pd.DataFrame:
        """WoE-encode the features (higher = safer)."""
        check_is_fitted(self, "columns_")
        X = pd.DataFrame(X)
        out = {c: self._specs[c]["woe"][self._bin_index(c, X[c])] for c in self.columns_}
        return pd.DataFrame(out, index=X.index)

    def iv_on(self, X, y) -> pd.Series:
        """Out-of-sample IV: ``sum_b (%good_b - %bad_b) * WoE_b`` with the learned WoE but the *new* data's distributions.

        Unlike the in-sample IV (inflated because the bins were optimised on the same rows) this is ~0 for a
        feature without signal, and can be negative when a pattern does not hold up.
        """
        check_is_fitted(self, "columns_")
        X = pd.DataFrame(X)
        y = np.asarray(y).astype(int)
        n_bad, n_good = max(int(y.sum()), 1), max(int((y == 0).sum()), 1)
        out = {}
        for c in self.columns_:
            woe = self._specs[c]["woe"]
            idx = self._bin_index(c, X[c])
            cnt = np.bincount(idx, minlength=len(woe)).astype(float)
            bad = np.bincount(idx, weights=y, minlength=len(woe)).astype(float)
            out[c] = float(np.sum(((cnt - bad) / n_good - bad / n_bad) * woe))
        return pd.Series(out, name="oof_iv")

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)

    # ------------------------------------------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------------------------------------------
    def binning_table(self, col: str, include_empty: bool = False) -> pd.DataFrame:
        check_is_fitted(self, "columns_")
        t = self.tables_[col].copy()
        if not include_empty:
            t = t[t["count"] > 0]
        t.insert(0, "feature", col)
        return t.reset_index(drop=True)

    def iv_summary(self) -> pd.DataFrame:
        """IV and a conventional strength label per feature."""
        check_is_fitted(self, "columns_")
        iv = self.iv_.rename("iv").to_frame()
        iv["n_bins"] = [int((self.tables_[c]["count"] > 0).sum()) for c in iv.index]
        iv["strength"] = pd.cut(iv["iv"], [-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
                                labels=["useless (<0.02)", "weak", "medium", "strong", "suspicious (>0.5)"])
        iv.index.name = "feature"
        return iv.reset_index()


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Pick a fixed list of columns (used to keep the selected WoE features inside a Pipeline)."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[list(self.columns)]


def cross_validated_iv(X: pd.DataFrame, y, binner_params: dict, n_splits: int = 3, seed: int = 0) -> pd.Series:
    """Mean out-of-fold IV: bins learned on (k-1) folds, IV measured on the held-out fold.

    Use it to *screen* features: supervised binning makes in-sample IV and Wald p-values optimistic
    (noise features can look significant after the tree has searched for their best split).
    """
    from sklearn.model_selection import StratifiedKFold
    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    parts = []
    for tr, va in skf.split(X, y):
        b = WoEBinner(**binner_params).fit(X.iloc[tr], y[tr])
        parts.append(b.iv_on(X.iloc[va], y[va]))
    return pd.concat(parts, axis=1).mean(axis=1).rename("oof_iv")
