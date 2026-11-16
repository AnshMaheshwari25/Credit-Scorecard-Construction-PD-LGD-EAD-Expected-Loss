"""Scorecard: turn a WoE logistic regression into points, with PDO scaling and reason codes.

Scaling (Siddiqi, *Credit Risk Scorecards*): with good:bad odds ``o = P(good)/P(bad)``

    score  = Offset + Factor * ln(o)             Factor = PDO / ln 2,   Offset = BaseScore - Factor * ln(BaseOdds)

A score of ``base_score`` therefore corresponds to odds of ``base_odds`` : 1 and every ``pdo`` points double the odds.
For a model ``logit P(bad) = a + sum_j b_j * WoE_j`` the score is additive over variables:

    points(j, bin i) = -(b_j * WoE_ji + a / n) * Factor + Offset / n        (n = number of variables)

so the final scorecard is a simple lookup table. ``Scorecard.score`` sums the *rounded* points (what would be
deployed); ``exact=True`` returns the un-rounded score, which equals ``Offset + Factor * ln(odds)`` exactly.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd


class Scorecard:
    def __init__(self, binner, model, features, base_score: float = 600, base_odds: float = 50,
                 pdo: float = 20, name: str = "scorecard"):
        self.binner = binner
        self.model = model
        self.features = list(features)
        self.base_score, self.base_odds, self.pdo = base_score, base_odds, pdo
        self.name = name
        self.factor = pdo / np.log(2.0)
        self.offset = base_score - self.factor * np.log(base_odds)
        self.intercept = float(model.intercept_[0])
        self.coefs = dict(zip(self.features, model.coef_[0].astype(float)))
        self._build()

    # ------------------------------------------------------------------------------------------------------
    def _build(self) -> None:
        n = len(self.features)
        self._pts_exact, self._pts_int, self._best, rows = {}, {}, {}, []
        for f in self.features:
            t = self.binner.tables_[f]
            woe = t["woe"].to_numpy()
            exact = -(self.coefs[f] * woe + self.intercept / n) * self.factor + self.offset / n
            self._pts_exact[f] = exact
            self._pts_int[f] = np.round(exact)
            populated = t["count"].to_numpy() > 0
            self._best[f] = self._pts_int[f][populated].max()
            for i, r in t.iterrows():
                rows.append({"variable": f, "bin": r["bin"], "count": int(r["count"]), "share": r["share"],
                             "bad_rate": r["bad_rate"], "woe": r["woe"], "coefficient": self.coefs[f],
                             "points": self._pts_int[f][i], "points_exact": exact[i]})
        self.points_table_all_ = pd.DataFrame(rows)

    @property
    def points_table(self) -> pd.DataFrame:
        """The scorecard: one row per populated bin (plus Missing/Unseen bins that occurred in training)."""
        t = self.points_table_all_
        return t[t["count"] > 0].reset_index(drop=True)

    @property
    def score_range(self) -> tuple[float, float]:
        lo = sum(self._pts_int[f][self.binner.tables_[f]["count"].to_numpy() > 0].min() for f in self.features)
        hi = sum(self._best[f] for f in self.features)
        return float(lo), float(hi)

    # ------------------------------------------------------------------------------------------------------
    def _bins(self, X) -> dict[str, np.ndarray]:
        b = self.binner.transform_bins(X, self.features)
        return {f: b[f].to_numpy() for f in self.features}

    def woe_matrix(self, X) -> pd.DataFrame:
        X = pd.DataFrame(X)
        bins = self._bins(X)
        return pd.DataFrame({f: self.binner.tables_[f]["woe"].to_numpy()[bins[f]] for f in self.features},
                            index=X.index)

    def predict_pd(self, X) -> pd.Series:
        """Probability of default from the underlying logistic regression."""
        W = self.woe_matrix(X)
        return pd.Series(self.model.predict_proba(W.to_numpy())[:, 1], index=W.index, name="pd")

    def score(self, X, exact: bool = False) -> pd.Series:
        """Credit score = sum of bin points (rounded unless ``exact``)."""
        X = pd.DataFrame(X)
        bins = self._bins(X)
        table = self._pts_exact if exact else self._pts_int
        total = np.zeros(len(X))
        for f in self.features:
            total += table[f][bins[f]]
        return pd.Series(total, index=X.index, name="score")

    # ------------------------------------------------------------------------------------------------------
    def pd_from_score(self, score) -> np.ndarray:
        """PD implied by a score: PD = 1 / (1 + odds), odds = exp((score - offset) / factor)."""
        return 1.0 / (1.0 + np.exp((np.asarray(score, dtype=float) - self.offset) / self.factor))

    def score_from_pd(self, pd_) -> np.ndarray:
        pd_ = np.clip(np.asarray(pd_, dtype=float), 1e-9, 1 - 1e-9)
        return self.offset + self.factor * np.log((1 - pd_) / pd_)

    def reason_codes(self, X, n: int = 3) -> pd.DataFrame:
        """Principal adverse-action style reasons: variables where the applicant lost the most points vs the best bin."""
        X = pd.DataFrame(X)
        bins = self._bins(X)
        pts = np.column_stack([self._pts_int[f][bins[f]] for f in self.features])
        best = np.array([self._best[f] for f in self.features])
        short = best - pts
        labels = {f: self.binner.tables_[f]["bin"].to_numpy() for f in self.features}
        order = np.argsort(-short, axis=1, kind="mergesort")[:, :n]
        rows = []
        for i in range(len(X)):
            rows.append([f"{self.features[j]}: {labels[self.features[j]][bins[self.features[j]][i]]} "
                         f"(-{short[i, j]:.0f} pts)" for j in order[i]])
        return pd.DataFrame(rows, index=X.index, columns=[f"reason_{k + 1}" for k in range(n)])

    def explain(self, X_row) -> pd.DataFrame:
        """Points breakdown for ONE applicant: variable, the bin the applicant falls in, and the points awarded."""
        X_row = pd.DataFrame(X_row)
        if len(X_row) != 1:
            raise ValueError("explain() expects exactly one row")
        bins = self._bins(X_row)
        rows = [{"variable": f, "bin": self.binner.tables_[f]["bin"].iloc[bins[f][0]],
                 "points": self._pts_int[f][bins[f][0]], "best_possible": self._best[f]} for f in self.features]
        out = pd.DataFrame(rows)
        out.loc[len(out)] = ["TOTAL SCORE", "", out["points"].sum(), out["best_possible"].sum()]
        return out

    def scaling(self) -> dict:
        return {"base_score": self.base_score, "base_odds": self.base_odds, "pdo": self.pdo,
                "factor": float(self.factor), "offset": float(self.offset),
                "intercept": self.intercept, "n_variables": len(self.features)}

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "Scorecard":
        return joblib.load(path)
