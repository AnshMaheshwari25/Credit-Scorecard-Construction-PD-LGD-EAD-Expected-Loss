"""Figures used in the notebook and saved to ``results/figures``. Every function returns a matplotlib Figure."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

BLUE, RED, GREY, GREEN, ORANGE = "#1f4e79", "#c0392b", "#7f8c8d", "#2e8b57", "#e08e0b"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 100, "axes.titlesize": 12, "axes.titleweight": "bold"})


def plot_vintage(df: pd.DataFrame) -> plt.Figure:
    """Bad rate and seasoning by origination month: shows right-censoring in the snapshot."""
    d = df[df["bad"].notna()].copy()
    g = d.groupby(d["issue_date"].dt.to_period("M")).agg(bad_rate=("bad", "mean"), loans=("bad", "size"),
                                                         mob=("months_on_book", "mean"))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(g.index.astype(str), 100 * g["bad_rate"], color=BLUE)
    ax.set_ylabel("Bad rate at snapshot (%)")
    ax.set_title("Observed bad rate (bars) and months on book (line) by issue month")
    ax2 = ax.twinx()
    ax2.plot(g.index.astype(str), g["mob"], color=RED, marker="o")
    ax2.set_ylabel("Months on book", color=RED)
    ax2.grid(False)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_iv(iv_table: pd.DataFrame, top: int = 20, col: str = "iv", title: str = "Information value by feature") -> plt.Figure:
    t = iv_table.sort_values(col, ascending=False).head(top).iloc[::-1]
    colors = [RED if v > 0.5 else (BLUE if v >= 0.1 else GREY) for v in t[col]]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(t) + 1.5))
    ax.barh(t["feature"], t[col], color=colors)
    for x in (0.02, 0.1, 0.3, 0.5):
        ax.axvline(x, color="k", lw=0.6, ls=":")
    ax.set_xlabel(col)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_woe(binner, col: str) -> plt.Figure:
    t = binner.binning_table(col)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(range(len(t)), t["woe"], color=[GREEN if w >= 0 else RED for w in t["woe"]])
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels([b if len(b) < 22 else b[:20] + "…" for b in t["bin"]], rotation=35, ha="right")
    ax.set_ylabel("WoE (higher = safer)")
    ax.axhline(0, color="k", lw=0.8)
    ax2 = ax.twinx()
    ax2.plot(range(len(t)), 100 * t["bad_rate"], color=BLUE, marker="o")
    ax2.set_ylabel("Bad rate (%)", color=BLUE)
    ax2.grid(False)
    ax.set_title(f"{col}: WoE and bad rate by bin (IV = {t['iv'].sum():.3f})")
    fig.tight_layout()
    return fig


def plot_roc(curves: dict) -> plt.Figure:
    """``curves``: name -> (y_true, predicted PD or score where higher = riskier)."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for (name, (y, p)), c in zip(curves.items(), [BLUE, RED, GREEN, ORANGE, GREY]):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=c, label=f"{name} (AUROC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test sample)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_ks(y, p) -> plt.Figure:
    y = np.asarray(y).astype(int)
    order = np.argsort(-np.asarray(p))
    ys = y[order]
    cum_bad = np.cumsum(ys) / ys.sum()
    cum_good = np.cumsum(1 - ys) / (1 - ys).sum()
    x = np.arange(1, len(ys) + 1) / len(ys)
    k = int(np.argmax(cum_bad - cum_good))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(x, cum_bad, color=RED, label="Cumulative bads")
    ax.plot(x, cum_good, color=BLUE, label="Cumulative goods")
    ax.vlines(x[k], cum_good[k], cum_bad[k], color="k", ls="--")
    ax.set_title(f"KS = {cum_bad[k] - cum_good[k]:.3f} (at {x[k]:.0%} of population, riskiest first)")
    ax.set_xlabel("Share of population, ordered by predicted PD (high to low)")
    ax.set_ylabel("Cumulative share")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_calibration(dec: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    w = 0.38
    x = np.arange(len(dec))
    axes[0].bar(x - w / 2, 100 * dec["avg_pd"], w, color=BLUE, label="Predicted PD")
    axes[0].bar(x + w / 2, 100 * dec["bad_rate"], w, color=RED, label="Observed bad rate")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dec["decile"].astype(str))
    axes[0].set_xlabel("Risk decile (1 = riskiest)")
    axes[0].set_ylabel("%")
    axes[0].set_title("Calibration by risk decile")
    axes[0].legend()
    lim = 100 * max(dec["avg_pd"].max(), dec["bad_rate"].max()) * 1.1
    axes[1].scatter(100 * dec["avg_pd"], 100 * dec["bad_rate"], color=BLUE)
    axes[1].plot([0, lim], [0, lim], "k--", lw=0.8)
    axes[1].set_xlabel("Predicted PD (%)")
    axes[1].set_ylabel("Observed bad rate (%)")
    axes[1].set_title("Reliability diagram")
    fig.tight_layout()
    return fig


def plot_score_dist(score, y, cutoff: float | None = None) -> plt.Figure:
    score = np.asarray(score, dtype=float)
    y = np.asarray(y).astype(int)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = np.linspace(score.min(), score.max(), 40)
    ax.hist(score[y == 0], bins=bins, density=True, alpha=0.6, color=BLUE, label="Good")
    ax.hist(score[y == 1], bins=bins, density=True, alpha=0.6, color=RED, label="Bad")
    if cutoff is not None:
        ax.axvline(cutoff, color="k", ls="--", label=f"Cut-off {cutoff:.0f}")
    ax.set_xlabel("Credit score")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution: good vs bad")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_strategy(st: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(100 * st["approval_rate"], 100 * st["approved_bad_rate"], marker="o", color=BLUE)
    ax.set_xlabel("Approval rate (%)")
    ax.set_ylabel("Bad rate of approved loans (%)")
    ax.invert_xaxis()
    ax.set_title("Approval vs bad-rate trade-off")
    for _, r in st.iterrows():
        ax.annotate(f"{r['cutoff_score']:.0f}", (100 * r["approval_rate"], 100 * r["approved_bad_rate"]),
                    textcoords="offset points", xytext=(4, 5), fontsize=7, color=GREY)
    fig.tight_layout()
    return fig


def plot_stability(csi: pd.DataFrame, title: str = "Characteristic stability index (train vs test)") -> plt.Figure:
    t = csi.sort_values("csi").tail(15)
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(t) + 1.5))
    ax.barh(t["variable"], t["csi"], color=[RED if v > 0.25 else (ORANGE if v > 0.1 else BLUE) for v in t["csi"]])
    ax.axvline(0.1, color="k", ls=":", lw=0.7)
    ax.axvline(0.25, color="k", ls=":", lw=0.7)
    ax.set_xlabel("CSI (<0.10 stable, >0.25 significant shift)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _zoomed_diag(ax, pred, actual):
    """Scatter axes zoomed to the data (LGD / CCF cluster near 1) with the 45-degree line."""
    pred, actual = np.asarray(pred, dtype=float), np.asarray(actual, dtype=float)
    lo = max(0.0, min(pred.min(), actual.min()) - 0.03)
    ax.plot([lo, 1.02], [lo, 1.02], "k--", lw=0.8)
    ax.set_xlim(lo, 1.02)
    ax.set_ylim(lo, 1.02)


def plot_loss_models(te_co: pd.DataFrame, lgd_pred, ccf_pred) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(te_co["lgd"], bins=25, color=BLUE)
    axes[0].set_title("Realised LGD of charged-off loans")
    axes[0].set_xlabel("LGD")
    axes[1].scatter(lgd_pred, te_co["lgd"], s=8, alpha=0.4, color=BLUE)
    _zoomed_diag(axes[1], lgd_pred, te_co["lgd"])
    axes[1].set_xlabel("Predicted LGD")
    axes[1].set_ylabel("Actual LGD")
    axes[1].set_title("LGD: predicted vs actual (test)")
    axes[2].scatter(ccf_pred, te_co["ccf"], s=8, alpha=0.4, color=GREEN)
    _zoomed_diag(axes[2], ccf_pred, te_co["ccf"])
    axes[2].set_xlabel("Predicted CCF")
    axes[2].set_ylabel("Actual CCF")
    axes[2].set_title("Exposure factor: predicted vs actual (test)")
    fig.tight_layout()
    return fig


def plot_el_by_segment(summary: pd.DataFrame, label: str = "grade") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(summary))
    ax.bar(x, summary["el_rate_%"], color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(summary[label].astype(str))
    ax.set_ylabel("Expected loss (% of funded amount)")
    ax.set_xlabel(label)
    ax.set_title(f"Expected loss rate by {label}")
    ax2 = ax.twinx()
    ax2.plot(x, 100 * summary["avg_pd"], color=RED, marker="o")
    ax2.set_ylabel("Average PD (%)", color=RED)
    ax2.grid(False)
    fig.tight_layout()
    return fig


def plot_el_vs_proxy(cmp: pd.DataFrame, label: str = "segment") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(cmp))
    w = 0.38
    ax.bar(x - w / 2, cmp["expected_loss_rate_%"], w, color=BLUE, label="Model expected loss")
    ax.bar(x + w / 2, cmp["loss_proxy_rate_%"], w, color=RED, label="Observed loss proxy")
    ax.set_xticks(x)
    ax.set_xticklabels(cmp[label].astype(str))
    ax.set_ylabel("% of funded amount")
    ax.set_title("Expected loss vs observed loss proxy (test sample)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_sensitivity(grid: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    im = ax.imshow(grid.to_numpy(), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels(grid.columns)
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels(grid.index)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.grid(False)
    ax.set_title("Portfolio EL (% of funded amount) under stress")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig
