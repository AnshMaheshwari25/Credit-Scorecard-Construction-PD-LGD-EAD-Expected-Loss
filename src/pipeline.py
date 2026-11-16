"""End-to-end pipeline (also importable step by step from the notebook).

    python -m src.pipeline --data data/raw/loan_data_2015.csv            # full run, writes results/ and models/
    python -m src.pipeline --data data/raw/loan_data_2015.csv --fast     # skip CV and the GBM benchmark
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import data_prep as dp
from . import expected_loss as E
from . import metrics as M
from .config import ROOT, load_config, resolve_data_path
from .lgd_ead import charged_off_sample, fit_loss_models
from .pd_model import cross_validate_pd, fit_gbm_benchmark, fit_pd_model


# --------------------------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------------------------
def prepare(cfg: dict, data_path=None, nrows: int | None = None) -> dict:
    """Load -> clean -> engineer -> define targets -> stratified train/test split of the development sample."""
    path = Path(data_path) if data_path else resolve_data_path(cfg)
    raw = dp.load_raw(path, nrows=nrows)
    df, info = dp.build_dataset(raw, cfg)
    train, test = dp.split_development_sample(df, cfg)
    return {"path": path, "raw": raw, "df": df, "info": info, "train": train, "test": test}


def leakage_audit(train: pd.DataFrame, test: pd.DataFrame, cfg: dict,
                  names=("standalone", "with_lc_grade", "leaky_reference"), reuse: dict | None = None):
    """Fit the same WoE-logit machinery on three feature sets and compare test discrimination.

    ``leaky_reference`` adds post-origination fields (out_prncp, total_pymnt, last payment ...). Its (implausibly) high
    AUROC is the signature of target leakage; it must never be deployed.
    """
    fs = dp.feature_sets()
    reuse = reuse or {}
    models, rows = {}, []
    for n in names:
        m = reuse[n] if n in reuse else fit_pd_model(train, fs[n], cfg, name=n)
        models[n] = m
        ev = M.evaluate_binary(test["bad"], m.predict_pd(test))
        rows.append({"feature_set": n, "candidates": len(fs[n]), "selected": len(m.features),
                     "test_auroc": ev["auroc"], "test_gini": ev["gini"], "test_ks": ev["ks"],
                     "uses_post_origination_info": any(f in dp.POST_ORIGINATION for f in m.features),
                     "selected_features": ", ".join(m.features)})
    return pd.DataFrame(rows), models


def seasoning_sensitivity(raw: pd.DataFrame, cfg: dict, months=(0, 3, 6, 9), feature_set: str = "standalone") -> pd.DataFrame:
    """Does the headline result depend on the minimum-months-on-book rule? Refit the scorecard for several values."""
    import copy
    rows = []
    for m in months:
        c = copy.deepcopy(cfg)
        c["sample"]["min_months_on_book"] = m
        df, _ = dp.build_dataset(raw, c)
        tr, te = dp.split_development_sample(df, c)
        pdm = fit_pd_model(tr, dp.feature_sets()[feature_set], c, name=feature_set)
        ev = M.evaluate_binary(te["bad"], pdm.predict_pd(te))
        rows.append({"min_months_on_book": m, "dev_loans": len(tr) + len(te), "bad_rate_%": 100 * ev["bad_rate"],
                     "selected_vars": len(pdm.features), "test_auroc": ev["auroc"], "test_ks": ev["ks"]})
    return pd.DataFrame(rows)


def evaluate_pd(pdm, train: pd.DataFrame, test: pd.DataFrame, cfg: dict, fast: bool = False) -> dict:
    """Discrimination, calibration, stability and strategy analysis for a fitted PD scorecard."""
    sc = pdm.scorecard
    y_tr, y_te = train["bad"].astype(int), test["bad"].astype(int)
    p_tr, p_te = pdm.predict_pd(train), pdm.predict_pd(test)
    s_tr, s_te = pdm.score(train), pdm.score(test)
    out = {"train": M.evaluate_binary(y_tr, p_tr), "test": M.evaluate_binary(y_te, p_te)}
    out["auroc_ci"] = M.bootstrap_ci(y_te, p_te, n_boot=cfg["model"].get("bootstrap_reps", 200))
    out["cv_auc"] = None if fast else cross_validate_pd(train, pdm, cfg)
    out["decile"] = M.decile_table(y_te, p_te)
    out["hosmer_lemeshow"] = M.hosmer_lemeshow(y_te, p_te)
    out["psi_score"] = M.psi(s_tr, s_te)
    dev = pd.concat([train, test])
    s_dev = pdm.score(dev)
    out["psi_by_month"] = pd.DataFrame(
        [{"issue_month": str(per), "loans": len(g), "psi_vs_train": M.psi(s_tr, s_dev.loc[g.index])}
         for per, g in dev.groupby(dev["issue_date"].dt.to_period("M"))])
    out["csi"] = M.csi_table(sc.binner.transform_bins(train, pdm.features),
                             sc.binner.transform_bins(test, pdm.features))
    out["strategy"] = M.strategy_table(s_te, y_te)
    out["ks_cutoff"] = M.ks_cutoff(s_te, y_te)
    out["score_bands"] = M.score_band_table(s_te, y_te)
    worst = s_te.sort_values().head(5).index
    out["reason_codes"] = sc.reason_codes(test.loc[worst]).assign(score=s_te.loc[worst].to_numpy())
    out["p_test"], out["score_test"], out["score_train"] = p_te, s_te, s_tr

    out["gbm"] = out["gbm_model"] = None
    rows = [{"model": f"Scorecard ({pdm.name}, {len(pdm.features)} vars)", **{k: out["test"][k] for k in
            ("auroc", "gini", "ks", "pr_auc", "brier")}}]
    if not fast:
        gbm = fit_gbm_benchmark(train, pdm.columns, cfg)
        p_gbm = gbm.predict_proba(test[pdm.columns])[:, 1]
        out["gbm"], out["gbm_model"], out["p_gbm"] = M.evaluate_binary(y_te, p_gbm), gbm, p_gbm
        rows.append({"model": "Gradient boosting (benchmark, raw features)", **{k: out["gbm"][k] for k in
                    ("auroc", "gini", "ks", "pr_auc", "brier")}})
    out["comparison"] = pd.DataFrame(rows)
    return out


def loss_report(train: pd.DataFrame, test: pd.DataFrame, pdm, lm, cfg: dict) -> dict:
    """Combine PD x LGD x EAD on the test sample; segment views, sensitivity and the proxy back-test."""
    ecfg = cfg["expected_loss"]
    el = E.compute_expected_loss(test, pdm.predict_pd(test), lm.predict_lgd(test), lm.predict_ccf(test))
    score = pdm.score(test)
    # higher score = safer, so rank the NEGATIVE score: Q1 = highest scores = lowest risk
    bands = pd.qcut((-score).rank(method="first"), 5, labels=["Q1 (lowest risk)", "Q2", "Q3", "Q4", "Q5 (highest risk)"])
    lgd_train = float(charged_off_sample(train, cfg["lgd_ead"]["min_ead"])["lgd"].mean())
    proxy = E.realized_loss_proxy(test, lgd_train)

    def _cmp(by: pd.Series, label: str) -> pd.DataFrame:
        s = E.summarize_el(el, by, label)
        g = proxy.groupby(by.reindex(proxy.index)).sum().rename("loss_proxy")
        s = s.merge(g, left_on=label, right_index=True, how="left")
        s["expected_loss_rate_%"] = s["el_rate_%"]
        s["loss_proxy_rate_%"] = 100 * s["loss_proxy"] / s["funded"]
        return s

    by_grade = _cmp(test["grade"], "grade")
    by_band = _cmp(bands, "risk_band")
    total = E.summarize_el(el)
    return {
        "el": el, "portfolio": total, "by_grade": by_grade, "by_risk_band": by_band,
        "sensitivity": E.sensitivity_grid(el, ecfg["pd_multipliers"], ecfg["lgd_add_pp"]),
        "totals": {"expected_loss": float(el["el"].sum()), "loss_proxy": float(proxy.sum()),
                   "funded": float(el["funded_amnt"].sum()), "lgd_train_mean": lgd_train,
                   "el_rate_pct": float(100 * el["el"].sum() / el["funded_amnt"].sum()),
                   "proxy_rate_pct": float(100 * proxy.sum() / el["funded_amnt"].sum())},
    }


# --------------------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------------------
def run_pipeline(cfg: dict | None = None, data_path=None, nrows: int | None = None, fast: bool = False,
                 save: bool = True, results_dir=None, verbose: bool = True, models_dir=None) -> dict:
    cfg = cfg or load_config()
    t0 = time.time()

    def log(msg):
        if verbose:
            print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    prep = prepare(cfg, data_path, nrows)
    train, test = prep["train"], prep["test"]
    log(f"data: {prep['path'].name} | snapshot {prep['info']['snapshot']:%b-%Y} | train {len(train):,} / test {len(test):,}")
    audit, models = leakage_audit(train, test, cfg)
    pdm = models[cfg["model"]["primary"]]
    log(f"PD scorecards fitted. Primary = {pdm.name} with {len(pdm.features)} variables")
    ev = evaluate_pd(pdm, train, test, cfg, fast=fast)
    log(f"test AUROC {ev['test']['auroc']:.3f} | KS {ev['test']['ks']:.3f}")
    try:
        lm = fit_loss_models(train, test, dp.LGD_EAD_NUMERIC, dp.LGD_EAD_CATEGORICAL,
                             min_ead=cfg["lgd_ead"]["min_ead"], seed=cfg["sample"]["random_state"])
        rep = loss_report(train, test, pdm, lm, cfg)
        log(f"LGD/EAD fitted on {lm.n_train:,} charged-off loans | portfolio EL {rep['totals']['el_rate_pct']:.2f}% of funded")
    except ValueError as exc:
        lm, rep = None, None
        log(f"LGD / EAD / EL skipped: {exc}")
    results = {"cfg": cfg, "prep": prep, "audit": audit, "models": models, "pdm": pdm, "pd_eval": ev,
               "loss_models": lm, "loss_report": rep}
    if save:
        save_outputs(results, results_dir or ROOT / cfg["output"]["results_dir"],
                     models_dir or ROOT / cfg["output"]["models_dir"])
        log("outputs written")
    return results


# --------------------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------------------
def _md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Minimal markdown table (keeps integer columns as integers; no tabulate dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (bool, np.bool_)):
                cells.append(str(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            elif isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_summary(res: dict) -> str:
    cfg, ev, audit, info = res["cfg"], res["pd_eval"], res["audit"], res["prep"]["info"]
    t, tr = ev["test"], ev["train"]
    cv = ev["cv_auc"]
    L = [f"# Results summary (auto-generated by `python -m src.pipeline`)", "",
         f"Data file: `{res['prep']['path'].name}` | snapshot: {info['snapshot']:%b-%Y} | primary PD model: "
         f"`{res['pdm'].name}`", "", "## Development sample", _md(info["waterfall"], "{:.2f}"), "",
         "## PD scorecard", "",
         f"- Test AUROC **{t['auroc']:.3f}** (95% bootstrap CI {ev['auroc_ci'][0]:.3f}-{ev['auroc_ci'][1]:.3f}), "
         f"Gini {t['gini']:.3f}, KS {t['ks']:.3f}, PR-AUC {t['pr_auc']:.3f} (bad-class base rate {t['bad_rate']:.3f})",
         f"- Train AUROC {tr['auroc']:.3f}" + (f" | {cfg['model']['cv_splits']}-fold CV AUROC "
                                              f"{cv.mean():.3f} +/- {cv.std():.3f}" if cv is not None else ""),
         f"- Score PSI train vs test: {ev['psi_score']:.4f} | Hosmer-Lemeshow p-value {ev['hosmer_lemeshow'][1]:.3f} "
         "(overly sensitive at large n; read with the decile table)",
         f"- Scorecard scaling: {res['pdm'].scorecard.base_score} points at {res['pdm'].scorecard.base_odds}:1 odds, "
         f"PDO {res['pdm'].scorecard.pdo}; KS-optimal cut-off score {ev['ks_cutoff']:.0f}", "",
         "### Scorecard vs benchmark", _md(ev["comparison"]), "",
         "### Leakage audit (same machinery, different feature sets; test sample)",
         _md(audit.drop(columns=["selected_features"])), ""]
    lm, rep = res["loss_models"], res["loss_report"]
    if lm is not None:
        L += ["## LGD and EAD", f"Trained on {lm.n_train:,} charged-off loans, evaluated on {lm.n_test:,}.", "",
              "### LGD models (test)", _md(lm.lgd_comparison), "", "### EAD / CCF models (test)", _md(lm.ead_comparison), "",
              "### Model-vs-naive check (CV on training data)", _md(lm.selection), "",
              "## Expected loss (test sample)", _md(rep["portfolio"].drop(columns=["segment"]), "{:,.4f}"), "",
              "### by LendingClub grade", _md(rep["by_grade"][["grade", "loans", "avg_pd", "avg_lgd", "avg_ccf",
                                                              "expected_loss_rate_%", "loss_proxy_rate_%"]]), "",
              "### Stress test: portfolio EL % of funded", _md(rep["sensitivity"].reset_index().rename(
                  columns={"index": "scenario"}), "{:.3f}")]
    return "\n".join(L)


def save_outputs(res: dict, results_dir, models_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import plotting as P

    results_dir, models_dir = Path(results_dir), Path(models_dir)
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    pdm, ev, rep, lm = res["pdm"], res["pd_eval"], res["loss_report"], res["loss_models"]
    train, test, df = res["prep"]["train"], res["prep"]["test"], res["prep"]["df"]
    tab = results_dir / "tables"

    res["prep"]["info"]["waterfall"].to_csv(tab / "sample_waterfall.csv", index=False)
    res["audit"].to_csv(tab / "leakage_audit.csv", index=False)
    pdm.scorecard.points_table.to_csv(tab / "scorecard_points.csv", index=False)
    pdm.binner.iv_summary().to_csv(tab / "iv_summary.csv", index=False)
    pdm.selection["log"].to_csv(tab / "feature_selection_log.csv", index=False)
    pdm.selection["summary"].to_csv(tab / "logit_coefficients.csv")
    ev["decile"].to_csv(tab / "calibration_deciles.csv", index=False)
    ev["strategy"].to_csv(tab / "cutoff_strategy.csv", index=False)
    ev["score_bands"].to_csv(tab / "score_bands.csv", index=False)
    ev["csi"].to_csv(tab / "csi.csv", index=False)
    ev["psi_by_month"].to_csv(tab / "psi_by_issue_month.csv", index=False)
    ev["comparison"].to_csv(tab / "pd_model_comparison.csv", index=False)
    ev["reason_codes"].to_csv(tab / "reason_codes_example.csv")
    if lm is not None:
        lm.lgd_comparison.to_csv(tab / "lgd_model_comparison.csv", index=False)
        lm.ead_comparison.to_csv(tab / "ead_model_comparison.csv", index=False)
        lm.selection.to_csv(tab / "lgd_ead_model_selection.csv", index=False)
        rep["portfolio"].to_csv(tab / "expected_loss_portfolio.csv", index=False)
        rep["by_grade"].to_csv(tab / "expected_loss_by_grade.csv", index=False)
        rep["by_risk_band"].to_csv(tab / "expected_loss_by_risk_band.csv", index=False)
        rep["sensitivity"].to_csv(tab / "expected_loss_sensitivity.csv")

    # figures --------------------------------------------------------------------------------------------------
    fig_dir = results_dir / "figures"
    figs = {"01_vintage_bad_rate": P.plot_vintage(df),
            "02_iv_ranking": P.plot_iv(pdm.binner.iv_summary()),
            "04_roc": P.plot_roc({"Scorecard": (test["bad"], ev["p_test"]),
                                  **({"Gradient boosting": (test["bad"], ev["p_gbm"])} if ev["gbm"] else {})}),
            "05_ks": P.plot_ks(test["bad"], ev["p_test"]),
            "06_calibration": P.plot_calibration(ev["decile"]),
            "07_score_distribution": P.plot_score_dist(ev["score_test"], test["bad"], ev["ks_cutoff"]),
            "08_strategy": P.plot_strategy(ev["strategy"]),
            "09_stability_csi": P.plot_stability(ev["csi"])}
    for i, f in enumerate(pdm.features[:4]):
        figs[f"03_woe_{i + 1}_{f}"] = P.plot_woe(pdm.binner, f)
    if lm is not None:
        te_co = charged_off_sample(test, res["cfg"]["lgd_ead"]["min_ead"])
        figs["10_loss_models"] = P.plot_loss_models(te_co, lm.predict_lgd(te_co), lm.predict_ccf(te_co))
        figs["11_el_by_grade"] = P.plot_el_by_segment(rep["by_grade"], "grade")
        figs["12_el_vs_proxy"] = P.plot_el_vs_proxy(rep["by_risk_band"], "risk_band")
        figs["13_el_sensitivity"] = P.plot_sensitivity(rep["sensitivity"])
    for name, fig in figs.items():
        fig.savefig(fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # models + metrics + summary ---------------------------------------------------------------------------------
    pdm.scorecard.save(models_dir / "pd_scorecard.joblib")
    if lm is not None:
        joblib.dump(lm, models_dir / "lgd_ead_models.joblib")
    metrics = {"primary_model": pdm.name, "features": pdm.features, "train": ev["train"], "test": ev["test"],
               "auroc_ci_95": ev["auroc_ci"], "cv_auc_mean": None if ev["cv_auc"] is None else float(ev["cv_auc"].mean()),
               "psi_score_train_vs_test": ev["psi_score"], "ks_cutoff_score": ev["ks_cutoff"],
               "scaling": pdm.scorecard.scaling(), "gbm_test": ev["gbm"],
               "audit": res["audit"].drop(columns=["selected_features"]).to_dict(orient="records"),
               "expected_loss": None if rep is None else rep["totals"]}
    with open(results_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    (results_dir / "summary.md").write_text(render_summary(res), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="path to the loan CSV (default: auto-detect via configs/config.yaml)")
    ap.add_argument("--config", help="path to a YAML config")
    ap.add_argument("--results-dir", help="where to write tables / figures / summary (default: results/)")
    ap.add_argument("--models-dir", help="where to write fitted models (default: models/)")
    ap.add_argument("--nrows", type=int, help="read only the first N rows (quick tests)")
    ap.add_argument("--fast", action="store_true", help="skip cross-validation and the gradient-boosting benchmark")
    a = ap.parse_args()
    run_pipeline(load_config(a.config), a.data, a.nrows, a.fast, True, a.results_dir, models_dir=a.models_dir)


if __name__ == "__main__":
    main()
