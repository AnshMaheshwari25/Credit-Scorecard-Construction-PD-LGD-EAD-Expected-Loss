"""Smoke test of the whole pipeline on a SYNTHETIC file (never real results)."""
import json

import numpy as np
import pytest

from src.config import load_config
from src.pipeline import run_pipeline


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory):
    from make_synthetic_sample import generate
    path = tmp_path_factory.mktemp("data") / "synthetic.csv"
    generate(40000, seed=5).to_csv(path, index=False)
    return path


def test_pipeline_runs_and_writes_outputs(synthetic_csv, tmp_path):
    cfg = load_config()
    res = run_pipeline(cfg, synthetic_csv, fast=True, save=True, results_dir=tmp_path / "results",
                       models_dir=tmp_path / "models", verbose=False)
    ev = res["pd_eval"]
    assert 0.55 < ev["test"]["auroc"] < 0.95
    # leakage audit: the feature set with post-origination fields must look far better than the clean ones
    a = res["audit"].set_index("feature_set")
    assert a.loc["leaky_reference", "test_auroc"] > a.loc["with_lc_grade", "test_auroc"] + 0.1
    assert not a.loc["standalone", "uses_post_origination_info"]
    # outputs
    out = tmp_path / "results"
    for f in ("summary.md", "metrics.json", "tables/scorecard_points.csv", "figures/04_roc.png",
              "figures/13_el_sensitivity.png"):
        assert (out / f).exists(), f
    assert (tmp_path / "models" / "pd_scorecard.joblib").exists()
    m = json.loads((out / "metrics.json").read_text())
    assert m["scaling"]["base_odds"] == 50 and m["expected_loss"]["el_rate_pct"] > 0
    # expected loss identity on the loan level
    el = res["loss_report"]["el"]
    assert np.allclose(el["el"], el["pd"] * el["lgd"] * el["ccf"] * el["funded_amnt"])
    assert el["pd"].between(0, 1).all() and el["lgd"].between(0, 1).all() and el["ccf"].between(0, 1).all()


def test_risk_bands_are_ordered_from_low_to_high_risk(synthetic_csv, tmp_path):
    """Regression test: Q1 must be the LOWEST-risk band (it was once mislabelled upside down)."""
    res = run_pipeline(load_config(), synthetic_csv, fast=True, save=False, verbose=False)
    b = res["loss_report"]["by_risk_band"].set_index("risk_band")
    assert b.index[0].startswith("Q1 (lowest") and b.index[-1].startswith("Q5 (highest")
    assert b.loc["Q5 (highest risk)", "avg_pd"] > b.loc["Q1 (lowest risk)", "avg_pd"]
    assert b.loc["Q5 (highest risk)", "expected_loss_rate_%"] > b.loc["Q1 (lowest risk)", "expected_loss_rate_%"]
