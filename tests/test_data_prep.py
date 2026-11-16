import numpy as np
import pandas as pd
import pytest

from src import data_prep as dp
from src.config import load_config


def test_parsers():
    assert dp.parse_term(pd.Series([" 36 months", " 60 months"])).tolist() == [36, 60]
    e = dp.parse_emp_length(pd.Series(["10+ years", "< 1 year", "1 year", "5 years", None, "n/a"]))
    assert e.iloc[:4].tolist() == [10, 0, 1, 5] and e.iloc[4:].isna().all()
    s = dp.normalize_status(pd.Series(["Does not meet the credit policy. Status:Charged Off", "Current"]))
    assert s.tolist() == ["Charged Off", "Current"]
    assert dp._to_float(pd.Series(["13.5%", "7%", None])).iloc[:2].tolist() == [13.5, 7.0]


def test_credit_history_century_fix():
    # 'Jan-62' must be read as 1962, not 2062: history of ~53 years at Mar-15
    h = dp.credit_history_months(pd.Series(["Jan-62", "Aug-94", "Mar-15"]), pd.Series(["Mar-15"] * 3))
    assert h.iloc[0] == (2015 - 1962) * 12 + 2
    assert h.iloc[1] == (2015 - 1994) * 12 - 5
    assert h.iloc[2] == 0


def test_feature_sets_have_no_post_origination_leakage():
    fs = dp.feature_sets()
    assert not set(fs["standalone"]) & set(dp.POST_ORIGINATION)
    assert not set(fs["with_lc_grade"]) & set(dp.POST_ORIGINATION)
    assert set(dp.POST_ORIGINATION) <= set(fs["leaky_reference"])
    assert not {"grade", "int_rate", "installment", "payment_to_income"} & set(fs["standalone"])
    assert not set(dp.POLICY_EXCLUDED) & set(fs["with_lc_grade"])


def test_load_raw_fills_optional_columns(tmp_path):
    """Older vintages may omit secondary fields; the loader creates explicit missing columns."""
    raw = {
        "loan_status": ["Fully Paid"],
        "issue_d": ["Dec-15"],
        "term": [" 36 months"],
        "loan_amnt": [1000],
        "funded_amnt": [1000],
        "int_rate": [10.0],
        "grade": ["B"],
        "annual_inc": [50000],
        "dti": [10.0],
        "earliest_cr_line": ["Jan-00"],
        "total_rec_prncp": [100.0],
        "recoveries": [0.0],
        "collection_recovery_fee": [0.0],
        "last_pymnt_d": ["Jan-16"],
    }
    path = tmp_path / "minimal_loans.csv"
    pd.DataFrame(raw).to_csv(path, index=False)

    loaded = dp.load_raw(path)
    assert "tot_cur_bal" in loaded and loaded["tot_cur_bal"].isna().all()
    assert "application_type" in loaded and loaded["application_type"].isna().all()


@pytest.fixture(scope="module")
def prepared():
    from make_synthetic_sample import generate
    cfg = load_config()
    raw = generate(6000, seed=3)
    return dp.build_dataset(raw, cfg)


def test_targets_and_snapshot(prepared):
    df, info = prepared
    assert info["snapshot"] == pd.Timestamp("2016-01-01")
    assert set(df.loc[df["bad"] == 1, "status_norm"]) <= dp.BAD_STATUSES
    assert df.loc[df["status_norm"] == "Issued", "bad"].isna().all()
    assert df["months_on_book"].between(0, 12).all()
    assert (df.loc[df["in_dev_sample"], "months_on_book"] >= 6).all()


def test_lgd_ead_definitions(prepared):
    df, _ = prepared
    co = df[df["is_charged_off"] & (df["ead_at_default"] > 1)]
    assert len(co) > 5
    assert np.allclose(co["ead_at_default"], co["funded_amnt"] - co["total_rec_prncp"])
    assert co["ccf"].between(0, 1).all() and co["recovery_rate"].between(0, 1).all()
    assert np.allclose(co["lgd"], 1 - co["recovery_rate"])
