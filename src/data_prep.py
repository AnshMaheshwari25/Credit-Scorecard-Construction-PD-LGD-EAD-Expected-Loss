"""Load, clean and engineer the Lending Club loan file; define the PD / LGD / EAD targets.

Design principles
-----------------
* **Origination-only predictors.** A PD scorecard is used to decide on an application, so it may only
  use information available at origination. Payment-history fields (``out_prncp``, ``total_pymnt``,
  ``total_rec_int``, ``last_pymnt_amnt``, ``last_pymnt_d`` ...) are generated *after* the loan is booked and
  mechanically reveal whether the borrower is paying, so they are kept out of the model
  (``POST_ORIGINATION``) and only used for the leakage audit and for the LGD / EAD targets.
* **Explicit observation window.** The 2015 file is a snapshot taken about Jan-2016, so every loan has at most
  ~12 months of history. ``months_on_book`` is computed against the snapshot and a minimum-seasoning filter
  is applied, instead of the arbitrary "today" date used in many tutorials.
* **Date parsing.** ``earliest_cr_line`` has two-digit years; ``%y`` maps 69-99 -> 19xx and 00-68 -> 20xx, so
  credit lines opened before 1969 would be parsed as 20xx. That is corrected explicitly.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------------------------------------
# Column groups
# --------------------------------------------------------------------------------------------------------
NUMERIC_RAW = [
    "loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs",
    "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "collections_12_mths_ex_med", "mths_since_last_major_derog",
    "acc_now_delinq", "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim",
]
CATEGORICAL_RAW = [
    "grade", "home_ownership", "verification_status", "purpose", "addr_state", "application_type",
    "initial_list_status",
]
OTHER_ORIGINATION_RAW = ["term", "emp_length", "issue_d", "earliest_cr_line"]
POST_ORIGINATION_RAW = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
    "next_pymnt_d", "last_credit_pull_d",
]
TARGET_RAW = ["loan_status"]
REQUIRED = [
    "loan_status", "issue_d", "term", "loan_amnt", "funded_amnt", "int_rate", "grade", "annual_inc", "dti",
    "earliest_cr_line", "total_rec_prncp", "recoveries", "collection_recovery_fee", "last_pymnt_d",
]

# Statuses ------------------------------------------------------------------------------------------------
BAD_STATUSES = {"Charged Off", "Default", "Late (31-120 days)"}
GOOD_STATUSES = {"Fully Paid", "Current"}
INDETERMINATE_STATUSES = {"In Grace Period", "Late (16-30 days)", "Issued"}

# Model feature sets ------------------------------------------------------------------------------------------
BASE_NUMERIC = [
    "loan_amnt", "term_months", "emp_length_yrs", "annual_inc", "dti", "delinq_2yrs", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "collections_12_mths_ex_med", "mths_since_last_major_derog", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim", "credit_history_months", "loan_to_income",
    "revol_bal_to_income", "tot_cur_bal_to_income",
]
BASE_CATEGORICAL = ["home_ownership", "verification_status", "purpose", "application_type"]

# LendingClub's own credit assessment. These are *outputs of another credit model* (and the interest rate is
# the price), so the headline scorecard is built without them and they are added only for a benchmark.
LC_ASSESSMENT_NUMERIC = ["int_rate", "payment_to_income"]  # payment_to_income embeds the rate via the instalment
LC_ASSESSMENT_CATEGORICAL = ["grade"]

# Excluded by policy: geography can act as a proxy for protected characteristics (fair-lending review needed);
# initial_list_status is an investor-side listing flag, not a borrower attribute.
POLICY_EXCLUDED = ["addr_state", "zip_code", "initial_list_status"]

# Generated after origination: used only for the leakage audit, never in a deployable model.
POST_ORIGINATION = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_amnt",
    "months_since_last_pymnt", "months_since_last_credit_pull", "months_on_book",
]

LGD_EAD_NUMERIC = [
    "loan_amnt", "term_months", "int_rate", "installment", "annual_inc", "dti", "emp_length_yrs", "revol_util",
    "credit_history_months", "inq_last_6mths", "delinq_2yrs", "loan_to_income", "open_acc",
]
LGD_EAD_CATEGORICAL = ["grade", "home_ownership", "verification_status", "purpose"]


def feature_sets() -> dict[str, list[str]]:
    """Named PD feature sets (column lists)."""
    standalone = BASE_NUMERIC + BASE_CATEGORICAL
    with_lc = standalone + LC_ASSESSMENT_NUMERIC + LC_ASSESSMENT_CATEGORICAL
    leaky = with_lc + POST_ORIGINATION
    return {"standalone": standalone, "with_lc_grade": with_lc, "leaky_reference": leaky}


# --------------------------------------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------------------------------------
def _to_float(s: pd.Series) -> pd.Series:
    """Numeric conversion that also accepts percent strings such as '13.5%'."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    cleaned = s.astype("object").astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def parse_month(s: pd.Series) -> pd.Series:
    """'Dec-15' -> Timestamp (first of month). Unparseable -> NaT."""
    return pd.to_datetime(s, format="%b-%y", errors="coerce")


def _month_index(ts: pd.Series) -> pd.Series:
    return ts.dt.year * 12 + ts.dt.month


def parse_term(s: pd.Series) -> pd.Series:
    """' 36 months' -> 36."""
    return pd.to_numeric(s.astype("object").astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def parse_emp_length(s: pd.Series) -> pd.Series:
    """'10+ years' -> 10, '< 1 year' -> 0, 'n/a' / missing -> NaN (kept missing, not imputed as 0)."""
    txt = s.astype("object").astype(str)
    num = pd.to_numeric(txt.str.extract(r"(\d+)")[0], errors="coerce")
    return num.where(~txt.str.contains("<", regex=False), 0.0)


def normalize_status(s: pd.Series) -> pd.Series:
    """Strip the legacy 'Does not meet the credit policy. Status:' prefix used in the 2007-2014 file."""
    return (s.astype("object").astype(str)
            .str.replace("Does not meet the credit policy. Status:", "", regex=False).str.strip())


def credit_history_months(earliest: pd.Series, issue: pd.Series) -> pd.Series:
    """Months between the earliest credit line and loan issue, with the two-digit-year century fix."""
    ecl = parse_month(earliest)
    iss = parse_month(issue) if not pd.api.types.is_datetime64_any_dtype(issue) else issue
    year = ecl.dt.year
    year = year.where(ecl <= iss, year - 100)  # e.g. 'Jan-62' parsed as 2062 -> 1962
    months = (iss.dt.year - year) * 12 + (iss.dt.month - ecl.dt.month)
    return months.clip(lower=0)


# --------------------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------------------
def load_raw(path, nrows: int | None = None) -> pd.DataFrame:
    """Read only the columns the project needs (the raw file has 74 columns, many of them ~all-null)."""
    needed = set(NUMERIC_RAW + CATEGORICAL_RAW + OTHER_ORIGINATION_RAW + POST_ORIGINATION_RAW + TARGET_RAW)
    df = pd.read_csv(path, usecols=lambda c: c in needed, low_memory=False, nrows=nrows)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from {path}: {missing}")

    # Some vintages omit secondary LendingClub fields. Add them as missing so the
    # modelling pipeline fails only on genuinely required target/date fields, not
    # with an obscure KeyError several steps downstream.
    optional = set(NUMERIC_RAW + CATEGORICAL_RAW + OTHER_ORIGINATION_RAW + POST_ORIGINATION_RAW)
    for col in optional.difference(df.columns):
        df[col] = np.nan
    return df


def infer_snapshot(df: pd.DataFrame, override=None) -> pd.Timestamp:
    """Reporting date of the file: latest payment / credit-pull month (unless overridden)."""
    if override:
        return pd.Timestamp(override).to_period("M").to_timestamp()
    cands = []
    for c in ("last_pymnt_d", "last_credit_pull_d"):
        if c in df.columns:
            cands.append(parse_month(df[c]).max())
    cands = [c for c in cands if pd.notna(c)]
    if not cands:
        raise ValueError("Cannot infer the snapshot date; set sample.snapshot_date in the config.")
    return max(cands)


# --------------------------------------------------------------------------------------------------------
# Main preparation
# --------------------------------------------------------------------------------------------------------
def build_dataset(df_raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Clean, engineer features, define targets and flag the development sample.

    Returns ``(df, info)``; ``info`` holds the snapshot date and the sample waterfall table.
    """
    scfg = cfg["sample"]
    df = df_raw.copy()

    # numeric conversion --------------------------------------------------------------------------------------
    for c in NUMERIC_RAW + [c for c in POST_ORIGINATION_RAW if c not in ("last_pymnt_d", "next_pymnt_d",
                                                                          "last_credit_pull_d")]:
        if c in df.columns:
            df[c] = _to_float(df[c])
    for c in ("recoveries", "collection_recovery_fee", "total_rec_prncp"):
        df[c] = df[c].fillna(0.0)

    # dates and seasoning -----------------------------------------------------------------------------------
    snapshot = infer_snapshot(df, scfg.get("snapshot_date"))
    issue = parse_month(df["issue_d"])
    snap_idx = snapshot.year * 12 + snapshot.month
    df["issue_date"] = issue
    df["months_on_book"] = (snap_idx - _month_index(issue)).clip(lower=0)

    last_pay = parse_month(df["last_pymnt_d"])
    df["months_since_last_pymnt"] = (snap_idx - _month_index(last_pay)).clip(lower=0)
    if "last_credit_pull_d" in df.columns:
        df["months_since_last_credit_pull"] = (snap_idx - _month_index(parse_month(df["last_credit_pull_d"]))
                                              ).clip(lower=0)
    else:
        df["months_since_last_credit_pull"] = np.nan

    # origination features ----------------------------------------------------------------------------------------
    df["term_months"] = parse_term(df["term"])
    df["emp_length_yrs"] = parse_emp_length(df["emp_length"]) if "emp_length" in df.columns else np.nan
    df["credit_history_months"] = credit_history_months(df["earliest_cr_line"], issue)
    income = df["annual_inc"].where(df["annual_inc"] > 0)
    df["loan_to_income"] = df["loan_amnt"] / income
    df["payment_to_income"] = df["installment"] * 12.0 / income
    df["revol_bal_to_income"] = df["revol_bal"] / income
    df["tot_cur_bal_to_income"] = df["tot_cur_bal"] / income

    # PD target: 1 = bad ----------------------------------------------------------------------------------------------
    status = normalize_status(df["loan_status"])
    df["status_norm"] = status
    known = BAD_STATUSES | GOOD_STATUSES | INDETERMINATE_STATUSES
    unknown = sorted(set(status.unique()) - known)
    if unknown:
        warnings.warn(f"Unrecognised loan_status values treated as indeterminate: {unknown}")
    df["bad"] = status.isin(BAD_STATUSES).astype(float).where(status.isin(BAD_STATUSES | GOOD_STATUSES))

    # LGD / EAD targets (meaningful for charged-off loans) ----------------------------------------------------
    min_ead = cfg.get("lgd_ead", {}).get("min_ead", 1.0)
    funded = df["funded_amnt"].where(df["funded_amnt"] > 0)
    ead = (df["funded_amnt"] - df["total_rec_prncp"]).clip(lower=0)
    df["ead_at_default"] = ead
    df["ccf"] = (ead / funded).clip(0, 1)
    df["net_recoveries"] = (df["recoveries"] - df["collection_recovery_fee"]).clip(lower=0)
    df["recovery_rate"] = (df["net_recoveries"] / ead.where(ead > min_ead)).clip(0, 1)
    df["lgd"] = 1.0 - df["recovery_rate"]
    df["is_charged_off"] = status.eq("Charged Off")

    # development sample + waterfall ------------------------------------------------------------------------------------
    n0 = len(df)
    determinate = df["bad"].notna() if scfg.get("exclude_indeterminate", True) else status.ne("Issued")
    seasoned = df["months_on_book"] >= scfg.get("min_months_on_book", 0)
    in_dev = determinate & seasoned
    df["in_dev_sample"] = in_dev
    df["bad_dev"] = df["bad"].where(in_dev)

    wf = pd.DataFrame({
        "step": ["All loans in file", "after dropping indeterminate / un-seasoned statuses",
                 f"after min {scfg.get('min_months_on_book', 0)} months on book (development sample)"],
        "loans": [n0, int(determinate.sum()), int(in_dev.sum())],
        "bad": [int((df["bad"] == 1).sum()), int(((df["bad"] == 1) & determinate).sum()),
                int((df["bad"] == 1)[in_dev].sum())],
    })
    wf["bad_rate_%"] = (100 * wf["bad"] / wf["loans"]).round(2)
    info = {"snapshot": snapshot, "waterfall": wf,
            "status_mix": status.value_counts(normalize=True).rename("share").to_frame()}
    return df, info


def split_development_sample(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train / test split of the development sample (stratified on the bad flag)."""
    scfg = cfg["sample"]
    dev = df[df["in_dev_sample"]]
    if dev.empty:
        raise ValueError("The development sample is empty. Check loan statuses and min_months_on_book.")
    class_counts = dev["bad"].astype(int).value_counts()
    if len(class_counts) != 2 or class_counts.min() < 2:
        raise ValueError("The development sample needs at least two good and two bad loans for a stratified split.")
    train, test = train_test_split(dev, test_size=scfg["test_size"], random_state=scfg["random_state"],
                                   stratify=dev["bad"].astype(int))
    return train.copy(), test.copy()
