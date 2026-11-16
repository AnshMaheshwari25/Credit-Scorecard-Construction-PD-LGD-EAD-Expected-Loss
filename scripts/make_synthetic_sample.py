#!/usr/bin/env python
"""Generate a SYNTHETIC Lending-Club-style loan file for offline smoke tests.

**This is NOT real data and must never be used to report results.**

It reproduces the *schema and string formats* of ``loan_data_2015.csv`` (2015 originations,
loan status observed at a Jan-2016 snapshot) so that every step of the pipeline can be
exercised without the Kaggle download: ' 36 months', '10+ years', 'Dec-15', percent-scale
``revol_util``, 2-digit-year ``earliest_cr_line``, the status mix, and internally consistent
post-origination payment fields (``out_prncp``, ``total_rec_prncp``, ``recoveries`` ...).

Usage
-----
    python scripts/make_synthetic_sample.py --n 60000 --out data/raw/synthetic_loans_2015.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SNAP_YEAR, SNAP_MONTH = 2016, 1  # snapshot of the status columns (Jan-2016)

STATES = ["CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH", "GA", "VA", "NC", "MI", "MA", "AZ", "MD",
          "WA", "CO", "MN", "MO", "IN", "CT", "TN", "NV", "WI", "OR", "SC", "AL", "LA", "KY", "KS",
          "OK", "UT", "AR", "MS", "NH", "NM", "RI", "HI", "WV", "DE", "NE", "MT", "DC", "AK", "WY",
          "VT", "SD", "ME", "ID", "ND"]


def fmt_month(year, month) -> np.ndarray:
    year = np.asarray(year)
    month = np.asarray(month)
    return np.array([f"{MONTHS[m - 1]}-{str(y)[-2:]}" for y, m in zip(year, month)], dtype=object)


def installment(P, rate_pct, n):
    r = rate_pct / 1200.0
    return P * r * (1 + r) ** n / ((1 + r) ** n - 1)


def balance_after(P, rate_pct, n, k):
    """Scheduled outstanding principal after k payments on an n-month amortising loan."""
    r = rate_pct / 1200.0
    k = np.minimum(k, n)
    return P * ((1 + r) ** n - (1 + r) ** k) / ((1 + r) ** n - 1)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n: int = 60000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---- origination month and seasoning ------------------------------------------------
    w = np.array([0.060, 0.060, 0.075, 0.075, 0.075, 0.080, 0.085, 0.085, 0.090, 0.095, 0.100, 0.120])
    w = w / w.sum()
    issue_m = rng.choice(np.arange(1, 13), size=n, p=w)
    months_on_book = (SNAP_YEAR * 12 + SNAP_MONTH) - (2015 * 12 + issue_m)  # Jan-15 -> 12, Dec-15 -> 1

    # ---- borrower / loan attributes ------------------------------------------------------
    term = rng.choice([36, 60], size=n, p=[0.68, 0.32])
    loan_amnt = np.clip(np.round(rng.lognormal(9.55, 0.62, n) / 25) * 25, 1000, 35000)
    annual_inc = np.clip(rng.lognormal(11.20, 0.55, n), 8000, 9_500_000)
    dti = np.clip(rng.gamma(4.5, 4.0, n), 0, 60)
    util = np.clip(rng.beta(2.0, 2.6, n) * 105, 0, 130)  # PERCENT scale, as in the real file
    util_missing = rng.random(n) < 0.0004
    rev_hi_lim = np.clip(rng.lognormal(9.9, 0.8, n), 1000, 500000)
    revol_bal = np.round(util / 100.0 * rev_hi_lim)
    hist_m = np.clip(rng.gamma(5.5, 38.0, n), 36, 700).astype(int)
    inq = rng.poisson(0.65, n)
    delinq = rng.poisson(0.28 * (1 + 0.5 * (dti > 25)), n)
    pub_rec = rng.poisson(0.15, n)
    open_acc = rng.poisson(11, n) + 1
    total_acc = open_acc + rng.poisson(14, n)
    home = rng.choice(["MORTGAGE", "RENT", "OWN", "ANY"], size=n, p=[0.50, 0.378, 0.12, 0.002])
    verif = rng.choice(["Source Verified", "Verified", "Not Verified"], size=n, p=[0.34, 0.31, 0.35])
    purposes = ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
                "small_business", "car", "medical", "moving", "vacation", "house", "wedding",
                "renewable_energy", "educational"]
    p_purpose = np.array([0.575, 0.222, 0.062, 0.058, 0.021, 0.012, 0.010, 0.012, 0.007, 0.006, 0.006, 0.0003,
                          0.0006, 0.0001])
    purpose = rng.choice(purposes, size=n, p=p_purpose / p_purpose.sum())
    w_state = np.r_[[0.14, 0.085, 0.08, 0.07, 0.04, 0.035, 0.03, 0.03, 0.03, 0.027, 0.025, 0.023, 0.022,
                     0.02, 0.019, 0.018, 0.018, 0.016, 0.016, 0.014], np.full(30, 0.012)]
    state = rng.choice(STATES, size=n, p=w_state / w_state.sum())
    emp_labels = ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years",
                  "8 years", "9 years", "10+ years"]
    p_emp = np.array([0.08, 0.065, 0.09, 0.08, 0.06, 0.06, 0.05, 0.045, 0.045, 0.04, 0.385])
    emp_length = rng.choice(emp_labels, size=n, p=p_emp / p_emp.sum()).astype(object)
    emp_length[rng.random(n) < 0.057] = np.nan

    # ---- latent risk index (some part observable, some unobservable) -----------------------
    emp_num = np.array([0 if (isinstance(e, float) or e is None) else
                        (0 if e.startswith("<") else int("".join(c for c in e if c.isdigit())))
                        for e in emp_length])
    lti = loan_amnt / annual_inc
    r = (0.020 * dti + 0.008 * util + 0.12 * inq + 0.15 * delinq + 0.45 * (term == 60)
         - 0.35 * (np.log(annual_inc) - 11.2) + 0.10 * (home == "RENT") + 0.15 * (verif == "Verified")
         + 0.45 * (purpose == "small_business") + 0.20 * (pub_rec > 0) - 0.0015 * (hist_m - 200)
         + 0.8 * np.clip(lti, 0, 0.6) - 0.01 * emp_num + rng.normal(0, 0.55, n))
    r_std = (r - r.mean()) / r.std()

    # LendingClub-style grade / rate: a noisy view of the same risk index
    gscore = r_std + rng.normal(0, 0.25, n)
    cuts = np.quantile(gscore, np.cumsum([0.17, 0.29, 0.29, 0.14, 0.08, 0.025])[:6])
    grade_idx = np.searchsorted(cuts, gscore)
    grade = np.array(list("ABCDEFG"))[grade_idx]
    base_rate = np.array([7.4, 11.0, 14.2, 17.5, 21.0, 24.5, 27.5])[grade_idx]
    int_rate = np.clip(base_rate + rng.normal(0, 1.2, n) + 1.2 * (term == 60), 5.32, 30.99).round(2)
    sub_grade = np.array([f"{g}{d}" for g, d in zip(grade, rng.integers(1, 6, n))])

    # ---- outcome by snapshot: lifetime risk x seasoning --------------------------------------
    season = 1 - np.exp(-months_on_book / 6.5)

    def mean_bad(b0):
        return float(np.mean(_sigmoid(b0 + 0.95 * r_std) * season))

    lo, hi = -9.0, 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean_bad(mid) < 0.0186 else (lo, mid)
    p_bad = _sigmoid((lo + hi) / 2 + 0.95 * r_std) * season
    bad = rng.random(n) < p_bad

    sub = rng.choice(["co", "late", "def"], size=n, p=[0.42, 0.535, 0.045])
    co_ok = months_on_book >= 5
    status = np.full(n, "Current", dtype=object)
    status[bad & (sub == "co") & co_ok] = "Charged Off"
    status[bad & ((sub == "late") | ((sub == "co") & ~co_ok))] = "Late (31-120 days)"
    status[bad & (sub == "def")] = "Default"
    nb = ~bad
    fp = nb & (rng.random(n) < 0.0085 * months_on_book)
    gp = nb & ~fp & (rng.random(n) < 0.0075)
    l16 = nb & ~fp & ~gp & (rng.random(n) < 0.0028)
    iss = nb & ~fp & ~gp & ~l16 & (issue_m == 12) & (rng.random(n) < 0.18)
    status[fp] = "Fully Paid"
    status[gp] = "In Grace Period"
    status[l16] = "Late (16-30 days)"
    status[iss] = "Issued"

    # ---- payment history consistent with status -------------------------------------------
    funded = np.where(rng.random(n) < 0.985, loan_amnt, np.floor(loan_amnt * rng.uniform(0.6, 1.0, n) / 25) * 25)
    inst = installment(funded, int_rate, term)
    m = months_on_book
    k = np.maximum(m - 1, 0)  # default: "Current"-like
    k = np.where(status == "Current", m, k)
    k_fp = np.minimum(rng.integers(1, 13, n), np.maximum(m, 1))
    k = np.where(status == "Fully Paid", k_fp, k)
    k = np.where(status == "Late (31-120 days)", np.maximum(m - rng.integers(1, 5, n), 0), k)
    k = np.where(status == "Default", np.maximum(m - rng.integers(4, 7, n), 0), k)
    k = np.where(status == "Charged Off", np.maximum(m - rng.integers(5, 8, n), 0), k)
    k = np.where(status == "Issued", 0, k)

    bal = balance_after(funded, int_rate, term, k)
    sched_prncp = funded - bal
    total_rec_prncp = np.where(status == "Fully Paid", funded, sched_prncp)
    total_rec_int = np.maximum(k * inst - sched_prncp, 0)
    late_fee = np.where(np.isin(status, ["Late (31-120 days)", "Default", "Charged Off"]) & (rng.random(n) < 0.5),
                        rng.uniform(0, 30, n), 0.0)
    out_prncp = np.where(np.isin(status, ["Fully Paid", "Charged Off"]), 0.0, bal)
    out_prncp = np.where(status == "Issued", funded, out_prncp)

    # recoveries (charged-off only): hurdle process - many zeros, small positive recovery rates
    ead = funded - total_rec_prncp
    has_rec = (status == "Charged Off") & (rng.random(n) < (0.55 - 0.05 * (grade_idx > 3)))
    rr = np.clip(rng.beta(0.9, 5.0, n) * (1.0 + 0.15 * (np.log(annual_inc) - 11.2)), 0, 1)
    recoveries = np.where(has_rec, rr * ead, 0.0).round(2)
    coll_fee = np.where(has_rec & (rng.random(n) < 0.6), 0.15 * recoveries, 0.0).round(2)
    total_pymnt = total_rec_prncp + total_rec_int + late_fee + recoveries

    # ---- dates -----------------------------------------------------------------------------
    snap_idx = SNAP_YEAR * 12 + SNAP_MONTH - 1
    issue_idx = 2015 * 12 + issue_m - 1
    last_idx = np.where(status == "Current", snap_idx - (rng.random(n) < 0.04), issue_idx + np.maximum(k, 0))
    last_idx = np.clip(last_idx, issue_idx, snap_idx)
    last_pymnt_d = fmt_month(last_idx // 12, last_idx % 12 + 1)
    no_pay = (k == 0) | (status == "Issued")
    last_pymnt_d = np.where(no_pay, None, last_pymnt_d)
    nxt = np.isin(status, ["Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)", "Default"])
    next_pymnt_d = np.where(nxt, fmt_month(np.full(n, SNAP_YEAR), np.full(n, SNAP_MONTH + 1)), None)
    pull_idx = np.where(rng.random(n) < 0.9, snap_idx, snap_idx - rng.integers(1, 8, n))
    last_credit_pull_d = fmt_month(pull_idx // 12, pull_idx % 12 + 1)
    last_pymnt_amnt = np.where(status == "Current", inst,
                      np.where(status == "Fully Paid", np.minimum(funded, inst * rng.uniform(1, 9, n)),
                      np.where(no_pay, 0.0, inst * rng.uniform(0.0, 1.0, n)))).round(2)
    earliest_idx = issue_idx - hist_m
    earliest_cr_line = fmt_month(earliest_idx // 12, earliest_idx % 12 + 1)

    # ---- assemble --------------------------------------------------------------------------
    def nanify(a, frac, rng_=rng):
        a = a.astype(float).copy()
        a[rng_.random(len(a)) < frac] = np.nan
        return a

    df = pd.DataFrame({
        "id": np.arange(60_000_000, 60_000_000 + n),
        "member_id": np.arange(70_000_000, 70_000_000 + n),
        "loan_amnt": loan_amnt.astype(int),
        "funded_amnt": funded.astype(int),
        "funded_amnt_inv": funded.astype(int),
        "term": [f" {t} months" for t in term],
        "int_rate": int_rate,
        "installment": inst.round(2),
        "grade": grade,
        "sub_grade": sub_grade,
        "emp_title": np.where(rng.random(n) < 0.057, None, "Employee"),
        "emp_length": emp_length,
        "home_ownership": home,
        "annual_inc": annual_inc.round(0),
        "verification_status": verif,
        "issue_d": fmt_month(np.full(n, 2015), issue_m),
        "loan_status": status,
        "pymnt_plan": "n",
        "url": "https://example.invalid/loan",
        "desc": np.nan,
        "purpose": purpose,
        "title": "Debt consolidation",
        "zip_code": [f"{z}xx" for z in rng.integers(100, 999, n)],
        "addr_state": state,
        "dti": dti.round(2),
        "delinq_2yrs": delinq,
        "earliest_cr_line": earliest_cr_line,
        "inq_last_6mths": inq,
        "mths_since_last_delinq": nanify(rng.integers(1, 100, n), 0.52),
        "mths_since_last_record": nanify(rng.integers(1, 120, n), 0.82),
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_bal": revol_bal.astype(int),
        "revol_util": np.where(util_missing, np.nan, util.round(1)),
        "total_acc": total_acc,
        "initial_list_status": rng.choice(["w", "f"], size=n, p=[0.55, 0.45]),
        "out_prncp": out_prncp.round(2),
        "out_prncp_inv": (out_prncp * 0.98).round(2),
        "total_pymnt": total_pymnt.round(2),
        "total_pymnt_inv": (total_pymnt * 0.98).round(2),
        "total_rec_prncp": total_rec_prncp.round(2),
        "total_rec_int": total_rec_int.round(2),
        "total_rec_late_fee": late_fee.round(2),
        "recoveries": recoveries,
        "collection_recovery_fee": coll_fee,
        "last_pymnt_d": last_pymnt_d,
        "last_pymnt_amnt": last_pymnt_amnt,
        "next_pymnt_d": next_pymnt_d,
        "last_credit_pull_d": last_credit_pull_d,
        "collections_12_mths_ex_med": rng.poisson(0.015, n),
        "mths_since_last_major_derog": nanify(rng.integers(1, 100, n), 0.71),
        "policy_code": 1,
        "application_type": rng.choice(["INDIVIDUAL", "JOINT"], size=n, p=[0.9988, 0.0012]),
        "acc_now_delinq": rng.poisson(0.004, n),
        "tot_coll_amt": np.where(rng.random(n) < 0.8, 0, rng.lognormal(6, 1.4, n)).round(0).astype(int),
        "tot_cur_bal": np.clip(rng.lognormal(10.9, 1.2, n), 0, 3_000_000).round(0).astype(int),
        "total_rev_hi_lim": rev_hi_lim.round(0).astype(int),
    })
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("data/raw/synthetic_loans_2015.csv"))
    args = ap.parse_args()
    df = generate(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} SYNTHETIC rows to {args.out}")
    print(df["loan_status"].value_counts(normalize=True).round(4).to_string())


if __name__ == "__main__":
    main()
