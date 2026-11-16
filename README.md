# Credit Scorecard Construction: PD, LGD, EAD & Expected Loss

End-to-end credit-risk modelling in Python (scikit-learn) on the **Lending Club 2015 loan file (Kaggle)**:

* a **PD scorecard**: supervised monotonic Weight-of-Evidence (WoE) binning, out-of-fold IV screening, logistic regression, PDO score scaling
* a two-stage (hurdle) **LGD** model and an exposure-factor **EAD** model
* **Expected loss** `EL = PD x LGD x EAD`, with segment views, a stress grid and a proxy back-test
* validation (AUROC / Gini / KS, calibration, bootstrap CI, repeated CV, PSI / CSI, gradient-boosting benchmark), a cut-off strategy and adverse-action style reason codes

> **Verification status.** The code is covered by 22 unit / end-to-end tests and has been run end to end on a **synthetic** file that mimics the schema of `loan_data_2015.csv`. It has **not** been run on the real Kaggle file in the environment where it was written, so this README contains **no real-data results**. Run the notebook (or the CLI), then copy the numbers from `results/summary.md` into the [Results](#results-fill-in-after-running) section.

---

## Pitfalls this project is built to avoid

| Pitfall | How it is handled |
|---|---|
| **Target leakage.** The raw file holds payment-history fields (`out_prncp`, `total_pymnt`, `total_rec_int`, `last_pymnt_amnt`, months since last payment ...) that only exist *after* origination and reveal whether the borrower is paying. A model using them can show an AUROC above 0.9 and be useless at decision time. | Predictors are **origination-only**. A *leakage audit* fits the same machinery with and without those fields so the inflation is **measured**, not assumed. |
| **Hand-picked bins.** Cut-offs chosen by eye cannot be re-fitted inside cross-validation, and silently break when a column's scale differs (e.g. `revol_util` is in percent, not a fraction). | `WoEBinner` learns monotone bins from training data only and works inside scikit-learn `Pipeline`s. |
| **Selection bias from supervised binning.** In-sample IV and p-values are optimistic because the bins were optimised on the same rows. | Variable screen uses **out-of-fold IV** (bins learned on part of the data, IV measured on held-out data), then sign / p-value / VIF checks. |
| **`class_weight='balanced'`** inflates predicted probabilities, so the mean PD no longer matches the default rate and EL is biased upwards. | No class re-weighting; calibration is checked by risk decile. |
| **Arbitrary "today" date** for seasoning features, and ignoring that the file is a snapshot with at most ~12 months of history. | Snapshot date is **inferred** from the data; a minimum months-on-book rule defines the sample; the PD is stated to be a *window* PD. |
| **Metrics that flatter an imbalanced target** (e.g. PR-AUC of the majority class). | AUROC, Gini, KS, Brier and PR-AUC of the **bad** class; bootstrap CI; CV that re-learns bins per fold. |
| **Deploying an LGD model that has no skill.** | A cross-validated check against the naive mean (training data only); the mean is used if the model does not beat it. |
| **Century bug in `earliest_cr_line`** (`%y` maps `Jan-62` to 2062). | Explicit correction, unit-tested. |

## Repository structure

```
credit-risk-scorecard/
├── README.md
├── requirements.txt
├── pytest.ini
├── configs/config.yaml            all thresholds, paths and scorecard scaling in one place
├── data/
│   ├── README.md                  where to get loan_data_2015.csv (not committed)
│   └── raw/
├── notebooks/
│   └── 01_credit_scorecard_pd_lgd_ead_el.ipynb   the narrated analysis (Run All)
├── src/
│   ├── data_prep.py               load, clean, features, targets, sample waterfall, split
│   ├── woe_binning.py             WoEBinner (sklearn transformer), out-of-fold IV
│   ├── feature_selection.py       IV screen, correlation filter, backward elimination, VIF, Wald p-values
│   ├── scorecard.py               points table, PDO scaling, PD <-> score, reason codes
│   ├── pd_model.py                fit / cross-validate the PD model, boosting benchmark
│   ├── metrics.py                 AUROC/Gini/KS, calibration, PSI/CSI, strategy tables
│   ├── lgd_ead.py                 two-stage LGD, CCF model, governance rule
│   ├── expected_loss.py           EL, segment summaries, stress grid, proxy back-test
│   ├── plotting.py                figures
│   └── pipeline.py                step functions + CLI (python -m src.pipeline)
├── scripts/make_synthetic_sample.py   SYNTHETIC data with the real schema, for smoke tests only
├── tests/                         22 tests (binning, scorecard maths, parsing, end-to-end)
├── results/                       tables, figures, summary.md, metrics.json (written by a run)
└── models/                        fitted scorecard and LGD/EAD models (written by a run)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. put loan_data_2015.csv in data/raw/   (see data/README.md)
# 2a. narrated analysis
jupyter notebook notebooks/01_credit_scorecard_pd_lgd_ead_el.ipynb      # Run All
# 2b. or headless
python -m src.pipeline --data data/raw/loan_data_2015.csv               # add --fast to skip CV + boosting benchmark

# tests (no real data needed)
pytest
```

On Kaggle: add the dataset to a notebook, `!git clone <your-repo-url>`, open the notebook from the clone (the loader also checks `/kaggle/input/loan-data-2015/`).

A run writes `results/summary.md`, `results/metrics.json`, `results/tables/*.csv` (including `scorecard_points.csv`, the finished scorecard), `results/figures/*.png` and `models/*.joblib`.

## Methodology

1. **Sample and target.** *Bad* = `Charged Off`, `Default`, `Late (31-120 days)`; *good* = `Fully Paid`, `Current`; `In Grace Period`, `Late (16-30 days)`, `Issued` excluded as indeterminate. Loans must have at least `min_months_on_book` months of history. Stratified 70/30 train/test split.
2. **Feature sets.** `standalone` (headline scorecard), `with_lc_grade` (adds LendingClub's own grade / rate, a benchmark because they are outputs of another credit model), `leaky_reference` (adds post-origination fields, **diagnostic only**). Geography and the listing flag are excluded by policy.
3. **WoE / IV.** `WoE = ln(%good / %bad)` (higher = safer), `IV = sum (%good - %bad) x WoE`. Numeric features: tree-based splits with at least 5% of the sample per bin, reduced until the bad rate is monotone. Categorical: rare levels pooled, levels grouped by bad rate. Missing values get their own bin.
4. **Selection.** Out-of-fold IV >= 0.02, then |corr| <= 0.70, then backward elimination (wrong sign first: every coefficient must be negative; then p > 0.05; then VIF > 5). Wald p-values are computed from the Fisher information in NumPy and cross-checked against statsmodels in the tests.
5. **Model and scaling.** Unpenalised logistic regression `logit P(bad) = a + sum b_j WoE_j`. Score `= Offset + Factor x ln(odds)`, `Factor = PDO / ln 2`, `Offset = BaseScore - Factor x ln(BaseOdds)` (defaults: 600 points at 50:1 odds, PDO 20). Points per bin are `-(b_j WoE_jb + a/n) x Factor + Offset/n`; the tests verify that the exact score reproduces `Offset + Factor x ln(odds)`.
6. **Validation.** Test AUROC / Gini / KS with bootstrap CI; repeated stratified CV in which the bins are re-learned per fold; risk-decile calibration and Hosmer-Lemeshow; score PSI (train vs test and by issue month) and per-variable CSI; gradient-boosting benchmark on raw features.
7. **Strategy.** Approval rate vs bad rate of the approved book, KS cut-off, score bands, reason codes for declined applicants.
8. **LGD / EAD** on charged-off loans. `EAD = funded_amnt - total_rec_prncp`; `CCF = EAD / funded_amnt`; `recovery rate = (recoveries - collection_recovery_fee) / EAD`; `LGD = 1 - recovery rate`. LGD: logistic regression for *P(any recovery)* x ridge for the recovery rate given recovery; benchmarks: single-stage ridge, gradient boosting, mean. EAD: ridge on CCF.
9. **Expected loss** on the test sample: `EL_i = PD_i x LGD_i x (CCF_i x funded_i)`; segment summaries by LC grade and risk quintile; stress grid (PD multiplier x additive LGD shock); proxy back-test against observed losses.

## Results (fill in after running)

Copy these from `results/summary.md` after a run on the real file. Nothing below has been run on real data.

| Item | Value |
|---|---|
| Development sample (loans / bad rate) | _ / _ % |
| PD scorecard: variables retained | _ |
| Test AUROC (95% CI) / Gini / KS | _ (_-_) / _ / _ |
| 5-fold CV AUROC | _ |
| Score PSI, train vs test | _ |
| Benchmarks: gradient boosting AUROC / scorecard + LC grade AUROC | _ / _ |
| Leakage audit: AUROC with post-origination fields | _ (not credible; shown to quantify leakage) |
| Charged-off loans used for LGD/EAD; mean LGD; mean CCF | _ ; _ % ; _ % |
| LGD / EAD models deployed (vs naive mean) | _ |
| Window expected loss, % of funded amount (model vs loss proxy) | _ % / _ % |

## Resume bullets

Choose one version. Replace `[ ]` with your own run's numbers; do not quote a figure you did not produce.

* Built an end-to-end credit-risk framework on [N] Lending Club loans in Python (scikit-learn): a WoE logistic-regression **PD scorecard** (AUROC [x], KS [x]), a two-stage **LGD** model and a CCF-based **EAD** model, combined into portfolio **expected loss** (EL = PD x LGD x EAD).
* Developed a custom scikit-learn **WoE/IV binning transformer** (monotone, tree-based) with out-of-fold IV screening to remove selection bias; validated with bootstrap CIs, repeated CV, calibration deciles and PSI/CSI stability monitoring.
* Ran a **leakage audit** showing that post-origination payment fields inflate AUROC from [x] to [y]; restricted the scorecard to origination-time predictors and scaled it to a 600-point / PDO-20 score with reason codes and a cut-off strategy.
* Modelled recoveries with a **hurdle (two-stage) LGD** model and a model-vs-naive governance check; stress-tested EL under PD multipliers and LGD shocks and back-tested by risk band.

Shorter version (no figures): *Built a PD/LGD/EAD credit-risk model suite with a WoE logistic-regression scorecard, leakage audit, calibration/stability validation and expected-loss stress testing (Python, scikit-learn).*

## Talking points for interviews

* **Why WoE + logistic regression?** Transparent, monotone, handles missing values and outliers, easy to explain and to audit; the boosting benchmark shows how much (or little) signal this gives up.
* **How do you know it is not leaking?** Predictors are origination-only, and the audit shows what the post-origination fields would have done to the AUROC.
* **Why no `class_weight='balanced'`?** PD feeds EL, so it must be a calibrated probability; re-weighting shifts the probabilities.
* **What does PDO mean?** Points to double the odds: +PDO points = 2x the good:bad odds; the tests check this identity.
* **Why a hurdle LGD?** Recoveries are zero-inflated and bounded; a single linear model mis-specifies both features.
* **What are the limits?** See below: the PD is a window PD, LGD is biased upwards by incomplete workouts, accepted loans only, one vintage.

## Limitations and extensions

* **Window PD, not lifetime PD.** Loans in the 2015 file have at most ~12 months of history; EL is a window EL (conceptually the building block of an IFRS 9 12-month ECL, undiscounted), not a lifetime loss. Extension: discrete-time hazard / vintage models on a more mature file.
* **LGD maturity.** Recoveries on recent charge-offs are incomplete, so LGD is biased upwards. Extension: the 2007-2014 file, or a recovery-development curve.
* **Accepted loans only**, so no reject inference.
* **Single vintage**, so no true out-of-time test. Extension: develop on the 2007-2014 file and validate out of time on 2015.
* **No macro overlay, pricing or capital.** PD is point-in-time for one origination year.
* **Fair lending.** Geography is excluded by policy; disparate-impact and explainability review would be needed before any real use.

## Tests

`pytest` runs 22 tests: WoE sign and monotonicity, out-of-fold IV behaviour, missing / unseen / rare handling, scikit-learn `Pipeline` / `clone` compatibility, the scorecard identity `score = Offset + Factor x ln(odds)`, the PDO property, PD <-> score round trip, PSI, VIF, p-values vs statsmodels, date and status parsing (including the century fix), the LGD / EAD definitions, and a full pipeline run on synthetic data (including the leakage audit and the expected-loss identity).

## References

* Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.* Wiley. (WoE / IV, PDO scaling)
* Basel Committee on Banking Supervision, internal-ratings-based (IRB) approach: `EL = PD x LGD x EAD`.
* IFRS 9 *Financial Instruments*: expected credit loss (12-month ECL for stage 1).
