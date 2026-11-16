# Credit Risk Scorecard: PD, LGD, EAD Proxy & Expected Loss

An end-to-end, reproducible Python project for retail credit-risk modelling using a historical LendingClub loan-file vintage. It builds an interpretable **Probability of Default (PD) scorecard**, models recoveries and exposure proxies for **LGD/EAD**, and combines the estimates into **Expected Loss (EL = PD × LGD × EAD)**.

The repository includes the modelling code, synthetic-data generator, automated tests and a command-line pipeline. The real LendingClub dataset is not redistributed here: download it separately and keep it outside version control.

> **Important verification note:** the checked-in tests use synthetic data. Synthetic runs verify that the pipeline executes; they do not establish real-world model performance. Do not put synthetic metrics on a CV or describe them as LendingClub results. Run the project against the real file before reporting sample sizes, AUROC, KS, EL or any other performance result.

## What the project does

- **Data preparation:** standardises LendingClub fields, parses dates and employment tenure, creates borrower-level ratios, defines the bad/good outcome and documents the sample waterfall.
- **PD scorecard:** custom scikit-learn-compatible Weight of Evidence (WoE) transformer, monotonic numeric binning, rare-category pooling, out-of-fold Information Value (IV) screening and logistic regression.
- **Feature selection:** out-of-fold IV, correlation filtering, coefficient-sign checks, Wald p-values and variance inflation factor (VIF) checks.
- **Score scaling:** additive points scorecard with configurable base score, base odds and Points to Double the Odds (PDO); includes applicant-level reason codes.
- **Validation:** AUROC, Gini, KS, bad-class PR-AUC, Brier score, bootstrap confidence intervals, cross-validation, calibration deciles, PSI/CSI, a gradient-boosting benchmark and approval cut-off analysis.
- **LGD and exposure:** a two-stage recovery model, an exposure-factor model and comparisons against simple benchmarks.
- **Expected loss:** portfolio and segment views, stress scenarios and a clearly labelled realised-loss proxy comparison.
- **Reproducibility:** central YAML configuration, CLI, pytest suite and GitHub Actions CI.

## Repository layout

```text
.
├── configs/
│   └── config.yaml
├── data/
│   ├── README.md
│   └── raw/                 # local CSVs only; ignored by Git
├── scripts/
│   └── make_synthetic_sample.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_prep.py
│   ├── expected_loss.py
│   ├── feature_selection.py
│   ├── lgd_ead.py
│   ├── metrics.py
│   ├── pd_model.py
│   ├── pipeline.py
│   ├── plotting.py
│   ├── scorecard.py
│   └── woe_binning.py
├── tests/
│   ├── conftest.py
│   ├── test_data_prep.py
│   ├── test_pipeline_end_to_end.py
│   ├── test_scorecard_metrics.py
│   └── test_woe_binning.py
├── .github/workflows/tests.yml
├── .gitignore
├── pytest.ini
├── requirements.txt
└── requirements-dev.txt
```

Generated `results/` reports and `models/` model files are created when the pipeline runs and are intentionally excluded from Git.

## Quick start

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 1. Run a synthetic end-to-end smoke test

This creates artificial records with a LendingClub-like schema. It is useful for verifying installation and output generation only.

```bash
python scripts/make_synthetic_sample.py --n 40000 --seed 5 --out data/raw/synthetic_loans_2015.csv
python -m src.pipeline --data data/raw/synthetic_loans_2015.csv --fast
```

The `--fast` flag skips repeated PD cross-validation and the gradient-boosting benchmark. The synthetic generator is deterministic for a fixed seed.

### 2. Run on the real LendingClub file

1. Obtain the historical LendingClub 2015 loan CSV from its data provider. The expected filename is `loan_data_2015.csv`; see [data/README.md](data/README.md) for the expected location and handling.
2. Place it at `data/raw/loan_data_2015.csv`, or set the `CREDIT_RISK_DATA` environment variable to its full path.
3. Run the fast or full pipeline:

```bash
# Quick run: no repeated CV or gradient-boosting benchmark
python -m src.pipeline --data data/raw/loan_data_2015.csv --fast

# Full run: includes repeated CV and the raw-feature benchmark
python -m src.pipeline --data data/raw/loan_data_2015.csv
```

You can override configuration and output locations:

```bash
python -m src.pipeline --data /path/to/loan_data_2015.csv \
  --config configs/config.yaml \
  --results-dir results \
  --models-dir models
```

The CLI can also auto-detect the data path from `configs/config.yaml` and `CREDIT_RISK_DATA`; see [src/config.py](src/config.py).

### 3. Run the tests

```bash
pytest -q
```

The suite covers parsing, target/feature definitions, missing optional columns, WoE binning, out-of-fold IV, estimator cloning, scorecard/PDO identities, metrics and a synthetic end-to-end run. GitHub Actions runs the same tests on pushes and pull requests.

## Methodology

### PD target and sample

The binary target is defined as:

- **Bad (1):** `Charged Off`, `Default`, `Late (31-120 days)`
- **Good (0):** `Fully Paid`, `Current`
- **Excluded as indeterminate:** `In Grace Period`, `Late (16-30 days)`, `Issued` and unrecognised statuses

The configuration applies a minimum months-on-book filter and a stratified train/test split. The resulting PD is a **snapshot/window outcome**, not a lifetime probability of default. A random split within one vintage is not an out-of-time validation.

### Origination-time feature controls

The headline scorecard uses origination-time borrower and loan characteristics. Payment-performance fields recorded after origination are excluded from the deployable feature set. A separate leakage audit fits a diagnostic model with those fields to demonstrate why their predictive lift is not suitable for an application-time model.

LendingClub grade and interest-rate fields are treated as a separate benchmark feature set because they reflect a prior lender assessment and pricing decision. They are not part of the standalone headline scorecard. Geography and investor-listing fields are excluded from the model feature sets by policy.

### WoE, IV and scorecard

For each feature, the transformer learns bins on the training sample only. Numeric bins are tree-derived and constrained to monotonic bad rates; rare categorical levels are pooled and grouped. Missing/unseen values are handled without refitting on validation data.

`WoE = ln(% good / % bad)` and `IV = Σ[(% good − % bad) × WoE]`. Out-of-fold IV is used for screening to reduce the optimism created by supervised binning. Correlation, coefficient sign, Wald significance and VIF checks then reduce the candidate set.

The selected WoE variables enter a logistic regression. The score is scaled using:

- `Factor = PDO / ln(2)`
- `Offset = BaseScore − Factor × ln(BaseOdds)`
- `Score = Offset + Factor × ln(good:bad odds)`

The default configuration uses a 600-point base score, 50:1 base odds and PDO 20. The scorecard table allocates the intercept across variables so the unrounded points reproduce the model odds exactly.

### LGD, EAD proxy and expected loss

The recovery model is two-stage: logistic regression estimates the probability of a positive recovery; ridge regression estimates recovery severity conditional on a positive recovery. Expected recovery is the product of those stages. The exposure-factor model estimates a bounded CCF-style factor. Candidate models are compared with a mean baseline and a training-only cross-validation rule determines whether the more complex model is retained.

**EAD limitation:** the source vintage does not provide a clean, independently observed outstanding balance at the exact default date. The project therefore estimates exposure using `funded_amnt − total_rec_prncp` from the snapshot and scales it as an exposure-factor proxy. This is not an exact contractual EAD-at-default observation and should not be represented as one.

`LGD = 1 − net recovery rate`; `EL = PD × LGD × EAD proxy`. The back-test is a realised-loss **proxy**, not an accounting loss or regulatory ECL calculation. Recent charge-offs may have incomplete recoveries, which can bias observed LGD upwards.

## Outputs

A successful run writes:

- `results/summary.md` — readable run summary and principal metrics
- `results/metrics.json` — machine-readable model metrics and audit results
- `results/tables/*.csv` — scorecard points, IV, selection log, coefficients, calibration, stability, strategy and loss tables
- `results/figures/*.png` — ROC, KS, calibration, score distribution, stability and expected-loss plots (where LGD/EAD can be fitted)
- `models/pd_scorecard.joblib` — fitted PD scorecard
- `models/lgd_ead_models.joblib` — fitted LGD/EAD models when sufficient charge-off observations exist

Generated data, model binaries and run outputs are ignored by Git to keep the repository focused on source code and reproducible instructions.

## Limitations and next improvements

- One historical vintage and accepted loans only; reject inference is not performed.
- A random split is not a time-based out-of-time validation.
- The PD target is a short observation-window outcome, not lifetime PD.
- Recovery maturity is incomplete for newer charge-offs.
- EAD is snapshot-derived and should be replaced with a true default-date exposure field where available.
- No macroeconomic scenario conditioning, discounting, staging, ECL lifetime term structure, pricing or capital model is implemented.
- This is an educational portfolio analytics project, not a production credit decision system. Any real deployment requires independent validation, governance, fair-lending review, data permissions and regulatory/legal review.

## Resume-ready project description

Use the wording below only after you have run the project on the real dataset and can explain the design choices. Add measured results from your own `results/summary.md`; do not invent metrics.

**Credit Risk Scorecard & Expected Loss Modelling | Python, pandas, scikit-learn**

- Developed an end-to-end retail credit-risk pipeline covering data preparation, origination-time feature controls, a monotonic WoE/IV logistic PD scorecard, LGD recovery modelling and an exposure-factor EAD proxy.
- Built a reusable scikit-learn WoE transformer with out-of-fold IV screening, rare-category pooling and PDO-based score scaling; implemented feature-selection diagnostics and applicant-level reason codes.
- Evaluated discrimination, calibration and stability using AUROC, Gini, KS, bootstrap confidence intervals, cross-validation, PSI/CSI and a gradient-boosting benchmark; performed a post-origination leakage audit.
- Combined PD, LGD and EAD proxy into portfolio expected-loss estimates, segment analysis and PD/LGD stress scenarios; documented data-window, recovery-maturity and exposure-measurement limitations.

**Add after a real-data run:** `N = [development loans]`, `AUROC = [ ]`, `KS = [ ]`, `selected variables = [ ]`, `EL proxy = [ ]`. Label each figure with its sample and evaluation split.

## References

- Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley.
- Basel Committee on Banking Supervision. Internal Ratings-Based (IRB) approach and the PD × LGD × EAD framework.
- IFRS 9, *Financial Instruments* (expected credit loss requirements).
