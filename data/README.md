# Dataset setup

This project does not include or redistribute the LendingClub loan dataset. Obtain the historical 2015 loan file from the data provider and confirm that your use complies with the provider's terms.

The pipeline expects a CSV named `loan_data_2015.csv` with LendingClub-style columns including loan status, issue date, term, funded amount, borrower characteristics, principal repayments, recoveries and payment dates.

Place the file here:

```text
data/raw/loan_data_2015.csv
```

Or point the pipeline to another path:

```bash
# macOS / Linux
export CREDIT_RISK_DATA="/absolute/path/to/loan_data_2015.csv"

# Windows PowerShell
$env:CREDIT_RISK_DATA = "C:\path\to\loan_data_2015.csv"
```

The loader validates fields required to define the sample and targets. Secondary fields absent from a vintage are added as missing columns, so feature pipelines can handle them without an unrelated `KeyError`.

All CSV and ZIP files under `data/raw/` are ignored by Git. Do not commit borrower-level or otherwise restricted data. The synthetic generator in `scripts/make_synthetic_sample.py` creates artificial data for software tests only; it is not a substitute for the real file and its metrics must not be presented as real portfolio results.
