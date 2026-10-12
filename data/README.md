# Dataset setup

## Source and acquisition

Use **PaySim: Synthetic Financial Datasets for Fraud Detection** from the
[publisher's Kaggle page](https://www.kaggle.com/datasets/ealaxi/paysim1).
The [author's simulator repository](https://github.com/EdgarLopezPhD/PaySim)
links this dataset and provides the paper citation. PaySim is published synthetic
mobile-money data; it is not a collection of actual individual bank transactions.

Download and extract the CSV using the publisher's site (sign-in may be required).
Place it at `data/raw/paysim.csv`, or retain its downloaded name and pass `--data`.
The commonly distributed filename is
`PS_20174392719_1491204439457_log.csv`; the loader accepts any explicit `.csv` path.
Archives are not accepted. No credentials are stored or downloads performed by the loader.

**Current workspace:** the supplied file is
`data/PS_20174392719_1491204439457_log.csv`. `configs/base.yaml` now points to it,
so the default command below works without moving or duplicating the CSV. Its
verified dimensions, target distribution and feature policy are in
[docs/DATASET.md](../docs/DATASET.md).

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py
# Or use an existing file without copying or renaming:
.\.venv\Scripts\python.exe scripts/prepare_data.py --data 'C:\datasets\PS_20174392719_1491204439457_log.csv'
```

## Expected schema

| Column | Role |
| --- | --- |
| `step` | Positive integer simulation time step; inspect before temporal splitting |
| `type` | CASH_IN, CASH_OUT, DEBIT, PAYMENT or TRANSFER |
| `amount` | Nonnegative transaction amount |
| `nameOrig`, `nameDest` | Origin/destination identifiers; not automatically model features |
| `oldbalanceOrg`, `newbalanceOrig` | Origin balance before/after transaction (spelling is intentional) |
| `oldbalanceDest`, `newbalanceDest` | Destination balance before/after transaction |
| `isFraud` | Target, nonmissing 0 or 1 |
| `isFlaggedFraud` | Existing flag, nonmissing 0 or 1; not the target |

Loader reports shape, inferred types, missing values, duplicates, label counts and
fraud percentage. Invalid schema/labels/categories/numeric values fail with a useful
message. Missing feature values and duplicates produce warnings and remain intact.
Single-class inputs are flagged as unsuitable for classification comparisons.
No training, cleaning, sampling, resampling or splitting occurs here.

Reports are uniquely named JSON files in `results/metrics/` with full-file SHA-256,
source path, byte count and validation UTC timestamp. Invalid data never produces a
successful report. Full-file validation loads the frame into RAM; large PaySim
files require adequate memory. A memory error does not trigger silent sampling.

## Use and provenance

The publisher metadata retrieved on 2026-09-15 lists **CC BY-SA 4.0**. Retain its
attribution, version, download date and license information with research notes;
the supplied file's download date/version are currently unknown. The simulator
code's license must not be assumed to govern the separately distributed data.

The publisher warns against all four balance fields because detected fraud is
cancelled. Our primary feature policy excludes them and balance-derived features,
as well as identifiers and `isFlaggedFraud`. `isFraud` is the target. Keep all
columns in raw validation; see [the decision](../docs/DATASET.md) before modeling.

Cite: E. A. Lopez-Rojas, A. Elmir, S. Axelsson, *PaySim: A financial mobile money
simulator for fraud detection*, EMSS, 2016 (citation supplied by the author repository).

Raw/interim/processed data are ignored by Git. Do not commit large CSVs, account
credentials or tokens. Handwritten rows in tests exist solely for software checks;
they must not be presented as dataset observations or experimental evidence.
