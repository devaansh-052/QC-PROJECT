# Phase 2: PaySim exploratory data analysis

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
.\.venv\Scripts\python.exe scripts/run_eda.py
.\.venv\Scripts\python.exe -m pytest -q -W error
```

The command validates and loads the CSV once, computes full-file aggregates,
releases the dataframe, and saves eight CSV tables plus six figures in PNG and PDF.
Each invocation uses a new `eda_<UTC>_<id>` subdirectory in `results/metrics/` and
`results/figures/`. `report.json` is written only after all outputs succeed and
contains the source fingerprint, validation report, policy, package versions,
seed, scope and artifact list. Runs are not overwritten. No sampling is used;
the seed is recorded for provenance but this EDA has no stochastic computation.

`notebooks/01_eda.ipynb` reads saved aggregate tables and figures. It never loads
the raw dataset. Select the project Python environment in your notebook-capable
IDE; the command-line workflow needs no Jupyter installation. For citation, set
the notebook's report path explicitly rather than relying on the latest run.

## Dataset and quality

Verified run: [report and artifact manifest](../results/metrics/eda_20260915T070857_9595acf31bfa/report.json).
Figures: [class balance](../results/figures/eda_20260915T070857_9595acf31bfa/class_balance.png),
[types](../results/figures/eda_20260915T070857_9595acf31bfa/transaction_types.png),
[amounts](../results/figures/eda_20260915T070857_9595acf31bfa/amounts.png),
[time](../results/figures/eda_20260915T070857_9595acf31bfa/time.png),
[balance diagnostics](../results/figures/eda_20260915T070857_9595acf31bfa/balance_diagnostics.png),
[correlations](../results/figures/eda_20260915T070857_9595acf31bfa/primary_numeric_correlations.png).

The input is `data/PS_20174392719_1491204439457_log.csv`, with SHA-256
`16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`.
It has 6,362,620 rows and 11 columns, no missing values and no exact duplicate rows.
See [DATASET.md](DATASET.md) for every column, dtype and leakage policy.
`isFraud` is the target: 8,213 positive and 6,354,407 negative rows; prevalence is
0.129082%. A constant nonfraud prediction would have about 99.871% accuracy, which
shows why accuracy cannot be the primary fraud metric. This arithmetic is not a
trained-model result.

## Findings from the complete file

### Transaction types

| Type | Transactions | Fraud | Fraud rate within type |
| --- | ---: | ---: | ---: |
| CASH_IN | 1,399,284 | 0 | 0% |
| CASH_OUT | 2,237,500 | 4,116 | 0.183955% |
| DEBIT | 41,432 | 0 | 0% |
| PAYMENT | 2,151,495 | 0 | 0% |
| TRANSFER | 532,909 | 4,097 | 0.768799% |

All observed fraud labels occur in TRANSFER or CASH_OUT. This is a property of
this synthetic file, not a guarantee that other transaction types are safe. Keep
the full transaction population for the primary task; do not silently remove easy
negative classes or compare models on different populations.

### Amounts

Median amount is 74,684.72 for nonfraud and 441,423.44 for fraud, in dataset-local
currency units. Amounts are strongly right-skewed: the maximum nonfraud amount is
92,445,516.64, while the maximum fraud amount is 10,000,000. Fraud includes zero
amounts; zero is therefore a valid observed value, not automatically a missing code.

Fixed descriptive amount bands show 2,419 fraud rows among 124,976 transactions
from 1 million up to (excluding) 10 million (1.935572%). The 10-million-and-above
band contains 287 fraud rows among 5,650 transactions (5.079646%). Amount alone
does not determine the label. These full-file bins are visualization choices,
not learned preprocessing bins or proposed decision thresholds.

### Time and suspicious simulator patterns

All 743 integer steps from 1 through 743 are represented. Step volume ranges from
2 to 51,352 transactions. **320 steps contain only fraud-labeled transactions**;
the last step containing any nonfraud is 718. Steps 719–743 contain 296 rows, all
fraud. Thus a late temporal slice can be single-class even though the full file
is highly imbalanced toward nonfraud.

For a descriptive division visible in the volume plot, steps through 400 have
fraud prevalence 0.077363%, versus 0.649073% after 400 (575,590 rows, 3,736 fraud).
This division is not a selected train/test cutoff. The change and sparse fraud-only
steps suggest simulator-specific time structure; this is an inference, not a
demonstrated cause. Phase 3 must use whole-step boundaries, inspect class coverage,
report partition prevalence and define how to handle single-class evaluation.
Do not shift boundaries to optimize model scores or hide the difficult tail.

### Balance diagnostics and existing flag

`newbalanceOrig` is zero in 8,053 of 8,213 fraud rows (98.051869%) and 3,601,513 of
6,354,407 nonfraud rows (56.677405%). These differences do not establish legitimate
predictive features: cancellation artifacts and transaction-type differences can
contribute. The Phase 1 exclusion of **all four balance columns and all derivatives**
remains unchanged. Zero balances should not automatically be interpreted as known
account states, especially where the dataset lacks merchant balance information.

The existing flag is set for only 16 rows, all labeled fraud; 8,197 fraud rows are
unflagged. This is descriptive inspection of a pre-existing flag, not evaluation
of a trained classifier. `isFlaggedFraud` remains excluded from primary inputs.

### Correlations

Pearson correlation is approximately 0.076688 between amount and target,
0.031578 between step and target, and 0.022373 between step and amount. These small
linear associations do not rule out nonlinear structure. Correlations do not
justify feature inclusion or establish causality. No arbitrary numeric codes are
assigned to transaction categories for correlation analysis.

## Figure guide

| Filename stem | Question and interpretation |
| --- | --- |
| `class_balance` | How imbalanced is the target? Counts use a labeled log scale. |
| `transaction_types` | Which types have high volume and fraud rates? Denominators are type totals. |
| `amounts` | How do class-conditional amount distributions differ? Histogram bins use ln(1+amount), normalized separately per class; amount-band bars use within-band prevalence. |
| `time` | How do volume and fraud prevalence change by step? Read the rate panel alongside volume and exact table counts; sparse steps can have 100% rates. |
| `balance_diagnostics` | How do zero balances vary by class? Diagnostic only; excluded fields. |
| `primary_numeric_correlations` | What linear relationships appear among eligible numeric fields and the target? Undefined values are labeled. |

## Limits and handoff to Phase 3

- EDA uses labels across the entire file. Future splits cannot be claimed to be
  completely untouched during research exploration. Freeze modeling choices and
  keep transforms fitted on training only, tuning on validation only; an independent
  dataset or later simulation is needed for a stronger external confirmation.
- No cleaning, imputation, encoding, scaling, selection, resampling or modeling
  was performed. Raw bytes remain unchanged and results stay outside `data/`.
- Missing-feature inputs are supported: summaries use observed values, group tables
  retain missing groups, and balance percentages use nonmissing denominators.
  Constant-column correlations are undefined rather than silently set to zero.
- The primary raw feature candidates remain `step`, `type`, `amount`. Later quantum
  dimensionality targets must respect the eligible encoded representation.
- These results describe published synthetic data, not production fraud behavior.
