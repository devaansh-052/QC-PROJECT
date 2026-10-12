# Dataset

PaySim is the initial dataset. See [source, schema, acquisition and use notes](../data/README.md).
The supplied CSV was inspected read-only on 2026-09-15 before configuration or code
changes. The loader in `src/quantum_fraud/data/paysim.py` validates input and returns
one dataframe with a JSON-compatible quality/provenance report.

## Verified local file

- Location: `data/PS_20174392719_1491204439457_log.csv` (kept in its supplied location).
- Size: 493,534,783 bytes.
- SHA-256: `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`.
- Dimensions: **6,362,620 rows x 11 columns**.
- Missing values: **0 in every column**; exact duplicate rows: **0**.
- Target: **`isFraud`**, binary; 0 = nonfraud, 1 = fraud.
- Target counts: **6,354,407 nonfraud**, **8,213 fraud**.
- Fraud prevalence: **0.1290820448%**. Accuracy alone will be misleading.
- Observed `step` range: **1 through 743**; these are observed bounds, not proof
  that every intervening step is present.

| Column | Inferred pandas dtype | Missing | Primary role |
| --- | --- | --- | --- |
| `step` | int64 | 0 | Temporal partition key; candidate time feature |
| `type` | object | 0 | Candidate categorical feature |
| `amount` | float64 | 0 | Candidate numeric feature |
| `nameOrig` | object | 0 | Identifier; excluded from primary features |
| `oldbalanceOrg` | float64 | 0 | Excluded: publisher leakage warning |
| `newbalanceOrig` | float64 | 0 | Excluded: publisher leakage warning |
| `nameDest` | object | 0 | Identifier; excluded from primary features |
| `oldbalanceDest` | float64 | 0 | Excluded: publisher leakage warning |
| `newbalanceDest` | float64 | 0 | Excluded: publisher leakage warning |
| `isFraud` | int64 | 0 | Target only |
| `isFlaggedFraud` | int64 | 0 | Existing detection flag; excluded from primary features |

Local download date and publisher version have not been supplied. The file hash
identifies our exact input; a familiar filename does not independently verify its
publisher version. The publisher page currently lists CC BY-SA 4.0; retain dataset
attribution and check its terms when sharing derivatives.

## Balance-field leakage investigation and decision

The [publisher's dataset documentation](https://www.kaggle.com/datasets/ealaxi/paysim1/metadata)
states that detected fraudulent transactions are cancelled and instructs users to
exclude **all four balance fields** for fraud detection. This warning includes the
two fields named `oldbalance...`, not just the post-transaction fields. The page
also distinguishes `isFraud` from the pre-existing rule flag `isFlaggedFraud` and
notes unavailable destination balance information for merchants.

**Primary policy: `paysim-primary-v1`.** Exclude `oldbalanceOrg`, `newbalanceOrig`,
`oldbalanceDest`, `newbalanceDest` and every feature computed from them (differences,
ratios, balance errors, zero-balance indicators). Also exclude `isFlaggedFraud`
and raw account identifiers; `isFraud` is used only as the supervised target.
Candidate raw predictors are `step`, `type`, and `amount`. The policy is available
in `src/quantum_fraud/data/policy.py` and recorded in validation reports. Future
preprocessing must select the allowlist explicitly rather than drop only the label.

Reasoning: a model could learn consequences of cancellation/detection that would
not be available when screening an incoming payment. Good predictive performance
would not prove those inputs are valid at decision time. Column names alone do not
establish trustworthy pre-decision provenance. We therefore follow the publisher's
warning before training, rather than choose features using whole-dataset label
correlations. No model or statistical leakage experiment was run in Phase 1.

All 11 source columns remain available for schema checks and explicitly labeled
diagnostic EDA. Any later balance-inclusive retrospective sensitivity analysis must
be separately identified as leakage-prone and excluded from primary real-time
performance claims. Reconsideration requires documented evidence of feature
availability before the decision, not merely better scores.

This leaves only three candidate raw predictors. Later 4/6/8-feature or qubit
experiments must document the valid encoded representation and its dimensional
limits; do not add excluded fields or invent independent features to fill a circuit.

## Raw-data and memory handling

The original CSV is neither rewritten, moved nor copied. Each validation process
reads one full dataframe; temporary per-column arrays/duplicate masks are needed
for checks but no second full dataframe is created. Fingerprinting streams bytes
without loading a second file copy. The inspection process ended before the final
CLI run began. The CLI releases the dataframe before saving its small JSON report.

Reports go to `results/metrics/`; intermediate and processed data, when implemented,
belong in `data/interim/` and `data/processed/`. The CLI rejects report destinations
inside `data/raw/`. CSVs anywhere below `data/` are ignored by Git.

## Policy for later phases

- Inspect `step` coverage and use whole-step temporal partitions where feasible.
- Enforce the primary feature policy above; separate any retrospective sensitivity
  analysis from the pre-decision research question.
- Investigate duplicates across partitions; never silently deduplicate raw input.
- Fit imputers, encoders, scalers and feature selectors on training data only.
- Record data hash and source version; synthetic-source performance does not
  establish operational banking performance.

The Phase 3 implementation is now available; see [PREPROCESSING.md](PREPROCESSING.md)
for the verified split protocol and fitted transformations.

Phase 2 EDA is complete; [EDA.md](EDA.md) records descriptive findings. All integer
steps 1–743 are present, but 320 steps have only fraud labels. Whole-file EDA does
not replace training-only fitting or validation-only tuning in subsequent phases.
