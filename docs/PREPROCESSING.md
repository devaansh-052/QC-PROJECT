# Phase 3 preprocessing

## Run

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
.\.venv\Scripts\python.exe scripts/preprocess.py
.\.venv\Scripts\python.exe -m pytest -q -W error
```

`configs/preprocessing.yaml` defines the split fractions and optional sampling.
The command writes a unique `data/processed/preprocessing_<UTC>_<id>/` directory.
Raw data is read once and stays unchanged. Only the three eligible feature columns
are retained after validation; the full raw dataframe is released before fitting.
Each transformed split is saved sequentially to avoid holding three feature matrices.

## Frozen split protocol

Sort distinct observed `step` values. Assign the first floor(70% of distinct steps)
to training, the next steps through floor(85%) to validation, and the remainder to
test. Fractions describe time-step ranks, **not transaction counts**. Equal steps
never cross a boundary; original zero-based CSV row positions are retained. No
shuffling, label stratification or score-based boundary selection occurs.

Labels are used to check that every partition has both classes. If a partition is
single-class, processing fails with an explicit error; boundaries are never moved
automatically. Missing, nonfinite or noninteger steps also fail before fitting.
Time cannot be imputed for split assignment. The later preprocessing imputer can
handle missing step values at inference, but partition creation requires known time.

This protocol preserves the difficult fraud-only tail identified in EDA. Within
the complete test partition, both classes are required; individual test steps may
still contain only fraud. Report temporal drift alongside model metrics. Account
IDs can recur across periods; this split evaluates later transactions, not generalization
to entirely unseen accounts. Whole-file EDA exposure remains a research limitation.

## One authoritative sklearn pipeline

1. Select the fixed `paysim-primary-v1` allowlist: `step`, `type`, `amount`.
   Excluded balances, derivatives, identifiers, labels and existing flags cannot
   enter via extra dataframe columns.
2. Generate `log_amount = log1p(amount)` and replace the raw amount predictor.
   This deterministic definition includes zero amounts and is not fitted to data.
3. Fit numeric median imputation and standard scaling on original training rows.
   An entirely missing training numeric feature uses the imputer's zero fallback.
4. Fill missing transaction types with `MISSING`; fit a one-hot vocabulary on
   training only. Unknown future categories map to zeros without refitting.
5. Remove zero-variance training features. This is the Phase 3 feature-selection
   baseline; supervised selection/PCA comparisons belong to Phase 5. A category
   or numeric field constant during training can therefore be removed permanently.

Validation and test call `transform` only. Standard scaling does not constrain
future values to a fixed interval: values outside the training range remain
possible. Quantum-specific angle scaling is deferred. Numeric outputs use float64;
labels use int8. NPY arrays can be opened with `mmap_mode='r'` for later training.

## Imbalance handling

Default `sampling: none` preserves original training imbalance and leaves validation
and test untouched. The manifest records balanced class weights `N / (2 * N_class)`
computed exclusively from training labels. Phase 4 models can use these through
`class_weight` or equivalent sample-weight parameters. Do not apply the same
weight twice through both mechanisms. Unweighted loss remains an explicit control.

Optional `sampling: random_undersample` selects the same number from each training
class without replacement, with the configured seed, and saves the selected source
positions. It discards majority-class information and is not the default. Learned
preprocessing still fits on all original training rows before this optional sampling.
The manifest records both original and effective training weights; effective weights
are 1/1 after balanced undersampling. Validation/test populations are never sampled.
SMOTE is not implemented; it is optional and must not precede partitioning.

## Artifacts and verification

- `config.yaml`: exact phase settings and seed.
- `preprocessor.joblib`: fitted feature generation, imputers, scaler, encoder and selector.
- `{train,validation,test}_X.npy`: transformed feature matrices.
- `{train,validation,test}_y.npy`: aligned labels.
- `{train,validation,test}_positions.npy`: aligned zero-based original row positions.
- `manifest.json`: written last after success; source hash, split ranges/counts,
  feature names, weights, settings, dependency versions and limitations.

The run checks finite outputs and verifies a saved/reloaded pipeline against an
in-memory transform of validation rows. Only load joblib artifacts from trusted
project runs. Failed runs may retain partial files but do not have a completed
manifest. New runs never overwrite existing directories.

No classifier is trained in this phase. Class weights are prepared metadata;
their estimator-specific use will be implemented and tested in Phase 4.

## Verified full-data run

[Manifest](../data/processed/preprocessing_20260915T135023_4073b139cc42/manifest.json)
records the completed run with seven features (two numeric, five transaction types).

| Partition | Steps | Rows | Fraud |
| --- | --- | ---: | ---: |
| Train | 1–520 | 6,082,007 | 5,781 |
| Validation | 521–631 | 191,147 | 1,180 |
| Test | 632–743 | 89,466 | 1,252 |

The original time fractions yield unequal transaction counts because volume is
concentrated early. All rows, including the fraud-only late tail, remain included.
Training-derived weights are approximately 0.500476 (nonfraud) and 526.034164
(fraud). The no-sampling run's effective weights equal these original weights.
