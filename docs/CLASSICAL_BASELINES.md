# Phase 4 classical baselines

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts/train_classical.py --prepared data/processed/preprocessing_20260915T135023_4073b139cc42
```

Select an explicit completed preprocessing run. The command reads its saved arrays
using memory mapping; the raw CSV is not loaded. It trains each model sequentially
and saves models, validation/test ranking scores, per-model metric JSON and a final
`results.json` under a unique `results/metrics/classical_<UTC>_<id>/` directory.
The final completion marker is absent if any fit/evaluation/reload check fails.

## Fixed initial model specifications

| Model | Specification |
| --- | --- |
| Logistic Regression | C=1, max_iter=1000, training-derived class weights |
| Random Forest | 100 trees, depth 12, min leaf 20, training-derived class weights; each tree bootstraps 20% of training rows |
| Linear SVM | LinearSVC, C=1, dual=False, max_iter=10000, training-derived class weights |
| XGBoost | 150 histogram trees, depth 5, learning rate 0.1; scale_pos_weight from training class ratio |

The exact settings are in `configs/classical.yaml` and copied into each run. Seed
42 is inherited from preprocessing; thread pools are limited to two. All four
models receive the same 6,082,007 training rows, seven features, 191,147 validation
rows and 89,466 test rows. Per-tree bootstrap sampling is a Random Forest setting,
not a different externally selected training partition.

A **linear SVM** makes fitting all training rows tractable. It is a valid classical
SVM baseline but does not replace the matched RBF-kernel SVM comparison. Phase 6
now includes that comparison on identical small subsets; see [QSVC.md](QSVC.md).

## Evaluation protocol

- Select each model's classification threshold to maximize validation F1; highest
  threshold wins exact ties. Apply that threshold unchanged to test scores.
- Calculate accuracy (secondary), precision, recall, F1, ROC-AUC, **average precision**
  as the explicitly named PR-AUC summary, MCC, confusion matrix, FPR and FNR.
  Average precision is not trapezoidal integration of the PR curve.
- Separately select thresholds maximizing validation recall subject to validation
  FPR <= 1% and <= 0.5%. Save both validation and test metrics at these thresholds.
  Test FPR is measured, not assumed to meet the same bound under temporal shift.
- A false-positive-free threshold can predict no positives; tied scores are not
  split artificially or interpolated into an unattainable classifier threshold.
- Report training duration separately from score-generation duration. Per-row
  prediction timing is a **batch-amortized** quantity, not single-request latency.
  Timings exclude preprocessing, API overhead, loading and serialization.
- Convergence warnings are errors, not silently accepted successful fits. Reload
  each persisted model and compare its scores with the in-memory model.

## Score interpretation and limitations

Linear SVM outputs margins. The other models output bounded model probabilities,
but class weighting and temporal prevalence differences make them **uncalibrated**;
they are not verified operational fraud probabilities. Calibration on a distinct
validation/calibration subset is needed before deployment or probability ensembles.

These are fixed initial baselines, not a hyperparameter search or a statistical
comparison. Report single-seed metrics as preliminary observations. No advantage,
quantum or otherwise, is inferred. Full-file EDA already exposed labels descriptively;
future claims must acknowledge that research limitation.

The automated runner isolation test flips only test labels and verifies that all
validation-selected thresholds are unchanged. This checks that test labels cannot
influence operating-point selection through the current training interface.

## Verified single-seed results

[Complete run](../results/metrics/classical_20260915T135718_d2739e452b90/results.json)
and [comparison CSV](../results/metrics/classical_20260915T135718_d2739e452b90/comparison.csv).
All metrics below are on the same 89,466 test rows, with validation-F1 thresholds.

| Model | Average precision | Precision | Recall | F1 | MCC | FPR | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.127552 | 0.191992 | 0.149361 | 0.168014 | 0.158975 | 0.008921 | 9.95 |
| Random Forest | 0.359654 | 0.639839 | 0.253994 | 0.363636 | 0.398210 | 0.002029 | 193.46 |
| Linear SVM | 0.127705 | 0.191207 | 0.149361 | 0.167713 | 0.158602 | 0.008967 | 14.49 |
| XGBoost | 0.325698 | 0.978448 | 0.181310 | 0.305930 | 0.418647 | 0.000057 | 39.31 |

The selected operating points have limited recall. High XGBoost precision does not
mean it catches most fraud. The metrics reflect the frozen validation threshold,
not the best threshold retrospectively available on test. Hyperparameters or
threshold objectives must not now be adjusted on these test observations.
Timings are one run on a shared workstation (tests briefly ran concurrently), so
they are diagnostic measurements, not isolated latency benchmarks.

Recompute the CSV without retraining:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --run results/metrics/classical_20260915T135718_d2739e452b90
```

`notebooks/03_classical_baselines.ipynb` reads completed results only.
