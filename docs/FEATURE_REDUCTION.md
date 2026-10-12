# Phase 5 feature reduction

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts/reduce_features.py --prepared data/processed/preprocessing_20260915T135023_4073b139cc42 --classical-run results/metrics/classical_20260915T135718_d2739e452b90
```

`configs/features.yaml` fixes dimensions and the mutual-information estimation cap.
The classical run must point to exactly the same completed preprocessing manifest.
The source CSV is not loaded: input matrices are memory-mapped from Phase 3.

## Methods and training boundaries

1. **Mutual information:** estimate scores on a deterministic, class-proportional
   sample of up to 100,000 original training-array rows. This bounds nearest-neighbor
   estimation cost. The two numeric fields are treated as continuous; one-hot
   columns are explicitly discrete. The random seed is fixed, scores are saved,
   and sampled training-array positions are saved for reproduction. Original CSV
   positions can be recovered through Phase 3 `train_positions.npy`.
2. **Random Forest importance:** reuse importances from the completed full-training
   Phase 4 forest. Its hyperparameters were fixed before test evaluation. Importances
   can favor certain feature types; they are not a causal explanation.
3. **PCA:** fit one unwhitened full-SVD decomposition exclusively on all training
   rows. Smaller projections use its leading components. Persist the fitted mean,
   components, explained variance ratios and projections. PCA captures variance,
   which need not correspond to fraud-discriminative information.

Score-based selectors rank descending; original feature order breaks ties. No
validation or test values fit any reducer. Existing numeric scaling was already
fitted on training in Phase 3. One-hot columns remain unscaled as previously defined.
An explicit worker cap defaults to two, avoiding unreliable physical-core discovery
on Windows systems without the legacy utility.

## Matched classical controls

For every method and feasible dimension (2, 4, 6), train a LinearSVC with C=1,
dual=False, max_iter=10000, seed 42 and the same training-derived class weights as
the Phase 4 linear SVM. Every control receives **all 6,082,007 training rows**;
the MI estimation sample is not its classifier training sample. Validation and test
use exactly the original 191,147 and 89,466 rows.

Thresholds maximize validation F1, with the same tie rule as Phase 4. Test results
are computed at frozen thresholds. Save average precision, ROC-AUC, precision,
recall, F1, MCC, accuracy, FPR/FNR and confusion matrices. Timing separates training
matrix projection from classifier fit. No projection is selected as the winner
using test results; this is a prespecified exploratory grid with a single seed.

These controls provide reduced-representation classical baselines. A later QSVC
comparison still requires matched sample sizes, encodings and an appropriate
kernel SVM; the full-data linear controls alone are not that quantum comparison.

## Dimensionality limitation

The approved input has **seven encoded columns**, with a linear dependency among
the five one-hot columns (their sum is one). A genuine reduced eight-feature dataset
cannot be obtained from this representation. The requested 8-feature case is
recorded explicitly in `unsupported_dimensions`; no duplicate, padded, random or
balance-derived feature is introduced. The feasible experiment grid is 2/4/6.
Any future change to representation requires a documented new preprocessing run
and matched controls. Eight qubits would not automatically mean eight independent
classical features. The frozen primary leakage policy remains intact.

## Artifacts

Each run uses a new `results/metrics/features_<UTC>_<id>/` directory containing:

- Exact `config.yaml`, score vectors, source/parent run references and seed.
- MI estimation positions and full train-fitted PCA artifact.
- Nine reducer/control joblib bundles and validation/test score arrays.
- Completed `results.json` only after every supported case and reload check passes.

Dense transformed training matrices are transient rather than duplicated on disk
for every representation. Later experiments can reload a reducer and apply it to
the same shared arrays. Inspect `unsupported_dimensions` and limitations before
interpreting a completed run. Failed attempts may leave incomplete directories;
they lack the completed results manifest and are not reported as experiments.

No quantum models, simulator runs or hardware results are included in this phase.

## Verified results

[Complete run](../results/metrics/features_20260915T140754_40734f02b041/results.json),
[comparison CSV](../results/metrics/features_20260915T140754_40734f02b041/comparison.csv),
and [comparison figure](../results/figures/features_20260915T140754_40734f02b041/feature_comparison.png).

| Method | Dimensions | Test average precision | Test F1 |
| --- | ---: | ---: | ---: |
| Mutual information | 2 | 0.217287 | 0.291476 |
| Mutual information | 4 | 0.123276 | 0.164899 |
| Mutual information | 6 | 0.127706 | 0.167713 |
| Forest importance | 2 | 0.117249 | 0.159521 |
| Forest importance | 4 | 0.123276 | 0.164899 |
| Forest importance | 6 | 0.127706 | 0.167713 |
| PCA | 2 | 0.235937 | 0.315135 |
| PCA | 4 | 0.192205 | 0.275891 |
| PCA | 6 | 0.127705 | 0.167713 |

These are exploratory grid results, not a selected winning representation or
statistical superiority finding. In particular, do not now choose PCA-2 based on
its test score. Model selection must use validation or a prespecified protocol.

MI ranks step and log-amount first. Forest ranks log-amount and TRANSFER first.
MI's 100,000-row estimation subset includes only **95 fraud observations**; ranking
stability across training samples/seeds remains untested. PCA cumulative explained
variance is about 80.963% for two components, 97.182% for four and 100% for six.
The one-hot dependency explains the redundant seventh direction. Variance retention
is not equivalent to retention of fraud information.

Recompute metrics and refresh derived exports without retraining:

```powershell
.\.venv\Scripts\python.exe scripts/summarize_features.py --run results/metrics/features_20260915T140754_40734f02b041
```

`notebooks/04_feature_selection.ipynb` displays saved metadata only. All 85 tests
pass, including a complete small grid and persisted PCA/column projection checks.
