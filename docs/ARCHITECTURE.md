# Architecture

## Current foundation

`src/quantum_fraud` is an installable package. `utils` validates YAML configuration,
configures package logging, and seeds Python/NumPy. `scripts/check_environment.py`
exercises imports and configuration; `tests` holds phase acceptance tests.

Phase 1 adds `data/paysim.py` for nonmutating validation and file provenance, plus
`scripts/prepare_data.py` for CLI execution and unique JSON quality reports. It
does not clean or prepare model features despite the forward-compatible script name.

Phase 2 adds `data/eda.py` with aggregate summaries and headless figures, called by
`scripts/run_eda.py`. One full dataframe is released after aggregation; plotting
uses only small tables. Unique run directories preserve previous outputs, and a
completed report lists artifact locations and source/dependency provenance.
`notebooks/01_eda.ipynb` reads saved results without loading the source again.

Phase 3 implements `preprocessing/pipeline.py` (deterministic allowed features,
whole-step partitions and unfitted sklearn pipeline) and `preprocessing/prepare.py`
(training-only fitting and artifact export). `scripts/preprocess.py` selects the
configuration; original row positions link every saved feature/label back to its
CSV position and fingerprint. The completed manifest defines reusable splits.

Phase 4 adds classical model factories, sequential training on the shared saved
partitions and evaluation utilities with validation-only threshold selection.
`scripts/train_classical.py` writes saved estimators, scores and per-model metrics
to a unique run. A complete results manifest is produced after every model succeeds.

Phase 5 adds training-only MI/forest ranking and PCA projections with nine matched
linear SVM controls. The reducer bundles reference the same completed Phase 3/4
runs and avoid duplicating full transformed datasets. Unsupported requested
dimensions are recorded explicitly in the completed feature-run manifest.

Phase 6 adds `quantum/data.py` for reproducible small subsets and training-only
bounded angles, `quantum/kernel.py` for exact statevector kernels with measured
construction/overlap timings, and `quantum/experiment.py` for QSVC and matched RBF
SVM runs. The shared evaluation code freezes validation-selected thresholds.

Phase 7 adds `quantum/variational.py` and `scripts/train_vqc.py` for the small VQC
grid, optimization traces, model reload checks and convergence figures.
`scripts/summarize_quantum.py` independently checks saved scores, including
validation-selected low-FPR operating points. See the phase tracker for run gates.

Phase 8 adds `hybrid/models.py` for the differentiable TorchConnector model and
matched classical network, plus `hybrid/train.py` for deterministic paired training,
validation checkpoint selection, gradient checks and saved weights. The package
entry point initializes Torch before sklearn on Windows to avoid a reproduced
native-DLL conflict. Data and models remain local; no service has been deployed.

## Phases 9-15

- `hybrid/ensemble.py`: disjoint validation calibration/selection and matched
  XGBoost/quantum probability blends.
- `quantum/ablation.py`: controlled circuit factors and three training resamples.
- `quantum/noise.py`: frozen exact/shot/noisy inference with verified bit mapping.
- `quantum/hardware.py`: immutable QPY requests, explicit IBM backend submission
  and nonblocking result collection. Physical execution is pending credentials.
- `inference`: saved train-fitted preprocessing/model and synthetic replay client.
- `api/app.py`: strict FastAPI transaction contract and local SQLite history.
- `dashboard`: Streamlit live predictions and saved research views.
- `evaluation/registry.py`: immutable configuration/result snapshots, source hashes,
  score-based verification, repeated-run tables and publication figures.
- `experiments`: configuration snapshots; `results`: metrics, figures and logs;
  `paper/tables`: generated publication artifacts, isolated by registry ID.

The live request path is transaction -> allowed inputs -> saved preprocessing ->
saved XGBoost -> frozen threshold -> SQLite journal -> dashboard. Research views
read saved scores and labels without loading raw data or training models.

Raw data remains unchanged. Transformations fit only on training partitions. PaySim
step-based temporal boundaries and feature availability must be examined before
modeling. Post-transaction balances can leak unavailable decision-time information.
Registry entries record dataset identity, split, seed and configuration; historical
metadata gaps are explicit rather than fabricated.
Simulator latency cannot establish physical quantum speedup.
