# Progress and phase gates

Legend: `[x]` verified; `[~]` current; `[ ]` pending. A phase advances only after
implementation, execution, tests, output inspection and documentation succeed.

- [x] Phase 0: project foundation
- [x] Phase 1: dataset pipeline
- [x] Phase 2: exploratory data analysis
- [x] Phase 3: preprocessing pipeline
- [x] Phase 4: classical baselines
- [x] Phase 5: feature reduction (2/4/6; 8 unsupported by approved representation)
- [x] Phase 6: QSVC / quantum kernel
- [x] Phase 7: VQC
- [x] Phase 8: hybrid neural model
- [x] Phase 9: hybrid ensemble
- [x] Phase 10: quantum ablations
- [x] Phase 11: noisy simulation
- [x] Phase 12: IBM hardware integration and offline verification
  - [ ] Physical hardware execution: API key and instance not configured
- [x] Phase 13: real-time transaction simulation / API
- [x] Phase 14: dashboard
- [x] Phase 15: research experiment management

## Initial inspection

Workspace empty; no Git repository, data or local instructions found. Python 3.11.9
available. Foundation was verified in a dedicated `.venv`.

## Phase 0 gate — passed (2026-09-15)

- Created src package, planned component directories, configs, scripts, tests,
  dependency pins, environment template, ignore rules and documentation foundation.
- Isolated Python 3.11.9 environment; editable installation succeeded.
- `python -m pytest -q --junitxml=results/experiment_logs/phase0-tests.xml`:
  **17 passed in 0.96s**, exit 0.
- `python scripts/check_environment.py`: exit 0; version 0.1.0, seed 42,
  deterministic draw 0.7739560485559633.
- `python -m pip check`: exit 0, no broken requirements.
- All commands used `.venv/Scripts/python.exe`. Full foundation dependency set is
  pinned in `requirements.txt`. Quantum compatibility remains unverified and
  quantum packages are not installed. No empirical results exist.
- Phase 1 may now begin; data is not present. Git is not initialized yet.

## Phase 1 software verification — passed; dataset gate pending (2026-09-15)

- Implemented PaySim loading, required-column/type/value/label validation, shape,
  missingness, duplicate counts, class distribution, provenance and useful errors.
- Added `scripts/prepare_data.py` with unique JSON reports and documented dataset
  acquisition, exact schema, use considerations and memory limitations.
- Added pandas 2.2.2 and pinned its dependency set; editable reinstall succeeded.
- Initial suite: 50 passed with one fixture dtype warning. Fixed the fixture cast.
- Final `.venv/Scripts/python.exe -m pytest -q -W error
  --junitxml=results/experiment_logs/phase1-tests.xml`: **50 passed in 4.17s**,
  exit 0, no warnings. Includes all 17 foundation tests and 33 data tests.
- CLI integration tests execute the command in subprocesses, verify successful
  JSON content, confirm reports are not overwritten and check missing-file exit 2.
- `.venv/Scripts/python.exe -m pip check`: no broken requirements, exit 0.
- Running `scripts/prepare_data.py` against the configured dataset path produced
  the expected actionable missing-CSV error. This verifies error handling only;
  it is **not** a successful published-dataset validation run.
- No dataset statistics or model results were generated. Unit-test fixtures are
  handwritten schema cases, not research data. No later phase was implemented.

### Original pending gate (resolved below)

Provide the published PaySim CSV at `data/raw/paysim.csv` or supply its existing
local path. Record publisher version/download date/license details, run
`scripts/prepare_data.py --data PATH`, inspect its full-file quality report and
resolve data issues. Rerun the suite and mark Phase 1 complete only then.
The next phase after this gate is **Phase 2: reproducible EDA**.

## Phase 1 full-dataset gate — passed (2026-09-15)

- Located the supplied `data/PS_20174392719_1491204439457_log.csv` and inspected it
  with the existing loader before modifying project code/configuration. An initial
  inline command failed due to shell quoting before reading data; reran successfully
  using Python source on stdin. No dataset or code error was skipped.
- Verified 6,362,620 rows, 11 columns, inferred datatypes, zero missing values,
  zero exact duplicates, 6,354,407 nonfraud and 8,213 fraud labels (0.1290820448%).
  Confirmed `isFraud` is the target; `isFlaggedFraud` is a separate existing flag.
- Updated the default CSV path without moving/copying/rewriting the source. Added
  CSV ignore coverage for its supplied location. Each validation run held only
  one full dataframe; inspection ended before the final CLI run.
- Investigated the publisher's cancellation warning and documented the decision
  in `docs/DATASET.md`. Added `paysim-primary-v1` metadata: exclude all four balance
  fields and their derivatives, existing flag and identifiers from primary inputs.
  This is eligibility policy, not an implemented training/preprocessing pipeline.
- `.venv/Scripts/python.exe scripts/prepare_data.py`: exit 0 on the complete file.
  Verified saved report:
  [dataset validation](../results/metrics/dataset_validation_20260915T063408_5c151783f6724a66afedadaf7f0897b1.json).
- Pre-change inspection and final validation SHA-256 both equal
  `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`.
  Source byte count and modification time were also unchanged.
- `.venv/Scripts/python.exe -m pytest -q -W error
  --junitxml=results/experiment_logs/phase1-dataset-tests.xml`:
  **53 passed in 4.86s**, exit 0. Includes raw-file preservation, single CSV read,
  forbidden raw report output, feature-policy checks and previous-phase regression tests.
- `.venv/Scripts/python.exe -m pip check`: exit 0, no broken requirements.
- No intermediate/processed dataset was created. Reports are outside `data/raw/`.
  Publisher metadata lists CC BY-SA 4.0; exact download date/publisher version of
  the supplied file remain unknown and are documented as provenance limitations.
- Phase 1 is complete. **Next: Phase 2 EDA**, not started in this Phase 1 request.
  No classical or quantum models were implemented or trained.

## Phase 2 EDA gate — passed (2026-09-15)

- Implemented reusable aggregate EDA in `src/quantum_fraud/data/eda.py`, CLI
  `scripts/run_eda.py`, a thin `notebooks/01_eda.ipynb`, and `docs/EDA.md`.
- Added Matplotlib 3.10.7 with fully pinned dependencies; editable installation
  and `pip check` succeeded. No quantum dependencies were installed.
- Full-file execution succeeded on all 6,362,620 rows. Final run:
  [report](../results/metrics/eda_20260915T070857_9595acf31bfa/report.json).
  The earlier successful run `eda_20260915T070507_405f86ff059b` is preserved.
- Saved eight aggregate CSV tables and six figures in both PNG and PDF, under
  unique metrics/figures run directories. All six initial figures were visually
  inspected; final PNGs/PDFs were verified and aggregate totals reconciled.
- Tests: `.venv/Scripts/python.exe -m pytest -q -W error
  --junitxml=results/experiment_logs/phase2-tests.xml`: **60 passed in 14.66s**,
  exit 0. Covers rate denominators, amount-bin boundaries, histogram coverage,
  missing groups, single-class and all-missing amounts, constant correlations,
  nonmutation, artifact creation, output preservation and prior-phase regression.
- All three notebook code cells executed against saved outputs headlessly with
  figure display suppressed. No notebook cell reloads the full CSV.
- Final source SHA-256 still matches the Phase 1 fingerprint; raw data unchanged.
- Documented concentration of fraud in TRANSFER/CASH_OUT, heavy amount tails,
  balance diagnostics, existing flag coverage and Pearson associations.
- Temporal finding: 320 fraud-only steps; steps 719–743 contain 296 fraud rows and
  no nonfraud. Future split design must validate class coverage and report shift.
- Full-file label exploration is explicitly disclosed as a limitation for claims
  of an untouched research holdout. No transformations or models were fitted.
- **Next phase: Phase 3 preprocessing**, not started. Phase 2 gate is complete.

## Phase 3 preprocessing gate — passed (2026-09-15)

- Added whole-step temporal splits, strict phase configuration, allowlisted log-amount
  features, training-only median/scaling/one-hot fitting and variance selection.
- Default retains all rows; training-derived class weights and optional deterministic
  training-only undersampling are implemented. Validation/test are never resampled.
- Full-data CLI succeeded:
  [manifest](../data/processed/preprocessing_20260915T135023_4073b139cc42/manifest.json).
- Train: steps 1–520, 6,082,007 rows (5,781 fraud). Validation: 521–631,
  191,147 rows (1,180 fraud). Test: 632–743, 89,466 rows (1,252 fraud).
- All three partitions have seven finite output features; original positional row
  indices cover the complete file without overlap. Saved pipeline reload verified.
- Full raw SHA-256 unchanged. Artifacts are outside raw data and previous runs persist.
- `python -m pytest -q -W error --junitxml=results/experiment_logs/phase3-tests.xml`:
  **75 passed in 21.96s**, exit 0, using `.venv/Scripts/python.exe`.
- `pip check`: no broken requirements. Added scikit-learn 1.4.2, SciPy 1.13.0,
  joblib 1.5.2 and threadpoolctl 3.6.0. No quantum package installed.
- Documented protocol and limitations in `docs/PREPROCESSING.md`. Phase 4 may begin.

## Phase 4 classical gate — passed (2026-09-15)

- Trained Logistic Regression, Random Forest, linear SVM and XGBoost on identical
  complete training features/rows. Thresholds use validation only; test is evaluation-only.
- Full run completed successfully:
  [results](../results/metrics/classical_20260915T135718_d2739e452b90/results.json),
  [comparison](../results/metrics/classical_20260915T135718_d2739e452b90/comparison.csv).
- All saved estimators were reloaded and scores checked. `scripts/evaluate.py`
  independently recomputed every stored test metric from saved scores and labels.
- Initial sandbox test failed on Windows Random Forest worker-pipe permission.
  Repeated with approved escalation; no model implementation failure was hidden.
- Final full suite: **80 passed in 23.23s**, warnings treated as errors, exit 0;
  `results/experiment_logs/phase4-tests.xml`. Includes a test-label flip isolation
  check for all validation-selected thresholds. `pip check` also passed.
- Executed `notebooks/03_classical_baselines.ipynb` code on saved results.
- Test average precision: LR 0.127552, RF 0.359654, linear SVM 0.127705,
  XGBoost 0.325698. These are single-seed observations, not statistical claims.
- Saved all required metrics, FPR-constrained validation operating points with
  measured test FPR, scores, durations and configurations. Uncalibrated probability
  outputs and batch-amortized timings are labeled. No quantum work performed.
- Phase 5 feature reduction may now begin.

## Phase 5 feature-reduction gate — passed for feasible grid (2026-09-15)

- Implemented mutual-information ranking, full-training forest importance and
  train-fitted PCA; saved 2/4/6-dimensional reducers and nine matched linear-SVM
  controls trained on exactly the same complete Phase 3 training partition.
- The 8-feature request is explicitly unsupported by the seven eligible input
  columns. No excluded or artificial feature was added. Phase 6 must retain this
  limitation or document a new justified representation and matched controls.
- Full run:
  [results](../results/metrics/features_20260915T140754_40734f02b041/results.json),
  [comparison](../results/metrics/features_20260915T140754_40734f02b041/comparison.csv),
  [figure](../results/figures/features_20260915T140754_40734f02b041/feature_comparison.png).
- MI used a saved deterministic training subset of 100,000 rows, including 95 fraud.
  All classifiers still used all training rows. PCA fitted all training rows.
- Initial test exposed unavailable Windows physical-core discovery; explicit
  worker cap fixed it. Initial full run was denied Random Forest worker pipes;
  approved escalation resolved it. The failed attempt has no completed manifest.
- Final full suite: **85 passed in 20.29s**, warnings treated as errors, exit 0;
  `results/experiment_logs/phase5-tests.xml`. All prior phase tests included.
- `scripts/summarize_features.py` independently reproduced all nine saved control
  test metric records and exported comparison CSV plus PNG/PDF figure. Figure
  visually inspected; saved reducer/control reload checks passed during the run.
- Feature notebook code executed using saved aggregates. `pip check` passed.
- Documentation updated in `docs/FEATURE_REDUCTION.md`; single-seed estimation
  limits and no test-based selection are explicit. No quantum work performed.
- **Stopped before Phase 6** under the user's instruction not to proceed to quantum
  models yet. Phases 0–5 are verified within the documented dimension constraint.

## Phase 6 gate — passed (2026-09-15)

- User subsequently authorized completion through Phase 8, with a hard stop there.
- Checked current official package metadata; installed Qiskit 2.5.2, QML 0.9.1,
  Aer 0.17.2, NumPy 2.2.6 and SciPy 1.15.3 with pinned transitive dependencies.
  All 85 prior tests passed migration; historical model probes still match saved scores.
- Implemented twelve exact statevector QSVCs (2/4/6 qubits, Z/ZZ, 1/2 repetitions)
  and three matched RBF SVMs. Each uses identical 128 training / 2,048 validation /
  2,048 test rows; positive counts are 64 / 13 / 29. No physical hardware.
- Initial clipped-angle run exposed clipping on every validation/test row and is
  retained as a diagnostic. Switched to train-standardized tanh encoding before
  proceeding and reran all fifteen models. This adaptive correction is disclosed.
- Authoritative [results](../results/metrics/qsvc_20260915T143628_87486f0e4a58/results.json)
  and [comparison](../results/metrics/qsvc_20260915T143628_87486f0e4a58/comparison.csv).
  Saved all sample positions, transforms, scores, models and circuit/kernel timings.
- Near-saturation rows: none at 2/4 dimensions; at 6 dimensions, 1 train, 15 validation,
  20 test. Recorded rather than silently discarded. Thresholds use validation only.
- Final full suite: **97 passed in 25.09s**, warnings treated as errors, exit 0;
  `results/experiment_logs/phase6-tests.xml`. Current serialization replaced a caught
  deprecated API. QSVC/Aer smoke, PSD, dimensionality, sampling and reload checks pass.
- Independently recomputed all validation/test metric records from saved scores;
  QSVC notebook code executed. `pip check` passed. Phase 7 may now begin.

## Phase 7 gate — passed (2026-09-15)

- Ran all 18 exact-simulator VQCs: 2/4/6 qubits, ansatz repetitions 1/2/3,
  linear/circular entanglement. Reused the exact Phase 6 samples and angles.
- [Results](../results/metrics/vqc_20260915T144414/results.json),
  [comparison](../results/metrics/vqc_20260915T144414/comparison.csv), and
  [convergence figure](../results/metrics/vqc_20260915T144414/convergence.png).
- Every model trained, produced finite scores, serialized and reproduced reload
  probe predictions. Saved objective traces, circuit metadata, parameter counts,
  optimizer evaluations and fit/inference durations. All 18 exhausted the fixed
  60-evaluation budget; completion does not imply optimizer convergence.
- Final full suite: **117 passed in 25.12s**, warnings treated as errors, exit 0;
  `results/experiment_logs/phase7-tests.xml`. `pip check` passed.
- Independently recomputed all 18 validation/test metric records and both low-FPR
  operating points from saved probabilities. Convergence PNG visually inspected;
  PDF exported. VQC notebook code executed and actual output saved.
- Test average precision spans 0.008540–0.100911. These single-seed results with
  29 test positives do not establish improvement; all variants remain reported.
  Matched RBF controls are in the shared Phase 6 parent run.
- Protocol and optimizer limitations documented in `docs/VQC.md`.
  Phase 8 may now begin. Stop after its validation gate; Phase 9 is not authorized.

## Phase 8 gate — passed; requested stopping point (2026-09-15)

- Installed/pinned PyTorch 2.10.0 CPU and transitive dependencies. Implemented the
  7 → 16 → 2 encoder, exact two-qubit EstimatorQNN/TorchConnector layer with
  parameter-shift input gradients, and classical probability output. Hybrid has
  169 parameters; matched classical neural control has 171.
- Both models ran all 20 epochs / 160 Adam updates on the same 128 training rows,
  sharing input features, common-layer initialization and deterministic batch order.
  Checkpoints selected by validation BCE: hybrid epoch 19, control epoch 20.
- [Results](../results/metrics/hybrid_20260915T151414_02ed38d0/results.json),
  [comparison](../results/metrics/hybrid_20260915T151414_02ed38d0/comparison.csv),
  [training figure](../results/metrics/hybrid_20260915T151414_02ed38d0/training.png).
- Gradient finite-difference tests pass for quantum and encoder weights. Actual
  training records nonzero gradients and parameter changes in every component.
  Independent checkpoint verification reproduced all 4,096 validation/test scores
  for each model, correct parameter counts and minimum-validation-loss selection.
- Two initial full-suite attempts exposed Windows native-DLL initialization order.
  Isolated probes identified sklearn-before-Torch as the failing order. Package
  startup now initializes Torch first on Windows; a fresh-process regression and
  selected prior-phase tests validate this fix. Custom-script import order and
  startup overhead are documented in `docs/HYBRID.md`.
- Final full suite: **123 passed in 48.16s**, warnings treated as errors, exit 0;
  `results/experiment_logs/phase8-tests.xml`. `pip check` passed. All four historical
  classical model probes and 2/6-qubit historical QSVC/VQC probes still reproduce
  their original saved predictions after the dependency/runtime change.
- Independently recomputed both models' full validation/test metric records and
  low-FPR operating points. Inspected training PNG; PDF exported. Hybrid notebook
  executed with real saved outputs. Documentation and experiment index updated.
- Observed test average precision: hybrid **0.074483**, control **0.200137**.
  Single seed and 29 test fraud cases do not support superiority claims. All VQC
  optimization budgets and hybrid epoch limits remain explicit. Timings are
  descriptive simulator measurements, with concurrent software checks disclosed.
- Raw CSV remains 493,534,783 bytes; streamed SHA-256 still matches
  `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`.
  No raw CSV changes or additional full-dataset copies. Balance exclusions persist.
- **Stopped after Phase 8. Phases 9–15 remain pending.** No ensemble, ablation/noise
  study, physical hardware, API, dashboard or deployment was started.

## Phase 9 gate — passed (2026-09-19)

- User authorized all remaining phases. Rechecked prior gate: 123 tests passed.
- Trained matched two-feature XGBoost; reused predeclared two-qubit QSVC/VQC.
  Disjoint validation halves support calibration and alpha/threshold selection.
  Evaluated three bases and six blends; selected alpha .50 for QSVC and .25 for VQC.
- [Completed run](../results/metrics/ensemble_20260919T082441_6070884a/results.json).
  All nine metric records, blends and validation-only selections independently verified.
- Full suite **125 passed in 53.96s**, warnings as errors;
  `results/experiment_logs/phase9-tests.xml`. Protocol and scarce-positive limits:
  `docs/ENSEMBLE.md`. Phase 10 may now begin.

## Phase 10 gate — passed (2026-09-19)

- Completed 24 QSVC ablations and nine matched RBF controls over three 96-row
  training resamples, with fixed evaluation rows. Covered 2/4/6 qubits, encoding
  repetitions 1/2/3, none/linear/full entanglement and Z/ZZ maps. Eight remains
  unsupported by the frozen feature policy; interactions between factors are not studied.
- [Run](../results/metrics/ablation_20260919T082721_8d49e44f/results.json),
  [mean/SD table](../results/metrics/ablation_20260919T082721_8d49e44f/mean_std.csv),
  [figure](../results/metrics/ablation_20260919T082721_8d49e44f/ablation.png).
- Reload probes passed; all 33 saved validation/test metric records independently
  reproduced. Interaction-free ZZ/Z scores agree. Inspected PNG; vector PDF saved.
- Full suite **126 passed in 56.34s**, warnings as errors;
  `results/experiment_logs/phase10-tests.xml`. Protocol: `docs/ABLATIONS.md`.
  Phase 11 may now begin. No significance or speedup conclusion is supported.

## Phase 11 gate — passed (2026-09-19)

- Compared exact, 256-shot noiseless, and illustrative noisy inference for one
  frozen two-qubit VQC on identical 4,096 evaluation rows. Sampled modes use three seeds.
- Exact-reference validation rejected an initial bit-count-parity assumption.
  Corrected to the trained VQC's integer-modulo-two interpretation; a new regression
  reproduces VQC probabilities from bound circuits. Failed attempt is retained and excluded.
- [Accepted run](../results/metrics/noise_20260919T083713_d55a9b52/results.json),
  [signed metric changes](../results/metrics/noise_20260919T083713_d55a9b52/degradation.csv),
  [inspected figure](../results/metrics/noise_20260919T083713_d55a9b52/noise.png).
- Every exact probability matches Phase 7; all seven saved metric records verified.
  Mean per-transaction SD is .02676 (shot) / .02719 (noisy). Signed metric changes
  are reported without assuming universal degradation. No hardware claim.
- Full suite **129 passed in 57.39s**, warnings as errors;
  `results/experiment_logs/phase11-tests.xml`. Protocol: `docs/NOISE.md`.
  Phase 12 integration may now begin; hardware execution remains credential-dependent.

## Phase 12 integration gate — passed; hardware execution pending (2026-09-19)

- Installed compatible Runtime SDK 0.49.0 with pinned transitive dependencies;
  existing numerical/quantum versions unchanged. `pip check` passed.
- Prepared [eight-circuit bundle](../results/hardware/hardware_20260919T084204_5fbc5fb8/request.json)
  with real PaySim positions, frozen weights/threshold and ideal reference metrics.
  QPY export and offline compilation succeeded. No IBM account key/instance is
  configured in the environment; no hardware job was submitted.
- Verified actual Runtime SamplerV2 against local Aer, actionable credential errors,
  immutable submission records, duplicate prevention and nonblocking collection.
- Full suite **132 passed in 48.44s**, warnings as errors;
  `results/experiment_logs/phase12-tests.xml`. Setup and submission/collection commands
  in `docs/HARDWARE.md`. Hardware results remain explicitly pending, as the master
  prompt permits. Phase 13 may now begin.

## Phase 13 gate - passed (2026-09-19)

- Saved full-training XGBoost serves strict transaction inputs through FastAPI and
  a persistent SQLite journal. Duplicate IDs and leakage fields are rejected.
- Real HTTP verification: `results/simulation/api_check_20260919T085042_225a7c63/verification.json`;
  20 predictions, duplicate 409 and target-field 422. Mean HTTP 24.74 ms;
  model pipeline 11.27 ms. Local demonstration only.
- Full suite **140 passed in 55.40s**, warnings as errors;
  `results/experiment_logs/phase13-tests.xml`. Dependency check passed.
- Contract and commands: `docs/API.md`. Phase 14 may now begin.

## Phase 14 gate - passed (2026-09-19)

- Streamlit live simulation and classical, quantum, hybrid, ablation/noise and
  feature-reduction pages implemented. Curves, confusion matrices, latency tables,
  model details, transaction history and alerts use saved artifacts/the real API.
- AppTest rendered all seven research runs, checked 88 model score/label pairs,
  rendered features and predicted using the actual saved model:
  `results/simulation/dashboard_check_20260919T085645_fc134139/verification.json`.
- Actual Streamlit server health and HTTP index passed:
  `results/simulation/dashboard_http_20260919T085726_49e61f72/verification.json`.
- Full suite **145 passed in 51.13s**, warnings as errors;
  `results/experiment_logs/phase14-tests.xml`. Dependency check passed.
- Commands and interpretation: `docs/DASHBOARD.md`. Phase 15 may now begin.

## Phase 15 gate - passed (2026-09-19)

- Created immutable, separate configurations and results for 97 reviewed model
  experiments and 15 excluded clipped-encoding diagnostics. Original run outputs
  are unchanged. Historical missing timings/versions are explicitly documented.
- Accepted [registry](../results/experiment_logs/registry_20260919T090207_faa6a792/registry.json)
  and [verification](../results/experiment_logs/registry_20260919T090207_faa6a792/verification.json):
  **112 experiment records and 661 artifact hashes verified**. Independent CLI
  revalidation with warnings as errors passed; saved metrics and validation-only
  thresholds, including low-FPR points, reproduced.
- Configurations: `experiments/registry_20260919T090207_faa6a792/`.
  [Publication tables and figures](../paper/tables/registry_20260919T090207_faa6a792/)
  include run-separated CSV/LaTeX tables and three-seed mean/sample-SD summaries.
  Inspected the 300-dpi PNG; vector PDF also saved. Shared evaluation samples and
  differing repetition protocols preclude a statistical-superiority claim.
- Streamed raw CSV verification matched original SHA256
  `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`
  and 493,534,783 bytes. No raw modifications or full CSV dataframe reload.
- Full suite **151 passed in 52.11s**, warnings as errors;
  `results/experiment_logs/phase15-tests.xml`. `pip check` and environment CLI passed.
- README, architecture, quantum design, experiment protocol and research notes
  updated. All requested software phases are complete. Physical IBM execution is
  pending credentials under the Phase 12 exception; no hardware results fabricated.
  Eight qubits remain unsupported by the approved feature representation.
