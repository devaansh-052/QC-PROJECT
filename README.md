# Hybrid Quantum-Classical Fraud Detection

Research platform investigating complementary predictive performance on imbalanced
payment-fraud data. Simulator results are exploratory; no quantum advantage is claimed.

## Setup (Windows PowerShell)

Python 3.11.9 is the tested interpreter. Run from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
$env:PYTHONHASHSEED = '42'
.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\python.exe -m pytest
```

Activation is optional: all commands explicitly select the isolated environment.
On Linux/macOS use `.venv/bin/python`. The `.env.example` file is documentation;
the application does not automatically load environment files.

## Architecture and workflow

Dataset validation → EDA → training-only preprocessing → classical baselines →
feature reduction → QSVC → VQC → hybrid models → controlled experiments → API/dashboard.

Implementation now covers all 15 phases, with physical IBM execution pending
credentials. See [architecture](docs/ARCHITECTURE.md)
and the [phase gate tracker](docs/PROGRESS.md). Run and pass the current phase's
tests and command-line smoke check, verify outputs and update documentation before
starting the next phase. A failed check blocks progression.

Configuration lives in `configs/base.yaml`. Relative data/output paths are resolved
against the project root, not the shell's current directory. Unknown keys are rejected.

## Dependencies

The verified stack includes NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.4.2,
Qiskit 2.5.2, Qiskit Machine Learning 0.9.1, Aer 0.17.2 and PyTorch 2.10.0 CPU.
All dependencies are pinned in `requirements.txt`. Historical run manifests retain
their original versions. See [quantum compatibility](docs/QUANTUM_DESIGN.md).

On Windows, custom scripts must import `quantum_fraud` (or `torch`) before
scikit-learn/XGBoost to initialize compatible native libraries. Project commands
and tests do this automatically. See [the verified workaround](docs/HYBRID.md#windows-compatibility).

The local dataset is untracked. See [data setup](data/README.md). Do not use test fixtures as
research evidence. Preserve comparable data, splits and representations for model
comparisons; fit transforms on training data only and choose thresholds on validation.

## Current status and dataset command

**Software implemented through Phase 15; physical IBM execution pending.** Feature reduction and quantum
grids cover 2/4/6 dimensions; 8 is unsupported by the approved representation.
The hybrid uses two qubits and a matched classical neural control. Latest full
suite: **151 passed**, warnings as errors. Individual gates and execution evidence
are recorded in [the phase tracker](docs/PROGRESS.md).

Phase 0 passed its 17-test gate before Phase 1 implementation began. Phase 1 uses
pandas 2.2.2 and validates the supplied 6,362,620-row PaySim CSV. `isFraud` is the
target (8,213 positive rows). See the [gate tracker](docs/PROGRESS.md) for verification.
Phase 2 EDA is implemented; see [findings and reproduction](docs/EDA.md).
All four balance fields and their derivatives are excluded
from primary model inputs under the [documented policy](docs/DATASET.md).

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py
```

This writes a unique quality/provenance JSON report and prints its path. Missing or
invalid inputs exit with code 2 and an actionable error. All pinned transitive
dependencies are listed in `requirements.txt`; rerun installation after updates.

## Exploratory data analysis

```powershell
.\.venv\Scripts\python.exe scripts/run_eda.py
```

Produces eight aggregate tables and six figures (PNG/PDF) in unique run directories
under `results/metrics/` and `results/figures/`. Open `notebooks/01_eda.ipynb` to
review saved outputs. EDA revealed strong temporal differences, including a
fraud-only tail; read [EDA.md](docs/EDA.md) before designing temporal partitions.
Matplotlib 3.10.7 and its pinned dependencies were added for this phase.

## Preprocessing

```powershell
.\.venv\Scripts\python.exe scripts/preprocess.py
```

Creates whole-step temporal partitions and a training-only fitted pipeline under
`data/processed/`. See [preprocessing protocol](docs/PREPROCESSING.md) for split
definitions, feature policy, imbalance handling and saved artifacts. The historical
Phase 3 run used SciPy 1.13.0; the current verified environment uses 1.15.3.

## Classical baselines

```powershell
.\.venv\Scripts\python.exe scripts/train_classical.py --prepared data/processed/preprocessing_20260915T135023_4073b139cc42
```

Trains Logistic Regression, Random Forest, linear SVM and XGBoost with shared
features/partitions. See [classical protocol](docs/CLASSICAL_BASELINES.md) for
validation-only thresholds, metrics and timing/score limitations. XGBoost 3.1.1
is pinned. Full-data tree training can take several minutes.

## Feature reduction

```powershell
.\.venv\Scripts\python.exe scripts/reduce_features.py --prepared data/processed/preprocessing_20260915T135023_4073b139cc42 --classical-run results/metrics/classical_20260915T135718_d2739e452b90
```

Compares mutual information, forest importance and PCA at 2/4/6 dimensions, each
with a matched full-training linear SVM. The requested 8-feature case is explicitly
unsupported by the seven-column eligible representation. See
[feature-reduction protocol](docs/FEATURE_REDUCTION.md).

## Quantum and hybrid simulator experiments

```powershell
.\.venv\Scripts\python.exe scripts/train_qsvc.py --prepared data/processed/preprocessing_20260915T135023_4073b139cc42 --feature-run results/metrics/features_20260915T140754_40734f02b041
.\.venv\Scripts\python.exe scripts/train_vqc.py --data results/metrics/qsvc_20260915T143628_87486f0e4a58/data
.\.venv\Scripts\python.exe scripts/train_hybrid.py --data results/metrics/qsvc_20260915T143628_87486f0e4a58/data
```

These are separate phase commands. QSVC creates a unique run; subsequent commands
above intentionally reuse the verified saved subset. For a new subset, replace
their `--data` path with the new QSVC run's data directory. Verify each phase before
starting the next. A complete VQC grid takes several minutes on a CPU simulator.

| Phase | Verified experiment | Protocol |
|---|---|---|
| 6 | 12 QSVCs and 3 matched RBF SVMs | [QSVC](docs/QSVC.md) |
| 7 | 18 VQCs with convergence traces | [VQC](docs/VQC.md) |
| 8 | Differentiable hybrid (169 parameters) and classical control (171) | [Hybrid](docs/HYBRID.md) |

Quantum experiments share 128 training / 2,048 validation / 2,048 test rows drawn
from the PaySim temporal partitions. Test contains only 29 fraud cases. These are
small research experiments, not full-dataset quantum training or production-ready
fraud probabilities. Raw CSV contents remain unchanged. The continuation adds
ensembles, ablations, noise, IBM integration, API, dashboard and experiment management.

## Run the local demo

Start these in separate terminals, then open http://127.0.0.1:8501:

```powershell
.\.venv\Scripts\python.exe -m uvicorn quantum_fraud.api.app:create_app --factory --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
```

The API uses the saved full-training XGBoost model and validation threshold.
The dashboard shows live transactions, alerts and model latency, plus research
tables, confusion matrices, ROC/PR curves, feature/qubit studies and noise results.
Research pages work with the API offline. Details: [API](docs/API.md),
[dashboard](docs/DASHBOARD.md). This is a local research demonstration.

## Later phases and reproducible reporting

| Phase | Work | Documentation |
| --- | --- | --- |
| 9 | Validation-calibrated probability ensembles | [Ensembles](docs/ENSEMBLE.md) |
| 10 | Circuit ablations over three training resamples | [Ablations](docs/ABLATIONS.md) |
| 11 | Exact, finite-shot and noisy simulation | [Noise](docs/NOISE.md) |
| 12 | IBM submission/collection integration; hardware execution pending | [Hardware](docs/HARDWARE.md) |
| 13 | Saved-model API and transaction generator | [API](docs/API.md) |
| 14 | Live and research dashboard | [Dashboard](docs/DASHBOARD.md) |
| 15 | Immutable configurations, metrics and publication outputs | [Experiments](docs/EXPERIMENTS.md) |

```powershell
.\.venv\Scripts\python.exe scripts/manage_experiments.py
```

Creates separate configuration/result snapshots and CSV/LaTeX tables with PNG/PDF
figures. It verifies saved metrics, artifact hashes and the original raw dataset
fingerprint. Historical missing durations are explicit. The registry contains
97 reviewed model records and 15 excluded diagnostic records. Repeated studies
report descriptive mean/SD; no statistical superiority or quantum speedup is claimed.
