# Phase 7: variational quantum classifier

## Protocol

The fixed grid uses 2/4/6 qubits, RealAmplitudes ansatz repetitions 1/2/3,
and linear/circular entanglement (18 fits). Repetitions are not literal gate
depth: each result separately records logical circuit depth and gate counts.
Eight qubits remain unsupported by the seven eligible features. Full connectivity
is deferred to a later resource study; linear and circular meet the small-circuit
scope. At two qubits these connectivity patterns can represent equivalent circuits.

Every fit uses the authoritative Phase 6 subset and its training-only PCA/angle
transforms: 128 training rows (64 fraud), 2,048 validation rows (13 fraud), and
2,048 test rows (29 fraud). No raw data is rewritten or loaded again. The matched
RBF SVM controls are saved in that same Phase 6 parent run; comparing against
full-data Phase 4 models would mix training sample size with model architecture.

ZZ encoding uses one repetition and linear connectivity. A parameterized
RealAmplitudes circuit follows the encoding. QMLSampler with `shots=None` computes
exact ideal statevector probabilities. VQC's binary measurement interpretation
maps basis-state outcomes to two classes. The positive output is an uncalibrated
fraud probability. Training minimizes cross entropy with COBYLA, a deterministic
seed-42 initial point, maximum 60 objective evaluations, and tolerance 1e-4.

The [current VQC API](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.algorithms.VQC.html)
supports a V2 sampler and optimizer callback. The installed QML 0.9.1
`QMLSampler` explicitly supports exact mode; no physical quantum device is used.

## Optimization and evaluation

Save every objective callback, optimizer evaluation count, available iteration
count, trainable parameter count, fit duration, and validation/test batch inference
durations. The SciPy COBYLA wrapper does not supply a separate iteration count
(`optimizer_iterations` is null); objective evaluations are the reported progress
axis. A 60-evaluation fit is marked budget-exhausted. A completed experiment means
the implementation and artifacts passed validation, not that the optimizer
converged. Convergence curves show observed loss, without inventing a convergence
or quantum advantage claim.

Select each decision threshold using validation F1 only. Also report validation-
selected 1% and 0.5% FPR operating points on test. Save accuracy, precision, recall,
F1, ROC-AUC, average precision (PR-AUC), MCC, false-positive/negative rates and
confusion matrices. All variants are reported; test results do not choose a winner.
Tiny positive counts, balanced training versus rare-fraud evaluation, one seed,
and the disclosed Phase 6 adaptive encoding correction limit research conclusions.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts/train_vqc.py --data results/metrics/qsvc_20260915T143628_87486f0e4a58/data
.\.venv\Scripts\python.exe scripts/summarize_quantum.py --run results/metrics/<vqc-run>
.\.venv\Scripts\python.exe -m pytest -q -W error
```

The unique output directory contains all models, score arrays, metric JSON,
`results.json`, and convergence plots in PNG/PDF. The summarizer independently
recomputes validation/test metrics from saved scores and exports `comparison.csv`.
Each fit also checks serialization by reloading and reproducing test probe scores.

See [PROGRESS.md](PROGRESS.md) for the executed run and phase gate evidence.
