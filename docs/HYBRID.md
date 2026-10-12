# Phase 8: differentiable hybrid neural model

## Architecture and matched control

The hybrid receives the seven eligible Phase 3 features for exactly the Phase 6
sample positions. It learns a 7 → 16 → 2 encoder with tanh activations. The bounded
two-dimensional latent vector enters a two-qubit ZZ feature map (one repetition),
followed by a linear RealAmplitudes ansatz (one repetition, four trainable weights).
Two Z expectation values feed a linear output; sigmoid yields a fraud score.
The trained representation uses no balance fields, identifiers or flag columns.

The [Qiskit TorchConnector integration](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/05_torch_connector.html)
supports joint training with PyTorch. This implementation explicitly enables
`EstimatorQNN.input_gradients=True` so gradients reach the upstream encoder.
`ParamShiftEstimatorGradient` supplies circuit derivatives. StatevectorEstimator
and EstimatorQNN both use precision 0 for exact ideal simulation. No shots, device
noise, quantum hardware, or gradient detachment is used.

| Component | Hybrid parameters | Classical control parameters |
|---|---:|---:|
| Shared encoder architecture | 162 | 162 |
| Quantum / dense 2 → 2 tanh bottleneck | 4 | 6 |
| Shared output architecture | 3 | 3 |
| Total | 169 | 171 |

The models receive identical seven-column features and labels, identical initial
encoder/output weights, and the same deterministic batch order. The two-qubit
design is a fixed small-resource choice, not a selection by VQC test performance.
The appropriate component comparison is this paired neural control. Across-phase
comparisons also change feature representation and must not attribute differences
solely to the quantum layer.

## Training and evaluation protocol

`configs/hybrid.yaml` specifies seed 42, 20 epochs, batch size 16 and Adam learning
rate 0.01. Train both models on all 128 sampled training rows (64 fraud). Minimize
`BCEWithLogitsLoss` on raw output logits. No extra class weighting is applied to
the already balanced training subset. CPU execution caps Torch threads at two
and enables deterministic algorithms.

Each epoch checks all model-component gradients for missing/nonfinite values and
records mean minibatch training BCE plus validation BCE. The training loss is the
weighted mean of losses encountered during updates, not a second post-epoch pass.
Choose the earliest checkpoint with minimum validation BCE among epochs 1–20.
The fitting function receives training and validation data only. Test predictions
are made after selecting the checkpoint. Choose F1 and low-FPR thresholds using
validation only and report their frozen test behavior.

Training uses 160 optimizer updates per model. Save all epoch losses, the selected
epoch, first-batch gradient norms and the selected checkpoint's parameter changes
for encoder, bottleneck and output. Save weights with explicit architecture metadata;
reconstruct the circuit and reload with `torch.load(weights_only=True)`. Reload
probes must reproduce saved test scores. Finite-difference tests independently
check a quantum weight and an encoder weight through the complete circuit.

Fit wall time includes validation monitoring and checkpoint selection. Separate
optimizer-training and validation-monitor times are also saved; these are nested
components and must not be added again to fit wall time. Final batch inference
times exclude model initialization and reload. No real-time or speedup conclusion
follows from these classical simulator timings.
Some training overlapped software regression checks on the same host. Wall times
describe these runs and are not an isolated, controlled latency benchmark.

## Windows compatibility

The verified environment adds PyTorch 2.10.0 (CPU) and pinned dependencies to the
Phase 7 stack. A full-suite run exposed `c10.dll` initialization failure when
scikit-learn 1.4.2 was imported first. Minimal fresh-process probes reproduced the
failure with `sklearn → torch` and succeeded with `torch → sklearn/XGBoost`.
The installed sklearn Windows wheel explicitly preloads bundled C++ runtime DLLs.

The package entry point initializes Torch first on Windows. Tests use that same
bootstrap, and a separate subprocess verifies classical-then-hybrid project imports.
In custom Windows scripts/notebooks, import `quantum_fraud` (or `torch`) **before**
scikit-learn/XGBoost. Restart a kernel that has already loaded them in the wrong
order. This adds Torch initialization overhead to Windows project startup; reported
training/inference timings begin after imports. Dependency versions of historical
results remain unchanged in their manifests.

## Reproduce and limitations

```powershell
.\.venv\Scripts\python.exe scripts/train_hybrid.py --data results/metrics/qsvc_20260915T143628_87486f0e4a58/data
.\.venv\Scripts\python.exe scripts/summarize_quantum.py --run results/metrics/<hybrid-run>
.\.venv\Scripts\python.exe scripts/verify_hybrid.py --run results/metrics/<hybrid-run>
.\.venv\Scripts\python.exe -m pytest -q -W error
```

The run directory contains checkpoints, configuration, saved probabilities, all
metric records, training curves (PNG/PDF), and a completed results manifest. The
summarizer independently recomputes all saved validation/test metrics and exports
a comparison CSV.
The checkpoint verifier recomputes every validation/test probability and checks
the selected checkpoint against the minimum recorded validation loss.

Only one paired seed is evaluated. Validation has 13 fraud cases and test has 29;
the epoch selection uses that same small validation set as threshold selection.
The 20-epoch budget does not guarantee convergence. Outputs are uncalibrated under
the large training/evaluation prevalence shift. The Phase 6 adaptive encoding
inspection and earlier EDA mean these are exploratory results, not an untouched
confirmatory study. No statistical superiority, production readiness or quantum
advantage is claimed. Phase 9 and later work are outside this request.

See [PROGRESS.md](PROGRESS.md) for executed results and the final gate.

## Executed result

[Run manifest](../results/metrics/hybrid_20260915T151414_02ed38d0/results.json),
[comparison CSV](../results/metrics/hybrid_20260915T151414_02ed38d0/comparison.csv),
and [training curves](../results/metrics/hybrid_20260915T151414_02ed38d0/training.png).

| Model | Selected epoch | Test average precision | Test ROC-AUC | Test F1 | Test FPR |
|---|---:|---:|---:|---:|---:|
| Hybrid | 19 | 0.074483 | 0.869635 | 0.107226 | 0.186726 |
| Classical neural control | 20 | 0.200137 | 0.911103 | 0.173913 | 0.006439 |

F1/FPR use each model's validation-selected F1 threshold. The hybrid has higher
recall at its threshold (0.793103 versus 0.137931), accompanied by many more false
positives. The control has higher observed average precision in this run. This
does not constitute a statistical superiority result. Full metrics include the
separate validation-selected low-FPR operating points. Both checkpoints reproduced
all 4,096 saved validation/test probabilities in independent reload verification.
