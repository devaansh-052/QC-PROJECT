# Quantum design

## Phase 6 dependency compatibility

Official PyPI metadata checked on 2026-09-15 identifies Qiskit 2.5.2, Qiskit Machine
Learning 0.9.1 and Aer 0.17.2 as current stable releases. Qiskit requires NumPy >=2
and SciPy >=1.14; QML also requires NumPy >=2. The working project set now pins
NumPy 2.2.6 and SciPy 1.15.3, plus every installed runtime dependency in
`requirements.txt`. Python remains 3.11.9. Earlier run manifests retain their
historical versions; old results are not relabeled as new experiments.

The original 85 tests passed after migration, `pip check` passed, and all four
historical classical model artifacts reproduced saved probe scores. A Bell-circuit
Aer test verifies simulator compatibility. Quantum code uses current circuit-library
functions and QML `to_dill`/`from_dill`, not deprecated blueprint/save APIs.

References: [Qiskit](https://pypi.org/project/qiskit/2.5.2/),
[QML](https://pypi.org/project/qiskit-machine-learning/0.9.1/),
[Aer](https://pypi.org/project/qiskit-aer/0.17.2/).

## Phase 6 subset and encoding protocol

- Use the completed Phase 3 temporal partitions and train-fitted Phase 5 PCA.
- Deterministically sample 64 fraud and 64 nonfraud training rows without replacement.
  Kernel cost grows quadratically with training sample count; this is an intentionally
  small exploratory supervised experiment, not a full-dataset quantum comparison.
- Sample 2,048 validation and 2,048 test rows approximately proportionally by class.
  Each retains both classes, but few positives make performance estimates unstable.
- Save original CSV row positions and partition-relative positions, labels and
  eligible seven-feature inputs. Reuse the exact same arrays for every control.
- Project to 2/4/6 dimensions with the same train-fitted PCA. Fit a separate
  standard scaler on the 128 supervised training rows only, then apply tanh to
  map smoothly into [-1,1] radians. Report rows outside training min/max and
  near-saturated angles; future values are not hard-clipped.
- PCA learned from all original training rows (unlabeled for this transformation);
  both quantum and classical controls receive this same representation. The prior
  full-file EDA and Phase 5 results were visible during research planning; results
  remain exploratory and require independent confirmation.

No balance field or derivative enters any representation. Eight-dimensional
experiments remain unsupported under the frozen feature policy.

The initial Phase 6 run `qsvc_20260915T143124_071ab7bb9001` used hard-clipped min-max
angles. Output inspection found all validation/test rows clipped in at least one
component. That diagnostic prompted the smooth `standard_tanh_v1` encoding before
proceeding. The initial run is retained, not silently overwritten. This is an
adaptive representation correction after inspecting the dataset, not a claim of
an untouched test set or a selection based on the highest test metric. Numerical
tanh saturation remains possible and is recorded explicitly.

## Phase 6 QSVC

`z_feature_map` and `zz_feature_map` encode the reduced angles on one qubit per
feature. The fixed grid is 2/4/6 qubits x Z/ZZ maps x 1/2 repetitions. ZZ uses linear
entanglement; Z has no entangling gates. Logical depth and gate counts describe the
constructed circuit, not a transpiled physical-device layout.

QML's [FidelityStatevectorKernel](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.kernels.FidelityStatevectorKernel.html)
computes squared state overlaps using exact statevectors (`shots=None`). QSVC uses
C=1. Every dimension also has an RBF SVM with C=1 and gamma='scale' on identical
samples and angle features. Predictions are SVM margins, not calibrated probabilities.
Thresholds and FPR operating points are chosen on validation only.

Kernel instrumentation uses the documented `statevector_type` extension point.
It measures statevector construction (simulator execution/allocation), remaining
kernel work (binding, caching, overlaps and loop overhead), and total kernel wall
time. Model training and batch prediction durations are recorded separately.
Nested kernel timings are components of model timings and must not be summed
again as independent work. Serialization/reload probes are outside reported fit
and prediction timings. Simulator runtime cannot establish quantum speedup.
Some runs overlapped software verification on the same machine; reported wall
times are descriptive rather than an isolated latency benchmark.

## Phases 7 and 8

Phase 7 runs the exact VQC grid described in [VQC.md](VQC.md). Phase 8 adds an
end-to-end differentiable two-qubit TorchConnector model and matched classical
neural control; see [HYBRID.md](HYBRID.md), including the Windows native-library
initialization requirement. Both reuse the authoritative Phase 6 sample positions.
Their training and evaluation artifacts are linked from [PROGRESS.md](PROGRESS.md).

## Phases 9-15

The continuation adds calibrated ensembles, controlled circuit ablations,
finite-shot/noisy Aer studies, IBM Runtime integration and local serving/research
views. See [ensembles](ENSEMBLE.md), [ablations](ABLATIONS.md), [noise](NOISE.md)
and [hardware](HARDWARE.md). Noise inference preserves the trained VQC's integer
outcome modulo-two mapping and validates bound circuits against saved exact
probabilities. The hybrid instead measures two Pauli-Z expectation values.

Physical hardware execution remains pending credentials. Prepared circuits and
local Runtime/Aer checks are not hardware results. The immutable
[experiment registry](EXPERIMENTS.md) links configurations, parent representations,
scores, metadata and separate timings. Repeated-run plots report descriptive SD,
without a quantum advantage or statistical superiority claim.
