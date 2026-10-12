# Experiments

Tests are software verification, not model results. Completed experiments retain
dataset identity, samples, features, split strategy, seed, preprocessing, model
and quantum settings, metrics and separate timings in an immutable registry.

| Run | Scope | Status |
| --- | --- | --- |
| `classical_20260915T135718_d2739e452b90` | Four full-training classical models, seven eligible features, seed 42, temporal splits | Complete; [results and limits](CLASSICAL_BASELINES.md) |
| `features_20260915T140754_40734f02b041` | Three reducers x 2/4/6 dimensions, nine matched full-training linear SVM controls | Complete feasible grid; 8 unsupported; [results and limits](FEATURE_REDUCTION.md) |
| `qsvc_20260915T143628_87486f0e4a58` | Twelve exact QSVCs plus three matched RBF controls, shared tiny temporal subsets | Complete; [protocol](QSVC.md) |
| `vqc_20260915T144414` | Eighteen exact VQCs, 2/4/6 qubits, repetitions 1/2/3, linear/circular | Complete budgeted experiment; [protocol](VQC.md) |
| `hybrid_20260915T151414_02ed38d0` | Two-qubit differentiable hybrid and capacity-matched classical neural control | Complete; [protocol](HYBRID.md) |

The earlier `qsvc_20260915T143124_071ab7bb9001` run is retained as a diagnostic
of hard-clipped encoding. The authoritative smooth-encoding correction is disclosed
in the quantum design. Simulator execution occurred on classical hardware only.
No physical quantum hardware execution has occurred; credentials remain pending.

## Continuation experiments

| Phase | Experiments | Protocol |
| --- | --- | --- |
| 9 | Three calibrated bases and six probability blends | [Ensembles](ENSEMBLE.md) |
| 10 | 24 QSVC ablations and nine RBF controls, three training resamples | [Ablations](ABLATIONS.md) |
| 11 | One exact reference and six shot/noisy evaluations, three simulator seeds | [Noise](NOISE.md) |
| 12 | Eight-circuit IBM submission bundle; physical execution pending | [Hardware](HARDWARE.md) |

## Create or validate a registry snapshot

```powershell
.\.venv\Scripts\python.exe scripts/manage_experiments.py
.\.venv\Scripts\python.exe scripts/manage_experiments.py --verify results/experiment_logs/registry_<timestamp>_<id>
```

Each invocation creates unique directories; exclusive writes reject collisions:

- `experiments/registry_<id>/`: one configuration JSON per model experiment.
- `results/experiment_logs/registry_<id>/`: result records, registry, verification.
- `paper/tables/registry_<id>/`: CSV/LaTeX tables and 300-dpi PNG/vector PDF plots.

The registry covers 97 reviewed records plus 15 explicitly excluded diagnostic
records from the first hard-clipped QSVC run. The catalog selects reviewed model
runs; feature controls and the disclosed diagnostic are also registered. Update
the reviewed catalog deliberately when adding validated runs. Incomplete attempts,
EDA and other ancillary runs appear in the inventory, outside model comparisons.
API/dashboard simulations are software demonstrations, not evaluation evidence.

Each configuration records experiment/run IDs, original UTC timestamp, dataset
identity/SHA256, sample and class counts, features, temporal boundaries, sampling,
seed and its role, preprocessing/feature policy, model/hyperparameters, quantum
configuration, source report and manifest hashes. Results separately record
metrics, training duration, partition inference duration, score/label locations,
timing scope and limits. Config hashes link the two. Model files, source reports,
saved arrays, preprocessing/position artifacts, dependency pins and current source
are fingerprinted. Validation recomputes metrics and validation-only thresholds,
including low-FPR operating points, and detects artifact changes.

Historical metadata is normalized retrospectively. Run timestamps are inferred
from original UTC run IDs, not invented per-model start times. Some fixed factory
constants are reconstructed from audited source. The current source snapshot is
not historical Git provenance. Original reports are retained without edits.

The feature and ensemble phases did not separately record pure inference duration;
those values are explicitly null with reasons. Blend/evaluation time is not
relabeled as inference. Ensemble times exclude reused parent-model training; noise
has zero additional training and points to its frozen source. Hybrid fit duration
includes validation monitoring. Nested simulator timings must not be added twice.
The feature report omitted runtime versions; earlier preprocessing versions are
not substituted. No retrospective timing or version measurements are fabricated.

The registry streams a raw CSV hash and checks it against the original fingerprint.
It never reloads the full CSV or changes raw data. All new outputs stay outside
`data/raw/`. Revalidation after an intentional source change will report a hash
mismatch; create a fresh snapshot to document the new source state.

## Statistical reporting

Tables are separated by run. The combined CSV includes sample sizes and should not
be used as a cross-run leaderboard. Repeated tables report mean and sample SD
(`ddof=1`) for three training resamples in Phase 10 and three simulator seeds in
Phase 11. The latter do not measure training or population uncertainty. Runs share
fixed evaluation rows; no independence or statistical superiority test is claimed.

Single-seed classical, VQC and hybrid results remain explicitly single-seed.
Quantum testing has only 29 fraud examples; ensemble selection uses six fraud
examples after the calibration split. Hyperparameter grids were not selected by
test winners. No production-calibration, quantum speedup or hardware-result claim
is justified. Eight qubits remain unsupported by the approved representation.

Accepted snapshot:
[registry_20260919T090207_faa6a792](../results/experiment_logs/registry_20260919T090207_faa6a792/registry.json).
Its [verification](../results/experiment_logs/registry_20260919T090207_faa6a792/verification.json)
checked 112 records and 661 artifact hashes. [Publication outputs](../paper/tables/registry_20260919T090207_faa6a792/)
include separate run tables and repeated-study figures. See [phase gates](PROGRESS.md)
for the full 151-test gate and execution evidence.
