# Phase 6 QSVC experiments

```powershell
.\.venv\Scripts\python.exe scripts/train_qsvc.py --prepared data/processed/preprocessing_20260915T135023_4073b139cc42 --feature-run results/metrics/features_20260915T140754_40734f02b041
```

The command uses `configs/quantum.yaml`, creates a unique `qsvc_<UTC>_<id>` run
under `results/metrics/`, and writes `results.json` only after every QSVC/control
fit, evaluation and saved-model reload succeeds. The `data/` subdirectory records
exact subsets, transformations, out-of-range counts and near-saturation counts for
reuse by later phases. The initial hard-clipping diagnostic is retained separately;
the authoritative encoding is `standard_tanh_v1`.

The grid has twelve ideal-statevector QSVCs and three matched RBF SVMs. Consult
[QUANTUM_DESIGN.md](QUANTUM_DESIGN.md) for dimensions, encodings, dependency checks,
timing interpretation and the sampling protocol. These runs cannot be compared
as algorithm superiority evidence against Phase 4's full-training baselines.

Tests verify exact-kernel symmetry, positive semidefiniteness, unit diagonal,
qubit/input consistency, saved QSVC reloads, dataset sampling and training-only
angle fitting, plus a real Aer Bell-circuit smoke check. No quantum device is used.

Verified [smooth-angle run](../results/metrics/qsvc_20260915T143628_87486f0e4a58/results.json)
and [comparison CSV](../results/metrics/qsvc_20260915T143628_87486f0e4a58/comparison.csv).
All 97 tests pass. `notebooks/05_qsvc.ipynb` reads this explicit run. The validation
and test subsets contain only 13 and 29 fraud cases, respectively; no statistical
superiority inference is justified. Six-dimensional near-saturation affects 15
validation and 20 test rows; 2/4-dimensional subsets have none at the 1e-7 boundary tolerance.
