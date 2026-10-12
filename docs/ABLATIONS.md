# Phase 10: controlled quantum ablations

The reference is a four-qubit QSVC with one ZZ encoding repetition and linear
entanglement. Change one factor at a time: qubits 2/4/6, encoding repetitions
1/2/3, entanglement none/linear/full, and Z versus ZZ encoding. Eight qubits are
unsupported by the approved seven inputs and six-dimensional PCA rank. No leakage
feature or artificial padding is introduced. Actual logical depth is recorded
separately from encoding repetitions. Phase 7 already covers VQC ansatz depth.

Three seeds (42/43/44) select 48 fraud and 48 nonfraud rows without replacement
from the fixed 128-row training pool. Within each seed every model receives the
same rows. Validation/test stay fixed; PCA and angle transforms are inherited
from the original training-only representation. These are training-resample
replicates, not stochastic initializations or independent datasets.

Eight quantum configurations and three dimension-matched RBF controls per seed
produce 33 fits. Removing ZZ pair interactions leaves exactly Z encoding; the
duplicate is an intentional numerical sanity check. The one-factor-at-a-time
design does not estimate interactions between factors. No test-based winner is
selected. Thresholds continue to use validation only.

Save models, sample positions, scores, actual circuit metadata and nested kernel
timings. Reload checks compare model probes. `study.csv` contains the requested
metrics/depth/timing columns. `mean_std.csv` reports means and sample standard
deviations across three training resamples. Error bars are SD, not confidence
intervals. With 29 test positives, no statistical superiority is established.
Simulator wall times may overlap software checks and are descriptive only.

```powershell
.\.venv\Scripts\python.exe scripts/run_ablations.py --data results/metrics/qsvc_20260915T143628_87486f0e4a58/data
.\.venv\Scripts\python.exe scripts/summarize_quantum.py --run results/metrics/<ablation-run>
```

The gate tracker links executed results. All outputs are outside raw data.
