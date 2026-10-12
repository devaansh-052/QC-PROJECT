# Phase 11: noisy simulation

Reuse the frozen two-qubit, linear, repetition-1 Phase 7 VQC and all original
2,048 validation and 2,048 test rows. No retraining or architecture selection occurs.
Transpile once into rz/sx/x/cx gates, bind input values and frozen weights, and
verify exact class probabilities against every saved Phase 7 probability.

Compare exact statevectors, noiseless finite-shot Aer density-matrix simulation,
and noisy finite-shot Aer density-matrix simulation. Shot runs use 256 shots and
seeds 42/43/44. Each measured bitstring contributes to positive-class probability
using the trained model's interpretation: integer outcome modulo two (the least
significant measured bit), not the parity of all set bits. The same compiled
circuits are used in both sampled modes.

The [Aer noise API](https://qiskit.github.io/qiskit-aer/tutorials/3_building_noise_models.html)
supports explicit gate and readout errors. This illustrative model applies .01
single-qubit depolarization to sx/x, .03 two-qubit depolarization to cx, and readout
matrix [[.98,.02],[.03,.97]]. Rz gates are treated as noiseless virtual rotations.
These rates are **not** measured hardware calibration or device results.

Each mode has ordinary validation-selected F1 and low-FPR metrics. In addition,
freeze the exact model's validation-F1 threshold and report signed changes in
precision, recall, F1 and average precision relative to exact test predictions.
A positive change is retained; noise is not assumed to worsen every metric.
`degradation.csv` records those fixed-threshold changes.

Report per-transaction sample SD across the three simulator seeds, its mean/max,
and mean absolute probability difference from exact outputs. The figure separates
probability bias from seed variation. Three seeds and 29 test positives are too
few for strong robustness claims. This is one noise profile on one small circuit.

Record transpilation, prediction wall time and Aer backend-reported simulator
time. Backend time is nested within prediction wall time, includes simulator work,
and is not physical circuit execution latency. Reused training is recorded as
zero additional fit time. All runs occur on classical hardware.

```powershell
.\.venv\Scripts\python.exe scripts/run_noise.py --data results/metrics/qsvc_20260915T143628_87486f0e4a58/data --vqc-run results/metrics/vqc_20260915T144414
.\.venv\Scripts\python.exe scripts/summarize_quantum.py --run results/metrics/<noise-run>
```

Saved configuration, probabilities, metrics, stability values and PNG/PDF figures
are linked from the phase tracker. Raw data is untouched.
