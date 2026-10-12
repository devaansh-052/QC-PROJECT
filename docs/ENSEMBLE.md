# Phase 9: probability ensembles

Run: [ensemble_20260919T082441_6070884a](../results/metrics/ensemble_20260919T082441_6070884a/results.json).
[All nine comparisons](../results/metrics/ensemble_20260919T082441_6070884a/comparison.csv).

Use the fixed two-dimensional Phase 6 angles, 128 training rows and identical
evaluation positions. Reuse the predeclared ZZ/repetition-1 QSVC and linear,
repetition-1 VQC; fit a small XGBoost on exactly their training rows and features.
These architectures were fixed without selecting a winning test score.

Stratify the 2,048-row validation set into disjoint calibration and selection
halves (seed 42), preserving both classes. Fit separate sigmoid calibrators to
each model's scalar output on the calibration half only. QSVC margins must not
be interpreted directly as probabilities. The same calibration procedure applies
to XGBoost and VQC. This differs from their earlier uncalibrated evaluations.

For each quantum model, evaluate alpha 0.25, 0.50 and 0.75 in
`alpha * P_xgboost + (1-alpha) * P_quantum`. Select alpha by maximum average
precision on the selection half only; first grid value wins ties. Decision
thresholds and low-FPR operating points use that same selection half. Test labels
are never used for calibration, alpha selection or thresholds. Report all base
models and grid candidates; selected alpha is explicit in the manifest.

Only 13 positive validation cases are divided between the two roles. Calibration
is therefore exploratory and cannot establish operational probability accuracy.
Both halves occupy the same original temporal validation window. The test still
has only 29 positives. Saved original row positions allow alignment audits.

Quantum fitting is reused: reported additional training time covers calibration,
not the historical QSVC/VQC fitting cost. Blend timings include metric evaluation
and are not pure model-inference benchmarks. No superiority claim is made.

```powershell
.\.venv\Scripts\python.exe scripts/train_ensemble.py --qsvc-run results/metrics/qsvc_20260915T143628_87486f0e4a58 --vqc-run results/metrics/vqc_20260915T144414
.\.venv\Scripts\python.exe scripts/summarize_quantum.py --run results/metrics/ensemble_20260919T082441_6070884a
```

Artifacts include XGBoost, three calibrators, calibration/selection indices,
configuration, score arrays and a completed result manifest. The independent
summarizer reproduces every reported metric, including low-FPR operating points.
