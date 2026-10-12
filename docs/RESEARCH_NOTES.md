# Research notes

No statistical superiority conclusions yet. Preserve class imbalance in evaluations, choose
thresholds on validation only, and distinguish synthetic public data, software test
fixtures, ideal simulation, noisy simulation and physical quantum hardware.

## Phase 1: 2026-09-15

Full-file validation found 6,362,620 rows, 8,213 fraud labels, no nulls and no exact
duplicates. These are data-quality observations, not model results. A clean schema
does not imply valid decision-time features. Follow `paysim-primary-v1` in
[DATASET.md](DATASET.md): exclude all four balance fields and derivatives from
primary models. No learned correlations or test-set scores informed this policy.
Do not infer that numerical zeros represent known actual balances in all records.

## Phase 2: descriptive EDA

See [EDA.md](EDA.md) for exact values and interpretation. All fraud labels occur
in TRANSFER/CASH_OUT; 320 steps are entirely fraud-labeled and the final 296 rows
after step 718 are all fraud. These simulator patterns make temporal partition
validation essential. Whole-file exploration also limits any future claim that
the research holdout was never inspected. Balance exclusions remain unchanged.

## Phases 3–5

Whole-step partitions preserved all rows and the late fraud-only tail. Train,
validation and test all contain both classes but their prevalences differ sharply.
All four classical baselines completed; RF has the largest observed average
precision in this fixed single-seed run, without a statistical superiority claim.
Validation-F1 operating points have limited fraud recall. Scores need calibration
before interpretation as operational fraud probabilities.

The prespecified reduction grid gives different linear-SVM scores at 2/4/6 dimensions.
Do not choose the apparently best test representation retrospectively. MI estimates
use only 95 positive examples and need stability checks. Eight reduced features
cannot be produced from seven approved encoded columns. See the linked phase
documents for exact experiments, saved artifacts and limits.

## Phases 6–8

Exact simulator experiments now cover QSVC, VQC and an end-to-end differentiable
hybrid. All quantum training uses the documented 128-row subset, with 13 validation
and 29 test positives. All balance exclusions persist. The initial clipped-angle
diagnostic and smooth-encoding correction are disclosed in `QUANTUM_DESIGN.md`.

All VQC variants exhaust their 60-evaluation optimization budget. The hybrid and
classical neural control each train for 20 epochs, selecting checkpoints on validation
only. Observed test average precision is 0.074483 for the hybrid versus 0.200137
for the neural control in this one paired run. The hybrid does not show a PR-AUC
improvement here; no statistical or quantum advantage claim is justified. Its
validation-F1 threshold gives higher recall but also substantially more false positives.
See full metrics and fixed low-FPR operating points before interpreting that tradeoff.

These experiments establish working implementations and reproducible artifacts.
They do not establish production readiness, calibration, convergence, robustness,
or a real-time quantum speed advantage.

## Phases 9-15: 2026-09-19

Validation-calibrated ensembles, three-resample quantum ablations and three-seed
shot/noisy simulations are complete. Ensemble alpha selection and thresholds use
only the selection half of validation; the other half fits calibrators. Six and
seven positive examples in these roles make calibration and selection unstable.
The noise experiment verifies the trained VQC's integer-modulo-two measurement
mapping; an initial incorrect parity assumption was caught and rejected before
accepting outputs. Synthetic noise rates do not represent an IBM calibration.

The IBM Runtime integration is tested locally with Aer and prepared real circuits,
but physical execution is pending account credentials. API and dashboard timings
measure a localhost demonstration; generated transactions are not research test
data. No service is deployed publicly.

The final registry preserves all reviewed results plus the excluded clipped-angle
diagnostic, hashes artifacts, separates configurations/results and regenerates
metrics. Descriptive mean/SD tables distinguish training-resample variation from
simulator-seed variation. Shared evaluation rows and scarce positives do not
justify significance tests or population confidence intervals here. Historical
missing inference times and feature-run versions remain explicitly missing.
