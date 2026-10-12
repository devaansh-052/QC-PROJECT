# Phase 12: optional IBM hardware integration

**Integration verified; physical execution pending.** No API key or IBM instance
was configured during this run. No physical hardware job was submitted and no
hardware counts, metrics or timings are claimed.

The prepared [bundle](../results/hardware/hardware_20260919T084204_5fbc5fb8/request.json)
contains eight frozen two-qubit VQC circuits (four examples of each class), original
row positions, labels, validation-frozen threshold and exact simulator reference.
This balanced diagnostic subset cannot estimate natural-prevalence performance.
Local QPY serialization and basis compilation are preparation, not hardware execution.

The pinned Runtime SDK is 0.49.0. Its SamplerV2 interface was exercised against
local Aer without credentials. The workflow follows IBM's
[Sampler quickstart](https://quantum.cloud.ibm.com/docs/en/guides/get-started-with-sampler)
and [Runtime service authentication](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/qiskit-runtime-service).
An explicit backend is selected, circuits are compiled to that backend's ISA,
and a single job uses 128 shots per circuit and a 120-second execution limit.
Device noise, calibration drift and queue time remain physical-run considerations.

## Setup and commands

Create an IBM Quantum Platform instance and API key using IBM's account interface.
Set `IBM_QUANTUM_API_KEY` and `IBM_QUANTUM_INSTANCE` (instance CRN) in your local
environment. Do not place secret values in source, result JSON or chat. This
implementation passes them directly to the SDK and does not save an account file.
The environment template contains empty names only and is not loaded automatically.

```powershell
.\.venv\Scripts\python.exe scripts/ibm_hardware.py prepare --data results/metrics/qsvc_20260915T143628_87486f0e4a58/data --vqc-run results/metrics/vqc_20260915T144414
.\.venv\Scripts\python.exe scripts/ibm_hardware.py submit --bundle results/hardware/<bundle> --backend <your-backend-name>
.\.venv\Scripts\python.exe scripts/ibm_hardware.py collect --bundle results/hardware/<bundle>
```

Preparation is offline. Submission is explicit and can consume account quota.
The command saves the returned job ID and exits; collection reports queued/running
status without waiting indefinitely. When done, it validates result/shot counts,
saves actual measured probabilities, metrics, compiled depths/widths, backend,
job timing/usage metadata and primitive metadata. Physical width can exceed two
after layout; logical qubits remain two. No noisy hardware result is filled in
from simulator data.

An exclusive submission reservation prevents duplicate jobs. If a network failure
occurs after reservation but before saving the job ID, check IBM's jobs view before
manual recovery; do not blindly resubmit. Previous request/submission/result files
are not overwritten. Credentials are required for submission and collection.

The master prompt explicitly allows later software phases to proceed after this
verified integration when credentials are absent. Hardware execution remains pending.
