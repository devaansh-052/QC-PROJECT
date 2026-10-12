# Phase 13: local transaction simulation and API

The local demonstration serves the full-training Phase 4 XGBoost artifact through
the saved Phase 3 preprocessing pipeline. Only step, type and amount enter the
model. Startup verifies that model and preprocessing provenance match. It loads
artifacts once and never trains on incoming transactions or rewrites raw data.

## Run

From the project root, in one terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn quantum_fraud.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
.\.venv\Scripts\python.exe scripts/simulate_transactions.py --url http://127.0.0.1:8000 --count 20 --seed 42
```

The simulator creates deterministic synthetic demonstration inputs, not evaluation
data or real payments. IDs include a unique run prefix. It records actual responses
and HTTP roundtrip durations in a unique `results/simulation/` directory. The API
uses `configs/serving.yaml`. Override the journal path with `QUANTUM_FRAUD_DATABASE`
or the configuration path with `QUANTUM_FRAUD_SERVING_CONFIG` if needed.

## Contract

- `GET /health`: readiness, model/run and frozen threshold.
- `POST /predict`: transaction details, score, predicted class, risk and decision.
- `GET /transactions?limit=50`: newest journal entries (limit 1–500).
- `GET /docs`: generated local API documentation.

Example input:

```json
{"transaction_id":"demo-001","step":632,"type":"TRANSFER","amount":15000.0}
```

Step is the PaySim hour index (1–743); type is one of its five known categories.
Amount must be finite, numeric, nonnegative and at most 1e12. Unknown fields,
including labels and balance fields, are rejected. Duplicate transaction IDs
return 409 and do not replace journal records. SQLite stores local demo history
using parameterized statements and a unique-ID constraint.

The response's `fraud_probability` is the model's **uncalibrated research score**.
Class weighting and prevalence shift prevent an operational probability claim.
The original validation-F1 threshold (0.9929561614990234) is frozen. At or above
it, predicted class is 1, risk HIGH and decision REVIEW. Below half that threshold
risk is LOW; otherwise MEDIUM. Both lower ranges return ALLOW. These are explicit
demo rules, not money-processing or banking approval actions.

`preprocessing_ms` and `model_ms` measure their components; `processing_ms` includes
lock waiting and processing but excludes HTTP serialization and journal insertion.
The generator measures HTTP roundtrip separately. The local sequential smoke run
does not establish production throughput or a real-time service guarantee.

## Verified execution

[Real HTTP verification](../results/simulation/api_check_20260919T085042_225a7c63/verification.json)
started an isolated Uvicorn server, sent 20 transactions through the actual saved
model, verified persistence and error responses, then stopped the server.

```powershell
.\.venv\Scripts\python.exe scripts/verify_api.py
```

Tests cover strict input validation, target/balance rejection, duplicate IDs,
journal persistence, bounds, deterministic generation and valid generated input.
FastAPI 0.141.1 / Starlette 1.6.0 uses HTTPX2 for its supported test client; the
caught deprecated HTTPX path was replaced. AnyIO 4.12.1 is pinned because 4.15.1
deprecates an alias still used by this Starlette test client. Warnings are not
suppressed. All dependencies are pinned.
The API is a localhost research demo; no public deployment was performed.
