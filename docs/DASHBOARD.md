# Research dashboard

From the project folder, start the API and dashboard in separate terminals:

```powershell
.\.venv\Scripts\python.exe -m uvicorn quantum_fraud.api.app:create_app --factory --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
```

Open http://127.0.0.1:8501. Live simulation shows transaction details, the saved
XGBoost model, uncalibrated probability, predicted class, risk, review decision,
processing latency, recent history and alerts. IDs must be unique. The API
validates all inputs and rejects targets and balances. No money moves.
`QUANTUM_FRAUD_API_URL` may point at a different local API port.

Research pages remain usable with the API offline. They read completed artifacts
selected in `configs/research_runs.json`, never the raw CSV or training models.
Classical, quantum and hybrid pages show metrics, confusion matrices, ROC and
precision-recall curves. Tables include training and test-batch timing; this is
not request latency. Feature reduction and ablation/noise pages provide the
2/4/6 feature/qubit studies, saved plots and downloadable repeated-run tables.
Eight qubits remain unsupported by the approved feature representation.

Full-data and subset runs are explicitly distinguished; cross-run rankings are
not controlled comparisons. Initial clipped QSVC diagnostics and failed noise
attempts are excluded. IBM execution remains pending, with no hardware metrics.

## Verification

`scripts/verify_dashboard.py` uses Streamlit's official
[AppTest interface](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
to render every research run, validate every model's score/label alignment,
render the feature page and submit a transaction through the real saved model.
The accepted record is
`results/simulation/dashboard_check_20260919T085645_fc134139/verification.json`.
`scripts/verify_dashboard_server.py` separately launches and checks the actual
HTTP server, then stops it. Unit tests cover offline behavior, predictions,
alerts, duplicate error display, rejected run paths and score alignment.

Streamlit 1.64.0 and its installed dependencies are pinned; numerical dependencies
remain unchanged. No browser screenshot or public deployment is claimed.
