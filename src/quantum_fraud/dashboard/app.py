"""Local Streamlit research dashboard; no training or raw CSV reads."""
from uuid import uuid4
import json
import os
import pandas as pd
import requests
import streamlit as st
from quantum_fraud.dashboard.research import ROOT, artifact_stem, catalog, curves, read_run, score_data, summary_table


def api_request(method, route, **kwargs):
    url = os.environ.get("QUANTUM_FRAUD_API_URL", "http://127.0.0.1:8000").rstrip("/")
    response = requests.request(method, url + route, timeout=10, **kwargs)
    response.raise_for_status()
    return response.json()


def live_page():
    st.header("Transaction simulation")
    st.caption("Local research demo. Scores are uncalibrated; REVIEW and ALLOW do not move or block money.")
    try:
        health = api_request("GET", "/health")
    except requests.RequestException:
        st.info("API is offline. Start the local API using the command in docs/API.md. Research pages remain available.")
        return
    st.write("Model: " + health["model"])
    st.caption(f'Frozen validation threshold: {health["threshold"]:.6f}')
    with st.form("transaction"):
        identifier = st.text_input("Transaction ID", value="demo-" + uuid4().hex[:10])
        left, right = st.columns(2)
        step = left.number_input("Hour (step)", min_value=1, max_value=743, value=650)
        kind = right.selectbox("Transaction type", ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"])
        amount = st.number_input("Amount", min_value=0.0, max_value=1e12, value=1000.0)
        submitted = st.form_submit_button("Predict")
    if submitted:
        try:
            result = api_request("POST", "/predict", json={"transaction_id": identifier, "step": step, "type": kind, "amount": amount})
            st.session_state["last_prediction"] = result
        except requests.RequestException as error:
            st.error(f"Prediction was not saved: {error}")
    if "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]
        st.subheader("Latest prediction")
        a, b, c = st.columns(3)
        a.metric("Fraud probability (uncalibrated)", f'{result["fraud_probability"]:.4f}')
        b.metric("Predicted class", str(result["predicted_class"]))
        c.metric("Processing latency", f'{result["processing_ms"]:.2f} ms')
        st.write(f'{result["transaction_id"]} | {result["type"]} | {result["amount"]:,.2f} | risk {result["risk_level"]} | {result["decision"]}')
        if result["predicted_class"]:
            st.warning("Fraud alert: review this simulated transaction.")
    st.subheader("Recent transactions")
    st.button("Refresh transactions")
    try:
        recent = api_request("GET", "/transactions?limit=50")["transactions"]
        if recent:
            columns = ["transaction_id", "step", "type", "amount", "fraud_probability", "predicted_class", "risk_level", "decision", "model", "processing_ms"]
            st.dataframe(pd.DataFrame(recent)[columns], hide_index=True)
            st.write(f'Fraud alerts in recent history: {sum(r["predicted_class"] for r in recent)}')
        else:
            st.info("No simulated transactions yet.")
    except requests.RequestException:
        st.error("Could not refresh transaction history.")


def research_page(page):
    st.header(page + " results")
    st.caption("Test results use frozen validation-selected thresholds. Full-data and subset experiments have different training/evaluation sizes; cross-run rankings do not establish superiority.")
    run_id = st.selectbox("Experiment run", catalog()[page])
    path, report = read_run(run_id)
    st.dataframe(summary_table(report), hide_index=True)
    st.caption("Timing is saved wall-clock duration. Test batch time is not single-request latency; reused models may record zero additional training. Simulator timings do not establish quantum speedup.")
    records = {artifact_stem(r): r for r in report["models"]}
    name = st.selectbox("Model details", list(records))
    record = records[name]
    labels, scores = score_data(path, report, record)
    st.write(f"Test transactions: {len(labels):,}; fraud: {int(labels.sum()):,}")
    st.subheader("Confusion matrix")
    st.dataframe(pd.DataFrame(record["test"]["confusion_matrix"], index=["Actual legitimate", "Actual fraud"], columns=["Predicted legitimate", "Predicted fraud"]))
    roc, pr = curves(labels, scores)
    left, right = st.columns(2)
    left.subheader("ROC curve")
    left.line_chart(roc, x="false positive rate", y="true positive rate")
    right.subheader("Precision-recall curve")
    right.line_chart(pr, x="recall", y="precision")
    if page == "Ablations and noise":
        st.info("Ablations repeat three training resamples; noise repeats three simulator seeds on one frozen model. Eight qubits are unsupported by the approved representation.")
        for filename in ("ablation.png", "noise.png"):
            if (path / filename).exists():
                st.image(str(path / filename))
        for filename in ("mean_std.csv", "degradation.csv"):
            if (path / filename).exists():
                st.download_button("Download " + filename, (path / filename).read_bytes(), file_name=filename, mime="text/csv")
    with st.expander("Saved experiment details"):
        st.json(record)


def feature_page():
    st.header("Feature reduction")
    path = ROOT / "results/metrics/features_20260915T140754_40734f02b041/results.json"
    report = json.loads(path.read_text())
    st.write("Training-only PCA, mutual information and forest importance; 2, 4 and 6 retained features.")
    st.info("Eight features/qubits are unsupported: seven eligible encoded inputs have PCA rank six. All balance fields, IDs and existing detection flags are excluded.")
    st.bar_chart(pd.DataFrame({"mutual information": report["mutual_information_scores"]}))
    st.bar_chart(pd.DataFrame({"forest importance": report["forest_importance_scores"]}))
    with st.expander("Reduction experiments and matched linear controls"):
        st.json(report["controls"])


def main():
    st.set_page_config(page_title="PaySim fraud research", layout="wide")
    st.title("PaySim fraud research")
    page = st.sidebar.radio("Page", ["Live simulation", *catalog(), "Feature reduction"])
    st.sidebar.caption("Physical IBM execution is pending credentials. All displayed quantum results are simulations.")
    if page == "Live simulation":
        live_page()
    elif page == "Feature reduction":
        feature_page()
    else:
        research_page(page)
