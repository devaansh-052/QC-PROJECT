"""Render all real research pages and exercise the UI against the saved model."""
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch
import quantum_fraud
from quantum_fraud.api.app import create_app
from quantum_fraud.dashboard.research import ROOT, catalog, read_run, score_data
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest


def main():
    output = ROOT / "results/simulation" / (datetime.now(timezone.utc).strftime("dashboard_check_%Y%m%dT%H%M%S_") + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    checks = []
    with TestClient(create_app(database=output / "transactions.sqlite3")) as client:
        def service(method, route, **kwargs):
            response = client.request(method, route, **kwargs)
            response.raise_for_status()
            return response.json()
        with patch("quantum_fraud.dashboard.app.api_request", service):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=60)
            assert not app.exception
            app.button[0].click().run(timeout=60)
            assert not app.exception and len(app.metric) == 3
            assert len(app.dataframe[0].value) == 1
            for page, runs in catalog().items():
                app.sidebar.radio[0].set_value(page).run(timeout=60)
                for run in runs:
                    app.selectbox[0].set_value(run).run(timeout=60)
                    assert not app.exception, str(app.exception)
                    path, report = read_run(run)
                    for record in report["models"]:
                        score_data(path, report, record)
                    checks.append({"page": page, "run": run, "models_checked": len(report["models"])})
            app.sidebar.radio[0].set_value("Feature reduction").run(timeout=60)
            assert not app.exception
    (output / "verification.json").write_text(json.dumps({"status": "complete", "live_saved_model_prediction": True, "research_pages": checks, "feature_page": True}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
