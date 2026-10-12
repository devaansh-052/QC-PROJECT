import pytest
from fastapi.testclient import TestClient
from quantum_fraud.api.app import create_app, Transaction
from quantum_fraud.inference.simulation import transactions


class FixturePredictor:
    name, run_id, threshold = "test-fixture", "not-research", .8

    def predict(self, transaction):
        assert set(transaction) == {"transaction_id", "step", "type", "amount"}
        return {"fraud_probability": .9, "predicted_class": 1, "risk_level": "HIGH", "decision": "REVIEW", "model": self.name, "processing_ms": 1., "simulation": True}


def test_api_prediction_journal_and_duplicate(tmp_path):
    app = create_app(FixturePredictor(), tmp_path / "journal.sqlite3")
    with TestClient(app) as client:
        assert client.get("/health").json()["simulation"]
        payload = transactions(1)[0]
        response = client.post("/predict", json=payload)
        assert response.status_code == 200 and response.json()["decision"] == "REVIEW"
        assert client.post("/predict", json=payload).status_code == 409
        assert len(client.get("/transactions").json()["transactions"]) == 1
        assert client.get("/transactions?limit=1000").status_code == 422
    with TestClient(create_app(FixturePredictor(), tmp_path / "journal.sqlite3")) as restored:
        assert len(restored.get("/transactions").json()["transactions"]) == 1


@pytest.mark.parametrize("change", [{"amount": -1}, {"amount": "nan"}, {"step": True}, {"type": "UNKNOWN"}, {"isFraud": 1}, {"oldbalanceOrg": 123}])
def test_api_rejects_invalid_or_leaking_inputs(tmp_path, change):
    with TestClient(create_app(FixturePredictor(), tmp_path / "journal.sqlite3")) as client:
        assert client.post("/predict", json={**transactions(1)[0], **change}).status_code == 422


def test_simulator_repeatability_and_validity():
    first = transactions(20, seed=42)
    assert first == transactions(20, seed=42)
    assert first != transactions(20, seed=43)
    assert len({r["transaction_id"] for r in first}) == 20
    for row in first:
        Transaction.model_validate(row)
