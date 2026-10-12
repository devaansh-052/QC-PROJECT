"""FastAPI demonstration; no real payment processing."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from quantum_fraud.inference.predictor import PROJECT_ROOT, SavedPredictor, serving_config


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    transaction_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    step: int = Field(ge=1, le=743)
    type: Literal["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
    amount: float = Field(ge=0, le=1e12)


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS transactions (sequence INTEGER PRIMARY KEY AUTOINCREMENT, transaction_id TEXT UNIQUE NOT NULL, record TEXT NOT NULL)")

    def insert(self, record):
        with sqlite3.connect(self.path, timeout=10) as connection:
            connection.execute("INSERT INTO transactions(transaction_id, record) VALUES (?, ?)", (record["transaction_id"], json.dumps(record, allow_nan=False)))

    def recent(self, limit):
        with sqlite3.connect(self.path) as connection:
            return [json.loads(row[0]) for row in connection.execute("SELECT record FROM transactions ORDER BY sequence DESC LIMIT ?", (limit,))]


def create_app(predictor=None, database=None):
    config = serving_config(os.getenv("QUANTUM_FRAUD_SERVING_CONFIG"))
    predictor = predictor or SavedPredictor(config)
    journal = Journal(database or os.getenv("QUANTUM_FRAUD_DATABASE") or PROJECT_ROOT / config["database"])
    app = FastAPI(title="PaySim Fraud Research API", version="0.1.0", description="Local research simulation. Scores are uncalibrated; no real funds are processed.")

    @app.get("/health")
    def health():
        return {"status": "ready", "simulation": True, "model": predictor.name, "model_run": predictor.run_id, "threshold": predictor.threshold}

    @app.post("/predict")
    def predict(transaction: Transaction):
        payload = transaction.model_dump()
        record = {**payload, **predictor.predict(payload), "created_utc": datetime.now(timezone.utc).isoformat()}
        try:
            journal.insert(record)
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail="transaction_id already exists") from exc
        return record

    @app.get("/transactions")
    def transactions(limit: int = Query(default=50, ge=1, le=500)):
        return {"transactions": journal.recent(limit), "simulation": True}

    return app
