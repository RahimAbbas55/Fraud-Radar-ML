"""
Tests for src/api.py.
Uses FastAPI's TestClient to exercise the real app (including lifespan
startup, which loads the actual XGBoost model) without needing a live
uvicorn server. 

<REQUIREMENTS>
Requires models/xgboost.joblib to exist, same limitation as test_consumer.py and test_explain.py.
"""

import pytest
from fastapi.testclient import TestClient
from src.api import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def _base_payload(amount=50.0, time=5000.0):
    payload = {f"V{i}": 0.1 for i in range(1, 29)}
    payload["Time"] = time
    payload["Amount"] = amount
    return payload

def test_health_endpoint_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

def test_score_low_amount_normal_hour_is_allowed_with_no_explanation(client):
    payload = _base_payload(amount=50.0, time=5000.0)  # hour_of_day = 1, not unusual
    response = client.post("/score", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "allow"
    assert body["fired_rules"] == []
    assert body["explanation"] is None

def test_score_high_amount_triggers_review_with_explanation(client):
    payload = _base_payload(amount=5000.0)
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "review"
    assert len(body["fired_rules"]) == 1
    assert body["fired_rules"][0]["rule"] == "high_amount"
    assert body["explanation"] is not None
    assert len(body["explanation"]) == 5

def test_score_missing_required_field_returns_422(client):
    incomplete_payload = {"Amount": 50.0}  # missing all V1-V28, Time
    response = client.post("/score", json=incomplete_payload)
    assert response.status_code == 422

def test_score_response_has_null_transaction_id_and_true_class(client):
    payload = _base_payload(amount=50.0)
    response = client.post("/score", json=payload)
    body = response.json()

    assert body["transaction_id"] is None
    assert body["true_class"] is None