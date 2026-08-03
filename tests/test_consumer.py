"""
    -> Tests for src/consumer.py.
        - extract_features() is tested with plain dicts (no dependencies).
        - score_transaction() is tested against the REAL saved XGBoost model
        - (models/xgboost.joblib), not a mock — the goal here is verifying the
        - raw-message-to-model-input wiring actually works end-to-end, which a mocked model couldn't confirm.
    - Known limitation: these tests require models/xgboost.joblib to exist
    (.joblib files are gitignored). Run `python -m src.train --model
    xgboost` first if these fail with a FileNotFoundError. 
"""

import pytest
import pandas as pd
from src.consumer import extract_features, score_transaction
from src.train import load_model

"""
    Build a complete, realistic message dict matching what the real
    producer would send: all V1-V28, Amount, Time, hour_of_day, plus
    the two non-feature fields the consumer must strip out.
"""
def _make_full_transaction_message(transaction_id=1, true_class=0):
    message = {f"V{i}": 0.1 for i in range(1, 29)}
    message.update({
        "Time": 5000.0,
        "Amount": 25.0,
        "hour_of_day": 1,
        "Class": true_class,
        "transaction_id": transaction_id,
    })
    return message


def test_extract_features_removes_non_feature_fields():
    message = _make_full_transaction_message(transaction_id=7, true_class=1)
    features = extract_features(message)
    assert "transaction_id" not in features
    assert "Class" not in features


def test_extract_features_preserves_actual_features():
    message = _make_full_transaction_message()
    features = extract_features(message)
    assert features["Amount"] == 25.0
    assert features["V1"] == 0.1
    assert features["hour_of_day"] == 1


"""
    Loads the real trained model once per test module, not once per
    test — model loading isn't free, and every test in this file that
    needs it can share the same loaded instance.
"""
@pytest.fixture(scope="module")
def xgboost_model():
    return load_model("xgboost")


def test_score_transaction_returns_expected_shape(xgboost_model):
    message = _make_full_transaction_message(transaction_id=99, true_class=0)
    result = score_transaction(xgboost_model, message)
    assert result["transaction_id"] == 99
    assert result["true_class"] == 0
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["prediction"] in (0, 1)

"""
    Confirms the 0.5 threshold logic in score_transaction is wired
    correctly: prediction should be 1 exactly when probability >= 0.5.
"""
def test_score_transaction_prediction_matches_probability_threshold(xgboost_model):
    message = _make_full_transaction_message()
    result = score_transaction(xgboost_model, message)
    expected_prediction = int(result["fraud_probability"] >= 0.5)
    assert result["prediction"] == expected_prediction