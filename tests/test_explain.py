"""
Tests for src/explain.py.
Uses the REAL trained XGBoost model (same approach as test_consumer.py)
since SHAP's TreeExplainer needs a real model's tree structure — a
mock wouldn't tell us anything meaningful here.
an expectation for).
<LIMITATIONS>
    Requires models/xgboost.joblib to exist (gitignored).
Run `python -m src.train --model xgboost` first if these fail.
"""

import pandas as pd
import pytest
from src.explain import explain_transaction, get_explainer, _explainer_cache
from src.train import load_model

@pytest.fixture(scope="module")
def xgboost_model():
    return load_model("xgboost")

@pytest.fixture
def sample_features(xgboost_model):
    expected = xgboost_model.get_booster().feature_names
    sample = {f: 0.1 for f in expected}
    sample["Amount"] = 5000.0
    sample["hour_of_day"] = 2
    return pd.DataFrame([sample])[expected]

def test_explain_transaction_returns_top_n_results(xgboost_model, sample_features):
    result = explain_transaction(xgboost_model, sample_features, top_n=5)
    assert len(result) == 5

def test_explain_transaction_respects_custom_top_n(xgboost_model, sample_features):
    result = explain_transaction(xgboost_model, sample_features, top_n=3)
    assert len(result) == 3

def test_explain_transaction_results_sorted_by_magnitude(xgboost_model, sample_features):
    result = explain_transaction(xgboost_model, sample_features, top_n=10)
    magnitudes = [abs(r["shap_value"]) for r in result]
    assert magnitudes == sorted(magnitudes, reverse=True)

def test_explain_transaction_each_result_has_expected_keys(xgboost_model, sample_features):
    result = explain_transaction(xgboost_model, sample_features, top_n=1)
    assert set(result[0].keys()) == {"feature", "shap_value"}

def test_explainer_is_cached_across_calls(xgboost_model):
    _explainer_cache.clear()  # ensure a clean state for this test
    explainer_1 = get_explainer(xgboost_model)
    explainer_2 = get_explainer(xgboost_model)
    assert explainer_1 is explainer_2  # same object, not rebuilt