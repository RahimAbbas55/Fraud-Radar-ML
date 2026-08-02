"""
-> Tests for src/train.py.
    Uses small mock/fake models rather than actually training XGBoost or
    Isolation Forest — these tests check that evaluate_model() correctly
    handles the output-format difference between the two model types, not
    whether the models themselves perform well (that requires real data
    and belongs in the notebook comparison, not a fast unit test).
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from src.train import evaluate_model

"""
    XGBoost path should call predict() and predict_proba(), and use
    predict_proba's second column directly as the score (no inversion).
"""
def test_evaluate_model_xgboost_path():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1, 0, 1])
    mock_model.predict_proba.return_value = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.7, 0.3],
        [0.1, 0.9],
    ])

    y_test = pd.Series([0, 1, 0, 1])
    X_test = pd.DataFrame({"feature": [1, 2, 3, 4]})

    results = evaluate_model(mock_model, "xgboost", X_test, y_test)

    # Predictions match y_test exactly, so precision/recall should be perfect
    assert results["precision"] == 1.0
    assert results["recall"] == 1.0
    mock_model.predict_proba.assert_called_once()

"""
    Isolation Forest path should convert -1/1 predictions to 0/1, and
    use the negated decision_function as the score.
"""
def test_evaluate_model_isolation_forest_path():
    mock_model = MagicMock()
    # -1 = anomaly (our "1"/fraud), 1 = normal (our "0"/not fraud)
    mock_model.predict.return_value = np.array([1, -1, 1, -1])
    mock_model.decision_function.return_value = np.array([0.5, -0.3, 0.4, -0.6])

    y_test = pd.Series([0, 1, 0, 1])
    X_test = pd.DataFrame({"feature": [1, 2, 3, 4]})

    results = evaluate_model(mock_model, "isolation_forest", X_test, y_test)

    assert results["precision"] == 1.0
    assert results["recall"] == 1.0
    mock_model.decision_function.assert_called_once()
    # predict_proba should never be called for Isolation Forest —
    # it doesn't have that method in real scikit-learn usage
    mock_model.predict_proba.assert_not_called()