"""
Tests for src/features.py.
Note: named test_features_module.py (not test_features.py) to avoid
colliding with the pre-existing placeholder test_features.py from the
initial scaffold.
"""

import pandas as pd
import pytest

from src.features import (
    add_time_of_day_feature,
    split_features_target,
    train_test_split_stratified,
)


def _make_imbalanced_df(n=1000, fraud_rate=0.05):
    n_fraud = int(n * fraud_rate)
    data = {
        "Time": range(n),
        "Amount": [10.0] * n,
        "Class": [1] * n_fraud + [0] * (n - n_fraud),
    }
    for i in range(1, 29):
        data[f"V{i}"] = [0.1] * n
    return pd.DataFrame(data)


def test_add_time_of_day_feature_range():
    df = _make_imbalanced_df(n=100)
    result = add_time_of_day_feature(df)
    assert "hour_of_day" in result.columns
    assert result["hour_of_day"].between(0, 23).all()


def test_split_features_target_shapes():
    df = _make_imbalanced_df(n=50)
    X, y = split_features_target(df)
    assert "Class" not in X.columns
    assert len(X) == len(y) == 50


def test_stratified_split_preserves_class_ratio():
    df = _make_imbalanced_df(n=1000, fraud_rate=0.05)
    X_train, X_test, y_train, y_test = train_test_split_stratified(df, test_size=0.2)

    train_ratio = y_train.mean()
    test_ratio = y_test.mean()

    # Both splits should closely match the original 5% fraud rate —
    # this is the whole point of stratification.
    assert train_ratio == pytest.approx(0.05, abs=0.01)
    assert test_ratio == pytest.approx(0.05, abs=0.01)