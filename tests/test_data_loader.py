"""
Tests for src/data_loader.py.

Uses small synthetic dataframes rather than the real 284K-row dataset —
these tests check validation *logic*, not the real data itself, so they
should stay fast and not depend on the Kaggle CSV being present.
"""

import pandas as pd
import pytest

from src.data_loader import validate_schema, get_class_distribution


def _make_valid_df(n=10):
    data = {"Time": range(n), "Amount": [10.0] * n, "Class": [0] * (n - 1) + [1]}
    for i in range(1, 29):
        data[f"V{i}"] = [0.1] * n
    return pd.DataFrame(data)


def test_validate_schema_passes_on_well_formed_data():
    df = _make_valid_df()
    validate_schema(df)  # should not raise


def test_validate_schema_raises_on_missing_column():
    df = _make_valid_df().drop(columns=["V1"])
    with pytest.raises(ValueError, match="missing expected columns"):
        validate_schema(df)


def test_validate_schema_raises_on_missing_values():
    df = _make_valid_df()
    df.loc[0, "Amount"] = None
    with pytest.raises(ValueError, match="missing values"):
        validate_schema(df)


def test_validate_schema_raises_on_unexpected_class_values():
    df = _make_valid_df()
    df.loc[0, "Class"] = 2
    with pytest.raises(ValueError, match="Unexpected values"):
        validate_schema(df)


def test_get_class_distribution_normalizes_correctly():
    df = _make_valid_df(n=10)  # 9 zeros, 1 one
    dist = get_class_distribution(df)
    assert dist[0] == pytest.approx(0.9)
    assert dist[1] == pytest.approx(0.1)