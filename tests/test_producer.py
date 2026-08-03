"""
    Tests for src/producer.py.
    This pytest file only tests the build_message function, which converts a pandas row into a JSON-serializable dict.
    The actual Kafka send/flush calls require a running broker and aren't unit-tested here.
"""

import pandas as pd
import numpy as np
import pytest
from src.producer import build_message


def test_build_message_includes_transaction_id():
    row = pd.Series({"Amount": 149.99, "V1": -1.23, "Class": 0})
    message = build_message(42, row)
    assert message["transaction_id"] == 42
    assert message["Amount"] == 149.99
    assert message["Class"] == 0


def test_build_message_transaction_id_is_plain_python_int():
    """
    Guards against the numpy-int-vs-python-int serialization gotcha
    described in Stage 4 — json.dumps() cannot handle numpy int types
    directly, so transaction_id must come back as a plain Python int.
    """
    row = pd.Series({"Amount": 10.0})
    # Simulate a numpy int64 index, as a real pandas DataFrame would produce
    numpy_index = np.int64(7)
    message = build_message(numpy_index, row)
    assert isinstance(message["transaction_id"], int)
    assert not isinstance(message["transaction_id"], np.integer)


def test_build_message_does_not_mutate_original_row():
    row = pd.Series({"Amount": 10.0, "Class": 1})
    original_row = row.copy()
    build_message(5, row)
    pd.testing.assert_series_equal(row, original_row)