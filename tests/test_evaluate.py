"""
-> Tests for src/evaluate.py.
Uses small, hand-constructed y_true/y_pred arrays with known, obvious
outcomes — not random or real data — so the expected metric values are
predictable and verifiable by hand.
"""
import pytest
from src.evaluate import evaluate_predictions, print_confusion_summary, compare_models

def test_perfect_prediction_score_1():
    y_true = [0, 0, 1, 1, 0, 1]
    y_pred = [0, 0, 1, 1, 0, 1]  # identical to y_true

    result = evaluate_predictions(y_true, y_pred, model_name="perfect")

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0

def test_model_predicting_all_zeros_does_not_crash():
    y_true = [0, 0, 1, 1, 0, 1]
    y_pred = [0, 0, 0, 0, 0, 0]  # never predicts fraud

    # Without zero_division=0, this would raise a warning/error rather
    # than a clean 0.0 — this test confirms that's handled.
    result = evaluate_predictions(y_true, y_pred, model_name="all_zeros")

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0

def test_pr_auc_only_computed_when_scores_provided():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]

    result_without_scores = evaluate_predictions(y_true, y_pred, model_name="no_scores")
    assert "pr_auc" not in result_without_scores

    result_with_scores = evaluate_predictions(
        y_true, y_pred, y_scores=[0.1, 0.2, 0.9, 0.8], model_name="with_scores"
    )
    assert "pr_auc" in result_with_scores

def test_compare_models_builds_correct_table():
    results = [
        {"model": "A", "precision": 0.8, "recall": 0.7, "f1": 0.75},
        {"model": "B", "precision": 0.6, "recall": 0.9, "f1": 0.72},
    ]

    table = compare_models(results)

    assert list(table.index) == ["A", "B"]
    assert table.loc["A", "precision"] == 0.8
    assert table.loc["B", "recall"] == 0.9