"""
-> Shared model evaluation utilities.
    Using one evaluation function for every model (Logistic Regression,
    XGBoost, Isolation Forest) guarantees a fair, consistent comparison —
    if each model's metrics were computed slightly differently, the
    comparison in the notebook wouldn't be trustworthy.
"""

import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

"""
    Compute precision, recall, F1, and (if scores are provided) PR-AUC
    for a set of predictions.

    y_true: true labels (0/1)
    y_pred: hard predictions (0/1) — the model's final decision
    y_scores: continuous risk scores or probabilities, used for PR-AUC.
              Optional because Isolation Forest's raw anomaly score
              needs separate handling (see Stage 6).
"""
def evaluate_predictions(y_true, y_pred, y_scores=None, model_name="model"):
    results = {
        "model": model_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_scores is not None:
        results["pr_auc"] = average_precision_score(y_true, y_scores)

    return results

"""
    Print a plain-language breakdown of the confusion matrix — not just
    raw numbers, but what each count actually means in a fraud context.
"""
def print_confusion_summary(y_true, y_pred, model_name="model"):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"--- {model_name} ---")
    print(f"True Positives  (fraud correctly caught):        {tp}")
    print(f"False Negatives (fraud missed — real loss):      {fn}")
    print(f"False Positives (legit txns wrongly flagged):    {fp}")
    print(f"True Negatives  (legit txns correctly cleared):  {tn}")


def compare_models(results_list):
    """Combine multiple evaluate_predictions() outputs into one table."""
    return pd.DataFrame(results_list).set_index("model")