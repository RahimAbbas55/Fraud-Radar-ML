"""
Reproducible model training script.

Run as: python -m src.train --model xgboost
    or: python -m src.train --model isolation_forest

Supports XGBoost and Isolation Forest only. Logistic Regression was a
useful baseline in the EDA/comparison notebook (02_baseline_models.ipynb)
but isn't a realistic production candidate given its precision at the
default threshold (see Day 2 notebook comparison), so it isn't wired
into this script.
"""

import argparse
import joblib
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from src.config import MODELS_DIR, RANDOM_STATE
from src.data_loader import load_and_validate
from src.features import add_time_of_day_feature, train_test_split_stratified
from src.evaluate import evaluate_predictions, print_confusion_summary


def train_xgboost(X_train, y_train):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Training XGBoost (scale_pos_weight={scale_pos_weight:.1f})...")

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_estimators=100,
    )
    model.fit(X_train, y_train)
    return model


def train_isolation_forest(X_train, y_train, contamination):
    X_train_normal = X_train[y_train == 0]
    print(f"Training Isolation Forest on {len(X_train_normal)} non-fraud "
          f"transactions (contamination={contamination})...")

    model = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_normal)
    return model


def evaluate_model(model, model_type, X_test, y_test):
    """
    Evaluation differs by model type: XGBoost outputs probabilities
    directly via predict_proba; Isolation Forest needs its -1/1 output
    converted to our 0/1 convention, and its decision_function negated
    so higher score consistently means "more fraud-like" across models.
    """
    if model_type == "xgboost":
        pred = model.predict(X_test)
        scores = model.predict_proba(X_test)[:, 1]
    else:  # isolation_forest
        raw_pred = model.predict(X_test)
        pred = (raw_pred == -1).astype(int)
        scores = -model.decision_function(X_test)

    results = evaluate_predictions(y_test, pred, scores, model_name=model_type)
    print(results)
    print_confusion_summary(y_test, pred, model_name=model_type)
    return results


def main():
    parser = argparse.ArgumentParser(description="Train a fraud detection model.")
    parser.add_argument(
        "--model",
        choices=["xgboost", "isolation_forest"],
        required=True,
        help="Which model to train.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.0017,
        help="Isolation Forest only: expected fraction of anomalies. "
             "Ignored for XGBoost.",
    )
    args = parser.parse_args()

    print("Loading and validating data...")
    df = load_and_validate()
    df = add_time_of_day_feature(df)
    X_train, X_test, y_train, y_test = train_test_split_stratified(df)

    if args.model == "xgboost":
        model = train_xgboost(X_train, y_train)
    else:
        model = train_isolation_forest(X_train, y_train, args.contamination)

    evaluate_model(model, args.model, X_test, y_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / f"{args.model}.joblib"
    joblib.dump(model, output_path)
    print(f"\nModel saved to {output_path}")


if __name__ == "__main__":
    main()