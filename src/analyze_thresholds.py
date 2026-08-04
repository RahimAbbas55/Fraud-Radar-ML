"""
Validates the decision layer's threshold choices (ALLOW_THRESHOLD,
BLOCK_THRESHOLD) against the real Day 2 test set, rather than trusting
they're sensible purely from reasoning. 
Run as:
    python -m src.analyze_thresholds
"""

import pandas as pd

from src.data_loader import load_and_validate
from src.features import add_time_of_day_feature, train_test_split_stratified
from src.train import load_model
from src.decision import make_decision


def main():
    print("Loading data and model...")
    df = load_and_validate()
    df = add_time_of_day_feature(df)
    X_train, X_test, y_train, y_test = train_test_split_stratified(df)

    model = load_model("xgboost")
    expected_features = model.get_booster().feature_names
    X_test_ordered = X_test[expected_features]

    probabilities = model.predict_proba(X_test_ordered)[:, 1]

    print(f"Scoring {len(X_test)} test transactions through the decision layer...")
    decisions = []
    for i, (idx, row) in enumerate(X_test.iterrows()):
        features = row.to_dict()
        result = make_decision(features, float(probabilities[i]))
        decisions.append({
            "decision": result["decision"],
            "true_class": int(y_test.iloc[i]),
            "rule_fired": len(result["fired_rules"]) > 0,
        })

    results_df = pd.DataFrame(decisions)

    print("\n=== Decision band distribution (all transactions) ===")
    print(results_df["decision"].value_counts())

    print("\n=== Decision band vs actual fraud/non-fraud ===")
    print(pd.crosstab(results_df["decision"], results_df["true_class"],
                       margins=True, margins_name="Total"))

    print("\n=== Recall by band (of all 98 real fraud cases, where did they land?) ===")
    fraud_only = results_df[results_df["true_class"] == 1]
    print(fraud_only["decision"].value_counts())

    print("\n=== How many transactions had a rule fire? ===")
    print(results_df["rule_fired"].value_counts())


if __name__ == "__main__":
    main()