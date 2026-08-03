"""
    -> Real-time transaction scoring consumer.
        This file contains a long-running Kafka consumer that continuously listens to the
        `transactions` topic and scores each incoming message using the trained XGBoost model.
        This is a long-running service, not a one-shot script — it runs until manually stopped.
    Run as: python -m src.consumer
"""
import pandas as pd
from kafka import KafkaConsumer
from src.config import TARGET_COL
from src.train import load_model
from src.kafka_utils import KAFKA_BOOTSTRAP_SERVERS, TRANSACTIONS_TOPIC, json_deserializer

'''
    Fields present in the raw Kafka message that are NOT model features —
    these must be stripped before scoring, or XGBoost will either error
    or silently misbehave on unexpected columns.
'''
NON_FEATURE_FIELDS = {"transaction_id", TARGET_COL}
def build_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        TRANSACTIONS_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=json_deserializer,
        auto_offset_reset="earliest",  # see explanation below
    )
def extract_features(message: dict) -> dict:
    """
        Strip non-feature fields (transaction_id, the real Class label)
        from a raw Kafka message, leaving only what the model expects.
    """
    return {k: v for k, v in message.items() if k not in NON_FEATURE_FIELDS}


def score_transaction(model, message: dict) -> dict:
    """
        Score a single transaction message and return a result dict
        combining the original transaction_id with the model's prediction.
    """
    features = extract_features(message)
    X = pd.DataFrame([features])
    fraud_probability = model.predict_proba(X)[:, 1][0]
    prediction = int(fraud_probability >= 0.5)
    return {
        "transaction_id": message.get("transaction_id"),
        "true_class": message.get(TARGET_COL),  # kept for our own evaluation, not shown to "production"
        "fraud_probability": float(fraud_probability),
        "prediction": prediction,
    }


def main():
    print("Loading XGBoost model...")
    model = load_model("xgboost")
    print(f"Connecting to Kafka, subscribing to '{TRANSACTIONS_TOPIC}'...")
    consumer = build_consumer()
    print("Listening for transactions (Ctrl+C to stop)...\n")
    try:
        for message in consumer:
            result = score_transaction(model, message.value)
            flag = "🚨 FRAUD" if result["prediction"] == 1 else "  ok"
            print(f"[{flag}] txn_id={result['transaction_id']:>6} "
                  f"prob={result['fraud_probability']:.4f} "
                  f"(actual class={result['true_class']})")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()