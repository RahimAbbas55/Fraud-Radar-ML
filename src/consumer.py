"""
    -> Real-time transaction scoring consumer.
        This file contains a long-running Kafka consumer that continuously listens to the
        `transactions` topic and scores each incoming message using the trained XGBoost model.
        This is a long-running service, not a one-shot script — it runs until manually stopped.
    Run as: python -m src.consumer
"""
from kafka import KafkaConsumer
from kafka import KafkaConsumer, KafkaProducer
from src.config import TARGET_COL
from src.train import load_model
from src.kafka_utils import (
    KAFKA_BOOTSTRAP_SERVERS,
    TRANSACTIONS_TOPIC,
    FRAUD_SCORES_TOPIC,
    json_deserializer,
    json_serializer,
)
from src.persistence import init_db, save_scored_result
from src.decision import make_decision
import pandas as pd
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
        auto_offset_reset="earliest", 
    )
"""
    Producer used to publish scored results downstream. Kept separate
    from build_consumer() since it's a genuinely different Kafka client
    role (producing, not consuming), even though both live in this file.
"""
def build_score_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=json_serializer,
    )
"""
    Strip non-feature fields (transaction_id, the real Class label)
    from a raw Kafka message, leaving only what the model expects.
"""
def extract_features(message: dict) -> dict:

    return {k: v for k, v in message.items() if k not in NON_FEATURE_FIELDS}

"""
    Score a single transaction message and return a result dict
    combining the original transaction_id with the model's prediction.
"""
def score_transaction(model, message: dict) -> dict:
    features = extract_features(message)
    X = pd.DataFrame([features])
    expected_features = model.get_booster().feature_names
    missing = set(expected_features) - set(X.columns)
    if missing:
        raise ValueError(f"Message is missing required features: {missing}")
    X = X[expected_features]

    fraud_probability = model.predict_proba(X)[:, 1][0]
    decision = make_decision(features, fraud_probability)

    return {
        "transaction_id": message.get("transaction_id"),
        "true_class": message.get(TARGET_COL),
        "fraud_probability": float(fraud_probability),
        "decision": decision["decision"],
        "fired_rules": decision["fired_rules"],
    }


def main():
    print("Loading XGBoost model...")
    model = load_model("xgboost")

    print("Initializing database...")
    init_db()

    print(f"Connecting to Kafka, subscribing to '{TRANSACTIONS_TOPIC}'...")
    consumer = build_consumer()
    score_producer = build_score_producer()

    print("Listening for transactions (Ctrl+C to stop)...\n")
    try:
        for message in consumer:
            result = score_transaction(model, message.value)
            save_scored_result(result)  # persist the scored result to the database
            # Publish the scored result downstream, so other services
            # (persistence, a future API, a dashboard) can consume
            # fully-scored results without needing the model themselves.
            score_producer.send(FRAUD_SCORES_TOPIC, value=result)

            decision = result["decision"]
            flag = {"block": "🚨 BLOCK", "review": "⚠️  REVIEW", "allow": "  allow "}[decision]
            rules_note = f" [{len(result['fired_rules'])} rule(s) fired]" if result["fired_rules"] else ""
            print(f"[{flag}] txn_id={result['transaction_id']:>6} "
                  f"prob={result['fraud_probability']:.4f} "
                  f"(actual class={result['true_class']}){rules_note}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        consumer.close()
        score_producer.flush()  # ensure any pending scored messages are sent, before the process actually exits
        score_producer.close()

if __name__ == "__main__":
    main()