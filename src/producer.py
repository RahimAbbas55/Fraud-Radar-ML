"""
Transaction replay producer.

Reads the credit card dataset and publishes each row as an individual
message to the `transactions` Kafka topic, with a small delay between
sends to simulate transactions arriving in near-real-time rather than
all at once.

Run as: python -m src.producer
    or: python -m src.producer --delay 0.05 --limit 1000
"""

import argparse
import time
from kafka import KafkaProducer
from src.data_loader import load_and_validate
from src.features import add_time_of_day_feature
from src.kafka_utils import KAFKA_BOOTSTRAP_SERVERS, TRANSACTIONS_TOPIC, json_serializer


def build_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=json_serializer,
    )

def build_message(row_index, row) -> dict:
    """
        Convert a single pandas row into a JSON-serializable message dict.(For testing without overloading kafka.)
    """
    message = row.to_dict()
    message["transaction_id"] = int(row_index)
    return message

def replay_transactions(producer: KafkaProducer, delay: float, limit: int | None):
    df = load_and_validate()
    df = add_time_of_day_feature(df)
    if limit is not None:
        df = df.head(limit)

    print(f"Replaying {len(df)} transactions to '{TRANSACTIONS_TOPIC}' "
          f"(delay={delay}s between messages)...")

    sent_count = 0
    for idx, row in df.iterrows():
        message = build_message(idx, row)
        producer.send(TRANSACTIONS_TOPIC, value=message)
        sent_count += 1
        # Flush periodically (not every message — that would be slow)
        # so we get confirmation messages are actually being delivered,
        # not just queued locally and silently lost.
        if sent_count % 100 == 0:
            producer.flush()
            print(f"  ...sent {sent_count} transactions")
        time.sleep(delay)
    producer.flush()  # final flush to catch any remaining unsent messages
    print(f"Done. Sent {sent_count} transactions total.")


def main():
    parser = argparse.ArgumentParser(description="Replay transactions to Kafka.")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Seconds to wait between sending each transaction (default: 0.05).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only replay the first N transactions (default: all 284,807 — "
             "useful for quick testing before running the full dataset).",
    )
    args = parser.parse_args()

    producer = build_producer()
    replay_transactions(producer, args.delay, args.limit)
    producer.close()


if __name__ == "__main__":
    main()