"""
Kafka consumer throughput benchmark. Times how long the consumer takes
to process a batch of transactions already sitting in the topic —
measures real sustained throughput, not a synthetic estimate.

Usage: run producer first to populate the topic, then run this against
the SAME transaction_id range to know exactly how many messages to expect.

Run as: python -m src.benchmark_kafka --expected 500
"""

import argparse
import time
from src.train import load_model
from src.persistence import init_db
from src.consumer import build_consumer, build_score_producer, score_transaction
from src.kafka_utils import TRANSACTIONS_TOPIC, FRAUD_SCORES_TOPIC
from src.persistence import save_scored_result


def main():
    parser = argparse.ArgumentParser(description="Benchmark Kafka consumer throughput.")
    parser.add_argument("--expected", type=int, required=True,
                         help="Number of messages to consume before reporting and exiting.")
    args = parser.parse_args()

    print("Loading model and connecting to Kafka...")
    model = load_model("xgboost")
    init_db()
    consumer = build_consumer()
    score_producer = build_score_producer()

    print(f"Waiting to consume {args.expected} messages...")
    processed = 0
    start = time.perf_counter()

    try:
        for message in consumer:
            result = score_transaction(model, message.value)
            save_scored_result(result)
            score_producer.send(FRAUD_SCORES_TOPIC, value=result)
            processed += 1

            if processed >= args.expected:
                break
    finally:
        elapsed = time.perf_counter() - start
        consumer.close()
        score_producer.flush()
        score_producer.close()

        print(f"\n=== Consumer throughput ({processed} messages) ===")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Throughput: {processed / elapsed:.1f} messages/sec")


if __name__ == "__main__":
    main()