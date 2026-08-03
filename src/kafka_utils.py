"""
->  Shared Kafka connection configuration.
    Centralizing the broker address and serialization logic here means
    producer.py and consumer.py both connect the same way — if the broker
    address ever changes, there's one place to fix it, not two.
"""
import json

# Matches the port exposed in docker-compose.yml (Stage 1) and the
# advertised listener Kafka itself reports to clients.
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TRANSACTIONS_TOPIC = "transactions"
FRAUD_SCORES_TOPIC = "fraud-scores"


"""
->  Convert a Python dict into bytes for sending over Kafka.
    Kafka only transports raw bytes — it has no concept of a Python
    object. JSON is chosen here for simplicity and human-readability
    (you can eyeball a raw message and understand it), not for
    performance — a production system at higher throughput would
    likely use a binary format like Avro or Protobuf instead.
"""
def json_serializer(data: dict) -> bytes:
    return json.dumps(data).encode("utf-8")

"""Convert bytes received from Kafka back into a Python dict."""
def json_deserializer(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"))