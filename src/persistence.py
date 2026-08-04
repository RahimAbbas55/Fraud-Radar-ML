"""
    SQLite persistence for scored fraud detection results.
"""

import sqlite3
from pathlib import Path

from src.config import ROOT_DIR

DB_PATH = ROOT_DIR / "data" / "fraud_scores.db"

"""
    Create the scored_transactions table if it doesn't already exist.
"""
def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scored_transactions (
            transaction_id INTEGER PRIMARY KEY,
            true_class INTEGER,
            fraud_probability REAL NOT NULL,
            decision TEXT NOT NULL,
            fired_rules TEXT,
            scored_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

"""
    Insert one scored transaction result into the database.
"""
def save_scored_result(result: dict, db_path: Path = DB_PATH) -> None:
    import json
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT OR REPLACE INTO scored_transactions
            (transaction_id, true_class, fraud_probability, decision, fired_rules)
        VALUES (?, ?, ?, ?, ?)
    """, (
        result["transaction_id"],
        result["true_class"],
        result["fraud_probability"],
        result["decision"],
        json.dumps(result["fired_rules"]),  # store as JSON text, since SQLite has no list type
    ))
    conn.commit()
    conn.close()

"""
    Retrieve the most recently scored transactions flagged as fraud
    (prediction == 1), most recent first.
"""
def get_flagged_transactions(db_path: Path = DB_PATH, limit: int = 50) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # lets us access columns by name, not just index
    rows = conn.execute("""
        SELECT * FROM scored_transactions
        WHERE decision IN ('block', 'review')
        ORDER BY scored_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()

    return [dict(row) for row in rows]