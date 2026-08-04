"""
FastAPI app for synchronous fraud scoring, alongside the Kafka pipeline.
Reuses score_transaction from consumer.py — same scoring logic, same
decision output, whether a transaction arrives via Kafka or via this API.
"""

from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from src.train import load_model
from src.features import add_time_of_day_feature
from src.consumer import score_transaction
_state = {}  # holds the loaded model across requests

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading XGBoost model...")
    _state["model"] = load_model("xgboost")
    yield
    _state.clear()

app = FastAPI(title="Fraud Radar API", lifespan=lifespan)
class TransactionInput(BaseModel):
    # Raw fields only — hour_of_day is derived server-side, not sent by the caller.
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

@app.post("/score")
def score(transaction: TransactionInput):
    model = _state["model"]

    df = pd.DataFrame([transaction.model_dump()])
    df = add_time_of_day_feature(df)  # same derivation the producer applies
    message = df.iloc[0].to_dict()

    return score_transaction(model, message)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _state}