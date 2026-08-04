"""
SHAP-based explainability. Computed only for review/block decisions —
avoids per-prediction latency cost on the majority of allow decisions.
"""

import numpy as np
import shap
_explainer_cache = {}

def get_explainer(model):
    # TreeExplainer is fast/exact for tree models; cache since building it isn't free.
    model_id = id(model)
    if model_id not in _explainer_cache:
        _explainer_cache[model_id] = shap.TreeExplainer(model)
    return _explainer_cache[model_id]

def explain_transaction(model, X, top_n: int = 5) -> list[dict]:
    # Returns top_n features by |SHAP value|, handles both old/new shap return formats.
    explainer = get_explainer(model)
    raw_shap_values = explainer.shap_values(X)
    if isinstance(raw_shap_values, list):
        row_shap = np.array(raw_shap_values[1])[0]  # older shap: class-1 (fraud) array
    else:
        row_shap = np.array(raw_shap_values)[0]  # modern shap: plain array
    feature_names = X.columns.tolist()
    contributions = [
        {"feature": name, "shap_value": float(value)}
        for name, value in zip(feature_names, row_shap)
    ]
    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    return contributions[:top_n]