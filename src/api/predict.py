"""Model loading + single-patient inference with SHAP explanation."""
import json
from functools import lru_cache

import joblib
import pandas as pd

from src.config import MODELS_DIR
from src.explain.shap_utils import explain_one


@lru_cache(maxsize=1)
def _load():
    pipe = joblib.load(MODELS_DIR / "pipeline.joblib")
    threshold = json.loads(
        (MODELS_DIR / "threshold.json").read_text()
    )["threshold"]
    return pipe, threshold


def predict_patient(features: dict) -> dict:
    pipe, threshold = _load()
    X = pd.DataFrame([features])

    proba = float(pipe.predict_proba(X)[:, 1][0])
    drivers = explain_one(pipe, X, top_k=5)

    return {
        "risk_score": proba,
        "prediction": int(proba >= threshold),
        "threshold": threshold,
        "top_5_drivers": drivers,
    }
