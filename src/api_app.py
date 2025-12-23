# src/api_app.py
"""
FastAPI app that exposes the heart disease model as a simple API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.schemas import PredictRequest, PredictResponse

app = FastAPI(title="Heart Risk Prediction API", version="1.0.0")

# ---------- 1) Load artifact at startup ----------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = PROJECT_ROOT / "models" / "heart_best_model.joblib"

try:
    artifact = joblib.load(ARTIFACT_PATH)
except FileNotFoundError:
    # Fail fast with a clear error (better than mysterious 500 later)
    raise RuntimeError(
        f"Model artifact not found at: {ARTIFACT_PATH}. "
        "Run: python src/train_model.py"
    )

model = artifact["model"]
feature_cols = artifact["feature_cols"]


# ---------- 2) Helper: convert + align features ----------

def build_X_single(features: dict) -> pd.DataFrame:
    """
    Convert incoming 'features' dict into a 1-row DataFrame aligned to training columns.

    Steps:
    1) DataFrame([features]) makes a single row
    2) reindex(columns=feature_cols, fill_value=0) enforces same columns/order as training
    3) to_numeric(coerce) ensures everything becomes float/int; non-numeric => NaN
    4) if NaN exists => reject with 400 (bad request)
    """
    df_input = pd.DataFrame([features])

    # Align to training schema: unknown keys dropped, missing keys added with 0
    X = df_input.reindex(columns=feature_cols, fill_value=0)

    # Force numeric so scaler/model won't crash on strings
    X = X.apply(pd.to_numeric, errors="coerce")

    # If conversion created NaN, input had invalid/non-numeric values
    if np.isnan(X.to_numpy()).any():
        bad_cols = X.columns[X.isna().any()].tolist()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid/non-numeric values for columns: {bad_cols}. Send numbers only."
        )

    return X


def predict_with_optional_proba(X: pd.DataFrame) -> Tuple[int, float | None]:
    """
    Predict class and optional probability (if model supports predict_proba).
    """
    pred = int(model.predict(X)[0])

    proba_1 = None
    if hasattr(model, "predict_proba"):
        proba_1 = float(model.predict_proba(X)[0][1])

    return pred, proba_1


# ---------- 3) Endpoints ----------

@app.get("/health")
def health():
    """
    Health check endpoint for monitoring.
    """
    return {"status": "ok"}


@app.get("/version")
def version():
    """
    Returns artifact/model metadata useful for debugging.
    """
    return {
        "api_version": app.version,
        "model_name": artifact.get("model_name", type(model).__name__),
        "n_features": len(feature_cols),
        "artifact_path": str(ARTIFACT_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict class label (0/1). Optionally includes probability if available.
    """
    X = build_X_single(req.features)
    pred, proba_1 = predict_with_optional_proba(X)
    return PredictResponse(prediction=pred, proba_1=proba_1)


@app.post("/predict_proba", response_model=PredictResponse)
def predict_proba(req: PredictRequest):
    """
    Always tries to return probability. If model does not support it, returns proba_1=None.
    """
    X = build_X_single(req.features)
    pred, proba_1 = predict_with_optional_proba(X)
    return PredictResponse(prediction=pred, proba_1=proba_1)