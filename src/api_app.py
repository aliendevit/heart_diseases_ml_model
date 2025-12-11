# src/api_app.py
"""
FastAPI app that exposes the heart disease model as a simple API.
"""

from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from preprocess import encode_features, DEFAULT_TARGET_COL


# ---------- 1. Load model artifact at startup ----------

project_root = Path(__file__).resolve().parents[1]
model_path = project_root / "models" / "heart_logreg_model.joblib"

artifact = joblib.load(model_path)
model = artifact["model"]
feature_cols: List[str] = artifact["feature_cols"]
target_col: str = artifact["target_col"]


# ---------- 2. Input schema (Pydantic model) ----------

class PatientInput(BaseModel):
    # Adjust field names/types to exactly match your heart.csv columns (except target)
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


# ---------- 3. Helper: preprocess for a single patient ----------

def preprocess_single(df_input: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Apply the same encoding as in training, and align columns to feature_cols.
    This logic mirrors what you used in predict_single.py.
    """
    # add dummy target to reuse encode_features
    df_input_with_dummy_target = df_input.copy()
    df_input_with_dummy_target[DEFAULT_TARGET_COL] = 0  # placeholder label

    X_encoded, _, _ = encode_features(
        df_input_with_dummy_target,
        target_col=DEFAULT_TARGET_COL,
    )

    # ensure all training feature columns exist
    for col in feature_cols:
        if col not in X_encoded.columns:
            X_encoded[col] = 0

    # keep only the columns used in training, in correct order
    X_encoded = X_encoded[feature_cols]

    return X_encoded


# ---------- 4. FastAPI app and endpoints ----------

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Simple API wrapping a Logistic Regression heart disease model.",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    """
    Basic health check endpoint.
    """
    return {"status": "ok", "model": "logistic_regression", "n_features": len(feature_cols)}


@app.post("/predict")
def predict(patient: PatientInput):
    """
    Predict heart disease for a single patient.
    """
    # Convert input to DataFrame
    df_input = pd.DataFrame([patient.dict()])

    # Preprocess to match training features
    X_single = preprocess_single(df_input, feature_cols)

    # Get prediction & probability
    y_pred = int(model.predict(X_single)[0])
    y_proba = float(model.predict_proba(X_single)[0, 1])  # prob of class 1

    return {
        "input": patient.dict(),
        "predicted_class": y_pred,
        "probability_class_1": y_proba,
    }
