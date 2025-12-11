# src/predict_single.py
# Loads the saved model artifact

# Constructs a one-row DataFrame for a new patient

# Applies the same feature engineering (age_group + get_dummies)

# Aligns columns with feature_cols

# Predicts class and probability
from pathlib import Path
import joblib
import pandas as pd

from preprocess import encode_features, DEFAULT_TARGET_COL


def load_model_artifact(model_path: str | Path):
    model_path = Path(model_path)
    artifact = joblib.load(model_path)
    return artifact


def build_input_dataframe():
    """
    For now, we hard-code one example patient.
    Later, this could be replaced by command-line input, API data, etc.
    """
    # Example patient data (keys MUST match raw CSV columns except target)
    # Adjust the field names to match your heart.csv columns exactly.
    data = {
        "age": 57,
        "sex": 1,
        "cp": 0,
        "trestbps": 130,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 140,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 2,
        "ca": 0,
        "thal": 2,
        # DO NOT include "target" here – we are predicting it
    }

    df_input = pd.DataFrame([data])  # one-row DataFrame
    return df_input


def preprocess_single(df_input: pd.DataFrame, feature_cols: list[str]):
    """
    Apply the same encoding as in training, and align columns to feature_cols.
    """
    # We need a dummy target col just to reuse encode_features
    # We'll drop it after encoding.
    df_input_with_dummy_target = df_input.copy()
    df_input_with_dummy_target[DEFAULT_TARGET_COL] = 0  # placeholder

    # encode_features will add age_group and one-hot encode it
    X_encoded, y_dummy, encoded_feature_cols = encode_features(
        df_input_with_dummy_target,
        target_col=DEFAULT_TARGET_COL
    )

    # Now X_encoded has some columns, but we must match the training feature_cols:
    # - All columns in feature_cols must exist
    # - If some columns are missing in X_encoded (e.g. dummy not created), we add them as 0

    for col in feature_cols:
        if col not in X_encoded.columns:
            X_encoded[col] = 0

    # And if X_encoded has extra columns not in feature_cols, we drop them
    X_encoded = X_encoded[feature_cols]

    return X_encoded


def main():
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "heart_logreg_model.joblib"


    artifact = load_model_artifact(model_path)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    # 1) Build an example input
    df_input = build_input_dataframe()
    print("Raw input:")
    print(df_input)

    # 2) Preprocess to match training features
    X_single = preprocess_single(df_input, feature_cols)

    # 3) Predict class and probability
    y_pred = model.predict(X_single)[0]
    y_proba = model.predict_proba(X_single)[0, 1]  # probability of class 1

    print("\nModel prediction:")
    print(f"Predicted class (target): {y_pred}")
    print(f"Predicted probability of class 1 (disease): {y_proba:.3f}")


if __name__ == "__main__":
    main()
