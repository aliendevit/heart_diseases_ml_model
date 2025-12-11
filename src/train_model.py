# src/train_model.py
# Loads and preprocesses the data
# Trains your final Logistic Regression model
# Saves it to models/heart_logreg_model.joblib

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from preprocess import load_and_preprocess, DEFAULT_TARGET_COL


def main():
    # 1) Paths
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "heart.csv"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Name of the model file we will save
    model_path = models_dir / "heart_logreg_model.joblib"

    # 2) Load and preprocess data
    df, X, y, feature_cols = load_and_preprocess(
        data_path,
        target_col=DEFAULT_TARGET_COL
    )
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Number of features after encoding: {X.shape[1]}")

    # 3) Define Logistic Regression model
    log_reg = LogisticRegression(max_iter=1000)

    # 4) Cross-validation to confirm performance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(log_reg, X, y, cv=cv, scoring="accuracy")

    print("Logistic Regression CV scores:", [f"{s:.3f}" for s in cv_scores])
    print(f"Mean CV accuracy: {cv_scores.mean():.3f}")
    print(f"Std: {cv_scores.std():.3f}")

    # 5) Train final model on ALL data
    log_reg.fit(X, y)

    # 6) Save model + metadata (feature names, target column)
    artifact = {
        "model": log_reg,
        "feature_cols": feature_cols,
        "target_col": DEFAULT_TARGET_COL,
    }

    joblib.dump(artifact, model_path)
    print(f"Saved model artifact to: {model_path}")


if __name__ == "__main__":
    main()
