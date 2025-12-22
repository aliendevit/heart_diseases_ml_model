# src/train_model.py
# Loads and preprocesses the data
# Trains a final Logistic Regression model (with scaling via Pipeline)
# Saves artifact to models/heart_best_model.joblib

from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from preprocess import load_and_preprocess, DEFAULT_TARGET_COL


def main():
    # 1) Paths (robust: works from anywhere)
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "heart.csv"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # One true artifact name (use the same in predict_single.py and api_app.py)
    model_path = models_dir / "heart_best_model.joblib"

    # 2) Load and preprocess data
    df, X, y, feature_cols = load_and_preprocess(data_path, target_col=DEFAULT_TARGET_COL)

    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Number of features after encoding: {X.shape[1]}")

    # 3) Model as a Pipeline (prevents leakage + improves LogReg performance)
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=5000,
            solver="lbfgs"
        )),
    ])

    # 4) Cross-validation (better than accuracy-only)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"acc": "accuracy", "auc": "roc_auc", "f1": "f1"}

    cv_out = cross_validate(model, X, y, cv=cv, scoring=scoring)

    acc_mean = cv_out["test_acc"].mean()
    acc_std = cv_out["test_acc"].std()
    auc_mean = cv_out["test_auc"].mean()
    auc_std = cv_out["test_auc"].std()
    f1_mean = cv_out["test_f1"].mean()
    f1_std = cv_out["test_f1"].std()

    print("CV results:")
    print(f"  Accuracy: {acc_mean:.3f} ± {acc_std:.3f}")
    print(f"  ROC-AUC:  {auc_mean:.3f} ± {auc_std:.3f}")
    print(f"  F1:       {f1_mean:.3f} ± {f1_std:.3f}")

    # 5) Train final model on ALL data
    model.fit(X, y)

    # 6) Save artifact (model + metadata)
    artifact = {
        "model": model,
        "feature_cols": list(feature_cols),  # ensure JSON-like serializable
        "target_col": DEFAULT_TARGET_COL,
        "model_name": "LogisticRegression+Scaler",
        "cv": {
            "n_splits": 5,
            "random_state": 42,
            "accuracy_mean": float(acc_mean),
            "accuracy_std": float(acc_std),
            "roc_auc_mean": float(auc_mean),
            "roc_auc_std": float(auc_std),
            "f1_mean": float(f1_mean),
            "f1_std": float(f1_std),
        },
    }

    joblib.dump(artifact, model_path)
    print(f"Saved model artifact to: {model_path}")


if __name__ == "__main__":
    main()
