# tests/conftest.py
from pathlib import Path
import sys
import pytest
import joblib
from fastapi.testclient import TestClient

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACT_PATH = PROJECT_ROOT / "models" / "heart_best_model.joblib"

@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT

@pytest.fixture(scope="session")
def artifact():
    assert ARTIFACT_PATH.exists(), f"Artifact not found: {ARTIFACT_PATH}. Run train_model.py first."
    return joblib.load(ARTIFACT_PATH)

@pytest.fixture(scope="session")
def model(artifact):
    return artifact["model"]

@pytest.fixture(scope="session")
def feature_cols(artifact):
    return artifact["feature_cols"]

@pytest.fixture(scope="session")
def client():
    # Import here so sys.path has been set
    from src.api_app import app
    return TestClient(app)
