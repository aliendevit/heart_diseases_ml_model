def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_endpoint_returns_prediction(client):
    payload = {"features": {"age": 54, "chol": 246}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert data["prediction"] in (0, 1)

def test_predict_schema_error_returns_422(client):
    # Missing "features" wrapper
    r = client.post("/predict", json={"age": 54})
    assert r.status_code == 422

def test_predict_bad_value_returns_400(client):
    payload = {"features": {"age": "abc", "chol": 246}}
    r = client.post("/predict", json=payload)
    # If your API converts to numeric + checks NaN, it should be 400
    # If you haven't implemented that check, it may be 200 or 500.
    assert r.status_code in (400, 422)
