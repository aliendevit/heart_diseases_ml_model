import pandas as pd
import numpy as np

def build_X_single_like_api(payload: dict, feature_cols):
    df = pd.DataFrame([payload])
    X = df.reindex(columns=feature_cols, fill_value=0)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X

def test_alignment_shape(feature_cols):
    payload = {"age": 54, "chol": 246}  # intentionally incomplete
    X = build_X_single_like_api(payload, feature_cols)
    assert X.shape[0] == 1
    assert X.shape[1] == len(feature_cols)

def test_alignment_column_order(feature_cols):
    payload = {"chol": 246, "age": 54}  # reversed order
    X = build_X_single_like_api(payload, feature_cols)
    assert list(X.columns) == list(feature_cols)

def test_alignment_no_nans(feature_cols):
    payload = {"age": 54, "chol": 246}
    X = build_X_single_like_api(payload, feature_cols)
    assert not np.isnan(X.to_numpy()).any()

def test_invalid_value_becomes_nan(feature_cols):
    payload = {"age": "abc", "chol": 246}
    X = build_X_single_like_api(payload, feature_cols)
    # If 'age' is a real feature column, it will become NaN
    if "age" in X.columns:
        assert X["age"].isna().any()
def test_model_predicts_one_row(model, feature_cols):
    import pandas as pd

    payload = {"age": 54, "chol": 246}
    X = pd.DataFrame([payload]).reindex(columns=feature_cols, fill_value=0).apply(pd.to_numeric, errors="coerce")

    pred = model.predict(X)
    assert len(pred) == 1
    assert int(pred[0]) in (0, 1)
