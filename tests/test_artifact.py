def test_artifact_has_required_keys(artifact):
    assert "model" in artifact
    assert "feature_cols" in artifact
    assert "target_col" in artifact

def test_feature_cols_look_valid(feature_cols):
    assert isinstance(feature_cols, (list, tuple))
    assert len(feature_cols) > 0
    # Columns should be strings
    assert all(isinstance(c, str) for c in feature_cols)
