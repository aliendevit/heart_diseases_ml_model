# src/schemas.py
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    Flexible input: accepts any features as key/value pairs.
    We keep it flexible because your model's feature_cols may be many
    (especially after encoding). We'll validate values later.
    """
    features: Dict[str, Any] = Field(
        ...,
        description="Feature dictionary: {feature_name: value}. Values should be numeric."
    )


class PredictResponse(BaseModel):
    prediction: int
    proba_1: Optional[float] = None
