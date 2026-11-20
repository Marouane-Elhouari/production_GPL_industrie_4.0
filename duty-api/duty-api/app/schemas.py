from typing import Dict, List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Mesures du procede : nom de la feature -> valeur. Voir GET /model-info pour la liste et un exemple.",
    )


class PredictResponse(BaseModel):
    duty_HT1_kW: float
    model_name: str
    model_version: str
    test_r2: float
    test_mae_kW: float
    warnings: List[str] = []


class BatchRequest(BaseModel):
    items: List[Dict[str, float]] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    duty_HT1_kW: List[float]
    model_version: str


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    target: str
    features: List[str]
    ranges: Dict[str, List[float]]
    example: Dict[str, float]
    metrics: Dict
