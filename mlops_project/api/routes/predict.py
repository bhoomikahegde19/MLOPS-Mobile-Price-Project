from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..deps import get_prediction_service
from ..services.prediction_service import PredictionService


router = APIRouter()


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Feature dictionary matching the training schema.")


@router.post("/predict")
def predict(request: PredictionRequest, service: PredictionService = Depends(get_prediction_service)):
    return service.predict_price(request.features)
