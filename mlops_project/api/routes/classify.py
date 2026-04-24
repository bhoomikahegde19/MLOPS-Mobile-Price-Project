from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..deps import get_prediction_service
from ..services.prediction_service import PredictionService


router = APIRouter()


class ClassificationRequest(BaseModel):
    features: dict[str, Any]


@router.post("/classify")
def classify(request: ClassificationRequest, service: PredictionService = Depends(get_prediction_service)):
    return service.classify_price_band(request.features)
