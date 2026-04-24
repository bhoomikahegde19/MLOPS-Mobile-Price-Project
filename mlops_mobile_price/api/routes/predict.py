from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.deps import get_prediction_service
from api.services.prediction_service import PredictionService


router = APIRouter()


class PredictRequest(BaseModel):
    features: dict[str, Any]


@router.post("/predict")
def predict(request: PredictRequest, service: PredictionService = Depends(get_prediction_service)):
    return service.predict(request.features)
