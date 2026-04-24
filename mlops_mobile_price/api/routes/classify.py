from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.deps import get_classification_service
from api.services.classification_service import ClassificationService


router = APIRouter()


class ClassifyRequest(BaseModel):
    features: dict[str, Any]


@router.post("/classify")
def classify(request: ClassifyRequest, service: ClassificationService = Depends(get_classification_service)):
    return service.classify(request.features)
