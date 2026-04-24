from __future__ import annotations

from functools import lru_cache

from api.services.classification_service import ClassificationService
from api.services.ner_service import NERService
from api.services.prediction_service import PredictionService


@lru_cache
def get_prediction_service() -> PredictionService:
    return PredictionService()


@lru_cache
def get_classification_service() -> ClassificationService:
    return ClassificationService()


@lru_cache
def get_ner_service() -> NERService:
    return NERService()
