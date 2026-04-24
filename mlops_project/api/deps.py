from __future__ import annotations

from functools import lru_cache

from .services.ner_service import NERService
from .services.prediction_service import PredictionService


@lru_cache
def get_prediction_service() -> PredictionService:
    return PredictionService()


@lru_cache
def get_ner_service() -> NERService:
    return NERService()
