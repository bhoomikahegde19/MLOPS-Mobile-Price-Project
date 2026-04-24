from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..deps import get_ner_service
from ..services.ner_service import NERService


router = APIRouter()


class NERRequest(BaseModel):
    text: str


@router.post("/ner")
def run_ner(request: NERRequest, service: NERService = Depends(get_ner_service)):
    return service.extract_entities(request.text)
