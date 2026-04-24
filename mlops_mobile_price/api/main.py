from __future__ import annotations

from fastapi import FastAPI

from api.routes.classify import router as classify_router
from api.routes.health import router as health_router
from api.routes.ner import router as ner_router
from api.routes.predict import router as predict_router


app = FastAPI(title="Mobile Phone Price Prediction API", version="1.0.0")
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(classify_router)
app.include_router(ner_router)
