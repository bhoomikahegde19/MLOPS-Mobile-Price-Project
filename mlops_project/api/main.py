from __future__ import annotations

from fastapi import FastAPI

from api.routes.classify import router as classify_router
from api.routes.ner import router as ner_router
from api.routes.predict import router as predict_router
from api.routes.stream import router as stream_router


app = FastAPI(title="Mobile Phone Price Prediction API", version="1.0.0")
app.include_router(predict_router)
app.include_router(classify_router)
app.include_router(ner_router)
app.include_router(stream_router)


@app.post("/health")
def health_check():
    return {"status": "healthy"}
